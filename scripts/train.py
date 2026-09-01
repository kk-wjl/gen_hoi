import json
import random
import sys
import time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Any
import setproctitle

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfg import Config, parse_args
from dataset import MotionLoader
from flow import LinearFlow
from models.dit import DiffusionTransformer1D
from models.unet import ConditionalUNet1D
from utils.checkpoint import load_training_checkpoint, save_training_checkpoint
from utils.optim import MuonAdamWWrapper


def seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_wandb(config: Config, run_dir: Path, payload: dict[str, Any]) -> Any | None:
    if not config.wandb.enabled:
        return None
    import wandb

    run_name = config.wandb.name or config.output.run_name or run_dir.name
    return wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=run_name,
        mode=config.wandb.mode,
        dir=str(run_dir),
        config=payload,
    )


def build_optimizer(config: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    if config.optim.name == "muon_adamw":
        return MuonAdamWWrapper(
            modules=[model],
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )


def update_process_title(
    *,
    run_name: str,
    train_start_time: float,
    completed_steps: int,
    total_steps: int,
) -> None:
    if completed_steps <= 0 or total_steps <= 0:
        return
    elapsed = max(time.perf_counter() - train_start_time, 0.0)
    seconds_per_step = elapsed / completed_steps
    remaining_steps = max(total_steps - completed_steps, 0)
    end_timestamp = time.time() + seconds_per_step * remaining_steps
    end_text = datetime.fromtimestamp(end_timestamp).strftime("%Y-%m-%d %H:%M")
    setproctitle.setproctitle(f"gen_hoi {run_name} end<={end_text}")


def build_model(config: Config, feature_dim: int) -> torch.nn.Module:
    if config.model.use_dit:
        model = DiffusionTransformer1D(
            input_dim=feature_dim,
            output_dim=feature_dim,
            hidden_size=config.model.hidden_size,
            depth=config.model.depth,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            max_seq_len=config.model.max_seq_len,
            dropout=config.model.dropout,
        )
    else:
        model = ConditionalUNet1D(
            input_dim=feature_dim,
            output_dim=feature_dim,
            base_channels=config.model.base_channels,
            channel_mults=tuple(config.model.channel_mults),
            cond_dim=config.model.cond_dim,
            dropout=config.model.dropout,
        )
    model.sample_shape = (config.dataset.seq_len, feature_dim)
    return model


def compute_metrics(
    *,
    flow: LinearFlow,
    dataset: MotionLoader,
    seed: int,
    eval_count: int,
    cond_steps: int,
    num_steps: int,
    device: torch.device,
) -> dict[str, float]:
    metric_sums = {
        "root_vel_fd_mse": 0.0,
        "joint_vel_fd_mse": 0.0,
        "object_vel_fd_mse": 0.0,
        "root_ang_vel_fd_mse": 0.0,
        "object_ang_vel_fd_mse": 0.0,
    }
    eval_rng = np.random.default_rng(seed)
    eval_indices = [
        int(index) for index in eval_rng.choice(len(dataset), size=eval_count, replace=False).tolist()
    ]
    if not eval_indices:
        return metric_sums

    with torch.inference_mode():
        for dataset_index in eval_indices:
            sample = dataset[dataset_index]
            chunk = sample["chunk"]
            _, start, _ = sample["meta"]

            cond_prefix = chunk.unsqueeze(0)[:, :cond_steps].to(device)
            pred_chunk = flow.sample(
                num_steps=num_steps,
                cond_prefix=cond_prefix,
                device=device,
            )[0].detach().cpu()

            pred_denorm = dataset.denormalize(pred_chunk) if dataset.normalize_enabled else pred_chunk
            anchor_frame = dataset.state_frames[start : start + 1].clone()
            anchor_pose = dataset.chunk_to_pose(anchor_frame)
            pred_world_chunk = dataset.accumulate_chunk_in_anchor_frame(
                pred_denorm,
                root0_pos=anchor_pose["root_pos"][0],
                root0_quat=anchor_pose["root_quat"][0],
                object0_pos=anchor_pose["object_pos"][0],
                object0_quat=anchor_pose["object_quat"][0],
            )
            metrics = dataset.metrics(pred_world_chunk, normalized=False)
            for key in metric_sums:
                metric_sums[key] += metrics[key]

    count = float(len(eval_indices))
    return {key: value / count for key, value in metric_sums.items()}

def main() -> None:
    config = parse_args()
    seed(config.train.seed)
    device = torch.device(config.train.device)
    root_dir = Path(config.output.root_dir)
    if config.output.run_name:
        run_name = config.output.run_name
    else:
        run_name = time.strftime("%Y-%m%d-%H%M")
    run_dir = root_dir / run_name
    checkpoint_dir = run_dir / "checkpoints"
    log_path = run_dir / "logs" / "train_metrics.jsonl"

    dataset = MotionLoader(
        npz_path=config.dataset.npz_path,
        seq_len=config.dataset.seq_len,
        stride=config.dataset.stride,
        make_relative=config.dataset.make_relative,
        normalize=config.dataset.normalize,
        use_vel=config.dataset.use_vel,
        use_stats_cache=config.dataset.use_stats_cache,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(config, dataset.feature_dim)
    model.to(device)
    num_parameters = sum(param.numel() for param in model.parameters())
    trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    optimizer = build_optimizer(config, model)
    flow = LinearFlow(
        model,
        noise_scale=config.flow.noise_scale,
        t_eps=config.flow.t_eps,
        condition=False,
    )
    flow.to(device)

    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **asdict(config),
        "data_files": [str(path) for path in dataset.npz_paths],
        "resolved_device": str(device),
        "feature_dim": dataset.feature_dim,
        "num_frames": dataset.num_frames,
        "num_windows": len(dataset),
        "run_dir": str(run_dir.resolve()),
        "num_parameters": num_parameters,
        "trainable_parameters": trainable_parameters,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "total_time_sec": None,
        "peak_gpu_memory_allocated_mb": None,
        "peak_gpu_memory_reserved_mb": None,
    }
    config_path = run_dir / "logs" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_stats_path = run_dir / "logs" / "dataset_stats.json"
    dataset_stats_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_stats_path.write_text(
        json.dumps(
            {
                "mean": dataset.state_mean.cpu().tolist(),
                "std": dataset.state_std.cpu().tolist(),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    wandb_run = setup_wandb(config, run_dir, run_config)

    start_epoch = 0
    latest_ckpt = checkpoint_dir / "latest.pt"
    train_start_time = time.perf_counter()
    total_steps = config.train.epochs * len(dataloader)
    completed_steps = 0
    peak_gpu_memory_allocated_mb = 0.0
    peak_gpu_memory_reserved_mb = 0.0
    previous_total_time_sec = 0.0
    previous_peak_gpu_memory_allocated_mb = 0.0
    previous_peak_gpu_memory_reserved_mb = 0.0
    if config.resume is not None:
        resume_path = Path(config.resume)
        resume_config_path = resume_path.resolve().parent.parent / "logs" / "config.json"
        if resume_config_path.is_file():
            try:
                previous_run_config = json.loads(resume_config_path.read_text(encoding="utf-8"))
            except Exception:
                previous_run_config = None
            if isinstance(previous_run_config, dict):
                previous_total_time_sec = float(previous_run_config.get("total_time_sec") or 0.0)
                previous_peak_gpu_memory_allocated_mb = float(
                    previous_run_config.get("peak_gpu_memory_allocated_mb") or 0.0
                )
                previous_peak_gpu_memory_reserved_mb = float(
                    previous_run_config.get("peak_gpu_memory_reserved_mb") or 0.0
                )
        start_epoch = load_training_checkpoint(resume_path, model, optimizer, map_location=device)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")
    elif latest_ckpt.is_file():
        start_epoch = load_training_checkpoint(latest_ckpt, model, optimizer, map_location=device)
        print(f"Auto-resumed from {latest_ckpt} at epoch {start_epoch}")
    completed_steps = start_epoch * len(dataloader)
    setproctitle.setproctitle(f"gen_hoi {run_name} end<=unknown")
    config_path.write_text(json.dumps(run_config, ensure_ascii=True, indent=2), encoding="utf-8")

    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_losses: list[float] = []
        epoch_grad_norms: list[float] = []
        progress = tqdm(
            enumerate(dataloader, start=1),
            total=len(dataloader),
            desc=f"epoch {epoch + 1}/{config.train.epochs}",
        )

        for step, batch in progress:
            chunks = batch["chunk"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = flow.compute_loss(x1=chunks, cond_steps=config.train.cond_steps)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip_norm)
            optimizer.step()

            loss_value = float(loss.detach().item())
            grad_norm_value = float(grad_norm)
            epoch_losses.append(loss_value)
            epoch_grad_norms.append(grad_norm_value)
            completed_steps += 1

            if step % config.train.log_every == 0 or step == len(dataloader):
                update_process_title(
                    run_name=run_name,
                    train_start_time=train_start_time,
                    completed_steps=completed_steps,
                    total_steps=total_steps,
                )
                progress.set_postfix(
                    loss=f"{loss_value:.5f}",
                )
        model.eval()
        metrics: dict[str, float | int] = {
            "epoch": epoch + 1,
            "loss": float(np.mean(epoch_losses)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": float(np.mean(epoch_grad_norms)),
            **compute_metrics(
                flow=flow,
                dataset=dataset,
                seed=config.train.seed,
                eval_count=6,
                cond_steps=config.train.cond_steps,
                num_steps=80,
                device=device,
            )
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=True) + "\n")
        print(json.dumps(metrics, ensure_ascii=True))
        if wandb_run is not None:
            epoch_end_step = (epoch + 1) * len(dataloader)
            wandb_run.log(metrics, step=epoch_end_step)

        save_training_checkpoint(
            latest_ckpt,
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            config=asdict(config),
        )
        if (epoch + 1) % config.train.save_every == 0:
            save_training_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                config=asdict(config),
            )

        if device.type == "cuda":
            peak_gpu_memory_allocated_mb = max(
                peak_gpu_memory_allocated_mb,
                torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                previous_peak_gpu_memory_allocated_mb,
            )
            peak_gpu_memory_reserved_mb = max(
                peak_gpu_memory_reserved_mb,
                torch.cuda.max_memory_reserved(device) / (1024 ** 2),
                previous_peak_gpu_memory_reserved_mb,
            )

    run_config["total_time_sec"] = time.perf_counter() - train_start_time + previous_total_time_sec
    setproctitle.setproctitle(f"gen_hoi {run_name} end<=done")
    if device.type == "cuda":
        run_config["peak_gpu_memory_allocated_mb"] = peak_gpu_memory_allocated_mb
        run_config["peak_gpu_memory_reserved_mb"] = peak_gpu_memory_reserved_mb
    config_path.write_text(json.dumps(run_config, ensure_ascii=True, indent=2), encoding="utf-8")

    if wandb_run is not None:
        wandb_run.summary["total_time_sec"] = run_config["total_time_sec"]
        wandb_run.summary["num_parameters"] = run_config["num_parameters"]
        wandb_run.summary["trainable_parameters"] = run_config["trainable_parameters"]
        wandb_run.summary["gpu_name"] = run_config["gpu_name"]
        wandb_run.summary["gpu_count"] = run_config["gpu_count"]
        wandb_run.summary["peak_gpu_memory_allocated_mb"] = run_config["peak_gpu_memory_allocated_mb"]
        wandb_run.summary["peak_gpu_memory_reserved_mb"] = run_config["peak_gpu_memory_reserved_mb"]
        wandb_run.finish()


if __name__ == "__main__":
    main()
