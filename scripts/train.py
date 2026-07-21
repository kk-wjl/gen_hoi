import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from model import DiffusionTransformer1D
from utils.checkpoint import load_training_checkpoint, save_training_checkpoint
from utils.optim import MuonAdamWWrapper


def seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_dir(config: Config) -> Path:
    root_dir = Path(config.output.root_dir)
    if config.output.run_name:
        run_name = config.output.run_name
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        data_path = Path(config.dataset.npz_path)
        data_stem = data_path.stem if data_path.suffix == ".npz" else data_path.name
        run_name = f"{stamp}_{data_stem}_T{config.dataset.seq_len}"
    return root_dir / run_name


def resolve_data_paths(npz_path: str) -> list[Path]:
    path = Path(npz_path).expanduser().resolve()
    if path.is_dir():
        npz_paths = sorted(path.glob("*.npz"))
        if not npz_paths:
            raise ValueError(f"No .npz files found under directory {path}")
        return npz_paths
    if path.is_file() and path.suffix == ".npz":
        return [path]
    raise ValueError(f"--data must point to a .npz file or a directory of .npz files, got {path}")


def setup_wandb(config: Config, run_dir: Path, payload: dict[str, Any]) -> Any | None:
    if not config.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb logging is enabled, but the 'wandb' package is not installed. "
            "Install dependencies and retry, or run without --wandb."
        ) from exc

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


def save_sample(
    *,
    flow: LinearFlow,
    dataset: MotionLoader,
    device: torch.device,
    batch: dict[str, Any],
    run_dir: Path,
    epoch: int,
    cond_steps: int,
    sample_steps: int,
) -> dict[str, float]:
    chunks = batch["chunk"][:1].to(device)
    cond_prefix = chunks[:, :cond_steps]
    pred = flow.sample(
        num_steps=sample_steps,
        cond_prefix=cond_prefix,
        device=device,
    )

    pred_cpu = pred.detach().cpu()
    target_cpu = chunks.detach().cpu()
    pred_denorm = dataset.denormalize(pred_cpu) if dataset.normalize_enabled else pred_cpu
    target_denorm = dataset.denormalize(target_cpu) if dataset.normalize_enabled else target_cpu
    metrics = dataset.metrics(pred_cpu, target_cpu, normalized=True)

    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sample_dir / f"epoch_{epoch:04d}.npz",
        pred=pred_cpu.numpy(),
        target=target_cpu.numpy(),
        pred_denorm=pred_denorm.numpy(),
        target_denorm=target_denorm.numpy(),
        cond_steps=np.asarray([cond_steps], dtype=np.int64),
        sample_steps=np.asarray([sample_steps], dtype=np.int64),
    )
    return metrics


def main() -> None:
    config = parse_args()
    seed(config.train.seed)
    device = torch.device(config.train.device)
    run_dir = build_run_dir(config)
    checkpoint_dir = run_dir / "checkpoints"
    log_path = run_dir / "logs" / "train_metrics.jsonl"
    data_paths = resolve_data_paths(config.dataset.npz_path)

    dataset = MotionLoader(
        npz_path=data_paths,
        seq_len=config.dataset.seq_len,
        stride=config.dataset.stride,
        make_relative=config.dataset.make_relative,
        normalize=config.dataset.normalize,
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

    model = DiffusionTransformer1D(
        input_dim=dataset.feature_dim,
        output_dim=dataset.feature_dim,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
    )
    model.sample_shape = (config.dataset.seq_len, dataset.feature_dim)
    model.to(device)

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
        "data_files": [str(path) for path in data_paths],
        "resolved_device": str(device),
        "feature_dim": dataset.feature_dim,
        "num_frames": dataset.num_frames,
        "num_windows": len(dataset),
        "run_dir": str(run_dir.resolve()),
    }
    config_path = run_dir / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(run_config, ensure_ascii=True, indent=2), encoding="utf-8")

    dataset_stats_path = run_dir / "dataset_stats.json"
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
    best_loss = float("inf")
    latest_ckpt = checkpoint_dir / "latest.pt"
    if config.resume is not None:
        resume_path = Path(config.resume)
        start_epoch = load_training_checkpoint(resume_path, model, optimizer, map_location=device)
        best_loss = torch.load(resume_path, map_location="cpu", weights_only=False).get("best_loss", best_loss)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")
    elif latest_ckpt.is_file():
        start_epoch = load_training_checkpoint(latest_ckpt, model, optimizer, map_location=device)
        best_loss = torch.load(latest_ckpt, map_location="cpu", weights_only=False).get("best_loss", best_loss)
        print(f"Auto-resumed from {latest_ckpt} at epoch {start_epoch}")

    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        epoch_losses: list[float] = []
        progress = tqdm(
            enumerate(dataloader, start=1),
            total=len(dataloader),
            desc=f"epoch {epoch + 1}/{config.train.epochs}",
        )
        last_batch: dict[str, Any] | None = None

        for step, batch in progress:
            chunks = batch["chunk"].to(device)
            last_batch = batch

            optimizer.zero_grad(set_to_none=True)
            loss = flow.compute_loss(x1=chunks, cond_steps=config.train.cond_steps)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip_norm)
            optimizer.step()

            loss_value = float(loss.detach().item())
            epoch_losses.append(loss_value)

            if step % config.train.log_every == 0 or step == len(dataloader):
                progress.set_postfix(
                    loss=f"{loss_value:.5f}",
                    grad=f"{float(grad_norm):.3f}",
                )

        epoch_loss = float(np.mean(epoch_losses))
        metrics: dict[str, float | int] = {
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        if last_batch is not None and (
            (epoch + 1) % config.train.sample_every == 0 or epoch + 1 == config.train.epochs
        ):
            model.eval()
            sample_metrics = save_sample(
                flow=flow,
                dataset=dataset,
                device=device,
                batch=last_batch,
                run_dir=run_dir,
                epoch=epoch + 1,
                cond_steps=config.train.cond_steps,
                sample_steps=config.train.sample_steps,
            )
            metrics.update(sample_metrics)

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=True) + "\n")
        print(json.dumps(metrics, ensure_ascii=True))
        if wandb_run is not None:
            wandb_run.log(metrics, step=epoch + 1)

        save_training_checkpoint(
            latest_ckpt,
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            config=asdict(config),
            extra={"best_loss": best_loss},
        )
        if (epoch + 1) % config.train.save_every == 0:
            save_training_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                config=asdict(config),
                extra={"best_loss": best_loss},
            )
        if best_loss == epoch_loss:
            save_training_checkpoint(
                checkpoint_dir / "best.pt",
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                config=asdict(config),
                extra={"best_loss": best_loss},
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
