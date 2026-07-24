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
        "data_files": [str(path) for path in dataset.npz_paths],
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
    latest_ckpt = checkpoint_dir / "latest.pt"
    if config.resume is not None:
        resume_path = Path(config.resume)
        start_epoch = load_training_checkpoint(resume_path, model, optimizer, map_location=device)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")
    elif latest_ckpt.is_file():
        start_epoch = load_training_checkpoint(latest_ckpt, model, optimizer, map_location=device)
        print(f"Auto-resumed from {latest_ckpt} at epoch {start_epoch}")

    for epoch in range(start_epoch, config.train.epochs):
        model.train()
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

            if step % config.train.log_every == 0 or step == len(dataloader):
                progress.set_postfix(
                    loss=f"{loss_value:.5f}",
                )

        metrics: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train/loss": float(np.mean(epoch_losses)),
            "train/lr": float(optimizer.param_groups[0]["lr"]),
            "train/grad_norm": float(np.mean(epoch_grad_norms)),
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

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
