from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MotionLoader
from flow import LinearFlow
from model import DiffusionTransformer1D
from utils.checkpoint import load_training_checkpoint, save_training_checkpoint
from utils.optim import MuonAdamWWrapper


@dataclass
class DatasetConfig:
    npz_path: str = "data"
    seq_len: int = 32
    stride: int = 1
    make_relative: bool = True
    normalize: bool = True
    use_stats_cache: bool = True


@dataclass
class ModelConfig:
    hidden_size: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_seq_len: int = 256


@dataclass
class OptimConfig:
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0


@dataclass
class FlowConfig:
    noise_scale: float = 1.0
    t_eps: float = 1e-5


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 100
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    log_every: int = 10
    save_every: int = 5
    sample_every: int = 5
    cond_steps: int = 8
    sample_steps: int = 50


@dataclass
class OutputConfig:
    root_dir: str = "outputs"
    run_name: str = ""


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "gen-hoi"
    entity: str | None = None
    name: str = ""
    mode: str = "online"


@dataclass
class Config:
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    flow: FlowConfig = dataclasses.field(default_factory=FlowConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    output: OutputConfig = dataclasses.field(default_factory=OutputConfig)
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    resume: str | None = None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train the HOI diffusion transformer.")
    parser.add_argument("--data", default=DatasetConfig.npz_path)
    parser.add_argument("--output-root", default=OutputConfig.root_dir)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume", default=None)

    parser.add_argument("--seq-len", type=int, default=DatasetConfig.seq_len)
    parser.add_argument("--stride", type=int, default=DatasetConfig.stride)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default=TrainConfig.device)

    parser.add_argument("--hidden-size", type=int, default=ModelConfig.hidden_size)
    parser.add_argument("--depth", type=int, default=ModelConfig.depth)
    parser.add_argument("--num-heads", type=int, default=ModelConfig.num_heads)
    parser.add_argument("--mlp-ratio", type=float, default=ModelConfig.mlp_ratio)
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    parser.add_argument("--max-seq-len", type=int, default=ModelConfig.max_seq_len)

    parser.add_argument("--optimizer", choices=["adamw", "muon_adamw"], default=OptimConfig.name)
    parser.add_argument("--lr", type=float, default=OptimConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=OptimConfig.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=OptimConfig.grad_clip_norm)

    parser.add_argument("--noise-scale", type=float, default=FlowConfig.noise_scale)
    parser.add_argument("--t-eps", type=float, default=FlowConfig.t_eps)

    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--sample-every", type=int, default=TrainConfig.sample_every)
    parser.add_argument("--cond-steps", type=int, default=TrainConfig.cond_steps)
    parser.add_argument("--sample-steps", type=int, default=TrainConfig.sample_steps)

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=WandbConfig.project)
    parser.add_argument("--wandb-entity", default=WandbConfig.entity)
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=WandbConfig.mode)
    args = parser.parse_args()

    return Config(
        dataset=DatasetConfig(
            npz_path=args.data,
            seq_len=args.seq_len,
            stride=args.stride,
        ),
        model=ModelConfig(
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
        ),
        optim=OptimConfig(
            name=args.optimizer,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
        ),
        flow=FlowConfig(
            noise_scale=args.noise_scale,
            t_eps=args.t_eps,
        ),
        train=TrainConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            log_every=args.log_every,
            save_every=args.save_every,
            sample_every=args.sample_every,
            cond_steps=args.cond_steps,
            sample_steps=args.sample_steps,
        ),
        output=OutputConfig(
            root_dir=args.output_root,
            run_name=args.run_name,
        ),
        wandb=WandbConfig(
            enabled=args.wandb,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
        ),
        resume=args.resume,
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


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
    if config.train.cond_steps >= config.dataset.seq_len:
        raise ValueError("cond_steps must be smaller than seq_len")
    if config.model.max_seq_len < config.dataset.seq_len:
        raise ValueError("max_seq_len must be >= seq_len")

    seed_everything(config.train.seed)
    device = resolve_device(config.train.device)
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
    write_json(run_dir / "config.json", run_config)
    write_json(
        run_dir / "dataset_stats.json",
        {
            "mean": dataset.state_mean.cpu().tolist(),
            "std": dataset.state_std.cpu().tolist(),
        },
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

        append_jsonl(log_path, metrics)
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
