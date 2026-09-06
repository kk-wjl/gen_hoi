from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    npz_path: str = "data"
    seq_len: int = 32
    stride: int = 1
    make_relative: bool = True
    normalize: bool = True
    use_vel: bool = True
    use_contact_label: bool = False
    use_stats_cache: bool = True


@dataclass
class ModelConfig:
    use_dit: bool = True
    # dit config
    hidden_size: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_seq_len: int = 256
    # unet config
    base_channels: int = 128
    channel_mults: tuple[int, ...] = (1, 2, 4, 8)
    cond_dim: int = 256


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
    epochs: int = 80
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    log_every: int = 10 # terminal logs every N steps
    save_every: int = 10 # save checkpoints every N epochs
    cond_steps: int = 8


@dataclass
class OutputConfig:
    root_dir: str = "outputs"
    run_name: str = "Baseline"


@dataclass
class WandbConfig:
    enabled: bool = True
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


def config_from_dict(values: dict[str, object]) -> Config:
    def convert(cls: type, raw: dict[str, object]) -> object:
        instance = cls()
        for field in dataclasses.fields(instance):
            if field.name not in raw:
                continue
            current_value = getattr(instance, field.name)
            raw_value = raw[field.name]
            if dataclasses.is_dataclass(current_value) and isinstance(raw_value, dict):
                setattr(instance, field.name, convert(type(current_value), raw_value))
            else:
                setattr(instance, field.name, raw_value)
        return instance

    return convert(Config, values)  # type: ignore[return-value]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train the HOI diffusion transformer.")
    parser.add_argument("--data", default=DatasetConfig.npz_path)
    parser.add_argument("--output-root", default=OutputConfig.root_dir)
    parser.add_argument("--run-name", default=OutputConfig.run_name)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--seq-len", type=int, default=DatasetConfig.seq_len)
    parser.add_argument("--stride", type=int, default=DatasetConfig.stride)
    parser.add_argument(
        "--use-contact-label",
        action=argparse.BooleanOptionalAction,
        default=DatasetConfig.use_contact_label,
    )
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default=TrainConfig.device)

    parser.add_argument("--use-dit", action=argparse.BooleanOptionalAction, default=ModelConfig.use_dit)
    parser.add_argument("--hidden-size", type=int, default=ModelConfig.hidden_size)
    parser.add_argument("--depth", type=int, default=ModelConfig.depth)
    parser.add_argument("--num-heads", type=int, default=ModelConfig.num_heads)
    parser.add_argument("--mlp-ratio", type=float, default=ModelConfig.mlp_ratio)
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    parser.add_argument("--max-seq-len", type=int, default=ModelConfig.max_seq_len)
    parser.add_argument("--base-channels", type=int, default=ModelConfig.base_channels)
    parser.add_argument("--channel-mults", type=int, nargs="+", default=list(ModelConfig.channel_mults))
    parser.add_argument("--cond-dim", type=int, default=ModelConfig.cond_dim)

    parser.add_argument("--optimizer", choices=["adamw", "muon_adamw"], default=OptimConfig.name)
    parser.add_argument("--lr", type=float, default=OptimConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=OptimConfig.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=OptimConfig.grad_clip_norm)

    parser.add_argument("--noise-scale", type=float, default=FlowConfig.noise_scale)
    parser.add_argument("--t-eps", type=float, default=FlowConfig.t_eps)

    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--cond-steps", type=int, default=TrainConfig.cond_steps)

    parser.add_argument("--wandb", default=WandbConfig.enabled)
    parser.add_argument("--wandb-project", default=WandbConfig.project)
    parser.add_argument("--wandb-entity", default=WandbConfig.entity)
    parser.add_argument("--wandb-name", default=WandbConfig.name)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=WandbConfig.mode)
    args = parser.parse_args()

    return Config(
        dataset=DatasetConfig(
            npz_path=args.data,
            seq_len=args.seq_len,
            stride=args.stride,
            use_contact_label=args.use_contact_label,
        ),
        model=ModelConfig(
            use_dit=args.use_dit,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
            base_channels=args.base_channels,
            channel_mults=tuple(args.channel_mults),
            cond_dim=args.cond_dim,
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
            cond_steps=args.cond_steps,
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
