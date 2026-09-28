"""Train the point-wise inner-hand object geometry representation."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometry.encoder import GeometryFieldEncoder
from utils.checkpoint import load_training_checkpoint, save_training_checkpoint


REQUIRED_KEYS = (
    "hand_inner_points_o", "hand_inner_signed_distance", "hand_inner_occupancy",
    "hand_inner_surface_normals_o", "hand_inner_valid", "motion_lengths",
)


class HandGeometryDataset(Dataset[dict[str, torch.Tensor]]):
    """Flatten valid ``(frame, query_point)`` pairs from motion-level splits."""

    def __init__(self, sources: list[dict[str, np.ndarray]], frame_refs: np.ndarray) -> None:
        self.sources = sources
        self.frame_refs = frame_refs
        self.points_per_frame = sources[0]["hand_inner_points_o"].shape[1]

    def __len__(self) -> int:
        return len(self.frame_refs) * self.points_per_frame

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frame_row, point_index = divmod(index, self.points_per_frame)
        source_index, frame_index = self.frame_refs[frame_row]
        source = self.sources[int(source_index)]
        return {
            "points_o": torch.from_numpy(source["hand_inner_points_o"][frame_index, point_index]).float(),
            "sdf": torch.tensor(source["hand_inner_signed_distance"][frame_index, point_index], dtype=torch.float32),
            "occupancy": torch.tensor(source["hand_inner_occupancy"][frame_index, point_index], dtype=torch.float32),
            "normal_o": torch.from_numpy(source["hand_inner_surface_normals_o"][frame_index, point_index]).float(),
        }


def load_data(paths: list[Path], val_fraction: float, seed: int) -> tuple[list[dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0, 1)")
    sources: list[dict[str, np.ndarray]] = []
    motions: list[tuple[int, int, int]] = []
    for source_index, path in enumerate(paths):
        with np.load(path, allow_pickle=True) as raw:
            missing = [key for key in REQUIRED_KEYS if key not in raw]
            if missing:
                raise KeyError(f"{path} is missing {missing}")
            source = {key: np.asarray(raw[key]) for key in REQUIRED_KEYS[:-1]}
            lengths = np.asarray(raw["motion_lengths"], dtype=np.int64)
        frames, points, xyz = source["hand_inner_points_o"].shape
        if xyz != 3 or source["hand_inner_signed_distance"].shape != (frames, points):
            raise ValueError(f"Unexpected geometry shapes in {path}")
        sources.append(source)
        start = 0
        for length in lengths.tolist():
            end = start + int(length)
            motions.append((source_index, start, end))
            start = end
        if start != frames:
            raise ValueError(f"motion_lengths in {path} sum to {start}, expected {frames}")
    if len(motions) < 2:
        raise ValueError("Need at least two motion segments for train/validation split")
    order = np.random.default_rng(seed).permutation(len(motions))
    target = val_fraction * sum(end - start for _, start, end in motions)
    val_ids: set[int] = set()
    count = 0
    for motion_id in order:
        if count >= target and val_ids:
            break
        val_ids.add(int(motion_id))
        _, start, end = motions[int(motion_id)]
        count += end - start
    def frames_for(ids: list[int]) -> np.ndarray:
        return np.asarray(
            [(file_id, frame) for motion_id in ids for file_id, start, end in [motions[motion_id]] for frame in range(start, end)],
            dtype=np.int64,
        )
    train = frames_for([index for index in range(len(motions)) if index not in val_ids])
    val = frames_for(sorted(val_ids))
    return sources, train, val


def loss_terms(model: GeometryFieldEncoder, batch: dict[str, torch.Tensor], args: argparse.Namespace) -> dict[str, torch.Tensor]:
    output = model(batch["points_o"])
    tsdf_target = (batch["sdf"] / args.tsdf_truncation).clamp(-1.0, 1.0)
    tsdf = F.smooth_l1_loss(output.tsdf, tsdf_target)
    positive = batch["occupancy"].sum()
    pos_weight = ((batch["occupancy"].numel() - positive) / positive.clamp_min(1.0)).clamp(max=100.0)
    occupancy = F.binary_cross_entropy_with_logits(output.occupancy_logits, batch["occupancy"], pos_weight=pos_weight)
    near = batch["sdf"].abs() < args.near_distance
    normal = (1.0 - (output.normals_o[near] * batch["normal_o"][near]).sum(-1)).mean() if near.any() else output.tsdf.sum() * 0.0
    return {"total": tsdf + args.occupancy_weight * occupancy + args.normal_weight * normal,
            "tsdf": tsdf, "occupancy": occupancy, "normal": normal}


def run_epoch(model: GeometryFieldEncoder, loader: DataLoader, optimizer: torch.optim.Optimizer | None, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {name: 0.0 for name in ("total", "tsdf", "occupancy", "normal")}
    count = 0
    for batch in tqdm(loader, leave=False, desc="train" if training else "val"):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.set_grad_enabled(training):
            losses = loss_terms(model, batch, args)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
        size = batch["sdf"].numel()
        count += size
        for name, value in losses.items():
            sums[name] += float(value.detach()) * size
    return {name: value / count for name, value in sums.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the smalltable geometry-field encoder.")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "hand")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "representation")
    parser.add_argument("--run-name", default="smalltable_field")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=40); parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0); parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--tsdf-truncation", type=float, default=0.05); parser.add_argument("--near-distance", type=float, default=0.05)
    parser.add_argument("--occupancy-weight", type=float, default=0.2); parser.add_argument("--normal-weight", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1); parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.tsdf_truncation <= 0 or args.near_distance <= 0:
        raise ValueError("TSDF truncation and near distance must be positive")
    if args.epochs < 1 or args.batch_size < 1 or args.save_every < 1:
        raise ValueError("epochs, batch size, and save interval must be positive")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_data_paths(data_path: Path) -> list[Path]:
    paths = [data_path] if data_path.suffix == ".npz" else sorted(data_path.glob("*_hand.npz"))
    if not paths:
        raise FileNotFoundError(f"No *_hand.npz files found under {data_path}")
    return paths


def build_dataloaders(
    paths: list[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, int, int]:
    sources, train_frames, val_frames = load_data(paths, args.val_fraction, args.seed)
    train_set = HandGeometryDataset(sources, train_frames)
    val_set = HandGeometryDataset(sources, val_frames)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(train_set, shuffle=True, **loader_args),
        DataLoader(val_set, shuffle=False, **loader_args),
        len(train_set),
        len(val_set),
    )


def prepare_run(
    args: argparse.Namespace,
    paths: list[Path],
    device: torch.device,
    train_samples: int,
    val_samples: int,
) -> tuple[Path, Path, Path, dict[str, object]]:
    run_dir = args.output_root / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    log_path = run_dir / "logs" / "metrics.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {
        **vars(args),
        "data": str(args.data),
        "output_root": str(args.output_root),
        "resume": None if args.resume is None else str(args.resume),
        "data_files": [str(path.resolve()) for path in paths],
        "device": str(device),
        "train_samples": train_samples,
        "val_samples": val_samples,
    }
    (log_path.parent / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return run_dir, checkpoint_dir, log_path, config


def load_resume_epoch(
    resume_path: Path | None,
    latest_path: Path,
    model: GeometryFieldEncoder,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if resume_path is not None:
        return load_training_checkpoint(resume_path, model, optimizer, map_location=device)
    if latest_path.is_file():
        return load_training_checkpoint(latest_path, model, optimizer, map_location=device)
    return 0


def train(
    model: GeometryFieldEncoder,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_dir: Path,
    log_path: Path,
    config: dict[str, object],
    start_epoch: int,
) -> None:
    latest_path = checkpoint_dir / "latest.pt"
    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, args, device)
        val_metrics = run_epoch(model, val_loader, None, args, device)
        metrics = {
            "epoch": epoch + 1,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        print(json.dumps(metrics))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        save_training_checkpoint(latest_path, epoch=epoch + 1, model=model, optimizer=optimizer, config=config)
        if (epoch + 1) % args.save_every == 0:
            save_training_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                config=config,
            )


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    paths = resolve_data_paths(args.data)
    train_loader, val_loader, train_samples, val_samples = build_dataloaders(paths, args, device)
    model = GeometryFieldEncoder(feature_dim=args.feature_dim, hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    _, checkpoint_dir, log_path, config = prepare_run(args, paths, device, train_samples, val_samples)
    start_epoch = load_resume_epoch(args.resume, checkpoint_dir / "latest.pt", model, optimizer, device)
    train(model, optimizer, train_loader, val_loader, args, device, checkpoint_dir, log_path, config, start_epoch)


if __name__ == "__main__":
    main()
