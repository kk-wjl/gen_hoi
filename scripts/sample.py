import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from cfg import Config, config_from_dict
from dataset import MotionLoader
from flow import LinearFlow
from models.dit import DiffusionTransformer1D
from models.unet import ConditionalUNet1D
from utils.checkpoint import read_training_checkpoint_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out trajectories from a saved checkpoint.")
    parser.add_argument("--run-dir", required=True, help="Run directory under outputs/, e.g. outputs/exp01")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to <run-dir>/checkpoints/latest.pt",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <run-dir>/validate")
    parser.add_argument("--num-rollouts", type=int, default=6)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--cond-steps", type=int, default=None)
    parser.add_argument("--num-windows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Override device. Defaults to the checkpoint config value.")
    return parser.parse_args()


def load_run_config(run_dir: Path, checkpoint_path: Path) -> Config:
    config_path = run_dir / "logs" / "config.json"
    if config_path.is_file():
        config_dict: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config_dict = read_training_checkpoint_config(checkpoint_path, map_location="cpu")
    return config_from_dict(config_dict)

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


def load_checkpoint_model(
    checkpoint_path: Path,
    config: Config,
    dataset: MotionLoader,
) -> tuple[torch.nn.Module, LinearFlow, int]:
    device = torch.device(config.train.device)
    model = build_model(config, dataset.feature_dim)
    model.to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    flow = LinearFlow(
        model,
        noise_scale=config.flow.noise_scale,
        t_eps=config.flow.t_eps,
        condition=False,
    )
    flow.to(device)
    flow.eval()
    return model, flow, int(payload["epoch"])


def choose_indices(dataset_len: int, num_rollouts: int, seed: int) -> list[int]:
    if dataset_len <= 0:
        raise ValueError("Dataset is empty, cannot sample rollouts")
    rng = np.random.default_rng(seed)
    replace = dataset_len < num_rollouts
    indices = rng.choice(dataset_len, size=num_rollouts, replace=replace)
    return [int(index) for index in indices.tolist()]


def save_rollout_npz(
    path: Path,
    pose: dict[str, torch.Tensor],
    fps: float,
    *,
    joint_vel: torch.Tensor | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "joint_pos": pose["joint_pos"].cpu().numpy(),
        "root_pos_w": pose["root_pos"].cpu().numpy(),
        "root_quat_w": pose["root_quat"].cpu().numpy(),
        "object_pos_w": pose["object_pos"].cpu().numpy(),
        "object_quat_w": pose["object_quat"].cpu().numpy(),
        "fps": np.asarray([fps], dtype=np.float32),
    }
    if joint_vel is not None:
        payload["joint_vel"] = joint_vel.cpu().numpy()
    np.savez_compressed(
        path,
        **payload,
    )


def cat_rollout_poses(poses: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not poses:
        raise ValueError("poses must contain at least one rollout")
    keys = ("joint_pos", "root_pos", "root_quat", "object_pos", "object_quat")
    return {key: torch.cat([pose[key] for pose in poses], dim=0) for key in keys}

def sliding_rollout(
    dataset: MotionLoader,
    flow: LinearFlow,
    target_chunk: torch.Tensor,
    anchor_frame: torch.Tensor,
    *,
    cond_steps: int,
    num_steps: int,
    num_windows: int,
    device: torch.device,
) -> torch.Tensor:
    if num_windows < 1:
        raise ValueError(f"num_windows must be >= 1, got {num_windows}")

    seq_len = target_chunk.shape[1]
    overlap_start = seq_len - cond_steps
    cond_prefix = target_chunk[:, :cond_steps].to(device)
    current_anchor = anchor_frame.clone()
    rollout_chunks: list[torch.Tensor] = []

    for window_idx in range(num_windows):
        with torch.inference_mode():
            pred_chunk = flow.sample(
                num_steps=num_steps,
                cond_prefix=cond_prefix,
                device=device,
            )[0].detach().cpu()

        pred_denorm = dataset.denormalize(pred_chunk) if dataset.normalize_enabled else pred_chunk
        anchor_pose = dataset.chunk_to_pose(current_anchor)
        pred_world_chunk = dataset.accumulate_chunk_in_anchor_frame(
            pred_denorm,
            root0_pos=anchor_pose["root_pos"][0],
            root0_quat=anchor_pose["root_quat"][0],
            object0_pos=anchor_pose["object_pos"][0],
            object0_quat=anchor_pose["object_quat"][0],
        )

        if window_idx == 0:
            rollout_chunks.append(pred_world_chunk)
        else:
            rollout_chunks.append(pred_world_chunk[cond_steps:])

        if window_idx == num_windows - 1:
            break

        next_cond_world = pred_world_chunk[overlap_start:].clone()
        current_anchor = next_cond_world[:1].clone()
        next_cond_chunk = dataset.make_relative(next_cond_world) if dataset.make_relative_enabled else next_cond_world
        if dataset.normalize_enabled:
            next_cond_chunk = dataset.normalize(next_cond_chunk)
        cond_prefix = next_cond_chunk.unsqueeze(0).to(device)

    return torch.cat(rollout_chunks, dim=0)

def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint is not None
        else run_dir / "checkpoints" / "latest.pt"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "validate"
    )

    config = load_run_config(run_dir, checkpoint_path)
    if args.device is not None:
        config.train.device = args.device
    device = torch.device(config.train.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cond_steps = args.cond_steps if args.cond_steps is not None else config.train.cond_steps
    if cond_steps < 1 or cond_steps >= config.dataset.seq_len:
        raise ValueError(
            f"cond_steps must be in [1, seq_len - 1], got cond_steps={cond_steps}, "
            f"seq_len={config.dataset.seq_len}"
        )
    if args.num_windows < 1:
        raise ValueError(f"num_windows must be >= 1, got {args.num_windows}")

    dataset = MotionLoader(
        npz_path=config.dataset.npz_path,
        seq_len=config.dataset.seq_len,
        stride=config.dataset.stride,
        make_relative=config.dataset.make_relative,
        normalize=config.dataset.normalize,
        use_vel=config.dataset.use_vel,
        use_stats_cache=config.dataset.use_stats_cache,
    )
    _, flow, checkpoint_epoch = load_checkpoint_model(checkpoint_path, config, dataset)
    indices = choose_indices(len(dataset), args.num_rollouts, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "rollout_metrics.jsonl"
    rollout_poses: list[dict[str, torch.Tensor]] = []
    rollout_joint_vels: list[torch.Tensor] = []
    metric_sums = {
        "root_vel_fd_mse": 0.0,
        "joint_vel_fd_mse": 0.0,
        "object_vel_fd_mse": 0.0,
        "root_ang_vel_fd_mse": 0.0,
        "object_ang_vel_fd_mse": 0.0,
    }

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for rollout_id, dataset_index in enumerate(indices):
            sample = dataset[dataset_index]
            chunk = sample["chunk"]
            motion_id, start, end = sample["meta"]

            target_chunk = chunk.unsqueeze(0)
            anchor_frame = dataset.state_frames[start : start + 1].clone()
            pred_world_chunk = sliding_rollout(
                dataset,
                flow,
                target_chunk,
                anchor_frame=anchor_frame,
                cond_steps=cond_steps,
                num_steps=args.num_steps,
                num_windows=args.num_windows,
                device=device,
            )
            pred_pose = dataset.chunk_to_pose(pred_world_chunk)
            pred_joint_vel = pred_world_chunk[..., dataset.joint_vel_slice]
            metrics = dataset.metrics(pred_world_chunk, normalized=False)

            rollout_name = f"rollout_{rollout_id:03d}.npz"
            rollout_path = output_dir / rollout_name
            save_rollout_npz(rollout_path, pred_pose, dataset.fps, joint_vel=pred_joint_vel)
            rollout_poses.append(pred_pose)
            rollout_joint_vels.append(pred_joint_vel)

            record = {
                "rollout_id": rollout_id,
                "rollout_file": rollout_name,
                "dataset_index": dataset_index,
                "motion_id": motion_id,
                "start": start,
                "end": end,
                "cond_steps": cond_steps,
                "num_steps": args.num_steps,
                "num_windows": args.num_windows,
                "num_frames": int(pred_world_chunk.shape[0]),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_epoch": checkpoint_epoch,
                "data_files": [str(path) for path in dataset.npz_paths],
                "root_vel_fd_mse": metrics["root_vel_fd_mse"],
                "joint_vel_fd_mse": metrics["joint_vel_fd_mse"],
                "object_vel_fd_mse": metrics["object_vel_fd_mse"],
                "root_ang_vel_fd_mse": metrics["root_ang_vel_fd_mse"],
                "object_ang_vel_fd_mse": metrics["object_ang_vel_fd_mse"],
            }
            metrics_file.write(json.dumps(record, ensure_ascii=True) + "\n")
            print(json.dumps(record, ensure_ascii=True, indent=2))

            for key in metric_sums:
                metric_sums[key] += record[key]

        mean_record = {
            "record_type": "summary",
            "num_rollouts": len(indices),
            "cond_steps": cond_steps,
            "num_steps": args.num_steps,
            "num_windows": args.num_windows,
            "seed": args.seed,
            "checkpoint_path": str(checkpoint_path),
            "mean_root_vel_fd_mse": metric_sums["root_vel_fd_mse"] / len(indices),
            "mean_joint_vel_fd_mse": metric_sums["joint_vel_fd_mse"] / len(indices),
            "mean_object_vel_fd_mse": metric_sums["object_vel_fd_mse"] / len(indices),
            "mean_root_ang_vel_fd_mse": metric_sums["root_ang_vel_fd_mse"] / len(indices),
            "mean_object_ang_vel_fd_mse": metric_sums["object_ang_vel_fd_mse"] / len(indices),
        }
        metrics_file.write(json.dumps(mean_record, ensure_ascii=True) + "\n")
        print(json.dumps(mean_record, ensure_ascii=True, indent=2))

    save_rollout_npz(
        output_dir / "rollout.npz",
        cat_rollout_poses(rollout_poses),
        dataset.fps,
        joint_vel=torch.cat(rollout_joint_vels, dim=0),
    )


if __name__ == "__main__":
    main()
