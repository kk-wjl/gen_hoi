import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from cfg import Config, config_from_dict
from dataset import MotionLoader
from flow import LinearFlow
from model import DiffusionTransformer1D
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Override device. Defaults to the checkpoint config value.")
    return parser.parse_args()

def load_checkpoint_model(
    checkpoint_path: Path,
    config: Config,
    dataset: MotionLoader,
) -> tuple[DiffusionTransformer1D, LinearFlow, int]:
    device = torch.device(config.train.device)
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


def rollout_to_world_pose(
    dataset: MotionLoader,
    pred_chunk: torch.Tensor,
    start: int,
    end: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
    target_world_chunk = dataset.state_frames[start:end].clone()
    anchor_pose = dataset.chunk_to_pose(target_world_chunk[0:1])
    pred_world_chunk = dataset.accumulate_chunk_in_anchor_frame(
        pred_chunk,
        root0_pos=anchor_pose["root_pos"][0],
        root0_quat=anchor_pose["root_quat"][0],
        object0_pos=anchor_pose["object_pos"][0],
        object0_quat=anchor_pose["object_quat"][0],
    )
    pred_pose = dataset.chunk_to_pose(pred_world_chunk)
    target_pose = dataset.chunk_to_pose(target_world_chunk)
    metrics = dataset.metrics(pred_world_chunk, target_world_chunk, normalized=False)
    return pred_pose, target_pose, metrics


def save_rollout_npz(path: Path, pose: dict[str, torch.Tensor], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        joint_pos=pose["joint_pos"].cpu().numpy(),
        root_pos=pose["root_pos"].cpu().numpy(),
        root_quat=pose["root_quat"].cpu().numpy(),
        object_pos=pose["object_pos"].cpu().numpy(),
        object_quat=pose["object_quat"].cpu().numpy(),
        fps=np.asarray([fps], dtype=np.float32),
    )


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

    config_dict = read_training_checkpoint_config(checkpoint_path, map_location="cpu")
    config = config_from_dict(config_dict)
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

    dataset = MotionLoader(
        npz_path=config.dataset.npz_path,
        seq_len=config.dataset.seq_len,
        stride=config.dataset.stride,
        make_relative=config.dataset.make_relative,
        normalize=config.dataset.normalize,
        use_stats_cache=config.dataset.use_stats_cache,
    )
    _, flow, checkpoint_epoch = load_checkpoint_model(checkpoint_path, config, dataset)
    indices = choose_indices(len(dataset), args.num_rollouts, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "rollout_metrics.jsonl"

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for rollout_id, dataset_index in enumerate(indices):
            sample = dataset[dataset_index]
            chunk = sample["chunk"]
            motion_id, start, end = sample["meta"]

            target_chunk = chunk.unsqueeze(0)
            cond_prefix = target_chunk[:, :cond_steps].to(device)
            with torch.inference_mode():
                pred_chunk = flow.sample(
                    num_steps=args.num_steps,
                    cond_prefix=cond_prefix,
                    device=device,
                )[0].detach().cpu()

            pred_denorm = dataset.denormalize(pred_chunk) if dataset.normalize_enabled else pred_chunk
            pred_pose, target_pose, metrics = rollout_to_world_pose(
                dataset,
                pred_chunk=pred_denorm,
                start=start,
                end=end,
            )

            rollout_name = f"rollout_{rollout_id:03d}.npz"
            rollout_path = output_dir / rollout_name
            save_rollout_npz(rollout_path, pred_pose, dataset.fps)

            record = {
                "rollout_id": rollout_id,
                "rollout_file": rollout_name,
                "dataset_index": dataset_index,
                "motion_id": motion_id,
                "start": start,
                "end": end,
                "cond_steps": cond_steps,
                "num_steps": args.num_steps,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_epoch": checkpoint_epoch,
                "data_files": [str(path) for path in dataset.npz_paths],
                "root_vel_fd_mse": metrics["root_vel_fd_mse"],
                "joint_vel_fd_mse": metrics["joint_vel_fd_mse"],
                "target_root_pos_0": target_pose["root_pos"][0].cpu().tolist(),
                "target_object_pos_0": target_pose["object_pos"][0].cpu().tolist(),
            }
            metrics_file.write(json.dumps(record, ensure_ascii=True) + "\n")
            print(json.dumps(record, ensure_ascii=True))


if __name__ == "__main__":
    main()
