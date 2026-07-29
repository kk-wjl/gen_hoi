import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "tests" / "data" / "rollout_2.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect per-joint velocity finite-difference errors.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--cond-steps", type=int, default=8)
    parser.add_argument("--num-windows", type=int, default=5)
    parser.add_argument("--num-rollouts", type=int, default=10)
    return parser.parse_args()


def compute_sq_error(joint_pos: np.ndarray, joint_vel: np.ndarray, fps: np.ndarray) -> np.ndarray:
    joint_vel_fd = np.diff(joint_pos, axis=0) * fps
    return (joint_vel[1:] - joint_vel_fd) ** 2


if __name__ == "__main__":
    args = parse_args()
    data = np.load(args.data_path)

    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    fps = data["fps"]

    rollout_len = args.seq_len + (args.num_windows - 1) * (args.seq_len - args.cond_steps)
    expected_total_frames = args.num_rollouts * rollout_len
    if joint_pos.shape[0] != expected_total_frames:
        raise ValueError(
            f"Expected total frames {expected_total_frames} from "
            f"num_rollouts={args.num_rollouts}, seq_len={args.seq_len}, "
            f"cond_steps={args.cond_steps}, num_windows={args.num_windows}, "
            f"but got {joint_pos.shape[0]}"
        )

    overall_sq_errors: list[np.ndarray] = []
    for rollout_idx in range(args.num_rollouts):
        start = rollout_idx * rollout_len
        end = start + rollout_len
        rollout_sq_error = compute_sq_error(joint_pos[start:end], joint_vel[start:end], fps)
        overall_sq_errors.append(rollout_sq_error)

    stacked_sq_error = np.concatenate(overall_sq_errors, axis=0)
    overall_mse = float(np.mean(stacked_sq_error))
    per_joint_mse = np.mean(stacked_sq_error, axis=0)
    body_joint_mse = float(np.mean(per_joint_mse[:29]))
    hand_joint_mse = float(np.mean(per_joint_mse[29:]))

    print(f"data_path: {args.data_path}")
    print(f"rollout_len: {rollout_len}")
    print(f"num_rollouts: {args.num_rollouts}")
    print(f"overall_joint_vel_fd_mse: {overall_mse:.10f}")
    print(f"body_joint_vel_fd_mse_mean: {body_joint_mse:.10f}")
    print(f"hand_joint_vel_fd_mse_mean: {hand_joint_mse:.10f}")
    print("per_joint_joint_vel_fd_mse:")
    for joint_idx, joint_mse in enumerate(per_joint_mse):
        print(f"  joint_dof[{joint_idx:02d}]: {float(joint_mse):.10f}")

    top_k = min(10, per_joint_mse.shape[0])
    top_indices = np.argsort(per_joint_mse)[::-1][:top_k]
    print(f"top_{top_k}_joint_errors:")
    for rank, joint_idx in enumerate(top_indices, start=1):
        print(f"  {rank:02d}. joint_dof[{joint_idx:02d}] -> {float(per_joint_mse[joint_idx]):.10f}")
