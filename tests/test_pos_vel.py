from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "smalltable_42clip.npz"
OBJECT_NAME = "smalltable"


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def build_motion_slices(num_frames: int, motion_lengths: np.ndarray | None) -> list[slice]:
    if motion_lengths is None:
        return [slice(0, num_frames)]

    lengths = [int(v) for v in np.asarray(motion_lengths).reshape(-1).tolist()]
    if sum(lengths) != num_frames:
        raise ValueError(
            f"motion_lengths sum {sum(lengths)} does not match number of frames {num_frames}"
        )

    motion_slices: list[slice] = []
    start = 0
    for length in lengths:
        end = start + length
        motion_slices.append(slice(start, end))
        start = end
    return motion_slices


def mse_per_clip(pos: np.ndarray, vel: np.ndarray, fps: float, motion_slices: list[slice]) -> float:
    sq_errors: list[np.ndarray] = []
    for motion_slice in motion_slices:
        clip_pos = pos[motion_slice]
        clip_vel = vel[motion_slice]
        if clip_pos.shape[0] < 2:
            continue

        clip_vel_fd = np.diff(clip_pos, axis=0) * fps
        sq_errors.append((clip_vel[1:] - clip_vel_fd) ** 2)

    if not sq_errors:
        return 0.0
    return float(np.mean(np.concatenate(sq_errors, axis=0)))


def sq_error_per_clip(pos: np.ndarray, vel: np.ndarray, fps: float, motion_slices: list[slice]) -> np.ndarray:
    sq_errors: list[np.ndarray] = []
    for motion_slice in motion_slices:
        clip_pos = pos[motion_slice]
        clip_vel = vel[motion_slice]
        if clip_pos.shape[0] < 2:
            continue

        clip_vel_fd = np.diff(clip_pos, axis=0) * fps
        sq_errors.append((clip_vel[1:] - clip_vel_fd) ** 2)

    if not sq_errors:
        return np.empty((0,) + pos.shape[1:], dtype=pos.dtype)
    return np.concatenate(sq_errors, axis=0)


if __name__ == "__main__":
    data = np.load(DATA_PATH)

    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    body_pos_w = data["body_pos_w"]
    body_lin_vel_w = data["body_lin_vel_w"]
    object_pos_w = data["object_pos_w"]
    object_lin_vel_w = data["object_lin_vel_w"]
    motion_lengths = data["motion_lengths"] if "motion_lengths" in data else None
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    root_pos_w = body_pos_w[:, 0, :]
    root_lin_vel_w = body_lin_vel_w[:, 0, :]
    motion_slices = build_motion_slices(joint_pos.shape[0], motion_lengths)
    joint_sq_error = sq_error_per_clip(joint_pos, joint_vel, fps, motion_slices)
    per_joint_mse = np.mean(joint_sq_error, axis=0) if joint_sq_error.size > 0 else np.zeros(joint_pos.shape[1])

    print(f"root_vel_fd_mse: {mse_per_clip(root_pos_w, root_lin_vel_w, fps, motion_slices):.10f}")
    print(f"joint_vel_fd_mse: {mse_per_clip(joint_pos, joint_vel, fps, motion_slices):.10f}")
    print(f"object_vel_fd_mse: {mse_per_clip(object_pos_w, object_lin_vel_w, fps, motion_slices):.10f}")
    print("per_joint_joint_vel_fd_mse:")
    for joint_idx, joint_mse in enumerate(per_joint_mse):
        print(f"  joint_dof[{joint_idx:02d}]: {float(joint_mse):.10f}")
