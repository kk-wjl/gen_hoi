# problematic
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.math import matrix_from_quat, quat_from_matrix, quat_normalize_wxyz, quat_standardize_wxyz

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "smalltable.npz"
TARGET_PATH = REPO_ROOT / "tests" / "data" / "smalltable_symmetry.npz"
OBJECT_NAME = "smalltable"
MIRROR_MAT = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float32))

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "R_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
]


def _joint_sign(name: str) -> float:
    if "yaw" in name or "roll" in name:
        return -1.0
    return 1.0


def _paired_joint_name(name: str) -> str | None:
    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    if name.startswith("L_"):
        return "R_" + name[len("L_") :]
    if name.startswith("R_"):
        return "L_" + name[len("R_") :]
    return None


def mirror_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
    name_to_idx = {name: idx for idx, name in enumerate(JOINT_NAMES)}
    mirrored = np.array(joint_pos, copy=True)
    visited: set[int] = set()

    for idx, name in enumerate(JOINT_NAMES):
        if idx in visited:
            continue
        pair_name = _paired_joint_name(name)
        sign = _joint_sign(name)

        if pair_name is None:
            mirrored[..., idx] = sign * joint_pos[..., idx]
            visited.add(idx)
            continue

        pair_idx = name_to_idx[pair_name]
        mirrored[..., idx] = sign * joint_pos[..., pair_idx]
        mirrored[..., pair_idx] = sign * joint_pos[..., idx]
        visited.add(idx)
        visited.add(pair_idx)

    return mirrored


def mirror_position(pos: np.ndarray) -> np.ndarray:
    mirrored = np.array(pos, copy=True)
    mirrored[..., 0] *= -1.0
    return mirrored


def mirror_quaternion(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = torch.as_tensor(quat_wxyz, dtype=torch.float32)
    quat = quat_standardize_wxyz(quat_normalize_wxyz(quat))
    rot = matrix_from_quat(quat)
    mirror_mat = MIRROR_MAT.to(device=rot.device, dtype=rot.dtype)
    mirrored_rot = mirror_mat @ rot @ mirror_mat
    mirrored_quat = quat_standardize_wxyz(quat_normalize_wxyz(quat_from_matrix(mirrored_rot)))
    return mirrored_quat.cpu().numpy()


def main() -> None:
    data = np.load(DATA_PATH, allow_pickle=True)

    joint_pos = data["joint_pos"]
    body_pos_w = data["body_pos_w"]
    body_quat_w = data["body_quat_w"]
    root_pos = body_pos_w[:, 0, :]
    root_quat = body_quat_w[:, 0, :]
    object_pos_w = data["object_pos_w"]
    object_quat_w = data["object_quat_w"]
    fps = data["fps"]

    converted = {
        "joint_pos": mirror_joint_pos(joint_pos),
        "root_pos": mirror_position(root_pos),
        "root_quat": mirror_quaternion(root_quat),
        "object_pos": mirror_position(object_pos_w),
        "object_quat": mirror_quaternion(object_quat_w),
        "fps": fps,
    }

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TARGET_PATH, **converted)
    print(f"saved mirrored motion to {TARGET_PATH}")


if __name__ == "__main__":
    main()
