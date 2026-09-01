from __future__ import annotations

from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "data" / "contact" / "train_retargeted_smalltable_replay_policy_success.npz"
TARGET_PATH = REPO_ROOT / "vis" / "data" / "processed" / "train_retargeted_smalltable_replay_policy_success_vis.npz"


if __name__ == "__main__":
    data = np.load(SOURCE_PATH)

    joint_pos = data["joint_pos"]
    body_pos_w = data["body_pos_w"]
    body_quat_w = data["body_quat_w"]
    object_pos_w = data["object_pos_w"]
    object_quat_w = data["object_quat_w"]
    fps = data["fps"]
    
    converted = {
        "joint_pos": joint_pos,
        "root_pos_w": body_pos_w[:, 0, :],
        "root_quat_w": body_quat_w[:, 0, :],
        "object_pos_w": object_pos_w,
        "object_quat_w": object_quat_w,
        "fps": fps,
    }

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TARGET_PATH, **converted)
    print(f"Saved converted motion to {TARGET_PATH}")
