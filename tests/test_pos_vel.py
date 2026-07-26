from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "smalltable.npz"
OBJECT_NAME = "smalltable"


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))

if __name__ == "__main__":
    data = np.load(DATA_PATH)

    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    body_pos_w = data["body_pos_w"]
    body_lin_vel_w = data["body_lin_vel_w"]
    object_pos_w = data["object_pos_w"]
    object_lin_vel_w = data["object_lin_vel_w"]
    fps = data["fps"]

    root_pos_w = body_pos_w[:, 0, :]
    root_lin_vel_w = body_lin_vel_w[:, 0, :]

    root_vel_fd = np.diff(root_pos_w, axis=0) * fps
    joint_vel_fd = np.diff(joint_pos, axis=0) * fps
    object_vel_fd = np.diff(object_pos_w, axis=0) * fps

    print(f"root_vel_fd_mse: {mse(root_lin_vel_w[1:], root_vel_fd):.10f}")
    print(f"joint_vel_fd_mse: {mse(joint_vel[1:], joint_vel_fd):.10f}")
    print(f"object_vel_fd_mse: {mse(object_lin_vel_w[1:], object_vel_fd):.10f}")
