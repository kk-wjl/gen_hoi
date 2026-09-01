# maybe change it to mujoco with contact detection
from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
import torch

from utils.math import (
    quat_from_matrix,
    quat_normalize_wxyz,
    quat_standardize_wxyz,
    quat_wxyz_to_xyzw,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF_PATH = REPO_ROOT / "assets" / "unitree_g1" / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"

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

BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "L_index_proximal",
    "L_middle_proximal",
    "L_pinky_proximal",
    "L_ring_proximal",
    "L_thumb_proximal_base",
    "R_index_proximal",
    "R_middle_proximal",
    "R_pinky_proximal",
    "R_ring_proximal",
    "R_thumb_proximal_base",
    "L_index_intermediate",
    "L_middle_intermediate",
    "L_pinky_intermediate",
    "L_ring_intermediate",
    "L_thumb_proximal",
    "R_index_intermediate",
    "R_middle_intermediate",
    "R_pinky_intermediate",
    "R_ring_intermediate",
    "R_thumb_proximal",
    "L_thumb_intermediate",
    "R_thumb_intermediate",
    "L_thumb_distal",
    "R_thumb_distal",
]

BODY_INDEX = {name: idx for idx, name in enumerate(BODY_NAMES)}


def build_robot_model(urdf_path: str | Path = DEFAULT_URDF_PATH) -> pin.Model:
    urdf_path = Path(urdf_path).expanduser().resolve()
    return pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())


def build_joint_q_index(model: pin.Model, joint_names: list[str] = JOINT_NAMES) -> np.ndarray:
    q_indices = np.empty(len(joint_names), dtype=np.int32)
    for i, joint_name in enumerate(joint_names):
        joint_id = model.getJointId(joint_name)
        joint = model.joints[joint_id]
        q_indices[i] = joint.idx_q
    return q_indices


class FKEngine:
    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_URDF_PATH,
        *,
        joint_names: list[str] = JOINT_NAMES,
        body_names: list[str] = BODY_NAMES,
    ) -> None:
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.joint_names = list(joint_names)
        self.body_names = list(body_names)
        self.body_index = {name: idx for idx, name in enumerate(self.body_names)}

        self.model = build_robot_model(self.urdf_path)
        self.data = self.model.createData()
        self.joint_q_indices = build_joint_q_index(self.model, self.joint_names)
        self.frame_ids = np.asarray([self.model.getFrameId(name) for name in self.body_names], dtype=np.int32)

        missing = [name for name, frame_id in zip(self.body_names, self.frame_ids.tolist()) if frame_id == 0]
        if missing:
            raise KeyError(f"Body frames not found in Pinocchio model: {missing}")

    def build_configuration(
        self,
        root_pos: torch.Tensor | np.ndarray,
        root_quat_wxyz: torch.Tensor | np.ndarray,
        joint_pos: torch.Tensor | np.ndarray,
    ) -> np.ndarray:
        root_pos_np = np.asarray(root_pos, dtype=np.float64)
        joint_pos_np = np.asarray(joint_pos, dtype=np.float64)
        root_quat_t = torch.as_tensor(root_quat_wxyz, dtype=torch.float32)
        root_quat_t = quat_standardize_wxyz(quat_normalize_wxyz(root_quat_t))
        root_quat_xyzw_np = quat_wxyz_to_xyzw(root_quat_t).cpu().numpy().astype(np.float64)

        q = np.zeros(self.model.nq, dtype=np.float64)
        q[:3] = root_pos_np
        q[3:7] = root_quat_xyzw_np
        q[self.joint_q_indices] = joint_pos_np
        return q

    def forward_kinematics(
        self,
        *,
        joint_pos: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = joint_pos.device
        dtype = joint_pos.dtype
        seq_len = int(joint_pos.shape[0])

        body_pos_w = np.zeros((seq_len, len(self.body_names), 3), dtype=np.float32)
        body_quat_w = np.zeros((seq_len, len(self.body_names), 4), dtype=np.float32)

        joint_pos_np = joint_pos.detach().cpu().numpy()
        root_pos_np = root_pos.detach().cpu().numpy()
        root_quat_np = root_quat.detach().cpu().numpy()

        for frame_idx in range(seq_len):
            q = self.build_configuration(
                root_pos=root_pos_np[frame_idx],
                root_quat_wxyz=root_quat_np[frame_idx],
                joint_pos=joint_pos_np[frame_idx],
            )
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            for body_idx, frame_id in enumerate(self.frame_ids.tolist()):
                placement = self.data.oMf[frame_id]
                body_pos_w[frame_idx, body_idx] = placement.translation.astype(np.float32)
                body_quat_t = quat_standardize_wxyz(
                    quat_normalize_wxyz(
                        quat_from_matrix(torch.as_tensor(placement.rotation, dtype=torch.float32))
                    )
                )
                body_quat_w[frame_idx, body_idx] = body_quat_t.cpu().numpy()

        return {
            "body_pos_w": torch.from_numpy(body_pos_w).to(device=device, dtype=dtype),
            "body_quat_w": torch.from_numpy(body_quat_w).to(device=device, dtype=dtype),
        }

    def select_bodies(
        self,
        body_tensor: torch.Tensor,
        body_names: list[str] | tuple[str, ...],
    ) -> torch.Tensor:
        if body_tensor.ndim < 2:
            raise ValueError(f"body_tensor must have at least 2 dims, got {tuple(body_tensor.shape)}")
        indices = [self.body_index[name] for name in body_names]
        index_tensor = torch.as_tensor(indices, device=body_tensor.device, dtype=torch.long)
        return body_tensor.index_select(dim=-2, index=index_tensor)
