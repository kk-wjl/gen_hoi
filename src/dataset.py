"""
Robot motion datasets.

G1 Human Object Interaction dataset(.npz)
Key Components:
['joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 
'body_ang_vel_w', 'object_pos_w', 'object_quat_w', 'object_lin_vel_w', 'object_ang_vel_w',
'contact_label', 'motion_lengths', 'object_names', 'motion_names', 'fps']

Main processes:
1. clips the motion sequences based on motion_lengths
2. transforms quaternions to rotation 6d representation
3. collect the root position, velocity, and quaternion
4. transform the full trajectory into sliding windows
5. make windows trajectory relative to the first frame
6. normalize the data based on the mean and std
7. autoregressively roll out the motion sequence
8. save the necessary metrics for evaluation
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.math import (
    quat_normalize_wxyz,
    quat_to_rot6d,
    quat_standardize_wxyz,
    rot6d_from_matrix,
    rot6d_to_matrix,
    rot6d_to_quat_wxyz,
    yaw_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data"

JOINT_POS_DIM = 53
JOINT_VEL_DIM = 53
ROOT_POS_DIM = 3
ROOT_LIN_VEL_DIM = 3
OBJECT_POS_DIM = 3
OBJECT_LIN_VEL_DIM = 3
ROT6D_DIM = 6

JOINT_POS_START = 0
JOINT_POS_END = JOINT_POS_START + JOINT_POS_DIM
JOINT_VEL_START = JOINT_POS_END
JOINT_VEL_END = JOINT_VEL_START + JOINT_VEL_DIM
ROOT_POS_START = JOINT_VEL_END
ROOT_POS_END = ROOT_POS_START + ROOT_POS_DIM
ROOT_ROT6D_START = ROOT_POS_END
ROOT_ROT6D_END = ROOT_ROT6D_START + ROT6D_DIM
ROOT_VEL_START = ROOT_ROT6D_END
ROOT_VEL_END = ROOT_VEL_START + ROOT_LIN_VEL_DIM
OBJECT_POS_START = ROOT_VEL_END
OBJECT_POS_END = OBJECT_POS_START + OBJECT_POS_DIM
OBJECT_ROT6D_START = OBJECT_POS_END
OBJECT_ROT6D_END = OBJECT_ROT6D_START + ROT6D_DIM
OBJECT_VEL_START = OBJECT_ROT6D_END
OBJECT_VEL_END = OBJECT_VEL_START + OBJECT_LIN_VEL_DIM
FEATURE_DIM = OBJECT_VEL_END
MOTION_STATS_CACHE_VERSION = 1


def _motion_stats_fingerprint(
    npz_path: Path,
    seq_len: int,
    stride: int,
    make_relative: bool,
) -> tuple[str, dict[str, Any]]:
    stat = npz_path.stat()
    payload: dict[str, Any] = {
        "cache_version": MOTION_STATS_CACHE_VERSION,
        "npz_name": npz_path.name,
        "npz_size": stat.st_size,
        "npz_mtime_ns": stat.st_mtime_ns,
        "seq_len": seq_len,
        "stride": stride,
        "make_relative": make_relative,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    return digest, payload


def _try_load_motion_stats_cache(
    cache_path: Path,
    expected_digest: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not cache_path.is_file():
        return None
    try:
        blob = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    if blob.get("digest") != expected_digest:
        return None
    if int(blob.get("cache_version", 0)) != MOTION_STATS_CACHE_VERSION:
        return None
    stats = blob.get("stats")
    if not isinstance(stats, dict):
        return None
    mean = stats.get("mean")
    std = stats.get("std")
    if not isinstance(mean, list) or not isinstance(std, list):
        return None
    if len(mean) != FEATURE_DIM or len(std) != FEATURE_DIM:
        return None
    if not all(isinstance(x, (int, float)) for x in mean):
        return None
    if not all(isinstance(x, (int, float)) for x in std):
        return None
    return torch.tensor(mean, dtype=dtype), torch.tensor(std, dtype=dtype)


def _save_motion_stats_cache(
    cache_path: Path,
    digest: str,
    payload: dict[str, Any],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, Any] = {
        "cache_version": MOTION_STATS_CACHE_VERSION,
        "digest": digest,
        "payload": payload,
        "stats": {
            "mean": mean.detach().cpu().float().numpy().astype(np.float64).tolist(),
            "std": std.detach().cpu().float().numpy().astype(np.float64).tolist(),
        },
    }
    cache_path.write_text(json.dumps(blob, ensure_ascii=True, indent=2), encoding="utf-8")


def arugment_motion():
    pass


class MotionLoader(Dataset):
    """
    Sliding-window dataset over HOI motion ``npz`` files.

    Unified per-frame state layout:
    ``[joint_pos, joint_vel, root_pos, root_rot6d, root_vel, object_pos, object_rot6d, object_vel]``

    A helper is provided to extract:
    ``[joint_pos, root_pos, root_quat_wxyz, object_pos, object_quat_wxyz]``
    """

    def __init__(
        self,
        npz_path: str | Path,
        seq_len: int = 32,
        stride: int = 1,
        *,
        make_relative: bool = True,
        normalize: bool = True,
        use_stats_cache: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        self.npz_path = Path(npz_path).expanduser().resolve()
        self.seq_len = seq_len
        self.stride = stride
        self.make_relative_enabled = make_relative
        self.normalize_enabled = normalize
        self.use_stats_cache = use_stats_cache
        self.dtype = dtype

        raw = np.load(self.npz_path, allow_pickle=True)
        self.raw_keys = tuple(raw.keys())
        self.fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
        self.dt = 1.0 / self.fps

        self.joint_pos = torch.as_tensor(raw["joint_pos"], dtype=dtype)
        self.joint_vel = torch.as_tensor(raw["joint_vel"], dtype=dtype)
        self.root_pos_w = torch.as_tensor(raw["body_pos_w"][:, 0, :], dtype=dtype)
        self.root_quat_w = quat_standardize_wxyz(
            quat_normalize_wxyz(torch.as_tensor(raw["body_quat_w"][:, 0, :], dtype=dtype))
        )
        self.root_vel_w = torch.as_tensor(raw["body_lin_vel_w"][:, 0, :], dtype=dtype)
        self.object_pos_w = torch.as_tensor(raw["object_pos_w"], dtype=dtype)
        self.object_quat_w = quat_standardize_wxyz(
            quat_normalize_wxyz(torch.as_tensor(raw["object_quat_w"], dtype=dtype))
        )
        self.object_vel_w = torch.as_tensor(raw["object_lin_vel_w"], dtype=dtype)

        self.num_frames = self.joint_pos.shape[0]

        self.joint_pos_slice = slice(JOINT_POS_START, JOINT_POS_END)
        self.joint_vel_slice = slice(JOINT_VEL_START, JOINT_VEL_END)
        self.root_pos_slice = slice(ROOT_POS_START, ROOT_POS_END)
        self.root_rot6d_slice = slice(ROOT_ROT6D_START, ROOT_ROT6D_END)
        self.root_vel_slice = slice(ROOT_VEL_START, ROOT_VEL_END)
        self.object_pos_slice = slice(OBJECT_POS_START, OBJECT_POS_END)
        self.object_rot6d_slice = slice(OBJECT_ROT6D_START, OBJECT_ROT6D_END)
        self.object_vel_slice = slice(OBJECT_VEL_START, OBJECT_VEL_END)
        self.feature_dim = FEATURE_DIM

        self.state_frames = self._build_state_frames()
        self.motion_slices, self.motion_lengths = self._build_motion_slices(raw)
        self.window_index = self._build_window_index()
        if not self.window_index:
            raise ValueError(
                f"No valid windows found in {self.npz_path} with seq_len={self.seq_len} "
                f"and stride={self.stride}."
            )
        self.state_mean, self.state_std = self._load_or_compute_stats()

    def _build_state_frames(self) -> torch.Tensor:
        root_rot6d = quat_to_rot6d(self.root_quat_w)
        object_rot6d = quat_to_rot6d(self.object_quat_w)
        return torch.cat(
            [
                self.joint_pos,
                self.joint_vel,
                self.root_pos_w,
                root_rot6d,
                self.root_vel_w,
                self.object_pos_w,
                object_rot6d,
                self.object_vel_w,
            ],
            dim=-1,
        )

    def _build_motion_slices(self, raw: Any) -> tuple[list[slice], list[int]]:
        if "motion_lengths" in raw:
            lengths = [int(v) for v in np.asarray(raw["motion_lengths"]).reshape(-1).tolist()]
        else:
            lengths = [self.num_frames]
        if sum(lengths) != self.num_frames:
            raise ValueError(
                f"motion_lengths sum {sum(lengths)} does not match number of frames {self.num_frames}"
            )

        motion_slices: list[slice] = []
        start = 0
        for length in lengths:
            end = start + length
            motion_slices.append(slice(start, end))
            start = end
        return motion_slices, lengths


    def _build_window_index(self) -> list[tuple[int, int, int]]:
        index: list[tuple[int, int, int]] = []
        for motion_id, motion_slice in enumerate(self.motion_slices):
            length = motion_slice.stop - motion_slice.start
            if length < self.seq_len:
                continue
            last = motion_slice.stop - self.seq_len
            for start in range(motion_slice.start, last + 1, self.stride):
                end = start + self.seq_len
                index.append((motion_id, start, end))
        return index

    def _compute_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        windows: list[torch.Tensor] = []
        for _, start, end in self.window_index:
            chunk = self.state_frames[start:end]
            if self.make_relative_enabled:
                chunk = self.make_relative(chunk)
            windows.append(chunk.reshape(-1, chunk.shape[-1]))
        stacked = torch.cat(windows, dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, unbiased=False).clamp_min(1e-8)
        mean[self.root_rot6d_slice] = 0.0
        mean[self.object_rot6d_slice] = 0.0
        std[self.root_rot6d_slice] = 1.0
        std[self.object_rot6d_slice] = 1.0
        return mean, std

    def _load_or_compute_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_stats_cache:
            return self._compute_stats()
        digest, payload = _motion_stats_fingerprint(
            self.npz_path,
            self.seq_len,
            self.stride,
            self.make_relative_enabled,
        )
        cache_dir = self.npz_path.parent / ".gen_hoi_cache"
        cache_path = cache_dir / f"{self.npz_path.stem}_stats_{digest}.json"
        cached = _try_load_motion_stats_cache(cache_path, digest, self.dtype)
        if cached is not None:
            return cached
        mean, std = self._compute_stats()
        _save_motion_stats_cache(cache_path, digest, payload, mean, std)
        return mean, std

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        motion_id, start, end = self.window_index[index]
        state_chunk = self.state_frames[start:end].clone()

        if self.make_relative_enabled:
            state_chunk = self.make_relative(state_chunk)

        if self.normalize_enabled:
            state_chunk = self.normalize(state_chunk)

        return {
            "chunk": state_chunk,
            "meta": (motion_id, start, end),
        }

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.state_mean) / self.state_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.state_std + self.state_mean

    def make_relative(self, chunk: torch.Tensor) -> torch.Tensor:
        out = chunk.clone()
        root_pos = out[..., self.root_pos_slice]
        root_rot6d = out[..., self.root_rot6d_slice]
        root_vel = out[..., self.root_vel_slice]
        object_pos = out[..., self.object_pos_slice]
        object_rot6d = out[..., self.object_rot6d_slice]
        object_vel = out[..., self.object_vel_slice]

        root_pos_rel = root_pos.clone()
        root_pos_rel[..., :2] = root_pos_rel[..., :2] - root_pos_rel[0:1, :2]
        object_pos_rel = object_pos.clone()
        object_pos_rel[..., :2] = object_pos_rel[..., :2] - object_pos_rel[0:1, :2]

        root_rotmat = rot6d_to_matrix(root_rot6d)
        object_rotmat = rot6d_to_matrix(object_rot6d)
        root_rotmat_0 = yaw_matrix(root_rotmat[0:1])
        root_rotmat_0_inv = root_rotmat_0.transpose(-1, -2)
        object_rotmat_0 = yaw_matrix(object_rotmat[0:1])
        object_rotmat_0_inv = object_rotmat_0.transpose(-1, -2)

        out[..., self.root_pos_slice] = torch.matmul(root_rotmat_0_inv, root_pos_rel.unsqueeze(-1)).squeeze(-1)
        out[..., self.root_vel_slice] = torch.matmul(root_rotmat_0_inv, root_vel.unsqueeze(-1)).squeeze(-1)
        out[..., self.root_rot6d_slice] = rot6d_from_matrix(torch.matmul(root_rotmat_0_inv, root_rotmat))

        out[..., self.object_pos_slice] = torch.matmul(object_rotmat_0_inv, object_pos_rel.unsqueeze(-1)).squeeze(-1)
        out[..., self.object_vel_slice] = torch.matmul(object_rotmat_0_inv, object_vel.unsqueeze(-1)).squeeze(-1)
        out[..., self.object_rot6d_slice] = rot6d_from_matrix(torch.matmul(object_rotmat_0_inv, object_rotmat))
        return out

    def accumulate_chunk_in_anchor_frame(
        self,
        chunk: torch.Tensor,
        root0_pos: torch.Tensor,
        root0_quat: torch.Tensor,
        object0_pos: torch.Tensor,
        object0_quat: torch.Tensor,
    ) -> torch.Tensor:
        out = chunk.clone()
        root_pos = out[..., self.root_pos_slice]
        root_vel = out[..., self.root_vel_slice]
        object_pos = out[..., self.object_pos_slice]
        object_vel = out[..., self.object_vel_slice]
        root_rotmat = rot6d_to_matrix(out[..., self.root_rot6d_slice])
        object_rotmat = rot6d_to_matrix(out[..., self.object_rot6d_slice])
        root0_rotmat = rot6d_to_matrix(quat_to_rot6d(quat_standardize_wxyz(quat_normalize_wxyz(root0_quat))).unsqueeze(0))
        root0_rotmat = yaw_matrix(root0_rotmat)
        object0_rotmat = rot6d_to_matrix(
            quat_to_rot6d(quat_standardize_wxyz(quat_normalize_wxyz(object0_quat))).unsqueeze(0)
        )
        object0_rotmat = yaw_matrix(object0_rotmat)

        root_pos_world = torch.matmul(root0_rotmat, root_pos.unsqueeze(-1)).squeeze(-1)
        root_pos_world[..., :2] = root_pos_world[..., :2] + root0_pos[:2]
        out[..., self.root_pos_slice] = root_pos_world
        out[..., self.root_vel_slice] = torch.matmul(root0_rotmat, root_vel.unsqueeze(-1)).squeeze(-1)
        out[..., self.root_rot6d_slice] = rot6d_from_matrix(torch.matmul(root0_rotmat, root_rotmat))

        object_pos_world = torch.matmul(object0_rotmat, object_pos.unsqueeze(-1)).squeeze(-1)
        object_pos_world[..., :2] = object_pos_world[..., :2] + object0_pos[:2]
        out[..., self.object_pos_slice] = object_pos_world
        out[..., self.object_vel_slice] = torch.matmul(object0_rotmat, object_vel.unsqueeze(-1)).squeeze(-1)
        out[..., self.object_rot6d_slice] = rot6d_from_matrix(torch.matmul(object0_rotmat, object_rotmat))
        return out

    def chunk_to_pose(self, chunk: torch.Tensor) -> dict[str, torch.Tensor]:
        root_quat = quat_standardize_wxyz(quat_normalize_wxyz(rot6d_to_quat_wxyz(chunk[..., self.root_rot6d_slice])))
        object_quat = quat_standardize_wxyz(
            quat_normalize_wxyz(rot6d_to_quat_wxyz(chunk[..., self.object_rot6d_slice]))
        )
        return {
            "joint_pos": chunk[..., self.joint_pos_slice],
            "root_pos": chunk[..., self.root_pos_slice],
            "root_quat": root_quat,
            "object_pos": chunk[..., self.object_pos_slice],
            "object_quat": object_quat,
        }


    def metrics(self, pred: torch.Tensor, target: torch.Tensor, *, normalized: bool = True) -> dict[str, float]:
        pred_chunk = pred.clone()
        target_chunk = target.clone()
        if normalized and self.normalize_enabled:
            pred_chunk = self.denormalize(pred_chunk)
            target_chunk = self.denormalize(target_chunk)

        pred_root_pos = pred_chunk[..., self.root_pos_slice]
        target_root_pos = target_chunk[..., self.root_pos_slice]
        pred_joint_pos = pred_chunk[..., self.joint_pos_slice]
        target_joint_pos = target_chunk[..., self.joint_pos_slice]

        if pred_chunk.shape[-2] < 2:
            return {"root_vel_fd_mse": 0.0, "joint_vel_fd_mse": 0.0}

        pred_root_vel = torch.diff(pred_root_pos, dim=-2) * self.fps
        target_root_vel = torch.diff(target_root_pos, dim=-2) * self.fps
        pred_joint_vel = torch.diff(pred_joint_pos, dim=-2) * self.fps
        target_joint_vel = torch.diff(target_joint_pos, dim=-2) * self.fps

        return {
            "root_vel_fd_mse": torch.mean((pred_root_vel - target_root_vel) ** 2).item(),
            "joint_vel_fd_mse": torch.mean((pred_joint_vel - target_joint_vel) ** 2).item(),
        }

    def get_stats(self) -> dict[str, torch.Tensor]:
        return {
            "mean": self.state_mean.clone(),
            "std": self.state_std.clone(),
        }
