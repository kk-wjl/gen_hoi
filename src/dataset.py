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
    quat_conjugate,
    quat_mul,
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
ROOT_ANG_VEL_DIM = 3
OBJECT_POS_DIM = 3
OBJECT_LIN_VEL_DIM = 3
OBJECT_ANG_VEL_DIM = 3
FINGERTIP_COUNT = 10
ROT6D_DIM = 6

JOINT_POS_START = 0
JOINT_POS_END = JOINT_POS_START + JOINT_POS_DIM
JOINT_VEL_START = JOINT_POS_END
JOINT_VEL_END = JOINT_VEL_START + JOINT_VEL_DIM
ROOT_POS_START = JOINT_VEL_END
ROOT_POS_END = ROOT_POS_START + ROOT_POS_DIM
ROOT_ROT6D_START = ROOT_POS_END
ROOT_ROT6D_END = ROOT_ROT6D_START + ROT6D_DIM
ROOT_LIN_VEL_START = ROOT_ROT6D_END
ROOT_LIN_VEL_END = ROOT_LIN_VEL_START + ROOT_LIN_VEL_DIM
ROOT_ANG_VEL_START = ROOT_LIN_VEL_END
ROOT_ANG_VEL_END = ROOT_ANG_VEL_START + ROOT_ANG_VEL_DIM
OBJECT_POS_START = ROOT_ANG_VEL_END
OBJECT_POS_END = OBJECT_POS_START + OBJECT_POS_DIM
OBJECT_ROT6D_START = OBJECT_POS_END
OBJECT_ROT6D_END = OBJECT_ROT6D_START + ROT6D_DIM
OBJECT_LIN_VEL_START = OBJECT_ROT6D_END
OBJECT_LIN_VEL_END = OBJECT_LIN_VEL_START + OBJECT_LIN_VEL_DIM
OBJECT_ANG_VEL_START = OBJECT_LIN_VEL_END
OBJECT_ANG_VEL_END = OBJECT_ANG_VEL_START + OBJECT_ANG_VEL_DIM
FEATURE_DIM = OBJECT_ANG_VEL_END
MOTION_STATS_CACHE_VERSION = 2


def _normalize_npz_paths(npz_path: str | Path | list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    if isinstance(npz_path, (str, Path)):
        paths = [Path(npz_path)]
    else:
        paths = [Path(path) for path in npz_path]
    resolved: list[Path] = []
    for path in paths:
        resolved_path = path.expanduser().resolve()
        if resolved_path.is_dir():
            resolved.extend(sorted(resolved_path.glob("*.npz")))
        else:
            resolved.append(resolved_path)
    if not resolved:
        raise ValueError("npz_path must contain at least one .npz file")
    if not all(path.is_file() and path.suffix == ".npz" for path in resolved):
        raise ValueError("npz_path must point to .npz files or directories containing .npz files")
    return resolved


def _motion_stats_fingerprint(
    npz_paths: list[Path],
    seq_len: int,
    stride: int,
    make_relative: bool,
    use_contact_label: bool,
    use_closest_points: bool,
    use_surface_normals: bool,
) -> tuple[str, dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for npz_path in npz_paths:
        stat = npz_path.stat()
        files.append(
            {
                "name": npz_path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload: dict[str, Any] = {
        "cache_version": MOTION_STATS_CACHE_VERSION,
        "files": files,
        "seq_len": seq_len,
        "stride": stride,
        "make_relative": make_relative,
        "use_contact_label": use_contact_label,
        "use_closest_points": use_closest_points,
        "use_surface_normals": use_surface_normals,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    return digest, payload


def _try_load_motion_stats_cache(
    cache_path: Path,
    expected_digest: str,
    dtype: torch.dtype,
    feature_dim: int,
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
    if len(mean) != feature_dim or len(std) != feature_dim:
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
    ``[joint_pos, joint_vel, root_pos, root_rot6d, root_lin_vel, root_ang_vel, object_pos, object_rot6d, object_lin_vel, object_ang_vel, contact_label, fingertip_geometry]``

    Contact labels and object-local fingertip geometry are included only when
    their corresponding ``use_*`` flags are enabled.

    A helper is provided to extract:
    ``[joint_pos, root_pos, root_quat_wxyz, object_pos, object_quat_wxyz]``
    """

    def __init__(
        self,
        npz_path: str | Path | list[str | Path] | tuple[str | Path, ...],
        seq_len: int = 32,
        stride: int = 1,
        *,
        make_relative: bool = True,
        normalize: bool = True,
        use_stats_cache: bool = True,
        use_vel: bool = True, # To avoid the raw dataset error
        use_contact_label: bool = False,
        use_closest_points: bool = False,
        use_surface_normals: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        self.npz_paths = _normalize_npz_paths(npz_path)
        self.seq_len = seq_len
        self.stride = stride
        self.make_relative_enabled = make_relative
        self.normalize_enabled = normalize
        self.use_stats_cache = use_stats_cache
        self.use_vel = use_vel
        self.use_contact_label = use_contact_label
        self.use_closest_points = use_closest_points
        self.use_surface_normals = use_surface_normals
        self.dtype = dtype

        raw_list = [np.load(path, allow_pickle=True) for path in self.npz_paths]
        self.raw_keys = tuple(raw_list[0].keys())
        self.fps = float(np.asarray(raw_list[0]["fps"]).reshape(-1)[0])
        self.dt = 1.0 / self.fps
        self.joint_pos = torch.cat(
            [torch.as_tensor(raw["joint_pos"], dtype=dtype) for raw in raw_list],
            dim=0,
        )
        self.joint_vel = torch.cat(
            [torch.as_tensor(raw["joint_vel"], dtype=dtype) for raw in raw_list],
            dim=0,
        )
        self.root_pos_w = torch.cat(
            [torch.as_tensor(raw["body_pos_w"][:, 0, :], dtype=dtype) for raw in raw_list],
            dim=0,
        )
        self.root_quat_w = quat_standardize_wxyz(
            quat_normalize_wxyz(
                torch.cat(
                    [torch.as_tensor(raw["body_quat_w"][:, 0, :], dtype=dtype) for raw in raw_list],
                    dim=0,
                )
            )
        )
        self.root_lin_vel_w = torch.cat(
            [torch.as_tensor(raw["body_lin_vel_w"][:, 0, :], dtype=dtype) for raw in raw_list],
            dim=0,
        )
        self.root_ang_vel_w = torch.cat(
            [torch.as_tensor(raw["body_ang_vel_w"][:, 0, :], dtype=dtype) for raw in raw_list],
            dim=0,
        )
        self.object_pos_w = torch.cat(
            [torch.as_tensor(raw["object_pos_w"], dtype=dtype) for raw in raw_list],
            dim=0,
        )
        self.object_quat_w = quat_standardize_wxyz(
            quat_normalize_wxyz(
                torch.cat(
                    [torch.as_tensor(raw["object_quat_w"], dtype=dtype) for raw in raw_list],
                    dim=0,
                )
            )
        )
        self.object_lin_vel_w = torch.cat(
            [torch.as_tensor(raw["object_lin_vel_w"], dtype=dtype) for raw in raw_list],
            dim=0,
        )
        self.object_ang_vel_w = torch.cat(
            [torch.as_tensor(raw["object_ang_vel_w"], dtype=dtype) for raw in raw_list],
            dim=0,
        )
        self.contact_label = (
            torch.cat([torch.as_tensor(raw["contact_label"], dtype=dtype) for raw in raw_list], dim=0)
            if self.use_contact_label
            else None
        )
        self.fingertip_closest_points_o = (
            torch.cat(
                [torch.as_tensor(raw["fingertip_closest_points_o"], dtype=dtype) for raw in raw_list],
                dim=0,
            )
            if self.use_closest_points
            else None
        )
        self.fingertip_surface_normals_o = (
            torch.cat(
                [torch.as_tensor(raw["fingertip_surface_normals_o"], dtype=dtype) for raw in raw_list],
                dim=0,
            )
            if self.use_surface_normals
            else None
        )

        self.num_frames = self.joint_pos.shape[0]
        self.motion_slices, self.motion_lengths = self._build_motion_slices(raw_list)

        self.joint_pos_slice = slice(JOINT_POS_START, JOINT_POS_END)
        self.joint_vel_slice = slice(JOINT_VEL_START, JOINT_VEL_END)
        self.root_pos_slice = slice(ROOT_POS_START, ROOT_POS_END)
        self.root_rot6d_slice = slice(ROOT_ROT6D_START, ROOT_ROT6D_END)
        self.root_lin_vel_slice = slice(ROOT_LIN_VEL_START, ROOT_LIN_VEL_END)
        self.root_ang_vel_slice = slice(ROOT_ANG_VEL_START, ROOT_ANG_VEL_END)
        self.object_pos_slice = slice(OBJECT_POS_START, OBJECT_POS_END)
        self.object_rot6d_slice = slice(OBJECT_ROT6D_START, OBJECT_ROT6D_END)
        self.object_lin_vel_slice = slice(OBJECT_LIN_VEL_START, OBJECT_LIN_VEL_END)
        self.object_ang_vel_slice = slice(OBJECT_ANG_VEL_START, OBJECT_ANG_VEL_END)
        self._optional_state_features: list[torch.Tensor] = []
        next_feature_start = FEATURE_DIM
        for slice_name, feature in (
            ("contact_label_slice", self.contact_label),
            (
                "fingertip_closest_points_slice",
                self.fingertip_closest_points_o.flatten(start_dim=1)
                if self.fingertip_closest_points_o is not None else None,
            ),
            (
                "fingertip_surface_normals_slice",
                self.fingertip_surface_normals_o.flatten(start_dim=1)
                if self.fingertip_surface_normals_o is not None else None,
            ),
        ):
            setattr(self, slice_name, None)
            if feature is not None:
                feature_end = next_feature_start + feature.shape[-1]
                setattr(self, slice_name, slice(next_feature_start, feature_end))
                self._optional_state_features.append(feature)
                next_feature_start = feature_end
        self.feature_dim = next_feature_start

        self.state_frames = self._build_state_frames()
        self.window_index = self._build_window_index()
        if not self.window_index:
            source_desc = self.npz_paths[0] if len(self.npz_paths) == 1 else f"{len(self.npz_paths)} files"
            raise ValueError(
                f"No valid windows found in {source_desc} with seq_len={self.seq_len} "
                f"and stride={self.stride}."
            )
        self.state_mean, self.state_std = self._compute_stats()

    def _pos_to_vel(self, pos: torch.Tensor) -> torch.Tensor:
        vel = torch.zeros_like(pos)
        for motion_slice in self.motion_slices:
            motion_pos = pos[motion_slice]
            if motion_pos.shape[0] > 1:
                motion_vel = torch.diff(motion_pos, dim=0) * self.fps
                vel[motion_slice][0] = motion_vel[0]
                vel[motion_slice][1:] = motion_vel
        return vel

    def _quat_to_vel(self, quat: torch.Tensor) -> torch.Tensor:
        ang_vel = torch.zeros(quat.shape[:-1] + (3,), dtype=quat.dtype, device=quat.device)
        for motion_slice in self.motion_slices:
            motion_quat = quat_standardize_wxyz(quat_normalize_wxyz(quat[motion_slice]))
            if motion_quat.shape[0] > 1:
                motion_ang_vel = self._quat_diff_to_ang_vel(motion_quat)
                ang_vel[motion_slice][0] = motion_ang_vel[0]
                ang_vel[motion_slice][1:] = motion_ang_vel
        return ang_vel

    def _quat_diff_to_ang_vel(self, quat: torch.Tensor) -> torch.Tensor:
        delta_q = quat_mul(quat[1:], quat_conjugate(quat[:-1]))
        delta_q = quat_standardize_wxyz(quat_normalize_wxyz(delta_q))
        delta_q_v = delta_q[..., 1:]
        v_norm = delta_q_v.norm(dim=-1, keepdim=True)
        theta = 2.0 * torch.atan2(v_norm, delta_q[..., 0:1].abs())

        axis = torch.zeros_like(delta_q_v)
        mask = v_norm.squeeze(-1) > 1e-7
        sign = torch.sign(delta_q[..., 0:1])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        axis[mask] = (delta_q_v[mask] / v_norm[mask]) * sign[mask]
        return (theta / self.dt) * axis

    def _build_state_frames(self) -> torch.Tensor:
        root_rot6d = quat_to_rot6d(self.root_quat_w)
        object_rot6d = quat_to_rot6d(self.object_quat_w)
        if self.use_vel:
            joint_vel = self.joint_vel
            root_lin_vel_w = self.root_lin_vel_w
            root_ang_vel_w = self.root_ang_vel_w
            object_lin_vel_w = self.object_lin_vel_w
            object_ang_vel_w = self.object_ang_vel_w
        else:
            joint_vel = self._pos_to_vel(self.joint_pos)
            root_lin_vel_w = self._pos_to_vel(self.root_pos_w)
            root_ang_vel_w = self._quat_to_vel(self.root_quat_w)
            object_lin_vel_w = self._pos_to_vel(self.object_pos_w)
            object_ang_vel_w = self._quat_to_vel(self.object_quat_w)
        state = [
            self.joint_pos,
            joint_vel,
            self.root_pos_w,
            root_rot6d,
            root_lin_vel_w,
            root_ang_vel_w,
            self.object_pos_w,
            object_rot6d,
            object_lin_vel_w,
            object_ang_vel_w,
        ]
        state.extend(self._optional_state_features)
        return torch.cat(state, dim=-1)

    def _build_motion_slices(self, raw_list: list[Any]) -> tuple[list[slice], list[int]]:
        lengths: list[int] = []
        for raw in raw_list:
            if "motion_lengths" in raw:
                lengths.extend(int(v) for v in np.asarray(raw["motion_lengths"]).reshape(-1).tolist())
            else:
                lengths.append(int(np.asarray(raw["joint_pos"]).shape[0]))
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
        if self.use_stats_cache:
            digest, payload = _motion_stats_fingerprint(
                self.npz_paths,
                self.seq_len,
                self.stride,
                self.make_relative_enabled,
                self.use_contact_label,
                self.use_closest_points,
                self.use_surface_normals,
            )
            cache_root = self.npz_paths[0].parent if len(self.npz_paths) == 1 else DEFAULT_DATA_PATH
            cache_stem = self.npz_paths[0].stem if len(self.npz_paths) == 1 else "multi_dataset"
            cache_dir = cache_root / ".gen_hoi_cache"
            cache_path = cache_dir / f"{cache_stem}_stats_{digest}.json"
            cached = _try_load_motion_stats_cache(cache_path, digest, self.dtype, self.feature_dim)
            if cached is not None:
                return cached

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
        if self.use_stats_cache:
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
        root_lin_vel = out[..., self.root_lin_vel_slice]
        root_ang_vel = out[..., self.root_ang_vel_slice]
        object_pos = out[..., self.object_pos_slice]
        object_rot6d = out[..., self.object_rot6d_slice]
        object_lin_vel = out[..., self.object_lin_vel_slice]
        object_ang_vel = out[..., self.object_ang_vel_slice]

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
        out[..., self.root_lin_vel_slice] = torch.matmul(root_rotmat_0_inv, root_lin_vel.unsqueeze(-1)).squeeze(-1)
        out[..., self.root_ang_vel_slice] = torch.matmul(root_rotmat_0_inv, root_ang_vel.unsqueeze(-1)).squeeze(-1)
        out[..., self.root_rot6d_slice] = rot6d_from_matrix(torch.matmul(root_rotmat_0_inv, root_rotmat))

        out[..., self.object_pos_slice] = torch.matmul(object_rotmat_0_inv, object_pos_rel.unsqueeze(-1)).squeeze(-1)
        out[..., self.object_lin_vel_slice] = torch.matmul(object_rotmat_0_inv, object_lin_vel.unsqueeze(-1)).squeeze(-1)
        out[..., self.object_ang_vel_slice] = torch.matmul(object_rotmat_0_inv, object_ang_vel.unsqueeze(-1)).squeeze(-1)
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
        root_lin_vel = out[..., self.root_lin_vel_slice]
        root_ang_vel = out[..., self.root_ang_vel_slice]
        object_pos = out[..., self.object_pos_slice]
        object_lin_vel = out[..., self.object_lin_vel_slice]
        object_ang_vel = out[..., self.object_ang_vel_slice]
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
        out[..., self.root_lin_vel_slice] = torch.matmul(root0_rotmat, root_lin_vel.unsqueeze(-1)).squeeze(-1)
        out[..., self.root_ang_vel_slice] = torch.matmul(root0_rotmat, root_ang_vel.unsqueeze(-1)).squeeze(-1)
        out[..., self.root_rot6d_slice] = rot6d_from_matrix(torch.matmul(root0_rotmat, root_rotmat))

        object_pos_world = torch.matmul(object0_rotmat, object_pos.unsqueeze(-1)).squeeze(-1)
        object_pos_world[..., :2] = object_pos_world[..., :2] + object0_pos[:2]
        out[..., self.object_pos_slice] = object_pos_world
        out[..., self.object_lin_vel_slice] = torch.matmul(object0_rotmat, object_lin_vel.unsqueeze(-1)).squeeze(-1)
        out[..., self.object_ang_vel_slice] = torch.matmul(object0_rotmat, object_ang_vel.unsqueeze(-1)).squeeze(-1)
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

    def chunk_to_rollout(self, chunk: torch.Tensor) -> dict[str, torch.Tensor]:
        rollout = self.chunk_to_pose(chunk)
        rollout.update(
            {
                "joint_vel": chunk[..., self.joint_vel_slice],
                "root_lin_vel": chunk[..., self.root_lin_vel_slice],
                "root_ang_vel": chunk[..., self.root_ang_vel_slice],
                "object_lin_vel": chunk[..., self.object_lin_vel_slice],
                "object_ang_vel": chunk[..., self.object_ang_vel_slice],
            }
        )
        if self.contact_label_slice is not None:
            rollout["contact_label"] = chunk[..., self.contact_label_slice]
        object_rot_w = rot6d_to_matrix(chunk[..., self.object_rot6d_slice])
        if self.fingertip_closest_points_slice is not None:
            closest_points_o = chunk[..., self.fingertip_closest_points_slice].reshape(*chunk.shape[:-1], 10, 3)
            rollout["fingertip_closest_points_o"] = closest_points_o
            rollout["fingertip_closest_points_w"] = (
                torch.matmul(object_rot_w.unsqueeze(-3), closest_points_o.unsqueeze(-1)).squeeze(-1)
                + rollout["object_pos"].unsqueeze(-2)
            )
        if self.fingertip_surface_normals_slice is not None:
            surface_normals_o = chunk[..., self.fingertip_surface_normals_slice].reshape(*chunk.shape[:-1], 10, 3)
            surface_normals_w = torch.matmul(
                object_rot_w.unsqueeze(-3),
                surface_normals_o.unsqueeze(-1),
            ).squeeze(-1)
            rollout["fingertip_surface_normals_o"] = surface_normals_o
            rollout["fingertip_surface_normals_w"] = surface_normals_w / surface_normals_w.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        return rollout


    def metrics(self, chunk: torch.Tensor, *, normalized: bool = True) -> dict[str, float]:
        chunk = chunk.clone()
        if normalized and self.normalize_enabled:
            chunk = self.denormalize(chunk)

        if chunk.shape[-2] < 2:
            return {
                "root_vel_fd_mse": 0.0,
                "joint_vel_fd_mse": 0.0,
                "object_vel_fd_mse": 0.0,
                "root_ang_vel_fd_mse": 0.0,
                "object_ang_vel_fd_mse": 0.0,
            }

        root_pos = chunk[..., self.root_pos_slice]
        root_lin_vel = chunk[..., self.root_lin_vel_slice]
        root_rot_quat = quat_standardize_wxyz(
            quat_normalize_wxyz(rot6d_to_quat_wxyz(chunk[..., self.root_rot6d_slice]))
        )
        root_ang_vel = chunk[..., self.root_ang_vel_slice]
        joint_pos = chunk[..., self.joint_pos_slice]
        joint_vel = chunk[..., self.joint_vel_slice]
        object_pos = chunk[..., self.object_pos_slice]
        object_rot_quat = quat_standardize_wxyz(
            quat_normalize_wxyz(rot6d_to_quat_wxyz(chunk[..., self.object_rot6d_slice]))
        )
        object_lin_vel = chunk[..., self.object_lin_vel_slice]
        object_ang_vel = chunk[..., self.object_ang_vel_slice]

        root_lin_vel_fd = torch.diff(root_pos, dim=-2) * self.fps
        root_ang_vel_fd = self._quat_diff_to_ang_vel(root_rot_quat)
        joint_vel_fd = torch.diff(joint_pos, dim=-2) * self.fps
        object_lin_vel_fd = torch.diff(object_pos, dim=-2) * self.fps
        object_ang_vel_fd = self._quat_diff_to_ang_vel(object_rot_quat)

        return {
            "root_vel_fd_mse": torch.mean((root_lin_vel[1:] - root_lin_vel_fd) ** 2).item(),
            "joint_vel_fd_mse": torch.mean((joint_vel[1:] - joint_vel_fd) ** 2).item(),
            "object_vel_fd_mse": torch.mean((object_lin_vel[1:] - object_lin_vel_fd) ** 2).item(),
            "root_ang_vel_fd_mse": torch.mean((root_ang_vel[1:] - root_ang_vel_fd) ** 2).item(),
            "object_ang_vel_fd_mse": torch.mean((object_ang_vel[1:] - object_ang_vel_fd) ** 2).item(),
        }

    def get_stats(self) -> dict[str, torch.Tensor]:
        return {
            "mean": self.state_mean.clone(),
            "std": self.state_std.clone(),
        }
