# not check yet
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from utils.math import matrix_from_quat, quat_normalize_wxyz, quat_standardize_wxyz

REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECTS_ROOT = REPO_ROOT / "objects"


def _build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    dims = [input_dim, *hidden_dims, output_dim]
    layers: list[nn.Module] = []
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


@lru_cache(maxsize=None)
def load_object_surface_points(
    object_name: str,
    *,
    objects_root: str | Path = OBJECTS_ROOT,
) -> torch.Tensor:
    object_dir = Path(objects_root)
    surface_path = object_dir / object_name / "surface.npy"
    if not surface_path.is_file():
        raise FileNotFoundError(f"Object surface point cloud not found: {surface_path}")

    surface_np = np.load(surface_path)
    if surface_np.ndim != 2 or surface_np.shape[-1] != 3:
        raise ValueError(
            f"Expected {surface_path} to have shape (N, 3), got {tuple(surface_np.shape)}"
        )
    return torch.from_numpy(surface_np.astype(np.float32, copy=False))


def sample_bps_basis(
    num_basis: int,
    *,
    radius: float = 0.3,
    seed: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if num_basis < 1:
        raise ValueError(f"num_basis must be >= 1, got {num_basis}")
    if radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    basis_points: list[torch.Tensor] = []
    while len(basis_points) < num_basis:
        points = torch.rand((num_basis * 2, 3), generator=gen, dtype=dtype) * 2.0 - 1.0
        mask = points.norm(dim=-1) <= 1.0
        if mask.any():
            basis_points.extend(points[mask])
    basis = torch.stack(basis_points[:num_basis], dim=0) * radius
    if device is not None:
        basis = basis.to(device=device)
    return basis


def transform_object_points(
    local_points: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    object_scale: torch.Tensor | float | None = None,
) -> torch.Tensor:
    if object_pos.shape[:-1] != object_quat.shape[:-1]:
        raise ValueError(
            f"object_pos and object_quat batch dims must match, got {tuple(object_pos.shape)} "
            f"and {tuple(object_quat.shape)}"
        )
    if local_points.ndim != 2 or local_points.shape[-1] != 3:
        raise ValueError(f"local_points must have shape (N, 3), got {tuple(local_points.shape)}")

    rot = matrix_from_quat(quat_standardize_wxyz(quat_normalize_wxyz(object_quat)))
    points = local_points.to(device=object_pos.device, dtype=object_pos.dtype)
    points = points.view(*((1,) * (object_pos.ndim - 1)), *points.shape)
    points = points.expand(*object_pos.shape[:-1], -1, -1)

    if object_scale is not None:
        scale = torch.as_tensor(object_scale, device=object_pos.device, dtype=object_pos.dtype)
        if scale.ndim == 0:
            scale = scale.view(*((1,) * (object_pos.ndim - 1)), 1, 1)
        else:
            scale = scale.reshape(*object_pos.shape[:-1], 1, 1)
        points = points * scale

    rotated = torch.matmul(rot.unsqueeze(-3), points.unsqueeze(-1)).squeeze(-1)
    return rotated + object_pos.unsqueeze(-2)


def compute_bps_feature(
    surface_points: torch.Tensor,
    basis_points: torch.Tensor,
) -> torch.Tensor:
    if surface_points.shape[:-2] != basis_points.shape[:-2]:
        raise ValueError(
            f"surface_points and basis_points must share leading dims, got "
            f"{tuple(surface_points.shape)} and {tuple(basis_points.shape)}"
        )
    if surface_points.shape[-1] != 3 or basis_points.shape[-1] != 3:
        raise ValueError("surface_points and basis_points must end with xyz coordinates")

    distances = torch.cdist(basis_points, surface_points)
    nearest_idx = distances.argmin(dim=-1)
    gather_idx = nearest_idx.unsqueeze(-1).expand(*nearest_idx.shape, 3)
    nearest_surface = torch.gather(surface_points, dim=-2, index=gather_idx)
    delta = nearest_surface - basis_points
    dist = delta.norm(dim=-1, keepdim=True)
    return torch.cat([delta, dist], dim=-1)


class BPSGeometryEncoder(nn.Module):
    """
    Encode an articulated object into a per-timestep low-dimensional geometry vector.

    The pipeline is:
    1. load a template surface point cloud from ``src/objects/<object_name>/surface.npy``
    2. transform it to the current world pose using ``object_pos`` and ``object_quat``
    3. compute basis point set (BPS) features against fixed basis points
    4. project flattened BPS features with an MLP

    Typical usage:
    ``encoder(object_pos, object_quat, object_names=["smalltable"] * batch_size)``
    """

    def __init__(
        self,
        *,
        num_basis: int = 128,
        basis_radius: float = 0.4,
        basis_seed: int = 0,
        hidden_dims: Sequence[int] = (256, 128),
        output_dim: int = 64,
        objects_root: str | Path = OBJECTS_ROOT,
        frame_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.output_dim = output_dim
        self.objects_root = Path(objects_root)
        self.frame_chunk_size = frame_chunk_size

        basis = sample_bps_basis(num_basis, radius=basis_radius, seed=basis_seed)
        self.register_buffer("basis_points", basis, persistent=True)
        self.feature_mlp = _build_mlp(num_basis * 4, hidden_dims, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _normalize_object_names(self, object_names: str | Sequence[str], batch_size: int) -> list[str]:
        if isinstance(object_names, str):
            return [object_names] * batch_size
        names = list(object_names)
        if len(names) != batch_size:
            raise ValueError(
                f"Expected {batch_size} object names, got {len(names)}: {names}"
            )
        return names

    def _basis_points_world(
        self,
        object_pos: torch.Tensor,
    ) -> torch.Tensor:
        basis = self.basis_points.to(device=object_pos.device, dtype=object_pos.dtype)
        return basis.view(*((1,) * (object_pos.ndim - 1)), self.num_basis, 3).expand(
            *object_pos.shape[:-1], -1, -1
        )

    def _encode_single_object(
        self,
        surface_local: torch.Tensor,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        object_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        basis_world = self._basis_points_world(object_pos) + object_pos.unsqueeze(-2)
        surface_world = transform_object_points(surface_local, object_pos, object_quat, object_scale)

        if self.frame_chunk_size is None:
            bps = compute_bps_feature(surface_world, basis_world)
            return self.feature_mlp(bps.flatten(start_dim=-2))

        flat_pos = object_pos.reshape(-1, 3)
        flat_quat = object_quat.reshape(-1, 4)
        flat_scale = None if object_scale is None else object_scale.reshape(-1)
        flat_basis = basis_world.reshape(-1, self.num_basis, 3)

        chunks: list[torch.Tensor] = []
        for start in range(0, flat_pos.shape[0], self.frame_chunk_size):
            end = start + self.frame_chunk_size
            surface_chunk = transform_object_points(
                surface_local,
                flat_pos[start:end],
                flat_quat[start:end],
                None if flat_scale is None else flat_scale[start:end],
            )
            bps_chunk = compute_bps_feature(surface_chunk, flat_basis[start:end])
            chunks.append(self.feature_mlp(bps_chunk.flatten(start_dim=-2)))
        encoded = torch.cat(chunks, dim=0)
        return encoded.view(*object_pos.shape[:-1], self.output_dim)

    def forward(
        self,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        object_names: str | Sequence[str],
        object_scale: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if object_pos.ndim != 3 or object_pos.shape[-1] != 3:
            raise ValueError(f"object_pos must have shape (B, T, 3), got {tuple(object_pos.shape)}")
        if object_quat.ndim != 3 or object_quat.shape[-1] != 4:
            raise ValueError(f"object_quat must have shape (B, T, 4), got {tuple(object_quat.shape)}")
        if object_pos.shape[:2] != object_quat.shape[:2]:
            raise ValueError(
                f"object_pos and object_quat must have matching (B, T), got "
                f"{tuple(object_pos.shape)} and {tuple(object_quat.shape)}"
            )

        batch_size, seq_len = object_pos.shape[:2]
        names = self._normalize_object_names(object_names, batch_size)
        out = torch.empty(
            batch_size,
            seq_len,
            self.output_dim,
            device=object_pos.device,
            dtype=object_pos.dtype,
        )

        if object_scale is None:
            scale_tensor = None
        else:
            scale_tensor = torch.as_tensor(object_scale, device=object_pos.device, dtype=object_pos.dtype)
            if scale_tensor.ndim == 0:
                scale_tensor = scale_tensor.expand(batch_size, seq_len)
            elif scale_tensor.shape == (batch_size,):
                scale_tensor = scale_tensor[:, None].expand(batch_size, seq_len)
            elif scale_tensor.shape != (batch_size, seq_len):
                raise ValueError(
                    f"object_scale must be scalar, (B,), or (B, T), got {tuple(scale_tensor.shape)}"
                )

        grouped_indices: dict[str, list[int]] = {}
        for idx, name in enumerate(names):
            grouped_indices.setdefault(name, []).append(idx)

        for name, batch_indices in grouped_indices.items():
            surface_local = load_object_surface_points(
                name,
                objects_root=self.objects_root,
            ).to(device=object_pos.device, dtype=object_pos.dtype)
            pos = object_pos[batch_indices]
            quat = object_quat[batch_indices]
            scale = None if scale_tensor is None else scale_tensor[batch_indices]
            out[batch_indices] = self._encode_single_object(surface_local, pos, quat, scale)
        return out


__all__ = [
    "BPSGeometryEncoder",
    "OBJECTS_ROOT",
    "compute_bps_feature",
    "load_object_surface_points",
    "sample_bps_basis",
    "transform_object_points",
]
