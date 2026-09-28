"""Precompute object-local fingertip geometry targets for contact motion data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metrics.query import FINGERTIP_NAMES, query_fingertip_object_geometry
from utils.math import matrix_from_quat


CONTACT_DIR = REPO_ROOT / "data" / "contact"
GEOMETRY_DIR = REPO_ROOT / "data" / "geometry"
REQUIRED_KEYS = (
    "body_pos_w",
    "body_quat_w",
    "object_pos_w",
    "object_quat_w",
    "motion_lengths",
    "object_names",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute fingertip-object geometry targets.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="*",
        default=None,
        help="Input NPZ files. Defaults to every .npz file under data/contact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GEOMETRY_DIR,
        help="Directory for <input_stem>_geometry.npz files.",
    )
    parser.add_argument("--batch-frames", type=int, default=1024)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch/Warp device used for mesh queries.",
    )
    return parser.parse_args()


def _motion_ranges(motion_lengths: np.ndarray, object_names: np.ndarray, frame_count: int) -> list[tuple[int, int, str]]:
    if len(motion_lengths) != len(object_names):
        raise ValueError("motion_lengths and object_names must contain one entry per motion")
    ranges: list[tuple[int, int, str]] = []
    start = 0
    for length, object_name in zip(motion_lengths.tolist(), object_names.tolist()):
        end = start + int(length)
        ranges.append((start, end, str(object_name)))
        start = end
    if start != frame_count:
        raise ValueError(f"motion_lengths sum to {start}, expected {frame_count} frames")
    return ranges


def _object_local_geometry(
        geometry: dict[str, torch.Tensor],
        object_pos_w: torch.Tensor,
        object_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    object_rot_w = matrix_from_quat(object_quat_w)
    object_rot_w_inv = object_rot_w.transpose(-1, -2)
    closest_points_o = torch.matmul(
        object_rot_w_inv.unsqueeze(1),
        (geometry["closest_points_w"] - object_pos_w.unsqueeze(1)).unsqueeze(-1),
    ).squeeze(-1)
    surface_normals_o = torch.matmul(
        object_rot_w_inv.unsqueeze(1),
        geometry["surface_normals_w"].unsqueeze(-1),
    ).squeeze(-1)
    return closest_points_o, surface_normals_o


def preprocess_file(input_path: Path, output_dir: Path, batch_frames: int, device: torch.device) -> Path:
    if batch_frames < 1:
        raise ValueError("batch_frames must be >= 1")

    with np.load(input_path, allow_pickle=True) as raw:
        missing = [key for key in REQUIRED_KEYS if key not in raw]
        if missing:
            raise KeyError(f"{input_path} is missing required keys: {missing}")
        payload = {key: raw[key] for key in raw.files}

    frame_count = int(payload["body_pos_w"].shape[0])
    motion_ranges = _motion_ranges(payload["motion_lengths"], payload["object_names"], frame_count)
    closest_points_o = np.empty((frame_count, len(FINGERTIP_NAMES), 3), dtype=np.float32)
    surface_normals_o = np.empty_like(closest_points_o)
    geometry_valid = np.empty((frame_count, len(FINGERTIP_NAMES)), dtype=bool)

    for motion_index, (motion_start, motion_end, object_name) in enumerate(motion_ranges):
        for start in range(motion_start, motion_end, batch_frames):
            end = min(start + batch_frames, motion_end)
            chunk = {
                key: torch.as_tensor(payload[key][start:end], dtype=torch.float32, device=device)
                for key in ("body_pos_w", "body_quat_w", "object_pos_w", "object_quat_w")
            }
            geometry = query_fingertip_object_geometry(chunk, object_name)
            points_o, normals_o = _object_local_geometry(
                geometry,
                chunk["object_pos_w"],
                chunk["object_quat_w"],
            )
            closest_points_o[start:end] = points_o.cpu().numpy()
            surface_normals_o[start:end] = normals_o.cpu().numpy()
            geometry_valid[start:end] = geometry["valid"].cpu().numpy()

        print(
            f"{input_path.name}: motion {motion_index + 1}/{len(motion_ranges)} "
            f"[{motion_start}:{motion_end}] object={object_name}"
        )

    if not geometry_valid.all():
        invalid_count = int((~geometry_valid).sum())
        raise RuntimeError(f"{input_path.name} has {invalid_count} fingertip geometry queries")

    payload["fingertip_closest_points_o"] = closest_points_o
    payload["fingertip_surface_normals_o"] = surface_normals_o
    payload["fingertip_geometry_valid"] = geometry_valid
    payload["fingertip_names"] = np.asarray(FINGERTIP_NAMES)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_geometry.npz"
    np.savez_compressed(output_path, **payload)
    return output_path


def main() -> None:
    args = parse_args()
    input_paths = args.input if args.input else sorted(CONTACT_DIR.glob("*.npz"))
    if not input_paths:
        raise FileNotFoundError(f"No input NPZ files found under {CONTACT_DIR}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")
    for input_path in input_paths:
        output_path = preprocess_file(input_path.resolve(), args.output_dir.resolve(), args.batch_frames, device)
        print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
