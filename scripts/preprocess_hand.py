"""Precompute object-local inner-hand geometry targets for encoder training.

The input files under ``data/contact`` remain untouched.  Each output file is
a compact sidecar under ``data/hand`` that preserves the motion/frame layout
while containing only the query points and mesh-derived training labels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometry.hand_queries import get_hand_inner_query_spec
from metrics.fk import BODY_INDEX
from metrics.query import query_hand_inner_object_geometry
from utils.math import matrix_from_quat


CONTACT_DIR = REPO_ROOT / "data" / "contact"
HAND_DIR = REPO_ROOT / "data" / "hand"
REQUIRED_KEYS = (
    "body_pos_w",
    "body_quat_w",
    "object_pos_w",
    "object_quat_w",
    "motion_lengths",
    "motion_names",
    "object_names",
    "fps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute inner-hand object geometry for encoder training.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="*",
        default=None,
        help="Input motion files. Defaults to every .npz file under data/contact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HAND_DIR,
        help="Directory for <input_stem>_hand.npz files.",
    )
    parser.add_argument("--points-per-hand", type=int, default=64)
    parser.add_argument("--batch-frames", type=int, default=1024)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch/Warp device used for mesh queries.",
    )
    return parser.parse_args()


def _motion_ranges(motion_lengths: np.ndarray, object_names: np.ndarray, frame_count: int) -> list[tuple[int, int, str]]:
    if len(motion_lengths) != len(object_names):
        raise ValueError("motion_lengths and object_names must have one entry per motion")
    ranges: list[tuple[int, int, str]] = []
    start = 0
    for length, object_name in zip(motion_lengths.tolist(), object_names.tolist()):
        end = start + int(length)
        ranges.append((start, end, str(object_name)))
        start = end
    if start != frame_count:
        raise ValueError(f"motion_lengths sum to {start}, expected {frame_count} frames")
    return ranges


def _to_object_local(
        values_w: torch.Tensor,
        object_pos_w: torch.Tensor,
        object_quat_w: torch.Tensor,
        *,
        is_direction: bool = False,
) -> torch.Tensor:
    object_rot_w_inv = matrix_from_quat(object_quat_w).transpose(-1, -2)
    if not is_direction:
        values_w = values_w - object_pos_w.unsqueeze(1)
    return torch.matmul(object_rot_w_inv.unsqueeze(1), values_w.unsqueeze(-1)).squeeze(-1)


def preprocess_file(input_path: Path, output_dir: Path, points_per_hand: int, batch_frames: int, device: torch.device) -> Path:
    if batch_frames < 1:
        raise ValueError("batch_frames must be >= 1")

    with np.load(input_path, allow_pickle=True) as raw:
        missing = [key for key in REQUIRED_KEYS if key not in raw]
        if missing:
            raise KeyError(f"{input_path} is missing required keys: {missing}")
        source = {key: raw[key] for key in REQUIRED_KEYS}

    frame_count = int(source["body_pos_w"].shape[0])
    query_spec = get_hand_inner_query_spec(points_per_hand)
    query_count = len(query_spec)
    motion_ranges = _motion_ranges(source["motion_lengths"], source["object_names"], frame_count)

    points_o = np.empty((frame_count, query_count, 3), dtype=np.float32)
    signed_distance = np.empty((frame_count, query_count), dtype=np.float32)
    surface_normals_o = np.empty_like(points_o)
    valid = np.empty((frame_count, query_count), dtype=bool)

    for motion_index, (motion_start, motion_end, object_name) in enumerate(motion_ranges):
        for start in range(motion_start, motion_end, batch_frames):
            end = min(start + batch_frames, motion_end)
            chunk = {
                key: torch.as_tensor(source[key][start:end], dtype=torch.float32, device=device)
                for key in ("body_pos_w", "body_quat_w", "object_pos_w", "object_quat_w")
            }
            geometry = query_hand_inner_object_geometry(
                chunk,
                object_name,
                points_per_hand=points_per_hand,
            )
            points_o[start:end] = _to_object_local(
                geometry["hand_inner_points_w"], chunk["object_pos_w"], chunk["object_quat_w"]
            ).cpu().numpy()
            surface_normals_o[start:end] = _to_object_local(
                geometry["surface_normals_w"],
                chunk["object_pos_w"],
                chunk["object_quat_w"],
                is_direction=True,
            ).cpu().numpy()
            signed_distance[start:end] = geometry["signed_distance"].cpu().numpy()
            valid[start:end] = geometry["valid"].cpu().numpy()

        print(
            f"{input_path.name}: motion {motion_index + 1}/{len(motion_ranges)} "
            f"[{motion_start}:{motion_end}] object={object_name}"
        )

    if not valid.all():
        invalid_count = int((~valid).sum())
        raise RuntimeError(f"{input_path.name} has {invalid_count} invalid hand-object mesh queries")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_hand.npz"
    np.savez_compressed(
        output_path,
        hand_inner_points_o=points_o,
        hand_inner_signed_distance=signed_distance,
        hand_inner_occupancy=signed_distance < 0.0,
        hand_inner_surface_normals_o=surface_normals_o,
        hand_inner_valid=valid,
        hand_inner_names=np.asarray([point.name for point in query_spec]),
        hand_inner_body_indices=np.asarray(
            [BODY_INDEX[point.body_name] for point in query_spec], dtype=np.int64
        ),
        motion_lengths=source["motion_lengths"],
        motion_names=source["motion_names"],
        object_names=source["object_names"],
        fps=source["fps"],
    )
    return output_path


def main() -> None:
    args = parse_args()
    if args.points_per_hand < 1:
        raise ValueError("--points-per-hand must be >= 1")
    input_paths = args.input if args.input else sorted(CONTACT_DIR.glob("*.npz"))
    if not input_paths:
        raise FileNotFoundError(f"No input NPZ files found under {CONTACT_DIR}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")
    for input_path in input_paths:
        output_path = preprocess_file(
            input_path.resolve(),
            args.output_dir.resolve(),
            args.points_per_hand,
            args.batch_frames,
            device,
        )
        print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
