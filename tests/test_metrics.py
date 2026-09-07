from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
DATA_PATH = REPO_ROOT / "tests" / "data" / "smalltable_42clip.npz"
# The full sequence has 11,098 frames. This budget covers every G1 collision
# geometry while keeping the point-by-point Warp query manageable.
NUM_QUERY_POINTS = 64


def load_chunk(data_path: Path = DATA_PATH) -> tuple[dict[str, torch.Tensor], str]:
    with np.load(data_path, allow_pickle=False) as data:
        object_names = np.unique(data["object_names"])
        if len(object_names) != 1:
            raise ValueError(f"Expected one object type per file, got {object_names.tolist()}")
        if int(data["motion_lengths"].sum()) != data["body_pos_w"].shape[0]:
            raise ValueError("motion_lengths must sum to the number of frames")
        chunk = {
            name: torch.from_numpy(data[name])
            for name in (
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "object_pos_w",
                "object_quat_w",
                "object_lin_vel_w",
                "object_ang_vel_w",
                "contact_label",
            )
        }
    return chunk, str(object_names[0])


def evaluate_full_chunk(
        chunk: dict[str, torch.Tensor],
        object_name: str,
        *,
        num_query_points: int = NUM_QUERY_POINTS,
) -> dict[str, float]:
    from metrics.evaluate import (
        compute_contact_percentage,
        compute_foot_floating,
        compute_foot_penetration,
        compute_foot_sliding,
        compute_hand_object_distance,
        compute_hand_sliding,
        compute_object_floating,
        compute_object_penetration,
        free_object_motion_penalty,
    )
    from metrics.query import (
        get_fingertip_query_points,
        query_fingertip_object_geometry,
        query_object_geometry,
    )

    contact_label = chunk["contact_label"]
    fingertip_points_w = get_fingertip_query_points(chunk)
    if fingertip_points_w.shape != (chunk["body_pos_w"].shape[0], 10, 3):
        raise RuntimeError(f"Unexpected fingertip-query shape: {tuple(fingertip_points_w.shape)}")
    if not torch.isfinite(fingertip_points_w).all():
        raise RuntimeError("Fingertip query contains non-finite values")
    fingertip_geometry = query_fingertip_object_geometry(chunk, object_name)
    if fingertip_geometry["closest_points_w"].shape != fingertip_points_w.shape:
        raise RuntimeError("Fingertip closest-point shape does not match fingertip query shape")
    if not fingertip_geometry["valid"].all():
        raise RuntimeError("Some fingertip query points did not find an object surface")
    normal_norm = torch.linalg.norm(fingertip_geometry["surface_normals_w"], dim=-1)
    if not torch.allclose(normal_norm, torch.ones_like(normal_norm), atol=1e-5):
        raise RuntimeError("Fingertip surface normals are not unit length")
    geometry = query_object_geometry(
        chunk,
        object_name=object_name,
        num_query_points=num_query_points,
    )
    hand_distance = compute_hand_object_distance(
        chunk,
        object_name=object_name,
        geometry=geometry,
    )
    metrics = {
        "foot_sliding": compute_foot_sliding(chunk, contact_label, geometry=geometry),
        "foot_penetration": compute_foot_penetration(chunk, geometry=geometry),
        "foot_floating": compute_foot_floating(chunk, contact_label, geometry=geometry),
        "hand_sliding": compute_hand_sliding(
            chunk,
            contact_label,
            object_name=object_name,
            geometry=geometry,
        ),
        "object_penetration": compute_object_penetration(
            chunk,
            object_name=object_name,
            geometry=geometry,
        ),
        "object_floating": compute_object_floating(
            chunk,
            contact_label,
            distance=hand_distance,
            geometry=geometry,
        ),
        "free_object_motion": free_object_motion_penalty(chunk, contact_label, geometry=geometry),
        "contact_percentage": compute_contact_percentage(chunk, contact_label, geometry=geometry),
    }
    if not geometry["valid"].all():
        raise RuntimeError("Some collision query points did not find an object surface")
    if hand_distance.shape != (chunk["body_pos_w"].shape[0], 2):
        raise RuntimeError(f"Unexpected hand-distance shape: {tuple(hand_distance.shape)}")

    results: dict[str, float] = {}
    for name, value in metrics.items():
        if not torch.isfinite(value):
            raise RuntimeError(f"Metric {name} is not finite: {value}")
        results[name] = float(value)
    return results


def main() -> None:
    chunk, object_name = load_chunk()
    print(f"data_path: {DATA_PATH}")
    print(f"object_name: {object_name}")
    print(f"num_frames: {chunk['body_pos_w'].shape[0]}")
    print(f"num_query_points: {NUM_QUERY_POINTS}")
    for name, value in evaluate_full_chunk(chunk, object_name).items():
        print(f"{name}: {value:.8f}")


if __name__ == "__main__":
    main()
