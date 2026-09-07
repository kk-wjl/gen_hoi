"""Executable smoke test for optional fingertip geometry dataset features."""

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
DATA_PATH = REPO_ROOT / "data" / "geometry" / "smalltable_g1_replay_policy_success_geometry.npz"


def main() -> None:
    from dataset import MotionLoader

    dataset = MotionLoader(
        DATA_PATH,
        seq_len=32,
        make_relative=True,
        normalize=True,
        use_contact_label=True,
        use_closest_points=True,
        use_surface_normals=True,
    )
    expected_feature_dim = 136 + 54 + 30 + 30
    if dataset.feature_dim != expected_feature_dim:
        raise RuntimeError(f"Expected feature_dim={expected_feature_dim}, got {dataset.feature_dim}")

    chunk = dataset.denormalize(dataset[0]["chunk"])
    rollout = dataset.chunk_to_rollout(dataset.accumulate_chunk_in_anchor_frame(
        chunk,
        root0_pos=dataset.root_pos_w[0],
        root0_quat=dataset.root_quat_w[0],
        object0_pos=dataset.object_pos_w[0],
        object0_quat=dataset.object_quat_w[0],
    ))
    for key in ("fingertip_closest_points_w", "fingertip_surface_normals_w"):
        if rollout[key].shape != (32, 10, 3) or not torch.isfinite(rollout[key]).all():
            raise RuntimeError(f"Invalid decoded {key}: {tuple(rollout[key].shape)}")
    normal_norm = torch.linalg.norm(rollout["fingertip_surface_normals_w"], dim=-1)
    if not torch.allclose(normal_norm, torch.ones_like(normal_norm), atol=1e-5):
        raise RuntimeError("Decoded surface normals are not unit length")

    print(f"data_path: {DATA_PATH}")
    print(f"feature_dim: {dataset.feature_dim}")
    print("decoded_geometry: (32, 10, 3)")


if __name__ == "__main__":
    main()
