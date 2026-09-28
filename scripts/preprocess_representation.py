"""Attach frozen geometry-encoder features to raw motion datasets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometry.encoder import GeometryFieldEncoder
from utils.checkpoint import read_training_checkpoint_config


CONTACT_DIR = REPO_ROOT / "data" / "contact"
HAND_DIR = REPO_ROOT / "data" / "hand"
REPRESENTATION_DIR = REPO_ROOT / "data" / "representation"
HAND_POINTS_KEY = "hand_inner_points_o"
HAND_NAMES_KEY = "hand_inner_names"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute frozen hand-geometry representations for motion data.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained GeometryFieldEncoder checkpoint.")
    parser.add_argument("--input", type=Path, nargs="*", default=None, help="Contact motion files; defaults to data/contact.")
    parser.add_argument("--hand-dir", type=Path, default=HAND_DIR, help="Directory containing <motion_stem>_hand.npz.")
    parser.add_argument("--output-dir", type=Path, default=REPRESENTATION_DIR)
    parser.add_argument("--batch-frames", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_encoder(checkpoint_path: Path, device: torch.device) -> tuple[GeometryFieldEncoder, int]:
    config = read_training_checkpoint_config(checkpoint_path, map_location="cpu")
    required = ("feature_dim", "hidden_dim", "depth")
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"{checkpoint_path} config is missing encoder fields: {missing}")
    encoder = GeometryFieldEncoder(
        feature_dim=int(config["feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        depth=int(config["depth"]),
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(payload["model_state_dict"])
    encoder.eval()
    return encoder, encoder.feature_dim


@torch.inference_mode()
def encode_points(
    encoder: GeometryFieldEncoder,
    points_o: np.ndarray,
    *,
    batch_frames: int,
    device: torch.device,
) -> np.ndarray:
    if points_o.ndim != 3 or points_o.shape[-1] != 3:
        raise ValueError(f"{HAND_POINTS_KEY} must have shape (T, M, 3), got {points_o.shape}")
    if batch_frames < 1:
        raise ValueError("--batch-frames must be >= 1")
    frames, point_count, _ = points_o.shape
    features = np.empty((frames, point_count, encoder.feature_dim), dtype=np.float32)
    for start in range(0, frames, batch_frames):
        end = min(start + batch_frames, frames)
        batch = torch.from_numpy(points_o[start:end]).reshape(-1, 3).to(device=device, dtype=torch.float32)
        features[start:end] = encoder(batch).features.reshape(end - start, point_count, -1).cpu().numpy()
    return features


def preprocess_file(
    motion_path: Path,
    hand_dir: Path,
    output_dir: Path,
    encoder: GeometryFieldEncoder,
    checkpoint_path: Path,
    batch_frames: int,
    device: torch.device,
) -> Path:
    hand_path = hand_dir / f"{motion_path.stem}_hand.npz"
    if not hand_path.is_file():
        raise FileNotFoundError(f"No matching hand sidecar for {motion_path.name}: {hand_path}")
    with np.load(motion_path, allow_pickle=True) as raw:
        payload = {key: raw[key] for key in raw.files}
    with np.load(hand_path, allow_pickle=True) as hand:
        if HAND_POINTS_KEY not in hand:
            raise KeyError(f"{hand_path} is missing {HAND_POINTS_KEY}")
        points_o = np.asarray(hand[HAND_POINTS_KEY], dtype=np.float32)
        if points_o.shape[0] != payload["joint_pos"].shape[0]:
            raise ValueError(f"Frame count mismatch between {motion_path.name} and {hand_path.name}")
        if "hand_inner_valid" in hand and not bool(np.asarray(hand["hand_inner_valid"]).all()):
            raise ValueError(f"{hand_path} has invalid geometry queries")
        names = np.asarray(hand[HAND_NAMES_KEY]) if HAND_NAMES_KEY in hand else None

    payload["hand_representation"] = encode_points(
        encoder, points_o, batch_frames=batch_frames, device=device
    )
    if names is not None:
        payload["hand_representation_names"] = names
    payload["hand_representation_checkpoint"] = np.asarray(str(checkpoint_path.resolve()))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{motion_path.stem}_representation.npz"
    np.savez_compressed(output_path, **payload)
    return output_path


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    encoder, feature_dim = load_encoder(args.checkpoint.resolve(), device)
    input_paths = args.input if args.input else sorted(CONTACT_DIR.glob("*.npz"))
    if not input_paths:
        raise FileNotFoundError(f"No contact motions found under {CONTACT_DIR}")
    for motion_path in input_paths:
        output_path = preprocess_file(
            motion_path.resolve(),
            args.hand_dir.resolve(),
            args.output_dir.resolve(),
            encoder,
            args.checkpoint,
            args.batch_frames,
            device,
        )
        print(f"wrote: {output_path} feature_dim={feature_dim}")


if __name__ == "__main__":
    main()
