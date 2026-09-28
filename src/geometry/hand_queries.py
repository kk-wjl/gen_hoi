"""Deterministic query points on the palm-facing G1 hand surfaces.

The points are sampled once from visual meshes and stored in the owning FK
link's local frame.  They therefore follow arbitrary motion using only the
``body_pos_w`` and ``body_quat_w`` tensors already present in a chunk.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh

SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metrics.fk import BODY_INDEX  # noqa: E402
from utils.math import matrix_from_quat  # noqa: E402


G1_URDF_PATH = SRC_ROOT / "assets" / "unitree_g1" / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
DEFAULT_POINTS_PER_HAND = 64
INNER_NORMAL_COSINE = 0.70

# These are the articulated links whose palm-facing surfaces can plausibly
# contact an object.  The base link owns the palm; the others own finger pads.
HAND_LINKS = {
    "left": (
        "L_hand_base_link",
        "L_thumb_proximal_base",
        "L_thumb_proximal",
        "L_thumb_intermediate",
        "L_thumb_distal",
        "L_index_proximal",
        "L_index_intermediate",
        "L_middle_proximal",
        "L_middle_intermediate",
        "L_ring_proximal",
        "L_ring_intermediate",
        "L_pinky_proximal",
        "L_pinky_intermediate",
    ),
    "right": (
        "R_hand_base_link",
        "R_thumb_proximal_base",
        "R_thumb_proximal",
        "R_thumb_intermediate",
        "R_thumb_distal",
        "R_index_proximal",
        "R_index_intermediate",
        "R_middle_proximal",
        "R_middle_intermediate",
        "R_ring_proximal",
        "R_ring_intermediate",
        "R_pinky_proximal",
        "R_pinky_intermediate",
    ),
}


@dataclass(frozen=True)
class HandQueryPoint:
    """One query point in the coordinate system of its FK owner link."""

    name: str
    side: str
    source_link: str
    body_name: str
    point_body: tuple[float, float, float]
    normal_body: tuple[float, float, float]


def _parse_xyz(value: str | None) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=np.float32)
    xyz = np.fromstring(value, sep=" ", dtype=np.float32)
    return xyz


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float32,
    )


def _origin_transform(origin: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if origin is None:
        return np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)
    return _parse_xyz(origin.get("xyz")), _rpy_to_matrix(_parse_xyz(origin.get("rpy")))


def _fixed_link_to_body_transform(
    link_name: str,
    joints_by_child: dict[str, ET.Element],
) -> tuple[str, np.ndarray, np.ndarray]:
    """Map a visual-only fixed link into the nearest link in ``BODY_INDEX``."""
    rotation = np.eye(3, dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)
    while link_name not in BODY_INDEX:
        joint = joints_by_child.get(link_name)
        if joint is None or joint.get("type") != "fixed":
            raise KeyError(f"No FK body found for hand link {link_name!r}")
        joint_xyz, joint_rot = _origin_transform(joint.find("origin"))
        translation = joint_rot @ translation + joint_xyz
        rotation = joint_rot @ rotation
        parent = joint.find("parent")
        if parent is None or parent.get("link") is None:
            raise KeyError(f"Fixed joint for {link_name!r} has no parent")
        link_name = parent.attrib["link"]
    return link_name, translation, rotation


def _sample_surface(
    mesh: trimesh.Trimesh,
    normal_direction: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    normals = np.asarray(mesh.face_normals, dtype=np.float32)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    alignment = normals @ normal_direction
    candidate = alignment >= INNER_NORMAL_COSINE
    if not candidate.any():
        raise ValueError("No mesh faces match the requested hand-inner normal direction")

    face_indices = np.flatnonzero(candidate)
    face_areas = areas[face_indices]
    chosen = rng.choice(face_indices, size=count, p=face_areas / face_areas.sum())
    triangles = np.asarray(mesh.triangles, dtype=np.float32)[chosen]
    uv = rng.random((count, 2), dtype=np.float32)
    u = 1.0 - np.sqrt(uv[:, :1])
    v = uv[:, 1:] * (1.0 - u)
    points = triangles[:, 0] + u * (triangles[:, 1] - triangles[:, 0]) + v * (triangles[:, 2] - triangles[:, 0])
    return points, normals[chosen]


def _link_point_counts(points_per_hand: int) -> dict[str, int]:
    if points_per_hand < len(HAND_LINKS["left"]):
        raise ValueError(
            f"points_per_hand must be at least {len(HAND_LINKS['left'])}, got {points_per_hand}"
        )
    # Give the palm four times a finger segment's sampling density.  Distribute
    # any remainder deterministically over the finger links.
    links = HAND_LINKS["left"]
    base_count = max(1, points_per_hand // (len(links) + 3))
    counts = {link: base_count for link in links}
    counts[links[0]] = 4 * base_count
    remainder = points_per_hand - sum(counts.values())
    for link in links[1 : 1 + remainder]:
        counts[link] += 1
    return counts


@lru_cache(maxsize=None)
def get_hand_inner_query_spec(
    points_per_hand: int = DEFAULT_POINTS_PER_HAND,
    urdf_path: str | Path = G1_URDF_PATH,
) -> tuple[HandQueryPoint, ...]:
    """Return deterministic palm-facing points in a stable left-then-right order.

    The G1 hand meshes use their local x-axis as the palm/back-of-hand axis.
    Both hands are mirrored around their local z-axis, so the palm-facing side
    is the negative local x normal for both left and right links.
    """
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints_by_child = {
        child.attrib["link"]: joint
        for joint in root.findall("joint")
        if (child := joint.find("child")) is not None and child.get("link") is not None
    }
    counts_left = _link_point_counts(points_per_hand)
    query_points: list[HandQueryPoint] = []
    rng = np.random.default_rng(20260926)
    inner_normal = np.asarray((-1.0, 0.0, 0.0), dtype=np.float32)

    for side in ("left", "right"):
        for link_index, link_name in enumerate(HAND_LINKS[side]):
            link = links[link_name]
            visual = link.find("visual")
            mesh_element = None if visual is None else visual.find("geometry/mesh")
            if mesh_element is None or mesh_element.get("filename") is None:
                raise ValueError(f"Hand link {link_name!r} has no visual mesh")
            mesh = trimesh.load_mesh(urdf_path.parent / mesh_element.attrib["filename"], process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.dump()))
            visual_xyz, visual_rot = _origin_transform(visual.find("origin"))
            body_name, body_xyz, body_rot = _fixed_link_to_body_transform(link_name, joints_by_child)

            count = counts_left[HAND_LINKS["left"][link_index]]
            points_link, normals_link = _sample_surface(mesh, inner_normal, count, rng)
            points_body = (points_link @ visual_rot.T + visual_xyz) @ body_rot.T + body_xyz
            normals_body = normals_link @ visual_rot.T @ body_rot.T
            normals_body /= np.linalg.norm(normals_body, axis=-1, keepdims=True)
            for point_index, (point, normal) in enumerate(zip(points_body, normals_body)):
                query_points.append(
                    HandQueryPoint(
                        name=f"{side}_{link_name[2:]}_{point_index:02d}",
                        side=side,
                        source_link=link_name,
                        body_name=body_name,
                        point_body=tuple(float(value) for value in point),
                        normal_body=tuple(float(value) for value in normal),
                    )
                )
    return tuple(query_points)


def get_hand_inner_query_points(
    chunk: dict[str, torch.Tensor],
    *,
    points_per_hand: int = DEFAULT_POINTS_PER_HAND,
) -> torch.Tensor:
    """Return palm-facing hand points in world coordinates with shape ``(T, 2M, 3)``."""
    body_pos_w = chunk["body_pos_w"].to(dtype=torch.float32)
    body_quat_w = chunk["body_quat_w"].to(dtype=torch.float32)

    spec = get_hand_inner_query_spec(points_per_hand)
    body_indices = torch.as_tensor([BODY_INDEX[item.body_name] for item in spec], device=body_pos_w.device)
    positions = body_pos_w.index_select(1, body_indices)
    rotations = matrix_from_quat(body_quat_w.index_select(1, body_indices))
    points_body = torch.tensor(
        [item.point_body for item in spec], device=body_pos_w.device, dtype=body_pos_w.dtype
    ).unsqueeze(0).expand(body_pos_w.shape[0], -1, -1)
    return positions + torch.matmul(rotations, points_body.unsqueeze(-1)).squeeze(-1)


def get_hand_inner_query_body_indices(
    points_per_hand: int = DEFAULT_POINTS_PER_HAND,
) -> torch.Tensor:
    """Return the FK owner body index for every point in the stable query order."""
    return torch.tensor([BODY_INDEX[item.body_name] for item in get_hand_inner_query_spec(points_per_hand)])
