from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import trimesh
import warp as wp

from metrics.fk import BODY_INDEX, FKEngine
from utils.math import matrix_from_quat

REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECTS_ROOT = REPO_ROOT / "objects"


def _foot_query_spec():
    # Midpoints of the foot collision capsules from
    # g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml.
    left_midpoints = [
        [0.075, -0.0265, -0.025],
        [0.0395, -0.0180, -0.025],
        [0.0390, -0.0100, -0.025],
        [0.0390, 0.0000, -0.025],
        [0.0390, 0.0100, -0.025],
        [0.0395, 0.0180, -0.025],
        [0.0750, 0.0260, -0.025],
    ]
    right_midpoints = [
        [0.075, -0.0260, -0.025],
        [0.0395, -0.0180, -0.025],
        [0.0390, -0.0100, -0.025],
        [0.0390, 0.0000, -0.025],
        [0.0390, 0.0100, -0.025],
        [0.0395, 0.0180, -0.025],
        [0.0750, 0.0260, -0.025],
    ]
    return [
        *(
            {"body": "left_ankle_roll_link", "point": point}
            for point in left_midpoints
        ),
        *(
            {"body": "right_ankle_roll_link", "point": point}
            for point in right_midpoints
        ),
    ]


def _hand_query_spec():
    # First-pass palm / fingertip proxy points based on the hand chain in
    # g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml. We keep only bodies present
    # in FKEngine.BODY_NAMES so the points can be transformed from chunk data.
    return [
        {"body": "left_wrist_yaw_link", "point": [0.0415, 0.0000, 0.0000]},
        {"body": "L_thumb_distal", "point": [0.0159104, -0.0156076, -0.00599021]},
        {"body": "L_index_intermediate", "point": [-0.00412053, -0.0433132, -0.00501047]},
        {"body": "L_middle_intermediate", "point": [-0.00426879, -0.0466043, -0.00498678]},
        {"body": "right_wrist_yaw_link", "point": [0.0415, 0.0000, 0.0000]},
        {"body": "R_thumb_distal", "point": [0.0159104, 0.0156076, -0.00599021]},
        {"body": "R_index_intermediate", "point": [-0.00412053, 0.0433132, -0.00501047]},
        {"body": "R_middle_intermediate", "point": [-0.00426879, 0.0466043, -0.00498678]},
    ]


def _trunk_query_spec():
    # First-pass torso / pelvis proxy points. These are coarse local-surface
    # anchors that can be refined later from mesh sampling if needed.
    return [
        {"body": "pelvis", "point": [0.0000, 0.0000, -0.0760]},
        {"body": "pelvis", "point": [0.0400, 0.0000, -0.0400]},
        {"body": "pelvis", "point": [-0.0400, 0.0000, -0.0400]},
        {"body": "pelvis", "point": [0.0000, 0.0800, -0.0400]},
        {"body": "pelvis", "point": [0.0000, -0.0800, -0.0400]},
        {"body": "torso_link", "point": [0.0000, 0.0000, 0.2600]},
        {"body": "torso_link", "point": [0.0600, 0.0000, 0.1800]},
        {"body": "torso_link", "point": [-0.0500, 0.0000, 0.1500]},
        {"body": "torso_link", "point": [0.0000, 0.1200, 0.1800]},
        {"body": "torso_link", "point": [0.0000, -0.1200, 0.1800]},
    ]


def _body_query_spec() -> list[dict[str, list[float] | str]]:
    return _foot_query_spec() + _hand_query_spec() + _trunk_query_spec()


def _query_body_local_to_world(
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        query_spec: list[dict[str, list[float] | str]],
    ) -> torch.Tensor:
    seq_len = body_pos_w.shape[0]
    body_indices = [BODY_INDEX[str(spec["body"])] for spec in query_spec]
    index_tensor = torch.as_tensor(body_indices, device=body_pos_w.device, dtype=torch.long)

    selected_pos = body_pos_w.index_select(dim=1, index=index_tensor)
    selected_quat = body_quat_w.index_select(dim=1, index=index_tensor)
    selected_rot = matrix_from_quat(selected_quat)

    local_points = torch.tensor(
        [spec["point"] for spec in query_spec],
        dtype=body_pos_w.dtype,
        device=body_pos_w.device,
    ).unsqueeze(0).expand(seq_len, -1, -1)

    rotated = torch.matmul(selected_rot, local_points.unsqueeze(-1)).squeeze(-1)
    return selected_pos + rotated


def _query_world_to_object_local(
        chunk: dict[str, torch.Tensor],
        query_points_w: torch.Tensor,
    ) -> torch.Tensor:
    # world frame to object local frame
    device = query_points_w.device
    dtype = query_points_w.dtype

    object_pos_w = chunk["object_pos_w"].to(device=device, dtype=dtype)
    object_rot_mat = matrix_from_quat(chunk["object_quat_w"].to(device=device, dtype=dtype))

    centered = query_points_w - object_pos_w.unsqueeze(1)
    query_points_local = torch.matmul(object_rot_mat.transpose(1, 2), centered.transpose(1, 2)).transpose(1, 2)
    return query_points_local


def get_query_point(
        chunk: dict[str, torch.Tensor],
        fk_engine: FKEngine | None = None,
    ):
    # Use representative surface points instead of body-frame origins.
    # Current default is the union of foot, hand, and trunk proxy points.
    if "body_pos_w" in chunk and "body_quat_w" in chunk:
        body_pos_w = chunk["body_pos_w"].to(dtype=torch.float32)
        body_quat_w = chunk["body_quat_w"].to(dtype=torch.float32)
    else:
        if fk_engine is None:
            fk_engine = FKEngine()
        fk_out = fk_engine.forward_kinematics(
            joint_pos=chunk["joint_pos"],
            root_pos=chunk["root_pos"],
            root_quat=chunk["root_quat"],
        )
        body_pos_w = fk_out["body_pos_w"].to(dtype=torch.float32)
        body_quat_w = fk_out["body_quat_w"].to(dtype=torch.float32)

    return _query_body_local_to_world(body_pos_w, body_quat_w, _body_query_spec())


@lru_cache(maxsize=None)
def _load_object_mesh_template(
        object_name: str,
        *,
        objects_root: str | Path = OBJECTS_ROOT,
    ) -> tuple[np.ndarray, np.ndarray]:
    object_dir = Path(objects_root) / object_name
    if not object_dir.is_dir():
        raise FileNotFoundError(f"Object directory not found: {object_dir}")

    mesh_path: Path | None = None
    for suffix in (".stl", ".obj", ".ply", ".glb"):
        candidates = sorted(object_dir.glob(f"*{suffix}"))
        if candidates:
            mesh_path = candidates[0]
            break
    if mesh_path is None:
        raise FileNotFoundError(
            f"No mesh file found under {object_dir}. Expected one of: .stl, .obj, .ply, .glb"
        )

    mesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type loaded from {mesh_path}: {type(mesh)!r}")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"Expected mesh vertices to have shape (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"Expected mesh faces to have shape (M, 3), got {faces.shape}")
    return vertices, faces

def query_object_signed_distance(
        chunk: dict[str, torch.Tensor],
        object_name: str,
        fk_engine: FKEngine | None = None,
        *,
        max_dist: float = 1.0e6,
        winding_accuracy: float = 2.0,
        winding_threshold: float = 0.5,
        normal_epsilon: float = 1.0e-3,
    ):
    # take several points on human body and compute signed distance to object mesh
    query_points_w = get_query_point(chunk, fk_engine=fk_engine) # (T, K, 3)

    template_vertices, template_faces = _load_object_mesh_template(object_name)
    query_points_local = _query_world_to_object_local(chunk, query_points_w)

    wp_device = wp.get_device("cuda" if query_points_local.is_cuda else "cpu")
    with wp.ScopedDevice(wp_device):
        mesh = wp.Mesh(
            points=wp.array(template_vertices, dtype=wp.vec3, device=wp_device),
            indices=wp.array(template_faces.reshape(-1), dtype=wp.int32, device=wp_device),
            support_winding_number=True,
        )

        signed_distance = torch.empty(
            query_points_local.shape[:-1],
            dtype=query_points_local.dtype,
            device=query_points_local.device,
        )
        query_points_local_np = query_points_local.detach().cpu().numpy()

        for t_idx in range(query_points_local_np.shape[0]):
            for p_idx in range(query_points_local_np.shape[1]):
                point_np = query_points_local_np[t_idx, p_idx]
                point_wp = wp.vec3(float(point_np[0]), float(point_np[1]), float(point_np[2]))

                result = wp.mesh_query_point_sign_winding_number(
                    mesh.id,
                    point_wp,
                    float(max_dist),
                    float(winding_accuracy),
                    float(winding_threshold),
                )
                if not result.result:
                    result = wp.mesh_query_point_sign_normal(
                        mesh.id,
                        point_wp,
                        float(max_dist),
                        float(normal_epsilon),
                    )
                if not result.result:
                    signed_distance[t_idx, p_idx] = float(max_dist)
                    continue

                face_idx = int(result.face)
                u = float(result.u)
                v = float(result.v)
                tri = template_faces[face_idx]
                a = template_vertices[tri[0]]
                b = template_vertices[tri[1]]
                c = template_vertices[tri[2]]
                closest = a * (1.0 - u - v) + b * u + c * v
                dist = float(np.linalg.norm(point_np - closest))
                sign = -1.0 if float(result.sign) < 0.0 else 1.0
                signed_distance[t_idx, p_idx] = sign * dist

    return signed_distance
