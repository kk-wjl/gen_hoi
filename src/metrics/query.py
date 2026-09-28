from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh
import warp as wp

from metrics.fk import BODY_INDEX
from utils.math import matrix_from_quat

REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECTS_ROOT = REPO_ROOT / "objects"
G1_URDF_PATH = REPO_ROOT / "assets" / "unitree_g1" / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"

# This is deliberately modest because object SDF queries are currently evaluated
# point-by-point.  The samples are cached and deterministic across evaluations.
DEFAULT_NUM_COLLISION_QUERY_POINTS = 1024
MIN_POINTS_PER_COLLISION = 8

# Fixed tip-link origins from the checked-in G1 URDF.  The tip links themselves
# are visual-only fixed children, so their centers are reconstructed from the
# parent FK body poses rather than included in collision surface sampling.
FINGERTIP_NAMES = (
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
)
FINGERTIP_QUERY_SPEC = (
    ("L_thumb_distal", (0.0159104, -0.0156076, -0.00599021)),
    ("L_index_intermediate", (-0.00412053, -0.0433132, -0.00501047)),
    ("L_middle_intermediate", (-0.00426879, -0.0466043, -0.00498678)),
    ("L_ring_intermediate", (-0.00366554, -0.0433541, -0.00501047)),
    ("L_pinky_intermediate", (-0.00260734, -0.0362776, -0.00501047)),
    ("R_thumb_distal", (0.0159104, 0.0156076, -0.00599021)),
    ("R_index_intermediate", (-0.00412053, 0.0433132, -0.00501047)),
    ("R_middle_intermediate", (-0.00426879, 0.0466043, -0.00498678)),
    ("R_ring_intermediate", (-0.00366554, 0.0433541, -0.00501047)),
    ("R_pinky_intermediate", (-0.00260734, 0.0362776, -0.00501047)),
)


@wp.kernel
def _query_object_mesh_kernel(
        mesh_id: wp.uint64,
        query_points: wp.array(dtype=wp.vec3),
        max_dist: float,
        winding_accuracy: float,
        winding_threshold: float,
        valid: wp.array(dtype=wp.int32),
        faces: wp.array(dtype=wp.int32),
        signs: wp.array(dtype=float),
        barycentric_u: wp.array(dtype=float),
        barycentric_v: wp.array(dtype=float),
    ):
    point_index = wp.tid()
    result = wp.mesh_query_point_sign_winding_number(
        mesh_id,
        query_points[point_index],
        max_dist,
        winding_accuracy,
        winding_threshold,
    )
    if result.result:
        valid[point_index] = 1
        faces[point_index] = result.face
        signs[point_index] = result.sign
        barycentric_u[point_index] = result.u
        barycentric_v[point_index] = result.v
    else:
        valid[point_index] = 0
        faces[point_index] = -1


def _parse_xyz(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float32)
    xyz = np.fromstring(value, sep=" ", dtype=np.float32)
    if xyz.shape != (3,):
        raise ValueError(f"Expected three coordinates, got {value!r}")
    return xyz


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def _load_collision_mesh(geometry: ET.Element, urdf_dir: Path) -> trimesh.Trimesh:
    primitive = next(iter(geometry), None)
    if primitive is None:
        raise ValueError("Collision geometry has no primitive")

    if primitive.tag == "mesh":
        # The checked-in G1 URDF uses unscaled STL files relative to itself.
        return trimesh.load_mesh(urdf_dir / primitive.attrib["filename"], process=False)
    if primitive.tag == "box":
        return trimesh.creation.box(extents=_parse_xyz(primitive.get("size"), (1.0, 1.0, 1.0)))
    if primitive.tag == "cylinder":
        return trimesh.creation.cylinder(
            radius=float(primitive.attrib["radius"]),
            height=float(primitive.attrib["length"]),
            sections=32,
        )
    if primitive.tag == "sphere":
        return trimesh.creation.uv_sphere(radius=float(primitive.attrib["radius"]), count=[16, 16])
    raise ValueError(f"Unsupported collision primitive: {primitive.tag}")


def _sample_mesh_surface(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> np.ndarray:
    triangles = np.asarray(mesh.triangles, dtype=np.float32)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if len(triangles) == 0 or not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Collision mesh must contain non-degenerate triangles")

    face_indices = rng.choice(len(triangles), size=count, p=areas / areas.sum())
    selected = triangles[face_indices]
    barycentric = rng.random((count, 2), dtype=np.float32)
    barycentric[:, 0] = 1.0 - np.sqrt(barycentric[:, 0])
    barycentric[:, 1] *= 1.0 - barycentric[:, 0]
    return (
        selected[:, 0]
        + barycentric[:, :1] * (selected[:, 1] - selected[:, 0])
        + barycentric[:, 1:] * (selected[:, 2] - selected[:, 0])
    )


def _fixed_link_to_body_transform(
        link_name: str,
        joints_by_child: dict[str, ET.Element],
    ) -> tuple[str, np.ndarray, np.ndarray] | None:
    """Return the nearest FK body and its transform from ``link_name``."""
    rotation = np.eye(3, dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)
    while link_name not in BODY_INDEX:
        joint = joints_by_child.get(link_name)
        if joint is None or joint.get("type") != "fixed":
            return None
        origin = joint.find("origin")
        joint_xyz = _parse_xyz(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
        joint_rpy = _parse_xyz(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
        joint_rot = _rpy_to_matrix(joint_rpy)
        translation = joint_rot @ translation + joint_xyz
        rotation = joint_rot @ rotation
        parent = joint.find("parent")
        if parent is None or parent.get("link") is None:
            return None
        link_name = parent.attrib["link"]
    return link_name, translation, rotation


@lru_cache(maxsize=None)
def _collision_query_spec(
        num_points: int = DEFAULT_NUM_COLLISION_QUERY_POINTS,
        urdf_path: str | Path = G1_URDF_PATH,
    ) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Build deterministic link-local surface points from URDF collision geometry."""
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    joints_by_child = {
        child.attrib["link"]: joint
        for joint in root.findall("joint")
        if (child := joint.find("child")) is not None and child.get("link") is not None
    }
    collision_meshes: list[tuple[str, trimesh.Trimesh, np.ndarray, np.ndarray]] = []

    for link in root.findall("link"):
        body_transform = _fixed_link_to_body_transform(link.attrib["name"], joints_by_child)
        # Collision links driven by a non-fixed joint missing from FKEngine cannot
        # be reconstructed from a rollout's body_pos_w/body_quat_w tensors.
        if body_transform is None:
            continue
        body_name, body_xyz, body_rot = body_transform
        for collision in link.findall("collision"):
            origin = collision.find("origin")
            origin_xyz = _parse_xyz(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
            origin_rpy = _parse_xyz(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
            geometry = collision.find("geometry")
            if geometry is None:
                continue
            mesh = _load_collision_mesh(geometry, urdf_path.parent)
            collision_rot = _rpy_to_matrix(origin_rpy)
            collision_meshes.append(
                (
                    body_name,
                    mesh,
                    body_rot @ origin_xyz + body_xyz,
                    body_rot @ collision_rot,
                )
            )

    if not collision_meshes:
        raise ValueError(f"No usable collision geometry found in {urdf_path}")
    if num_points < len(collision_meshes):
        raise ValueError(
            f"num_points={num_points} is smaller than the {len(collision_meshes)} collision geometries"
        )

    areas = np.asarray([mesh.area for _, mesh, _, _ in collision_meshes], dtype=np.float64)
    minimum = min(MIN_POINTS_PER_COLLISION, num_points // len(collision_meshes))
    counts = np.full(len(collision_meshes), minimum, dtype=np.int32)
    remaining = num_points - int(counts.sum())
    if remaining:
        fractional = areas / areas.sum() * remaining
        extra = np.floor(fractional).astype(np.int32)
        counts += extra
        for index in np.argsort(fractional - extra)[::-1][: remaining - int(extra.sum())]:
            counts[index] += 1

    rng = np.random.default_rng(0)
    query_spec: list[tuple[str, tuple[float, float, float]]] = []
    for (body_name, mesh, origin_xyz, origin_rot), count in zip(collision_meshes, counts.tolist()):
        points_collision = _sample_mesh_surface(mesh, count, rng)
        points_link = points_collision @ origin_rot.T + origin_xyz
        query_spec.extend((body_name, tuple(point.tolist())) for point in points_link)
    return tuple(query_spec)


def _query_body_local_to_world(
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        query_spec: tuple[tuple[str, tuple[float, float, float]], ...],
    ) -> torch.Tensor:
    seq_len = body_pos_w.shape[0]
    body_indices = [BODY_INDEX[body_name] for body_name, _ in query_spec]
    index_tensor = torch.as_tensor(body_indices, device=body_pos_w.device, dtype=torch.long)

    selected_pos = body_pos_w.index_select(dim=1, index=index_tensor)
    selected_quat = body_quat_w.index_select(dim=1, index=index_tensor)
    selected_rot = matrix_from_quat(selected_quat)

    local_points = torch.tensor(
        [point for _, point in query_spec],
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
        *,
        num_points: int = DEFAULT_NUM_COLLISION_QUERY_POINTS,
) -> torch.Tensor:
    """Return world-space samples from the G1 collision surfaces.

    The metric layer consumes rollout body poses directly.  FK, when needed,
    should be performed while constructing the chunk rather than implicitly
    inside this geometry query.
    """
    required = ("body_pos_w", "body_quat_w")
    missing = [name for name in required if name not in chunk]
    if missing:
        raise KeyError(f"Collision query requires {missing}")
    body_pos_w = chunk["body_pos_w"].to(dtype=torch.float32)
    body_quat_w = chunk["body_quat_w"].to(dtype=torch.float32)

    return _query_body_local_to_world(body_pos_w, body_quat_w, _collision_query_spec(num_points))


def get_fingertip_query_points(chunk: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the fixed left/right fingertip centers in ``FINGERTIP_NAMES`` order.

    The returned tensor has shape ``(T, 10, 3)``.  This landmark query is
    independent of the random full-body collision surface samples.
    """
    required = ("body_pos_w", "body_quat_w")
    missing = [name for name in required if name not in chunk]
    if missing:
        raise KeyError(f"Fingertip query requires {missing}")
    body_pos_w = chunk["body_pos_w"].to(dtype=torch.float32)
    body_quat_w = chunk["body_quat_w"].to(dtype=torch.float32)
    return _query_body_local_to_world(body_pos_w, body_quat_w, FINGERTIP_QUERY_SPEC)


def get_query_body_indices(
        *,
        num_points: int = DEFAULT_NUM_COLLISION_QUERY_POINTS,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
    """Return the FK body index that owns each collision query point."""
    query_spec = _collision_query_spec(num_points)
    return torch.as_tensor(
        [BODY_INDEX[body_name] for body_name, _ in query_spec],
        dtype=torch.long,
        device=device,
    )


@lru_cache(maxsize=None)
def _load_object_mesh_template(
        object_name: str,
        *,
        objects_root: str | Path = OBJECTS_ROOT,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``objects/<object_name>/<object_name>.stl`` in object-local coordinates."""
    mesh_path = Path(objects_root) / object_name / f"{object_name}.stl"
    mesh = trimesh.load_mesh(mesh_path, process=False)
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)

def _query_object_points_geometry(
        chunk: dict[str, torch.Tensor],
        object_name: str,
        query_points_w: torch.Tensor,
        *,
        max_dist: float = 1.0e6,
        winding_accuracy: float = 2.0,
        winding_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
    """Query the object mesh from arbitrary world-space body query points.

    ``surface_normal_w`` follows the object mesh triangle winding and should be
    interpreted as the object's outward normal when the mesh is consistently
    oriented.  Every returned tensor is indexed by ``(frame, query_point)``.
    """
    template_vertices, template_faces = _load_object_mesh_template(object_name)
    query_points_local = _query_world_to_object_local(chunk, query_points_w)

    wp_device = wp.get_device("cuda" if query_points_local.is_cuda else "cpu")
    with wp.ScopedDevice(wp_device):
        mesh = wp.Mesh(
            points=wp.array(template_vertices, dtype=wp.vec3, device=wp_device),
            indices=wp.array(template_faces.reshape(-1), dtype=wp.int32, device=wp_device),
            support_winding_number=True,
        )
        query_shape = query_points_local.shape[:-1]
        flat_query_points = query_points_local.reshape(-1, 3).contiguous()
        query_points_wp = wp.from_torch(flat_query_points, dtype=wp.vec3)
        point_count = flat_query_points.shape[0]
        valid_wp = wp.empty(point_count, dtype=wp.int32, device=wp_device)
        faces_wp = wp.empty(point_count, dtype=wp.int32, device=wp_device)
        signs_wp = wp.empty(point_count, dtype=float, device=wp_device)
        barycentric_u_wp = wp.empty(point_count, dtype=float, device=wp_device)
        barycentric_v_wp = wp.empty(point_count, dtype=float, device=wp_device)
        wp.launch(
            kernel=_query_object_mesh_kernel,
            dim=point_count,
            inputs=[
                mesh.id,
                query_points_wp,
                float(max_dist),
                float(winding_accuracy),
                float(winding_threshold),
            ],
            outputs=[valid_wp, faces_wp, signs_wp, barycentric_u_wp, barycentric_v_wp],
            device=wp_device,
        )

    valid_np = valid_wp.numpy().astype(bool)
    faces_np = faces_wp.numpy()
    signs_np = signs_wp.numpy()
    barycentric_u_np = barycentric_u_wp.numpy()
    barycentric_v_np = barycentric_v_wp.numpy()
    query_points_local_np = flat_query_points.detach().cpu().numpy()
    closest_points_local_np = np.full_like(query_points_local_np, np.nan)
    surface_normals_local_np = np.full_like(query_points_local_np, np.nan)
    signed_distance_np = np.full(point_count, float(max_dist), dtype=np.float32)

    valid_indices = np.flatnonzero(valid_np)
    if len(valid_indices):
        triangles = template_vertices[template_faces[faces_np[valid_indices]]]
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        u = barycentric_u_np[valid_indices, None]
        v = barycentric_v_np[valid_indices, None]
        closest = a * (1.0 - u - v) + b * u + c * v
        normal = np.cross(b - a, c - a)
        normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        non_degenerate = normal_norm[:, 0] > 0.0
        valid_indices = valid_indices[non_degenerate]
        closest = closest[non_degenerate]
        normal = normal[non_degenerate] / normal_norm[non_degenerate]
        closest_points_local_np[valid_indices] = closest
        surface_normals_local_np[valid_indices] = normal
        signed_distance_np[valid_indices] = (
            np.where(signs_np[valid_indices] < 0.0, -1.0, 1.0)
            * np.linalg.norm(query_points_local_np[valid_indices] - closest, axis=-1)
        )

    closest_points_local = torch.as_tensor(
        closest_points_local_np.reshape(*query_shape, 3),
        dtype=query_points_local.dtype,
        device=query_points_local.device,
    )
    surface_normals_local = torch.as_tensor(
        surface_normals_local_np.reshape(*query_shape, 3),
        dtype=query_points_local.dtype,
        device=query_points_local.device,
    )
    signed_distance = torch.as_tensor(
        signed_distance_np.reshape(*query_shape),
        dtype=query_points_local.dtype,
        device=query_points_local.device,
    )
    query_valid = torch.as_tensor(
        np.isfinite(surface_normals_local_np).all(axis=-1).reshape(*query_shape),
        dtype=torch.bool,
        device=query_points_local.device,
    )

    object_pos_w = chunk["object_pos_w"].to(
        device=query_points_w.device,
        dtype=query_points_w.dtype,
    )
    object_rot_w = matrix_from_quat(
        chunk["object_quat_w"].to(
            device=query_points_w.device,
            dtype=query_points_w.dtype,
        )
    )
    closest_points_w = (
        torch.matmul(object_rot_w.unsqueeze(1), closest_points_local.unsqueeze(-1)).squeeze(-1)
        + object_pos_w.unsqueeze(1)
    )
    surface_normals_w = torch.matmul(
        object_rot_w.unsqueeze(1),
        surface_normals_local.unsqueeze(-1),
    ).squeeze(-1)

    return {
        "query_points_w": query_points_w,
        "signed_distance": signed_distance,
        "closest_points_w": closest_points_w,
        "surface_normals_w": surface_normals_w,
        "valid": query_valid,
    }


def query_object_geometry(
        chunk: dict[str, torch.Tensor],
        object_name: str,
        *,
        num_query_points: int = DEFAULT_NUM_COLLISION_QUERY_POINTS,
        max_dist: float = 1.0e6,
        winding_accuracy: float = 2.0,
        winding_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
    """Query the object mesh from the full G1 collision-surface point set."""
    query_points_w = get_query_point(chunk, num_points=num_query_points)
    geometry = _query_object_points_geometry(
        chunk,
        object_name,
        query_points_w,
        max_dist=max_dist,
        winding_accuracy=winding_accuracy,
        winding_threshold=winding_threshold,
    )
    geometry["query_body_indices"] = get_query_body_indices(
        num_points=num_query_points,
        device=query_points_w.device,
    )
    return geometry


def query_fingertip_object_geometry(
        chunk: dict[str, torch.Tensor],
        object_name: str,
        *,
        max_dist: float = 1.0e6,
        winding_accuracy: float = 2.0,
        winding_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
    """Query object geometry at the ten fixed fingertip centers.

    Every returned tensor is indexed by ``(frame, fingertip)`` using the order
    in ``FINGERTIP_NAMES``.  The query-point dimension is always 10.
    """
    fingertip_points_w = get_fingertip_query_points(chunk)
    geometry = _query_object_points_geometry(
        chunk,
        object_name,
        fingertip_points_w,
        max_dist=max_dist,
        winding_accuracy=winding_accuracy,
        winding_threshold=winding_threshold,
    )
    geometry["fingertip_points_w"] = fingertip_points_w
    return geometry


def query_hand_inner_object_geometry(
        chunk: dict[str, torch.Tensor],
        object_name: str,
        *,
        points_per_hand: int = 64,
        max_dist: float = 1.0e6,
        winding_accuracy: float = 2.0,
        winding_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
    """Query object geometry at fixed palm-facing hand surface points.

    The returned point dimension is ``2 * points_per_hand`` in the stable
    left-then-right order defined by :mod:`geometry.hand_queries`.
    """
    # Import locally so metric-only workflows do not initialize the hand-query
    # module unless this denser contact representation is requested.
    from geometry.hand_queries import (
        get_hand_inner_query_body_indices,
        get_hand_inner_query_points,
    )

    hand_points_w = get_hand_inner_query_points(chunk, points_per_hand=points_per_hand)
    geometry = _query_object_points_geometry(
        chunk,
        object_name,
        hand_points_w,
        max_dist=max_dist,
        winding_accuracy=winding_accuracy,
        winding_threshold=winding_threshold,
    )
    geometry["hand_inner_points_w"] = hand_points_w
    geometry["query_body_indices"] = get_hand_inner_query_body_indices(
        points_per_hand=points_per_hand,
    ).to(device=hand_points_w.device)
    return geometry
