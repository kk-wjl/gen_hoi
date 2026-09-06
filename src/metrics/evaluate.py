"""Geometry-based metrics for generated G1-object motions."""

from __future__ import annotations

import torch

from metrics.fk import BODY_INDEX
from metrics.query import (
    get_query_body_indices,
    get_query_point,
    query_object_geometry,
)


def _region_body_indices(region: str) -> list[int]:
    if region == "left_foot":
        names = {"left_ankle_roll_link"}
    elif region == "right_foot":
        names = {"right_ankle_roll_link"}
    elif region == "left_hand":
        names = {"left_wrist_yaw_link"}
        names.update(name for name in BODY_INDEX if name.startswith("L_"))
    elif region == "right_hand":
        names = {"right_wrist_yaw_link"}
        names.update(name for name in BODY_INDEX if name.startswith("R_"))
    elif region == "foot":
        return _region_body_indices("left_foot") + _region_body_indices("right_foot")
    elif region == "hand":
        return _region_body_indices("left_hand") + _region_body_indices("right_hand")
    else:
        raise ValueError(f"Unsupported collision-query region: {region}")
    return [BODY_INDEX[name] for name in names]


def query_region_mask(query_body_indices: torch.Tensor, region: str) -> torch.Tensor:
    """Select collision query points belonging to a semantic G1 region."""
    region_indices = torch.as_tensor(
        _region_body_indices(region),
        dtype=torch.long,
        device=query_body_indices.device,
    )
    mask = torch.isin(query_body_indices, region_indices)
    if not mask.any():
        raise ValueError(f"No collision query points found for region {region!r}")
    return mask


def _collision_geometry(
        chunk: dict[str, torch.Tensor],
        geometry: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
    if geometry is not None:
        return geometry
    query_points_w = get_query_point(
        chunk,
    )
    return {
        "query_points_w": query_points_w,
        "query_body_indices": get_query_body_indices(
            device=query_points_w.device,
        ),
    }


def _query_contact_weight(
        chunk: dict[str, torch.Tensor],
        query_body_indices: torch.Tensor,
        contact_label: torch.Tensor | None,
    ) -> torch.Tensor:
    if contact_label is None:
        contact_label = chunk.get("contact_label")
    if contact_label is None:
        raise KeyError("Collision-query metrics require per-link contact_label")
    if contact_label.ndim != 2 or contact_label.shape[1] != len(BODY_INDEX):
        raise ValueError(
            f"contact_label must have shape (T, {len(BODY_INDEX)}), got {tuple(contact_label.shape)}"
        )
    return contact_label.to(
        device=query_body_indices.device,
        dtype=torch.float32,
    ).index_select(1, query_body_indices).clamp(0.0, 1.0)


def _region_contact_weight(
        chunk: dict[str, torch.Tensor],
        owner: torch.Tensor,
        region: str,
        contact_label: torch.Tensor | None,
    ) -> torch.Tensor:
    return _query_contact_weight(chunk, owner, contact_label) * query_region_mask(owner, region).unsqueeze(0)


def _region_contact_frames(
        chunk: dict[str, torch.Tensor],
        owner: torch.Tensor,
        region: str,
        contact_label: torch.Tensor | None,
    ) -> torch.Tensor:
    return _region_contact_weight(chunk, owner, region, contact_label).amax(dim=1)


def _query_point_velocity(
        chunk: dict[str, torch.Tensor],
        geometry: dict[str, torch.Tensor],
    ) -> torch.Tensor:
    required = ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w")
    missing = [name for name in required if name not in chunk]
    if missing:
        raise KeyError(f"Query-point velocity requires {missing}")

    owner = geometry["query_body_indices"]
    query_points_w = geometry["query_points_w"]
    body_pos_w = chunk["body_pos_w"].to(device=query_points_w.device, dtype=query_points_w.dtype)
    body_lin_vel_w = chunk["body_lin_vel_w"].to(device=query_points_w.device, dtype=query_points_w.dtype)
    body_ang_vel_w = chunk["body_ang_vel_w"].to(device=query_points_w.device, dtype=query_points_w.dtype)
    owner_pos = body_pos_w.index_select(1, owner)
    owner_lin_vel = body_lin_vel_w.index_select(1, owner)
    owner_ang_vel = body_ang_vel_w.index_select(1, owner)
    return owner_lin_vel + torch.cross(owner_ang_vel, query_points_w - owner_pos, dim=-1)


def compute_hand_object_distance(
        chunk: dict[str, torch.Tensor],
        object_name: str,
        *,
        geometry: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
    """Return left/right hand minimum outside distances to the object, shape ``(T, 2)``."""
    if geometry is None:
        geometry = query_object_geometry(
            chunk,
            object_name,
        )
    owner = geometry["query_body_indices"]
    outside_distance = geometry["signed_distance"].clamp_min(0.0)
    valid = geometry["valid"]
    distances: list[torch.Tensor] = []
    for region in ("left_hand", "right_hand"):
        mask = query_region_mask(owner, region)
        distances.append(outside_distance.masked_fill(~(valid & mask.unsqueeze(0)), float("inf")).min(dim=1).values)
    return torch.stack(distances, dim=-1)


def compute_hand_object_sliding(
        chunk: dict[str, torch.Tensor],
        object_name: str,
        contact_label: torch.Tensor | None = None,
        *,
        geometry: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
    """Contact-label-weighted tangential velocity across hand collision surfaces."""
    if geometry is None:
        geometry = query_object_geometry(
            chunk,
            object_name,
        )
    required = ("object_pos_w", "object_lin_vel_w", "object_ang_vel_w")
    missing = [name for name in required if name not in chunk]
    if missing:
        raise KeyError(f"Hand-object sliding requires {missing}")

    query_points_w = geometry["query_points_w"]
    owner = geometry["query_body_indices"]
    object_pos_w = chunk["object_pos_w"].to(device=query_points_w.device, dtype=query_points_w.dtype)
    object_lin_vel_w = chunk["object_lin_vel_w"].to(device=query_points_w.device, dtype=query_points_w.dtype)
    object_ang_vel_w = chunk["object_ang_vel_w"].to(device=query_points_w.device, dtype=query_points_w.dtype)
    object_surface_velocity_w = object_lin_vel_w.unsqueeze(1) + torch.cross(
        object_ang_vel_w.unsqueeze(1),
        geometry["closest_points_w"] - object_pos_w.unsqueeze(1),
        dim=-1,
    )
    relative_velocity_w = _query_point_velocity(chunk, geometry) - object_surface_velocity_w
    normal_w = geometry["surface_normals_w"]
    tangent_speed = torch.linalg.norm(
        relative_velocity_w - (relative_velocity_w * normal_w).sum(dim=-1, keepdim=True) * normal_w,
        dim=-1,
    )
    weight = _region_contact_weight(chunk, owner, "hand", contact_label)
    weight = weight * geometry["valid"].to(dtype=weight.dtype)
    return (weight * tangent_speed).sum() / (weight.shape[0] * query_region_mask(owner, "hand").sum())


def compute_hand_sliding(
        chunk: dict[str, torch.Tensor],
        contact_label: torch.Tensor | None,
        object_name: str,
        *,
        geometry: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
    """Compatibility entry point for collision-query hand-object sliding."""
    return compute_hand_object_sliding(
        chunk,
        object_name,
        contact_label=contact_label,
        geometry=geometry,
    )


def compute_foot_sliding(
        chunk: dict[str, torch.Tensor],
        contact_label: torch.Tensor | None,
        *,
        geometry: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
    """Contact-label-weighted horizontal velocity of foot collision points."""
    geometry = _collision_geometry(chunk, geometry)
    owner = geometry["query_body_indices"]
    foot_mask = query_region_mask(owner, "foot")
    horizontal_speed = torch.linalg.norm(_query_point_velocity(chunk, geometry)[..., :2], dim=-1)
    weight = _region_contact_weight(chunk, owner, "foot", contact_label)
    return (weight * horizontal_speed).sum() / (weight.shape[0] * foot_mask.sum())


def compute_foot_penetration(
        chunk: dict[str, torch.Tensor],
        *,
        geometry: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Mean floor penetration of the lowest point on each foot collision surface, in cm."""
    geometry = _collision_geometry(chunk, geometry)
    owner = geometry["query_body_indices"]
    foot_height = torch.stack(
        [
            geometry["query_points_w"][:, query_region_mask(owner, region), 2].min(dim=1).values
            for region in ("left_foot", "right_foot")
        ],
        dim=-1,
    )
    return foot_height.clamp_max(0.0).neg().mean() * 100.0


def compute_foot_floating(
        chunk: dict[str, torch.Tensor],
        contact_label: torch.Tensor | None,
        *,
        geometry: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Mean clearance above the floor when each foot is labelled in contact, in cm."""
    geometry = _collision_geometry(chunk, geometry)
    owner = geometry["query_body_indices"]
    foot_height = torch.stack(
        [
            geometry["query_points_w"][:, query_region_mask(owner, region), 2].min(dim=1).values
            for region in ("left_foot", "right_foot")
        ],
        dim=-1,
    )
    contact = torch.stack(
        [_region_contact_frames(chunk, owner, region, contact_label) for region in ("left_foot", "right_foot")],
        dim=-1,
    )
    return (foot_height.clamp_min(0.0) * contact).mean() * 100.0


def get_object_height(chunk: dict[str, torch.Tensor]) -> torch.Tensor:
    """Legacy smalltable resting height used by ``compute_object_floating``."""
    return torch.full(
        (chunk["object_pos_w"].shape[0],),
        0.361,
        dtype=chunk["object_pos_w"].dtype,
        device=chunk["object_pos_w"].device,
    )


def compute_object_floating(
        chunk: dict[str, torch.Tensor],
        contact_label: torch.Tensor | None,
        distance: torch.Tensor,
        *,
        geometry: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
    """Object clearance when free, or hand-object gap when the hand is in contact, in cm."""
    geometry = _collision_geometry(chunk, geometry)
    hand_contact = _region_contact_frames(chunk, geometry["query_body_indices"], "hand", contact_label)
    hand_distance = distance.min(dim=-1).values
    free_clearance = (chunk["object_pos_w"][..., 2] - get_object_height(chunk)).clamp_min(0.0)
    return (free_clearance * (1.0 - hand_contact) + hand_distance * hand_contact).mean() * 100.0


def compute_object_penetration(
        chunk: dict[str, torch.Tensor],
        object_name: str,
        *,
        geometry: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
    if geometry is None:
        geometry = query_object_geometry(
            chunk,
            object_name=object_name,
        )
    return geometry["signed_distance"].clamp_max(0.0).neg().mean() * 100.0


def free_object_motion_penalty(
        chunk: dict[str, torch.Tensor],
        contact_label: torch.Tensor | None,
        *,
        geometry: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
    geometry = _collision_geometry(chunk, geometry)
    hand_contact = _region_contact_frames(chunk, geometry["query_body_indices"], "hand", contact_label)
    return ((1.0 - hand_contact) * torch.linalg.norm(chunk["object_lin_vel_w"], dim=-1)).mean()


def compute_contact_percentage(
        chunk: dict[str, torch.Tensor],
        contact_label: torch.Tensor | None,
        *,
        geometry: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
    geometry = _collision_geometry(chunk, geometry)
    return _region_contact_frames(chunk, geometry["query_body_indices"], "hand", contact_label).mean()


def compute_contact_accuracy():
    """Reserved for precision/recall/F1 against geometry-derived contacts."""
    pass
