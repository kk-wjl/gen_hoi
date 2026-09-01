"""
1. hand sliding
2. feet sliding
3. object sliding

what we need:
1. determine when G1 is in contact
support two formats:
contact label and calculated distance
2. relative velocity of contact points

what we try to calculate:
Foot Sliding Score

support two different datasets:
1. generative datasets
root + object + joint + contact
2. rl rollout datasets
body + object + joint + contact

chunk(generated/rl collected):
root/body(pos + lin_vel + ang_vel + quat/rot6d) +
object(pos + lin_vel + ang_vel + quat/rot6d) + joint (pos + vel) 
+ contact(contact label/or more)

missing part: 
normal vector calculation
hand distance calculation: query + minimum
"""

import torch
import torch.nn.functional as F

import numpy as np

from metrics.fk import BODY_INDEX, FKEngine
from metrics.query import query_object_signed_distance

def contact_candidates(body: str):
    # foot: left_ankle_roll_link/ right_ankle_roll_link
    # hand: left_ankle_roll_link/ right_ankle_roll_link
    if body == "foot":
        body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    elif body == "hand":
        body_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    else:
        raise ValueError(f"Unsupported contact body: {body}")
    body_indices = [BODY_INDEX[name] for name in body_names]
    return body_names, body_indices

def compute_surface_normal(
        chunk: dict[str, torch.Tensor],
        body: str,
        surface_normal: torch.Tensor | None = None,
    ):
    # option one: calculate from pose
    # option two: get from simulation
    # foot/ground: z vector
    # hand/object: object surface vector
    _, body_indices = contact_candidates(body)
    seq_len = int(chunk["body_pos_w"].shape[0]) if "body_pos_w" in chunk else int(chunk["joint_pos"].shape[0])
    num_candidates = len(body_indices)

    if surface_normal is not None:
        normal = surface_normal.to(dtype=torch.float32)
        if normal.shape != (seq_len, num_candidates, 3):
            raise ValueError(
                f"surface_normal must have shape {(seq_len, num_candidates, 3)}, got {tuple(normal.shape)}"
            )
        return F.normalize(normal, dim=-1, eps=1e-8)

    if body == "foot":
        normal = torch.zeros(seq_len, num_candidates, 3, dtype=torch.float32)
        normal[..., 2] = 1.0
        return normal

    normal = torch.zeros(seq_len, num_candidates, 3, dtype=torch.float32)
    return normal

def compute_relative_distance(
        chunk: dict[str, torch.Tensor],
        body: str,
        fk_engine: FKEngine | None = None,
        distance: torch.Tensor | None = None, # (T, K hand) setting as larger than 0
    ):
    # calculate relative distance between ground and feet or between hand and object
    # option one: calculate from pose
    # option two: get from simulation
    # input: chunk
    # output: calculated distance
    _, body_indices = contact_candidates(body)

    if "body_pos_w" in chunk:
        body_pos_w = chunk["body_pos_w"].to(dtype=torch.float32)
    else:
        if fk_engine is None:
            fk_engine = FKEngine()
        fk_out = fk_engine.forward_kinematics(
            joint_pos=chunk["joint_pos"],
            root_pos=chunk["root_pos"],
            root_quat=chunk["root_quat"],
        )
        body_pos_w = fk_out["body_pos_w"].to(dtype=torch.float32)

    selected_body_pos = body_pos_w[:, body_indices, :]

    if body == "foot":
        return selected_body_pos[..., 2]
    else:
        return distance
    
def compute_relative_velocity(
        chunk: dict[str, torch.Tensor],
        body: str,
        fk_engine: FKEngine | None = None,
    ):
    # calculate the relative velocity on contact surface if there is contact(by def determine_whether_in_contact)
    # input: chunk
    # output: calculated velocity
    _, body_indices = contact_candidates(body)

    if "body_lin_vel_w" in chunk:
        body_lin_vel_w = chunk["body_lin_vel_w"].to(dtype=torch.float32)
    else:
        if "body_pos_w" in chunk:
            body_pos_w = chunk["body_pos_w"].to(dtype=torch.float32)
        else:
            if fk_engine is None:
                fk_engine = FKEngine()
            fk_out = fk_engine.forward_kinematics(
                joint_pos=chunk["joint_pos"],
                root_pos=chunk["root_pos"],
                root_quat=chunk["root_quat"],
            )
            body_pos_w = fk_out["body_pos_w"].to(dtype=torch.float32)
        body_lin_vel_w = torch.zeros_like(body_pos_w)
        if body_pos_w.shape[0] > 1:
            body_lin_vel_w[1:] = body_pos_w[1:] - body_pos_w[:-1]
            body_lin_vel_w[0] = body_lin_vel_w[1]

    selected_body_vel = body_lin_vel_w[:, body_indices, :]

    if body == "foot":
        return selected_body_vel

    if "object_lin_vel_w" in chunk:
        object_lin_vel_w = chunk["object_lin_vel_w"].to(dtype=torch.float32)
    else:
        if "object_pos_w" not in chunk:
            raise KeyError("Hand relative velocity requires object_pos_w or object_lin_vel_w in chunk")
        object_pos_w = chunk["object_pos_w"].to(dtype=torch.float32)
        object_lin_vel_w = torch.zeros_like(object_pos_w)
        if object_pos_w.shape[0] > 1:
            object_lin_vel_w[1:] = object_pos_w[1:] - object_pos_w[:-1]
            object_lin_vel_w[0] = object_lin_vel_w[1]
    return selected_body_vel - object_lin_vel_w.unsqueeze(1)

def determine_whether_in_contact(
        chunk: dict[str, torch.Tensor],
        contact_label: torch.Tensor | None, # (T, K)
        body: str,
        fk_engine: FKEngine | None = None,
        distance: torch.Tensor | None = None, # (T, K hand)
    ) -> torch.Tensor:
    # determine whether there is a contact
    # option one: by contact label(for each body) 1 is in contact, 0 is not in contact
    # option two: by geometrical relation and forward kinematic(if there is only root not body information)
    _, body_indices = contact_candidates(body)

    if contact_label is None and "contact_label" in chunk:
        contact_label = chunk["contact_label"]

    if contact_label is not None:
        label = contact_label.to(dtype=torch.float32)
        if label.shape[-1] == len(BODY_INDEX):
            label = label[:, body_indices]
        return label > 0.5

    distance = compute_relative_distance(chunk, body=body, fk_engine=fk_engine, distance=distance)
    if body == "foot":
        threshold = torch.tensor(0.05, dtype=distance.dtype)
    else:
        threshold = torch.tensor(0.05, dtype=distance.dtype)
    return distance <= threshold

def compute_contact_weight(
        chunk: dict[str, torch.Tensor],
        contact_label: torch.Tensor | None,
        body: str,
        fk_engine: FKEngine | None = None,
        distance: torch.Tensor | None = None, # (T, K hand)
    ):
    # we use contact_label as probability to be in contact, where 0 means no contact, 1 means in contact
    # we use relative distance as threshold to determine, transform it to probabilty
    # when using contact_label, contact_weight = contact_label
    # when using threshold, contact_weight = f(relative_distance)
    contact_mask = determine_whether_in_contact(
        chunk,
        contact_label=contact_label,
        body=body,
        fk_engine=fk_engine,
        distance=distance
    )
    _, body_indices = contact_candidates(body)

    if contact_label is None and "contact_label" in chunk:
        contact_label = chunk["contact_label"]

    if contact_label is not None:
        label = contact_label.to(dtype=torch.float32)
        if label.shape[-1] == len(BODY_INDEX):
            label = label[:, body_indices]
        return contact_mask.to(dtype=torch.float32) * label.clamp(0.0, 1.0)

    distance = compute_relative_distance(chunk, body=body, fk_engine=fk_engine, distance=distance)
    if body == "foot":
        threshold = torch.tensor(0.05, dtype=distance.dtype)
        softness = torch.tensor(0.01, dtype=distance.dtype)
    else:
        threshold = torch.tensor(0.05, dtype=distance.dtype)
        softness = torch.tensor(0.02, dtype=distance.dtype)
    softness = torch.clamp(softness, min=torch.finfo(distance.dtype).eps)
    raw_weight = torch.sigmoid((threshold - distance) / softness)
    return contact_mask.to(dtype=raw_weight.dtype) * raw_weight

def compute_foot_sliding(
        chunk: dict[str, torch.Tensor], 
        contact_label: torch.Tensor | None,
        distance: torch.Tensor | None = None, # (T, K hand)
    ):
    # calculate the foot sliding score for the given chunk
    # input: chunk
    # output: FS score
    v_rel = compute_relative_velocity(chunk, body="foot")
    n = compute_surface_normal(chunk, body="foot")
    contact_weight = compute_contact_weight(chunk, contact_label=contact_label, body="foot", distance=distance)

    tangent_v = v_rel - torch.sum(v_rel * n, dim=-1, keepdim=True) * n
    tangent_speed = torch.linalg.norm(tangent_v, dim=-1)

    weighted_speed = contact_weight * tangent_speed
    return weighted_speed.mean()

def compute_hand_sliding(
        chunk: dict[str, torch.Tensor], 
        contact_label: torch.Tensor | None,
        distance: torch.Tensor | None = None, # (T, K hand)
    ):
    # calculate the hand sliding score for the given chunk
    # input: chunk
    # output: HS score
    v_rel = compute_relative_velocity(chunk, body="hand")
    n = compute_surface_normal(chunk, body="hand")
    contact_weight = compute_contact_weight(chunk, contact_label=contact_label, body="hand", distance=distance)

    tangent_v = v_rel - torch.sum(v_rel * n, dim=-1, keepdim=True) * n
    tangent_speed = torch.linalg.norm(tangent_v, dim=-1)

    weighted_speed = contact_weight * tangent_speed
    return weighted_speed.mean()


"""
This scripts are used to determine the penetration scores and floating score.

"""

# maybe adding contact_margin
def compute_foot_penetration(
        chunk: dict[str, torch.Tensor],
    ):
    # get foot height and calculate penetration depth for each body during the frame
    # penetration depth: (T, K)
    # penetration score
    foot_height = compute_relative_distance(chunk, body="foot")
    penetration_depth = torch.clamp(-foot_height, min=0.0)
    penetration_score = penetration_depth.mean() * 100.0
    return penetration_score

def compute_foot_floating(
        chunk: dict[str, torch.Tensor],
        contact_label: torch.Tensor | None,
    ):
    # when contact_label says true
    # get foot height and compare with threholds
    # floating height: (T, K)
    # floating score
    foot_height = compute_relative_distance(chunk, body="foot")
    floating_height = torch.clamp(foot_height, min=0.0) 
    contact_weight = compute_contact_weight(chunk, contact_label=contact_label, body="foot")
    floating_score = (floating_height * contact_weight).mean() * 100.0
    return floating_score

# after getting collision information between object and ground
def get_object_height(chunk: dict[str, torch.Tensor],):
    # object baseline height (T,)
    object_pos_w = chunk["object_pos_w"]   # (T, 3)
    seq_len = object_pos_w.shape[0]
    object_baseline_height = torch.full(
        (seq_len,),
        0.361,
        dtype=object_pos_w.dtype,
        device=object_pos_w.device,
    )
    return object_baseline_height

def compute_object_floating(
        chunk: dict[str, torch.Tensor],
        contact_label: torch.Tensor | None,
        distance: torch.Tensor | None = None, # (T, K hand)
    ):
    # object should not be floating when in contact
    # object should be in ground when no contact
    contact_weight = compute_contact_weight(chunk, contact_label=contact_label, body="hand", distance=distance)
    contact_weight = contact_weight.max(dim=-1).values # either hand in contact

    distance = compute_relative_distance(chunk, body="hand", distance = distance) # (T, K)
    hand_distance = distance.min(dim=-1).values # (T, )
    object_height = get_object_height(chunk) # (T, )
    object_pos_w = chunk["object_pos_w"] # (T, 3)
    floating_height = torch.clamp(object_pos_w[..., 2] - object_height, min=0.0)

    floating_score = (floating_height * (1 - contact_weight) + hand_distance * contact_weight).mean() * 100.0
    return floating_score

def compute_object_penetration(
        chunk: dict[str, torch.Tensor],
        object_name: str,
        fk_engine: FKEngine | None = None,
    ):
    # penetration with body
    # not with ground yet
    signed_distance = query_object_signed_distance(
        chunk,
        object_name=object_name,
        fk_engine=fk_engine,
    )
    penetration_depth = torch.clamp(-signed_distance, min=0.0)
    penetration_score = penetration_depth.mean() * 100.0
    return penetration_score

def free_object_motion_penalty(
        chunk: dict[str, torch.Tensor],
        contact_label: torch.Tensor | None,
        distance: torch.Tensor | None = None, # (T, K hand)
    ):
    # object should be still when no contact
    # OS score = v * contact_weight
    v_rel = chunk["object_lin_vel_w"]
    contact_weight = compute_contact_weight(chunk, contact_label=contact_label, body="hand", distance=distance)
    contact_weight = contact_weight.max(dim=-1).values # either hand in contact
    object_contact_weight = 1 - contact_weight

    speed = torch.linalg.norm(v_rel, dim=-1)

    weighted_speed = object_contact_weight * speed
    return weighted_speed.mean()

def compute_contact_percentage(
        chunk: dict[str, torch.Tensor],
        contact_label: torch.Tensor | None,
        distance: torch.Tensor | None = None, # (T, K hand)
    ):
    # contact time/ total time
    contact_mask = determine_whether_in_contact(
            chunk,
            contact_label=contact_label,
            body="hand",
            fk_engine=None,
            distance=distance
        )
    contact_frame_mask = contact_mask.any(dim=-1)   # (T,)
    contact_percentage = contact_frame_mask.float().mean()
    return contact_percentage

def compute_contact_accuracy():
    # not sure yet
    # precision/recall/F1 score
    # ground truth: contact label?
    # predicted: distance?
    pass

"""
smoothness metrics:
acceleration
jerk
diversity metrics:
"""
