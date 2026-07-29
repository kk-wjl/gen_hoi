from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION_PATH = REPO_ROOT / "vis" / "data" / "smalltable_vis.npz"
DEFAULT_ROBOT_XML = REPO_ROOT / "vis" / "assets" / "unitree_g1" / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml"
DEFAULT_OBJECT_STL = REPO_ROOT / "vis" / "assets" / "smalltable.stl"
DEFAULT_OBJECT_NAME = "smalltable"

KEY_SPACE = 32
KEY_LEFT = 263
KEY_RIGHT = 262
KEY_R = 82

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "R_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
]


def build_scene_xml(robot_xml: Path, object_stl: Path, object_name: str) -> str:
    xml = robot_xml.read_text()
    meshdir = robot_xml.parent.resolve()
    object_stl = object_stl.resolve()

    if 'meshdir="' not in xml:
        xml = xml.replace(
            '<compiler angle="radian"/>',
            f'<compiler angle="radian" meshdir="{meshdir}"/>',
        )

    object_asset = f'\n    <mesh name="{object_name}" file="{object_stl}"/>\n  '
    object_body = f"""
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.85 0.85 0.85 1"/>
    <body name="{object_name}" mocap="true" pos="0 0 0" quat="1 0 0 0">
      <geom type="mesh" mesh="{object_name}" rgba="0.2 0.7 0.3 1" contype="0" conaffinity="0" group="1"/>
    </body>
  """
    xml = xml.replace("</asset>", f"{object_asset}</asset>")
    xml = xml.replace("</worldbody>", f"{object_body}</worldbody>")
    return xml


def load_motion(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    motion = {
        "joint_pos": np.asarray(data["joint_pos"], dtype=np.float64),
        "root_pos": np.asarray(data["root_pos"], dtype=np.float64),
        "root_quat": np.asarray(data["root_quat"], dtype=np.float64),
        "object_pos": np.asarray(data["object_pos"], dtype=np.float64),
        "object_quat": np.asarray(data["object_quat"], dtype=np.float64),
        "fps": float(np.asarray(data["fps"]).reshape(-1)[0]),
    }
    num_frames = motion["joint_pos"].shape[0]
    expected_shapes = {
        "root_pos": (num_frames, 3),
        "root_quat": (num_frames, 4),
        "object_pos": (num_frames, 3),
        "object_quat": (num_frames, 4),
    }
    for key, expected in expected_shapes.items():
        if motion[key].shape != expected:
            raise ValueError(f"Unexpected shape for {key}: {motion[key].shape}, expected {expected}")
    if motion["joint_pos"].shape[1] != len(JOINT_NAMES):
        raise ValueError(
            f"joint_pos has {motion['joint_pos'].shape[1]} dims, expected {len(JOINT_NAMES)}"
        )
    return motion


def joint_qpos_addresses(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    addrs = np.empty(len(joint_names), dtype=np.int32)
    for i, name in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Joint {name!r} not found in MuJoCo model")
        addrs[i] = model.jnt_qposadr[jid]
    return addrs


def apply_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion: dict[str, np.ndarray],
    frame_idx: int,
    root_qposadr: int,
    joint_qposadr: np.ndarray,
    object_mocap_id: int,
) -> None:
    data.qpos[root_qposadr : root_qposadr + 3] = motion["root_pos"][frame_idx]
    data.qpos[root_qposadr + 3 : root_qposadr + 7] = motion["root_quat"][frame_idx]
    data.qpos[joint_qposadr] = motion["joint_pos"][frame_idx]

    data.mocap_pos[object_mocap_id] = motion["object_pos"][frame_idx]
    data.mocap_quat[object_mocap_id] = motion["object_quat"][frame_idx]
    mujoco.mj_forward(model, data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize G1 HOI motion in the MuJoCo native viewer.")
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION_PATH, help="Path to motion npz.")
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML, help="Path to the robot MJCF XML.")
    parser.add_argument("--object-stl", type=Path, default=DEFAULT_OBJECT_STL, help="Path to the object STL mesh.")
    parser.add_argument("--object-name", type=str, default=DEFAULT_OBJECT_NAME, help="Object name used in the MuJoCo scene.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion = load_motion(args.motion.resolve())

    scene_xml = build_scene_xml(args.robot_xml.resolve(), args.object_stl.resolve(), args.object_name)
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)

    free_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if free_jid < 0:
        raise KeyError("Joint 'floating_base_joint' not found in MuJoCo model")
    root_qposadr = int(model.jnt_qposadr[free_jid])
    joint_qposadr = joint_qpos_addresses(model, JOINT_NAMES)

    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.object_name)
    if object_body_id < 0:
        raise KeyError(f"Body {args.object_name!r} not found in MuJoCo model")
    object_mocap_id = int(model.body(args.object_name).mocapid[0])

    num_frames = motion["joint_pos"].shape[0]
    dt = 1.0 / max(motion["fps"], 1.0)

    frame = 0
    paused = False
    step_delta = 0
    reset_requested = False

    def key_callback(keycode: int) -> None:
        nonlocal paused, frame, step_delta, reset_requested
        if keycode == KEY_SPACE:
            paused = not paused
            state = "paused" if paused else "playing"
            print(f"[viewer] {state} at frame {frame}")
        elif keycode == KEY_LEFT:
            paused = True
            step_delta = -1
        elif keycode == KEY_RIGHT:
            paused = True
            step_delta = 1
        elif keycode == KEY_R:
            paused = True
            reset_requested = True

    print(f"[INFO] Loaded {num_frames} frames @ {motion['fps']:.1f} fps")
    print("[INFO] Controls: Space = pause/resume, Left/Right = step, R = reset")

    apply_frame(model, data, motion, frame, root_qposadr, joint_qposadr, object_mocap_id)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if reset_requested:
                frame = 0
                reset_requested = False
            elif step_delta != 0:
                frame = (frame + step_delta) % num_frames
                step_delta = 0
            elif not paused:
                frame = (frame + 1) % num_frames

            apply_frame(model, data, motion, frame, root_qposadr, joint_qposadr, object_mocap_id)
            viewer.sync()
            time.sleep(dt if not paused else 0.01)


if __name__ == "__main__":
    main()
