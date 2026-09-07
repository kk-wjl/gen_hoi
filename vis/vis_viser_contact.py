from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np

import pinocchio as pin
import trimesh
import viser


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION_PATH = REPO_ROOT / "data" / "geometry" / "smalltable_42clip_var5_policy_success_geometry.npz"
DEFAULT_URDF_PATH = REPO_ROOT / "vis" / "assets" / "unitree_g1" / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
DEFAULT_OBJECT_MESH_PATH = REPO_ROOT / "vis" / "assets" / "smalltable.stl"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
OBJECT_SCALE = 1.0
CONTACT_COLOR = np.array([255, 0, 0, 255], dtype=np.uint8)
CLOSEST_POINT_COLOR = np.array([0, 102, 255], dtype=np.uint8)
NORMAL_COLOR = np.array([255, 196, 0], dtype=np.uint8)
NORMAL_LENGTH = 0.06
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

BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "L_index_proximal",
    "L_middle_proximal",
    "L_pinky_proximal",
    "L_ring_proximal",
    "L_thumb_proximal_base",
    "R_index_proximal",
    "R_middle_proximal",
    "R_pinky_proximal",
    "R_ring_proximal",
    "R_thumb_proximal_base",
    "L_index_intermediate",
    "L_middle_intermediate",
    "L_pinky_intermediate",
    "L_ring_intermediate",
    "L_thumb_proximal",
    "R_index_intermediate",
    "R_middle_intermediate",
    "R_pinky_intermediate",
    "R_ring_intermediate",
    "R_thumb_proximal",
    "L_thumb_intermediate",
    "R_thumb_intermediate",
    "L_thumb_distal",
    "R_thumb_distal",
]

BODY_INDEX = {name: idx for idx, name in enumerate(BODY_NAMES)}

def normalize_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    quat = quat / norm
    sign = np.where(quat[..., :1] < 0.0, -1.0, 1.0).astype(np.float32)
    return quat * sign


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = normalize_wxyz(quat_wxyz)
    return np.concatenate([quat_wxyz[..., 1:], quat_wxyz[..., :1]], axis=-1)


def wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = normalize_wxyz(quat_wxyz)
    w, x, y, z = (quat_wxyz[..., index] for index in range(4))
    return np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(*quat_wxyz.shape[:-1], 3, 3)


def rotmat_to_wxyz(rotmat: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotmat))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotmat[2, 1] - rotmat[1, 2]) * s
        y = (rotmat[0, 2] - rotmat[2, 0]) * s
        z = (rotmat[1, 0] - rotmat[0, 1]) * s
    else:
        diag = np.diag(rotmat)
        if diag[0] > diag[1] and diag[0] > diag[2]:
            s = 2.0 * np.sqrt(1.0 + rotmat[0, 0] - rotmat[1, 1] - rotmat[2, 2])
            w = (rotmat[2, 1] - rotmat[1, 2]) / s
            x = 0.25 * s
            y = (rotmat[0, 1] + rotmat[1, 0]) / s
            z = (rotmat[0, 2] + rotmat[2, 0]) / s
        elif diag[1] > diag[2]:
            s = 2.0 * np.sqrt(1.0 + rotmat[1, 1] - rotmat[0, 0] - rotmat[2, 2])
            w = (rotmat[0, 2] - rotmat[2, 0]) / s
            x = (rotmat[0, 1] + rotmat[1, 0]) / s
            y = 0.25 * s
            z = (rotmat[1, 2] + rotmat[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotmat[2, 2] - rotmat[0, 0] - rotmat[1, 1])
            w = (rotmat[1, 0] - rotmat[0, 1]) / s
            x = (rotmat[0, 2] + rotmat[2, 0]) / s
            y = (rotmat[1, 2] + rotmat[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float32)
    return normalize_wxyz(quat)


def load_motion(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        if "root_pos_w" in data and "root_quat_w" in data:
            root_pos_w = np.asarray(data["root_pos_w"], dtype=np.float32)
            root_quat_w = normalize_wxyz(data["root_quat_w"])
        elif "body_pos_w" in data and "body_quat_w" in data:
            root_pos_w = np.asarray(data["body_pos_w"][:, 0, :], dtype=np.float32)
            root_quat_w = normalize_wxyz(data["body_quat_w"][:, 0, :])
        else:
            raise KeyError(
                "Motion NPZ requires root_pos_w/root_quat_w or body_pos_w/body_quat_w for the root pose"
            )
        motion = {
            "joint_pos": np.asarray(data["joint_pos"], dtype=np.float32),
            "root_pos_w": root_pos_w,
            "root_quat_w": root_quat_w,
            "object_pos_w": np.asarray(data["object_pos_w"], dtype=np.float32),
            "object_quat_w": normalize_wxyz(data["object_quat_w"]),
            "fps": np.asarray(data["fps"], dtype=np.float32),
        }
        if "contact_label" in data:
            motion["contact_label"] = np.asarray(data["contact_label"], dtype=np.float32)
        object_rot_w = wxyz_to_rotmat(motion["object_quat_w"])
        if "fingertip_closest_points_w" in data:
            motion["fingertip_closest_points_w"] = np.asarray(data["fingertip_closest_points_w"], dtype=np.float32)
        elif "fingertip_closest_points_o" in data:
            closest_points_o = np.asarray(data["fingertip_closest_points_o"], dtype=np.float32)
            motion["fingertip_closest_points_w"] = (
                np.einsum("tij,tkj->tki", object_rot_w, closest_points_o)
                + motion["object_pos_w"][:, None, :]
            )
        if "fingertip_surface_normals_w" in data:
            motion["fingertip_surface_normals_w"] = np.asarray(data["fingertip_surface_normals_w"], dtype=np.float32)
        elif "fingertip_surface_normals_o" in data:
            surface_normals_o = np.asarray(data["fingertip_surface_normals_o"], dtype=np.float32)
            motion["fingertip_surface_normals_w"] = np.einsum(
                "tij,tkj->tki",
                object_rot_w,
                surface_normals_o,
            )

    num_frames = motion["joint_pos"].shape[0]
    expected_shapes = {
        "root_pos_w": (num_frames, 3),
        "root_quat_w": (num_frames, 4),
        "object_pos_w": (num_frames, 3),
        "object_quat_w": (num_frames, 4),
    }
    for key, expected_shape in expected_shapes.items():
        if motion[key].shape != expected_shape:
            raise ValueError(f"Unexpected shape for {key}: {motion[key].shape}, expected {expected_shape}")
    if "contact_label" in motion:
        expected_shape = (num_frames, len(BODY_NAMES))
        if motion["contact_label"].shape != expected_shape:
            raise ValueError(
                f"Unexpected shape for contact_label: {motion['contact_label'].shape}, "
                f"expected {expected_shape}"
            )
    for key in ("fingertip_closest_points_w", "fingertip_surface_normals_w"):
        if key in motion and motion[key].shape != (num_frames, 10, 3):
            raise ValueError(
                f"Unexpected shape for {key}: {motion[key].shape}, expected {(num_frames, 10, 3)}"
            )
    return motion


def build_robot_models(urdf_path: Path) -> tuple[pin.Model, pin.GeometryModel]:
    model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
    geometry_model = pin.buildGeomFromUrdf(
        model,
        str(urdf_path),
        pin.GeometryType.VISUAL,
        package_dirs=[str(urdf_path.parent)],
    )
    return model, geometry_model


def build_joint_q_index(model: pin.Model, joint_names: list[str]) -> np.ndarray:
    q_indices = np.empty(len(joint_names), dtype=np.int32)
    for i, joint_name in enumerate(joint_names):
        joint_id = model.getJointId(joint_name)
        if joint_id == 0:
            raise KeyError(f"Joint {joint_name!r} not found in Pinocchio model")
        joint = model.joints[joint_id]
        if joint.nq != 1:
            raise ValueError(f"Joint {joint_name!r} has nq={joint.nq}, expected 1")
        q_indices[i] = joint.idx_q
    return q_indices


def build_configuration(
    model: pin.Model,
    joint_q_indices: np.ndarray,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_pos: np.ndarray,
) -> np.ndarray:
    if joint_pos.shape[-1] != len(joint_q_indices):
        raise ValueError(f"joint_pos has {joint_pos.shape[-1]} dims, but expected {len(joint_q_indices)}")
    q = np.zeros(model.nq, dtype=np.float64)
    q[:3] = root_pos.astype(np.float64)
    q[3:7] = wxyz_to_xyzw(root_quat_wxyz).astype(np.float64)
    q[joint_q_indices] = joint_pos.astype(np.float64)
    return q


class MotionViewer:
    def __init__(
        self,
        motion_path: Path,
        urdf_path: Path,
        object_mesh_path: Path,
        host: str,
        port: int,
    ) -> None:
        self.motion_path = motion_path
        self.object_mesh_path = object_mesh_path
        self.motion = load_motion(motion_path)
        self.model, self.visual_model = build_robot_models(urdf_path)
        self.data = self.model.createData()
        self.visual_data = pin.GeometryData(self.visual_model)
        self.joint_q_indices = build_joint_q_index(self.model, JOINT_NAMES)

        self.server = viser.ViserServer(host=host, port=port)
        self.num_frames = self.motion["joint_pos"].shape[0]
        self.fps = float(self.motion["fps"].reshape(-1)[0])
        self.frame_dt = 1.0 / max(self.fps, 1e-6)

        self.frame_lock = threading.Lock()
        self.current_frame = 0
        self.mesh_handles: dict[str, object] = {}
        self.contact_mesh_handles: dict[str, object] = {}
        self.geometry_body_indices: dict[str, int] = {}
        self.mesh_cache: dict[Path, trimesh.Trimesh] = {}
        self.closest_points_handle: object | None = None
        self.surface_normals_handle: object | None = None

        self._setup_scene()
        self._setup_gui()
        self._render_frame(0)

    def _setup_scene(self) -> None:
        self.server.scene.world_axes.visible = True
        self.server.scene.add_grid(
            "/ground",
            width=4.0,
            height=4.0,
            plane="xy",
            cell_size=0.2,
            section_size=1.0,
        )

        for geometry_object in self.visual_model.geometryObjects:
            mesh_path = Path(geometry_object.meshPath)
            if not mesh_path.suffix:
                continue
            mesh = self._load_mesh(mesh_path)
            handle = self.server.scene.add_mesh_trimesh(
                f"/robot/{geometry_object.name}",
                mesh=mesh,
            )
            self.mesh_handles[geometry_object.name] = handle

            body_index = self._geometry_body_index(geometry_object)
            if body_index is not None and "contact_label" in self.motion:
                contact_mesh = mesh.copy()
                contact_mesh.visual = trimesh.visual.ColorVisuals(
                    mesh=contact_mesh,
                    face_colors=np.tile(CONTACT_COLOR, (len(contact_mesh.faces), 1)),
                )
                contact_handle = self.server.scene.add_mesh_trimesh(
                    f"/robot_contact/{geometry_object.name}",
                    mesh=contact_mesh,
                    visible=False,
                )
                self.contact_mesh_handles[geometry_object.name] = contact_handle
                self.geometry_body_indices[geometry_object.name] = body_index

        object_mesh = self._load_mesh(self.object_mesh_path)
        self.object_handle = self.server.scene.add_mesh_trimesh(
            "/object",
            mesh=object_mesh,
        )

        if "fingertip_closest_points_w" in self.motion:
            self.closest_points_handle = self.server.scene.add_point_cloud(
                "/geometry/fingertip_closest_points",
                points=self.motion["fingertip_closest_points_w"][0],
                colors=CLOSEST_POINT_COLOR[:3],
                point_size=0.025,
                point_shape="circle",
            )
        if {
            "fingertip_closest_points_w",
            "fingertip_surface_normals_w",
        }.issubset(self.motion):
            closest_points_w = self.motion["fingertip_closest_points_w"][0]
            normals_w = self._normalized_normals(self.motion["fingertip_surface_normals_w"][0])
            self.surface_normals_handle = self.server.scene.add_arrows(
                "/geometry/fingertip_surface_normals",
                points=np.stack([closest_points_w, closest_points_w + NORMAL_LENGTH * normals_w], axis=1),
                colors=NORMAL_COLOR[:3],
                shaft_radius=0.003,
                head_radius=0.008,
                head_length=0.015,
            )

    def _geometry_body_index(self, geometry_object: object) -> int | None:
        """Map a visual mesh to the contact-label body that drives its color."""
        parent_joint = int(geometry_object.parentJoint)
        if parent_joint == 1:
            return BODY_INDEX["pelvis"]
        joint_name = self.model.names[parent_joint]
        try:
            return JOINT_NAMES.index(joint_name) + 1
        except ValueError:
            return None

    def _setup_gui(self) -> None:
        self.frame_slider = self.server.gui.add_slider(
            "frame",
            min=0,
            max=self.num_frames - 1,
            step=1,
            initial_value=0,
        )
        self.play_checkbox = self.server.gui.add_checkbox("play", initial_value=True)
        self.fps_number = self.server.gui.add_number("fps", initial_value=self.fps, step=1.0)
        self.reset_button = self.server.gui.add_button("reset")

        @self.frame_slider.on_update
        def _(_: object) -> None:
            frame_idx = int(self.frame_slider.value)
            with self.frame_lock:
                self.current_frame = frame_idx
            self._render_frame(frame_idx)

        @self.reset_button.on_click
        def _(_: object) -> None:
            with self.frame_lock:
                self.current_frame = 0
            self.frame_slider.value = 0
            self._render_frame(0)

    def _load_mesh(self, mesh_path: Path) -> trimesh.Trimesh:
        resolved_path = mesh_path if mesh_path.is_absolute() else (mesh_path if mesh_path.exists() else self.object_mesh_path.parent / mesh_path)
        resolved_path = resolved_path.resolve()
        if resolved_path not in self.mesh_cache:
            mesh = trimesh.load_mesh(resolved_path, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.dump()))
            self.mesh_cache[resolved_path] = mesh
        return self.mesh_cache[resolved_path].copy()

    @staticmethod
    def _normalized_normals(normals: np.ndarray) -> np.ndarray:
        lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals / np.maximum(lengths, 1e-8)

    def _render_frame(self, frame_idx: int) -> None:
        q = build_configuration(
            self.model,
            self.joint_q_indices,
            self.motion["root_pos_w"][frame_idx],
            self.motion["root_quat_w"][frame_idx],
            self.motion["joint_pos"][frame_idx],
        )
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(self.model, self.data, self.visual_model, self.visual_data)

        for geometry_object, placement in zip(self.visual_model.geometryObjects, self.visual_data.oMg):
            handle = self.mesh_handles.get(geometry_object.name)
            if handle is None:
                continue
            position = placement.translation.astype(np.float32)
            wxyz = rotmat_to_wxyz(placement.rotation)
            handle.position = position
            handle.wxyz = wxyz

            contact_handle = self.contact_mesh_handles.get(geometry_object.name)
            if contact_handle is not None:
                contact_handle.position = position
                contact_handle.wxyz = wxyz
                body_index = self.geometry_body_indices[geometry_object.name]
                in_contact = bool(self.motion["contact_label"][frame_idx, body_index] > 0.5)
                handle.visible = not in_contact
                contact_handle.visible = in_contact

        self.object_handle.position = self.motion["object_pos_w"][frame_idx]
        self.object_handle.wxyz = self.motion["object_quat_w"][frame_idx]
        if self.closest_points_handle is not None:
            self.closest_points_handle.points = self.motion["fingertip_closest_points_w"][frame_idx]
        if self.surface_normals_handle is not None:
            closest_points_w = self.motion["fingertip_closest_points_w"][frame_idx]
            normals_w = self._normalized_normals(self.motion["fingertip_surface_normals_w"][frame_idx])
            self.surface_normals_handle.points = np.stack(
                [closest_points_w, closest_points_w + NORMAL_LENGTH * normals_w],
                axis=1,
            )

    def run(self) -> None:
        while True:
            if self.play_checkbox.value:
                with self.frame_lock:
                    next_frame = (self.current_frame + 1) % self.num_frames
                    self.current_frame = next_frame
                self.frame_slider.value = next_frame
                self._render_frame(next_frame)
                time.sleep(1.0 / max(float(self.fps_number.value), 1e-6))
            else:
                time.sleep(0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize G1 HOI motion in viser with Pinocchio FK.")
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION_PATH, help="Path to converted motion npz.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH, help="Path to the G1 URDF.")
    parser.add_argument("--object-mesh", type=Path, default=DEFAULT_OBJECT_MESH_PATH, help="Path to the object mesh.")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Viser host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Viser port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer = MotionViewer(
        motion_path=args.motion.resolve(),
        urdf_path=args.urdf.resolve(),
        object_mesh_path=args.object_mesh.resolve(),
        host=args.host,
        port=args.port,
    )
    viewer.run()


if __name__ == "__main__":
    main()
