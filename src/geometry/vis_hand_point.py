"""Visualize deterministic palm-facing hand query points with Viser.

Run from the repository root:
``python geometry/vis_hand_point.py``
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pinocchio as pin
import trimesh
import viser

SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geometry.hand_queries import (  # noqa: E402
    DEFAULT_POINTS_PER_HAND,
    HAND_LINKS,
    get_hand_inner_query_spec,
)
from metrics.fk import BODY_INDEX  # noqa: E402


DEFAULT_URDF_PATH = SRC_ROOT / "assets" / "unitree_g1" / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
VISUAL_SURFACE_OFFSET = 0.0015


def _rotmat_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        root = np.sqrt(trace + 1.0) * 2.0
        quat = np.array((0.25 * root, (rotation[2, 1] - rotation[1, 2]) / root,
                         (rotation[0, 2] - rotation[2, 0]) / root,
                         (rotation[1, 0] - rotation[0, 1]) / root), dtype=np.float32)
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            root = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.array(((rotation[2, 1] - rotation[1, 2]) / root, 0.25 * root,
                             (rotation[0, 1] + rotation[1, 0]) / root,
                             (rotation[0, 2] + rotation[2, 0]) / root), dtype=np.float32)
        elif index == 1:
            root = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.array(((rotation[0, 2] - rotation[2, 0]) / root,
                             (rotation[0, 1] + rotation[1, 0]) / root, 0.25 * root,
                             (rotation[1, 2] + rotation[2, 1]) / root), dtype=np.float32)
        else:
            root = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.array(((rotation[1, 0] - rotation[0, 1]) / root,
                             (rotation[0, 2] + rotation[2, 0]) / root,
                             (rotation[1, 2] + rotation[2, 1]) / root, 0.25 * root), dtype=np.float32)
    return quat / np.linalg.norm(quat)


def _transform(position: tuple[float, float, float], rotation: np.ndarray) -> pin.SE3:
    return pin.SE3(rotation.astype(np.float64), np.asarray(position, dtype=np.float64))


class HandPointViewer:
    def __init__(self, urdf_path: Path, points_per_hand: int, host: str, port: int) -> None:
        self.model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
        self.geometry_model = pin.buildGeomFromUrdf(
            self.model, str(urdf_path), pin.GeometryType.VISUAL, package_dirs=[str(urdf_path.parent)]
        )
        self.data = self.model.createData()
        self.geometry_data = pin.GeometryData(self.geometry_model)
        self.points_per_hand = points_per_hand
        self.server = viser.ViserServer(host=host, port=port)
        self._setup_neutral_pose()
        self._setup_scene()

    def _setup_neutral_pose(self) -> None:
        q = pin.neutral(self.model)
        q[2] = 0.0
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(self.model, self.data, self.geometry_model, self.geometry_data)

    def _setup_scene(self) -> None:
        self.server.scene.world_axes.visible = True
        self.server.scene.add_grid("/ground", width=1.2, height=0.8, plane="xy", cell_size=0.05, section_size=0.2)
        self.server.scene.add_frame("/left_hand", axes_length=0.05, axes_radius=0.002)
        self.server.scene.add_frame("/right_hand", axes_length=0.05, axes_radius=0.002)

        # Rebase each neutral hand subtree independently.  A 180-degree z flip
        # makes the two palms face the viewer with thumbs toward the center.
        display_bases = {
            "left": _transform((-0.12, 0.0, 0.10), np.diag((-1.0, -1.0, 1.0))),
            "right": _transform((0.12, 0.0, 0.10), np.diag((1.0, -1.0, -1.0))),
        }
        base_frames = {
            side: self.model.getFrameId(links[0]) for side, links in HAND_LINKS.items()
        }
        base_world = {side: self.data.oMf[frame_id] for side, frame_id in base_frames.items()}

        meshes: dict[Path, trimesh.Trimesh] = {}
        for geometry, placement in zip(self.geometry_model.geometryObjects, self.geometry_data.oMg):
            frame_name = self.model.frames[geometry.parentFrame].name
            side = "left" if frame_name.startswith("L_") else "right" if frame_name.startswith("R_") else None
            if side is None or frame_name not in HAND_LINKS[side]:
                continue
            mesh_path = Path(geometry.meshPath).resolve()
            if mesh_path not in meshes:
                loaded = trimesh.load_mesh(mesh_path, process=False)
                meshes[mesh_path] = trimesh.util.concatenate(tuple(loaded.dump())) if isinstance(loaded, trimesh.Scene) else loaded
            display_placement = display_bases[side] * base_world[side].inverse() * placement
            self.server.scene.add_mesh_trimesh(
                f"/{side}/mesh/{geometry.name}",
                mesh=meshes[mesh_path],
                position=display_placement.translation.astype(np.float32),
                wxyz=_rotmat_to_wxyz(display_placement.rotation),
            )

        spec = get_hand_inner_query_spec(self.points_per_hand)
        points_by_side: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        for item in spec:
            body_frame = self.model.getFrameId(item.body_name)
            point_world = self.data.oMf[body_frame].act(np.asarray(item.point_body, dtype=np.float64))
            display_point = display_bases[item.side].act(base_world[item.side].inverse().act(point_world))
            normal_world = self.data.oMf[body_frame].rotation @ np.asarray(item.normal_body, dtype=np.float64)
            display_normal = display_bases[item.side].rotation @ base_world[item.side].rotation.T @ normal_world
            # The query coordinate remains exactly on the mesh.  Offset only
            # this drawing point so the small marker is never depth-occluded.
            points_by_side[item.side].append(
                np.asarray(display_point + VISUAL_SURFACE_OFFSET * display_normal, dtype=np.float32)
            )

        for side, points in points_by_side.items():
            self.server.scene.add_point_cloud(
                f"/{side}/inner_query_points",
                points=np.stack(points),
                colors=np.tile(np.asarray((220, 30, 45), dtype=np.uint8), (len(points), 1)),
                point_size=0.004,
                point_shape="circle",
            )

        self.server.gui.add_markdown(
            f"Showing {self.points_per_hand} palm-facing mesh points per hand ({2 * self.points_per_hand} total)."
        )

    def run(self) -> None:
        while True:
            time.sleep(1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show G1 palm-facing hand query points in Viser.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--points-per-hand", type=int, default=DEFAULT_POINTS_PER_HAND)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer = HandPointViewer(args.urdf.resolve(), args.points_per_hand, args.host, args.port)
    viewer.run()


if __name__ == "__main__":
    main()
