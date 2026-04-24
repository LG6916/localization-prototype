"""Simple box-gripper approach planner.

Given a Detection (model -> scene pose) and a gripper shape, sample a hemisphere
of approach directions; for each direction, check whether a swept box gripper
collides with scene points (via KD-tree distance). Return the best-clearance
direction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class GripperSpec:
    width_mm: float = 30.0      # jaw opening when closed (outer width)
    height_mm: float = 30.0     # thickness perpendicular to both approach & jaw
    finger_length_mm: float = 60.0  # how far fingers extend along approach
    approach_offset_mm: float = 5.0 # clearance above TCP when closed
    palm_depth_mm: float = 20.0

    def swept_box_half_extents(self) -> tuple[float, float, float]:
        # Along-approach, along-jaw, along-height (local frame)
        return (self.finger_length_mm / 2.0,
                self.width_mm / 2.0,
                self.height_mm / 2.0)


@dataclass
class GripperPlan:
    feasible: bool                  # True iff a collision-free approach was found
    approach_dir: np.ndarray        # unit vector, FROM gripper TO target
    grasp_point: np.ndarray         # mm, scene frame
    clearance_mm: float             # clearance of the chosen direction (mm)
    tried_directions: int = 0       # how many candidate directions were scored
    feasible_count: int = 0         # how many of those were collision-free
    best_infeasible_clearance_mm: float = 0.0   # best clearance if none feasible
    top_candidates: list = field(default_factory=list)  # [(dir, clearance)], best first


def _sample_hemisphere(n: int, axis: np.ndarray, cone_deg: float = 180.0, seed: int = 0) -> np.ndarray:
    """Fibonacci sphere, cropped to the hemisphere around `axis` (cone_deg half-angle).

    cone_deg=180 gives the full hemisphere; smaller values constrain more.
    """
    rng = np.random.default_rng(seed)
    # Use golden angle to spread directions
    k = np.arange(n)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (k / max(n - 1, 1)) * 2.0
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * k
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    pts = np.stack([x, y, z], axis=1)
    # Align y-axis with given axis
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    base = np.array([0.0, 1.0, 0.0])
    if np.allclose(ax, base):
        R = np.eye(3)
    elif np.allclose(ax, -base):
        R = np.diag([1, -1, 1])
    else:
        v = np.cross(base, ax)
        s = np.linalg.norm(v)
        c = np.dot(base, ax)
        vx = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / max(s * s, 1e-12))
    pts = pts @ R.T
    # dot with axis, keep cos(angle) >= cos(cone_deg)
    cos_lim = np.cos(np.deg2rad(cone_deg))
    dots = pts @ ax
    pts = pts[dots >= cos_lim - 1e-6]
    # permute
    rng.shuffle(pts)
    return pts


def _clearance_along_direction(
    scene_tree: cKDTree,
    grasp_point: np.ndarray,
    direction: np.ndarray,
    spec: GripperSpec,
    *,
    target_radius_mm: float = 0.0,
) -> float:
    """Minimum clearance from obstacles to the swept gripper volume along
    `direction` (points FROM gripper TO target).

    We approximate the gripper volume as a capsule: segment from `A`
    (`target_radius_mm + approach_offset_mm` behind the grasp point, on the
    gripper's side) to `B` (another `finger_length + palm_depth` behind A),
    with radius = 0.5 * max(width, height).

    Scene points within `target_radius_mm` of the grasp point are considered
    *the target* and are excluded from the collision check — they're what the
    gripper is trying to grasp, not obstacles to avoid.
    """
    d = direction / (np.linalg.norm(direction) + 1e-12)
    L = spec.finger_length_mm + spec.palm_depth_mm

    # Start the gripper capsule beyond the target's far side (relative to the
    # approach direction) so we never count the target itself as an obstacle.
    A = grasp_point - d * (target_radius_mm + spec.approach_offset_mm)
    B = A - d * L
    r = 0.5 * max(spec.width_mm, spec.height_mm)

    mid = 0.5 * (A + B)
    query_r = 0.5 * L + r + 2.0
    idx = scene_tree.query_ball_point(mid, query_r)
    if not idx:
        return float("inf")
    P = scene_tree.data[idx]

    # Also filter out scene points that belong to the target (within a sphere
    # around the grasp point). Belt-and-suspenders with the shifted capsule.
    if target_radius_mm > 0.0:
        d_target = np.linalg.norm(P - grasp_point, axis=1)
        keep = d_target > target_radius_mm
        if not keep.any():
            return float("inf")
        P = P[keep]

    AB = B - A
    ab2 = float(np.dot(AB, AB))
    if ab2 < 1e-12:
        seg_d = np.linalg.norm(P - A, axis=1)
    else:
        t = np.clip(((P - A) @ AB) / ab2, 0.0, 1.0)
        proj = A + np.outer(t, AB)
        seg_d = np.linalg.norm(P - proj, axis=1)
    return float(seg_d.min()) - r


def plan_approach(
    scene_points: np.ndarray,
    detection_pose: np.ndarray,
    spec: GripperSpec,
    *,
    preferred_axis: Optional[np.ndarray] = None,
    n_directions: int = 64,
    cone_deg: float = 90.0,
    safety_margin_mm: float = 0.0,
    target_radius_mm: float = 0.0,
) -> GripperPlan:
    """Score `n_directions` approach vectors on a hemisphere around `preferred_axis`
    and return the best **feasible** one (collision-free with at least
    `safety_margin_mm` of clearance). If no direction is feasible, return the
    best infeasible candidate with `feasible=False` — caller is expected to
    render it differently from a real plan.

    `preferred_axis` defaults to +Z (the scanner's look direction); an approach
    direction points FROM the gripper TO the target, so the natural top-down
    pick is +Z.
    """
    if preferred_axis is None:
        preferred_axis = np.array([0.0, 0.0, 1.0])
    grasp_point = detection_pose[:3, 3].copy()
    tree = cKDTree(scene_points)
    dirs = _sample_hemisphere(n_directions, preferred_axis, cone_deg=cone_deg)
    if len(dirs) == 0:
        dirs = preferred_axis[None, :]

    scored = []
    for d in dirs:
        c = _clearance_along_direction(
            tree, grasp_point, d, spec, target_radius_mm=target_radius_mm,
        )
        scored.append((d, c))

    scored.sort(key=lambda p: p[1], reverse=True)
    feasible = [(d, c) for d, c in scored if c >= safety_margin_mm]

    if feasible:
        best_dir, best_clr = feasible[0]
        return GripperPlan(
            feasible=True,
            approach_dir=best_dir,
            grasp_point=grasp_point,
            clearance_mm=best_clr,
            tried_directions=len(scored),
            feasible_count=len(feasible),
            best_infeasible_clearance_mm=scored[0][1],
            top_candidates=feasible[:5],
        )
    # No direction clears the margin — report the least-bad one, but flag it.
    best_dir, best_clr = scored[0]
    return GripperPlan(
        feasible=False,
        approach_dir=best_dir,
        grasp_point=grasp_point,
        clearance_mm=best_clr,
        tried_directions=len(scored),
        feasible_count=0,
        best_infeasible_clearance_mm=best_clr,
        top_candidates=scored[:5],
    )
