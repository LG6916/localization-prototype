"""Plotly traces for gripper box and approach arrows."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from gripper.planner import GripperSpec, GripperPlan


def _box_wireframe(center: np.ndarray, axes: np.ndarray, half_extents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Nx3 vertices, list of line pair indices) for an oriented box."""
    ex, ey, ez = axes[:, 0], axes[:, 1], axes[:, 2]
    hx, hy, hz = half_extents
    # 8 corners
    signs = np.array([
        (-1, -1, -1), (+1, -1, -1), (+1, +1, -1), (-1, +1, -1),
        (-1, -1, +1), (+1, -1, +1), (+1, +1, +1), (-1, +1, +1),
    ], dtype=np.float32)
    corners = center + signs[:, 0:1] * hx * ex + signs[:, 1:2] * hy * ey + signs[:, 2:3] * hz * ez
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return corners, edges


def gripper_box_traces(plan: GripperPlan, spec: GripperSpec, *, color: str = "#7ed492"):
    """Return a list of Plotly Scatter3d traces for one gripper plan.

    Feasible plans render as a green wireframe box + bright arrow (the box
    shows where the gripper would sit; the arrow points along the approach).

    Infeasible plans render compactly — a small red ✗ marker at the grasp
    point and no wireframe. That way dense scenes aren't swamped with red
    boxes when every direction collides.
    """
    if plan is None or plan.approach_dir is None:
        return []

    hover = (
        f"grasp point ({plan.grasp_point[0]:.0f}, "
        f"{plan.grasp_point[1]:.0f}, {plan.grasp_point[2]:.0f}) mm<br>"
        f"{plan.tried_directions} directions tried, "
        f"{plan.feasible_count} feasible<br>"
    )

    if not plan.feasible:
        hover += (
            f"<b>no collision-free approach</b><br>"
            f"best clearance would be {plan.clearance_mm:.1f} mm<br>"
            f"(shrink the gripper or widen the approach cone)"
        )
        gp = plan.grasp_point
        return [go.Scatter3d(
            x=[gp[0]], y=[gp[1]], z=[gp[2]],
            mode="markers+text",
            marker=dict(size=9, color="#ff6b6b", symbol="x",
                          line=dict(color="#ffffff", width=1)),
            text=["no approach"], textposition="top center",
            textfont=dict(size=10, color="#ff6b6b"),
            name="gripper: infeasible",
            hovertext=[hover],
            hoverinfo="text",
            showlegend=False,
        )]

    d = plan.approach_dir / (np.linalg.norm(plan.approach_dir) + 1e-12)
    # Gripper frame: ex points away from target (gripper sits behind grasp point)
    ex = -d
    up = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    ez = np.cross(ex, up); ez /= np.linalg.norm(ez) + 1e-12
    ey = np.cross(ez, ex)
    axes = np.stack([ex, ey, ez], axis=1)

    center = plan.grasp_point - d * (
        spec.approach_offset_mm + 0.5 * (spec.finger_length_mm + spec.palm_depth_mm)
    )
    he = np.array([
        0.5 * (spec.finger_length_mm + spec.palm_depth_mm),
        0.5 * spec.width_mm,
        0.5 * spec.height_mm,
    ])
    corners, edges = _box_wireframe(center, axes, he)

    xs, ys, zs = [], [], []
    for a, b in edges:
        xs.extend([corners[a, 0], corners[b, 0], None])
        ys.extend([corners[a, 1], corners[b, 1], None])
        zs.extend([corners[a, 2], corners[b, 2], None])

    hover_full = hover + (
        f"<b>feasible</b>, {plan.clearance_mm:.1f} mm clearance<br>"
        f"approach dir = ({d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f})"
    )
    wire = go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=3),
        name=f"gripper: {plan.clearance_mm:.1f} mm clr",
        hovertext=[hover_full] * len(xs),
        hoverinfo="text",
    )

    # Bright arrow along approach — grasp point is the tip
    arrow_len = 0.45 * (spec.finger_length_mm + spec.palm_depth_mm)
    tip = plan.grasp_point
    tail = plan.grasp_point - d * arrow_len
    arr = go.Scatter3d(
        x=[tail[0], tip[0]], y=[tail[1], tip[1]], z=[tail[2], tip[2]],
        mode="lines+markers",
        line=dict(color=color, width=6),
        marker=dict(size=[0, 6], color=color, symbol="diamond"),
        name="approach",
        showlegend=False,
        hoverinfo="skip",
    )
    return [wire, arr]
