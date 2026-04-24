"""Build the main plotly 3D scene figure from pipeline outputs."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from .color_modes import fit_quality_colors, segment_colors


# Scanner-frame default camera: look along +Z (the scanner's view direction),
# with -Y as screen-up (OpenCV camera convention that Photoneo follows: +X right,
# +Y down, +Z forward). Users can still orbit freely; uirevision preserves their
# rotation across figure updates.
_DEFAULT_CAMERA = dict(
    eye=dict(x=0.0, y=0.0, z=-1.25),   # behind origin, facing +Z
    up=dict(x=0.0, y=-1.0, z=0.0),     # -Y up = image-top up
    center=dict(x=0.0, y=0.0, z=0.0),  # Plotly snaps this to the scene center
)


def empty_figure(message: str = "Upload a scene and a model, then click Run.") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        scene=dict(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
            aspectmode="data",
            camera=_DEFAULT_CAMERA,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="#1d2128",
        plot_bgcolor="#1d2128",
        font=dict(color="#e8ecf1"),
        title=dict(text=message, x=0.5, xanchor="center", font=dict(size=13, color="#b8c1cc")),
        uirevision="scene-static",
    )
    return fig


def _colors_to_str(colors: Optional[np.ndarray]) -> list[str]:
    if colors is None:
        return ["#c0c0c0"] * 0
    c = (np.clip(colors, 0, 1) * 255).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in c]


def build_scene_figure(
    display_points: np.ndarray,
    display_colors: Optional[np.ndarray],
    *,
    color_mode: str = "rgb",             # 'rgb' | 'segment' | 'fit_quality'
    cluster_labels: Optional[np.ndarray] = None,
    fit_distances: Optional[np.ndarray] = None,
    removed_points: Optional[np.ndarray] = None,
    model_points: Optional[np.ndarray] = None,   # already transformed to scene frame
    detections: Sequence = (),
    gripper_traces: Sequence = (),
    show_removed: bool = False,
    show_model: bool = True,
    point_size: float = 1.2,
    selected_instance_id: Optional[int] = None,
) -> go.Figure:
    fig = go.Figure()

    # --- Color mapping ---
    if color_mode == "rgb" and display_colors is not None:
        colors = _colors_to_str(display_colors)
    elif color_mode == "segment" and cluster_labels is not None and len(cluster_labels) == len(display_points):
        colors = _colors_to_str(segment_colors(cluster_labels))
    elif color_mode == "fit_quality" and fit_distances is not None and len(fit_distances) == len(display_points):
        colors = _colors_to_str(fit_quality_colors(fit_distances))
    else:
        colors = ["#9fb3c8"] * len(display_points)

    fig.add_trace(go.Scatter3d(
        x=display_points[:, 0], y=display_points[:, 1], z=display_points[:, 2],
        mode="markers",
        marker=dict(size=point_size, color=colors, opacity=0.85),
        name=f"scene ({len(display_points):,} pts)",
        hoverinfo="skip",
    ))

    if show_removed and removed_points is not None and len(removed_points) > 0:
        # Downsample aggressively for display
        step = max(1, len(removed_points) // 20000)
        rp = removed_points[::step]
        fig.add_trace(go.Scatter3d(
            x=rp[:, 0], y=rp[:, 1], z=rp[:, 2],
            mode="markers",
            marker=dict(size=1.0, color="#554", opacity=0.25),
            name=f"background ({len(rp):,} shown)",
            hoverinfo="skip",
        ))

    # --- Model overlay per detection ---
    if show_model and model_points is not None and len(detections) > 0:
        for det in detections:
            T = det.pose
            pts = model_points
            ones = np.ones((len(pts), 1), dtype=np.float32)
            scene_pts = (np.hstack([pts, ones]) @ T.T)[:, :3]
            is_sel = (selected_instance_id is not None and det.instance_id == selected_instance_id)
            sub = scene_pts if is_sel else scene_pts[::max(1, len(scene_pts) // 300)]
            fig.add_trace(go.Scatter3d(
                x=sub[:, 0], y=sub[:, 1], z=sub[:, 2],
                mode="markers",
                marker=dict(
                    size=2.5 if is_sel else 1.8,
                    color="#ff6b6b" if is_sel else "#ffd166",
                    opacity=0.95 if is_sel else 0.75,
                ),
                name=f"det #{det.instance_id} (conf {det.confidence:.2f})",
                hovertext=[
                    f"instance #{det.instance_id}<br>"
                    f"conf: {det.confidence:.3f}<br>"
                    f"fit: {det.fitness:.3f}<br>"
                    f"rmse: {det.inlier_rmse:.2f} mm<br>"
                    f"method: {det.method}"
                ] * len(sub),
                hoverinfo="text",
            ))
            # Pose axes gizmo at each detection
            R = T[:3, :3]; t = T[:3, 3]
            L = 40.0  # mm
            for i, col in enumerate(("red", "lime", "deepskyblue")):
                a = t
                b = t + R[:, i] * L
                fig.add_trace(go.Scatter3d(
                    x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
                    mode="lines",
                    line=dict(color=col, width=3),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    # --- Gripper traces ---
    for tr in gripper_traces:
        fig.add_trace(tr)

    fig.update_layout(
        scene=dict(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
            aspectmode="data",
            camera=_DEFAULT_CAMERA,
            xaxis=dict(backgroundcolor="#1d2128", color="#b8c1cc", gridcolor="#3d4452"),
            yaxis=dict(backgroundcolor="#1d2128", color="#b8c1cc", gridcolor="#3d4452"),
            zaxis=dict(backgroundcolor="#1d2128", color="#b8c1cc", gridcolor="#3d4452"),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#1d2128",
        plot_bgcolor="#1d2128",
        font=dict(color="#e8ecf1", size=11),
        legend=dict(bgcolor="rgba(17,20,24,0.8)", bordercolor="#3d4452", borderwidth=1),
        uirevision="scene-static",
        showlegend=True,
    )
    return fig
