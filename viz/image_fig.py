"""2D image figure with detection overlays and back-projected 3D poses.

Images are encoded once as a base64 PNG and attached via `layout.images`
rather than `go.Image(z=...)`. This avoids shipping ~2 MB of JSON pixel
values on every pan/zoom and lets the browser cache the decoded bitmap as
a single texture — pan/zoom becomes trivial resize operations.
"""
from __future__ import annotations

import base64
import io
from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from PIL import Image as PILImage


def _rgb_to_datauri(rgb_hw: np.ndarray) -> str:
    """Encode (H, W, 3) uint8 array as a data-URI PNG. Low compression so
    the Python encode is fast; the image is small enough that the wire size
    difference is negligible."""
    if rgb_hw.dtype != np.uint8:
        rgb_hw = np.clip(rgb_hw, 0, 255).astype(np.uint8)
    img = PILImage.fromarray(rgb_hw, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def empty_image_figure(message: str = "Upload an ordered-grid scene to view its RGB texture.") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#1d2128", plot_bgcolor="#1d2128",
        font=dict(color="#e8ecf1"),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=message, x=0.5, xanchor="center", font=dict(size=13, color="#b8c1cc")),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def build_image_figure(
    rgb_hw: np.ndarray,                         # (H, W, 3) uint8
    *,
    twod_detections: Sequence = (),
    backprojected_points: Optional[np.ndarray] = None,   # (K, 2) int, (-1,-1) = behind camera
    backprojected_labels: Optional[Sequence[str]] = None,
    show_boxes: bool = True,
    show_backproj: bool = True,
    show_masks: bool = True,
    selected_instance_id: Optional[int] = None,
) -> go.Figure:
    H, W, _ = rgb_hw.shape
    fig = go.Figure()
    # Background image via layout (decoded + cached once by the browser).
    fig.add_layout_image(dict(
        source=_rgb_to_datauri(rgb_hw),
        xref="x", yref="y",
        x=0, y=0,           # upper-left corner in data coords (axis is flipped)
        sizex=W, sizey=H,
        xanchor="left", yanchor="top",
        sizing="stretch",
        layer="below",
    ))
    # Invisible scatter trace to establish the axis extents — without any
    # traces Plotly won't render the axes with the image behind them.
    fig.add_trace(go.Scatter(
        x=[0, W], y=[0, H], mode="markers",
        marker=dict(size=0.1, color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=False, name="",
    ))

    # 2D detection bboxes as shapes
    if show_boxes:
        for i, d in enumerate(twod_detections):
            x1, y1, x2, y2 = [float(v) for v in d.bbox]
            fig.add_shape(
                type="rect", x0=x1, x1=x2, y0=y1, y1=y2,
                line=dict(color="#4a9eff" if d.confidence > 0.15 else "#f0ca7f",
                           width=2),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )
            fig.add_annotation(
                x=x1, y=max(0, y1 - 4),
                text=f"{d.class_name} {d.confidence:.2f}",
                showarrow=False, xanchor="left", yanchor="bottom",
                font=dict(color="#4a9eff" if d.confidence > 0.15 else "#f0ca7f", size=11),
                bgcolor="rgba(17,20,24,0.75)",
            )

    # Back-projected 3D detection markers (from 3D pipeline results)
    if show_backproj and backprojected_points is not None and len(backprojected_points):
        valid = (backprojected_points[:, 0] >= 0) & (backprojected_points[:, 1] >= 0)
        u = backprojected_points[valid, 0]
        v = backprojected_points[valid, 1]
        labels = [backprojected_labels[i] for i in np.where(valid)[0]] if backprojected_labels else [""]*len(u)
        colors = [
            ("#ff6b6b" if selected_instance_id is not None and i == selected_instance_id else "#7ed492")
            for i in np.where(valid)[0]
        ]
        fig.add_trace(go.Scatter(
            x=u, y=v, mode="markers+text",
            marker=dict(size=14, color=colors, line=dict(color="white", width=2), symbol="circle"),
            text=[f"#{lbl}" for lbl in labels], textposition="top center",
            textfont=dict(size=11, color="white"),
            name="3D detections (back-projected)",
            hovertext=[f"3D detection #{l}" for l in labels],
            hoverinfo="text",
        ))

    fig.update_xaxes(range=[0, W], showgrid=False, visible=False, constrain="domain")
    fig.update_yaxes(range=[H, 0], showgrid=False, visible=False,
                      scaleanchor="x", scaleratio=1)
    fig.update_layout(
        paper_bgcolor="#1d2128", plot_bgcolor="#1d2128",
        margin=dict(l=0, r=0, t=5, b=0),
        showlegend=False, uirevision="img-static",
    )
    return fig
