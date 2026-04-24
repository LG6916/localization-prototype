"""Dash app entry point for the Localization Prototype."""
from __future__ import annotations

import base64
import json
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html, no_update
from scipy.spatial import cKDTree

import config as cfg
from ioutil import build_display_copy, load_model, load_scene
from pipelines import PRESETS, ProgressReporter, build_pipeline, list_methods
from pipelines.base import PipelineContext
from gripper.planner import GripperSpec, plan_approach
from viz import (build_scene_figure, empty_figure, gripper_box_traces,
                  build_image_figure, empty_image_figure,
                  backproject_points_to_pixels, build_tree, preprocess_rgb)
from detector import detect as yolo_detect, is_available as yolo_available


# ---------------------------------------------------------------- Session ----
# In-process state: point clouds are too big to round-trip through the browser,
# and Dash background callbacks run in a separate process so we can't use them
# for objects that live in main-process memory. A single-threaded worker that
# shares memory keeps things simple.
SESSION: dict[str, Any] = {
    "scene": None,          # SceneData
    "model": None,          # ModelData
    "display": None,        # DisplayCopy of the scene
    "last_ctx": None,       # PipelineContext after last run
    "last_trace": None,
    "last_elapsed_s": 0.0,
    "last_preset": "",
    # Run state (written by worker thread, read by polling callback)
    "run_id": None,
    "run_progress": {"frac": 0.0, "label": "idle"},
    "run_done": True,
    "run_error": None,
    "run_result_ts": 0.0,
    # 2D detector state
    "twod_detections": [],           # list[Detection2D]
    "twod_status": "",
    "twod_ts": 0.0,
}
SESSION_LOCK = threading.Lock()


def _run_pipeline_thread(run_id: str, preset: str, overrides: list, scene, model):
    """Worker thread body: run the pipeline, keep progress in SESSION."""
    def sink(label, frac):
        with SESSION_LOCK:
            if SESSION.get("run_id") != run_id:
                return  # superseded / cancelled
            SESSION["run_progress"] = {"frac": float(frac), "label": label}

    try:
        pipe = build_pipeline(preset, overrides=overrides)
        ordered_grid = None
        if scene.is_ordered:
            ordered_grid = dict(
                xyz_hw=scene.xyz_hw, valid_mask_hw=scene.valid_mask_hw,
                pixel_of_point=scene.pixel_of_point,
                point_of_pixel=scene.point_of_pixel,
                shape=scene.xyz_hw.shape[:2],
            )
        with SESSION_LOCK:
            twod = list(SESSION.get("twod_detections") or [])
        ctx_ = PipelineContext(
            scene_points=scene.points.astype(np.float32),
            scene_colors=scene.colors,
            model_points=model.sample_points(3000),
            model_mesh=model.mesh,
            model_radius_mm=model.sphere_radius_mm,
            scene_ordered_grid=ordered_grid,
            twod_detections=twod,
            progress=ProgressReporter(sink),
        )
        t0 = time.time()
        pipe.run(ctx_)
        elapsed = time.time() - t0
        with SESSION_LOCK:
            if SESSION.get("run_id") != run_id:
                return
            SESSION["last_ctx"] = ctx_
            SESSION["last_trace"] = ctx_.trace
            SESSION["last_elapsed_s"] = elapsed
            SESSION["last_preset"] = preset
            SESSION["run_progress"] = {
                "frac": 1.0,
                "label": f"done — {len(ctx_.detections)} detections in {elapsed:.1f}s",
            }
            SESSION["run_done"] = True
            SESSION["run_error"] = None
            SESSION["run_result_ts"] = time.time()
    except Exception as e:
        traceback.print_exc()
        with SESSION_LOCK:
            SESSION["run_progress"] = {"frac": 0.0, "label": f"ERROR: {e}"}
            SESSION["run_done"] = True
            SESSION["run_error"] = str(e)


# ---------------------------------------------------------------- App setup --
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Localization Prototype",
)
server = app.server


# ---------------------------------------------------------------- Helpers ----
def _save_upload(contents: str, filename: str, kind: str) -> str:
    """Save a dcc.Upload base64 contents to disk; returns file path."""
    if contents is None or not filename:
        return None
    _, b64 = contents.split(",", 1)
    data = base64.b64decode(b64)
    out = cfg.UPLOAD_DIR / f"{kind}_{uuid.uuid4().hex[:8]}_{Path(filename).name}"
    with open(out, "wb") as f:
        f.write(data)
    return str(out)


def _pipeline_spec_from_controls(preset: str, overrides: dict | None) -> list[tuple[str, str, dict]]:
    """Produce a concrete pipeline spec, applying any UI overrides."""
    base = [list(s) for s in PRESETS[preset]]
    if overrides:
        for slot, new_method, new_params in overrides:
            for row in base:
                if row[0] == slot:
                    if new_method:
                        row[1] = new_method
                    if new_params:
                        merged = dict(row[2])
                        merged.update(new_params)
                        row[2] = merged
                    break
    return [(s, m, p) for s, m, p in base]


# ---------------------------------------------------------------- Layout -----
def _stage_card(slot: str, preset: str) -> dbc.Card:
    row = next((s for s in PRESETS[preset] if s[0] == slot), None)
    method = row[1] if row else list_methods(slot)[0]
    params = row[2] if row else {}
    return html.Div(className="stage-card", children=[
        html.Div(slot, className="slot-label"),
        html.Div(className="stage-body", children=[
            dcc.Dropdown(
                id={"type": "stage-method", "slot": slot},
                options=[{"label": m, "value": m} for m in list_methods(slot)],
                value=method, clearable=False, style={"fontSize": "12px"},
            ),
            dcc.Textarea(
                id={"type": "stage-params", "slot": slot},
                value=json.dumps(params, indent=2),
                style={"width": "100%", "height": "110px", "marginTop": "6px",
                       "fontSize": "11px", "fontFamily": "monospace"},
            ),
        ]),
    ])


def layout_localization() -> html.Div:
    preset_default = list(PRESETS.keys())[0]
    return html.Div(children=[
        dbc.Row([
            # ---------- LEFT: Config ----------
            dbc.Col(md=3, children=[
                html.Div(className="panel", children=[
                    html.H6("Inputs"),
                    html.Div("Scene (PLY)", style={"fontSize": "12px", "marginTop": "4px"}),
                    dcc.Upload(
                        id="upload-scene",
                        children=html.Div("Drag & drop or click to upload",
                                            style={"fontSize": "12px", "color": "var(--text-dim)"}),
                        style={"borderRadius": "5px", "border": "1px dashed var(--border)",
                               "padding": "10px", "textAlign": "center", "cursor": "pointer"},
                        multiple=False,
                    ),
                    html.Div(id="scene-info", style={"marginTop": "6px", "fontSize": "11px"}),
                    html.Div("Model (STL/OBJ)", style={"fontSize": "12px", "marginTop": "10px"}),
                    dcc.Upload(
                        id="upload-model",
                        children=html.Div("Drag & drop or click to upload",
                                            style={"fontSize": "12px", "color": "var(--text-dim)"}),
                        style={"borderRadius": "5px", "border": "1px dashed var(--border)",
                               "padding": "10px", "textAlign": "center", "cursor": "pointer"},
                        multiple=False,
                    ),
                    html.Div(id="model-info", style={"marginTop": "6px", "fontSize": "11px"}),
                    html.Div(style={"display": "flex", "gap": "6px", "marginTop": "8px"}, children=[
                        dbc.Button("Load calib.ply + sphere.stl example", id="btn-load-example",
                                   color="secondary", size="sm", outline=True, className="w-100"),
                    ]),
                ]),
                html.Div(className="panel", children=[
                    html.H6("Pipeline"),
                    dcc.Dropdown(
                        id="preset-selector",
                        options=[{"label": k, "value": k} for k in PRESETS.keys()],
                        value=preset_default, clearable=False,
                    ),
                    html.Div(id="preset-hint", className="muted",
                              style={"marginTop": "6px", "fontSize": "11px"},
                              children=(
                                  "Pick a preset and hit Run. To customize methods or "
                                  "params per stage, expand 'Advanced' below."
                              )),
                    html.Details(className="pipeline-advanced", open=False, children=[
                        html.Summary("Advanced — edit pipeline stages"),
                        html.Div(id="pipeline-stages",
                                  style={"marginTop": "8px"}, children=[
                            _stage_card("preprocess", preset_default),
                            _stage_card("background", preset_default),
                            _stage_card("candidates", preset_default),
                            _stage_card("refine", preset_default),
                            _stage_card("scoring", preset_default),
                        ]),
                    ]),
                ]),
                html.Div(className="panel", children=[
                    html.H6("Gripper"),
                    dbc.Checklist(
                        options=[{"label": "Plan approach vectors", "value": 1}],
                        value=[], id="gripper-enable", switch=True,
                        inputStyle={"marginRight": "6px"},
                        style={"fontSize": "12px"},
                    ),
                    html.Div(style={"marginTop": "8px"}, children=[
                        html.Label("Width (mm)", style={"fontSize": "11px"}),
                        dcc.Input(id="gripper-w", type="number", value=30, min=5, max=400,
                                  style={"width": "100%", "fontSize": "12px"}),
                        html.Label("Height (mm)", style={"fontSize": "11px", "marginTop": "6px"}),
                        dcc.Input(id="gripper-h", type="number", value=30, min=5, max=200,
                                  style={"width": "100%", "fontSize": "12px"}),
                        html.Label("Finger length (mm)", style={"fontSize": "11px", "marginTop": "6px"}),
                        dcc.Input(id="gripper-fl", type="number", value=60, min=5, max=300,
                                  style={"width": "100%", "fontSize": "12px"}),
                        html.Label("Palm depth (mm)", style={"fontSize": "11px", "marginTop": "6px"}),
                        dcc.Input(id="gripper-pd", type="number", value=20, min=5, max=200,
                                  style={"width": "100%", "fontSize": "12px"}),
                    ]),
                ]),
            ]),

            # ---------- CENTER: 3D view ----------
            dbc.Col(md=6, children=[
                html.Div(className="panel", style={"padding": "6px"}, children=[
                    html.Div(style={"display": "flex", "gap": "8px", "alignItems": "center",
                                     "marginBottom": "6px"}, children=[
                        html.Div("View:", style={"fontSize": "11px", "color": "var(--text-dim)"}),
                        dcc.RadioItems(
                            id="color-mode",
                            options=[
                                {"label": " RGB", "value": "rgb"},
                                {"label": " Segment", "value": "segment"},
                                {"label": " Fit quality", "value": "fit_quality"},
                            ],
                            value="rgb", inline=True,
                            labelStyle={"fontSize": "11px", "marginRight": "10px"},
                        ),
                        dcc.Checklist(
                            id="layer-toggles",
                            options=[
                                {"label": " model overlay", "value": "model"},
                                {"label": " gripper", "value": "gripper"},
                            ],
                            value=["model", "gripper"], inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"fontSize": "11px", "marginRight": "10px"},
                        ),
                    ]),
                    dcc.Graph(id="scene-fig", figure=empty_figure(),
                              style={"height": "72vh"},
                              config={"scrollZoom": True, "displaylogo": False}),
                ]),
                html.Div(className="panel", children=[
                    html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px"}, children=[
                        html.Button("Run Localization", id="btn-run",
                                    className="btn-run", n_clicks=0),
                        html.Button("Cancel", id="btn-cancel", className="btn",
                                    n_clicks=0, style={"padding": "8px 14px"}),
                        html.Div(id="run-status", style={"flex": "1"}, children=[
                            html.Div(className="progress-wrap", children=[
                                html.Div(className="bar", id="progress-bar",
                                         style={"width": "0%"}),
                                html.Div(className="label", id="progress-label",
                                         children="idle"),
                            ]),
                        ]),
                    ]),
                    dcc.Interval(id="progress-ticker", interval=400, disabled=True),
                    dcc.Store(id="selected-instance-id"),
                    dcc.Store(id="result-timestamp", data=0),
                ]),
            ]),

            # ---------- RIGHT: Details ----------
            dbc.Col(md=3, children=[
                html.Div(className="panel", children=[
                    html.H6("Detections"),
                    html.Div(id="detection-list",
                             style={"maxHeight": "38vh", "overflowY": "auto"},
                             children=html.Div("No run yet.", className="muted")),
                ]),
                html.Div(className="panel", children=[
                    html.H6("Pipeline Trace"),
                    html.Div(id="pipeline-trace",
                             style={"maxHeight": "28vh", "overflowY": "auto"},
                             children=html.Div("No run yet.", className="muted")),
                ]),
                html.Div(className="panel", children=[
                    html.H6("Run Summary"),
                    html.Div(id="run-summary", children=html.Div("No run yet.", className="muted")),
                ]),
            ]),
        ]),
    ])


def layout_2d_rgb() -> html.Div:
    yolo_ok = yolo_available()
    return html.Div(children=[
        dbc.Row([
            # --- Left: prompts, controls, 2D detections list ---
            dbc.Col(md=3, children=[
                html.Div(className="panel", children=[
                    html.H6("Open-vocab 2D detector"),
                    html.Div(
                        ("YOLO-World via ultralytics. Text-prompt open-vocabulary "
                         "detection on the RGB texture. Detections can then be used "
                         "as 3D ROIs by pipelines D / E on the Localization tab.")
                        if yolo_ok else
                        ("ultralytics not installed. Run "
                         "`pip install ultralytics` in the loc-proto env to enable."),
                        className="muted", style={"fontSize": "11px"}),
                    html.Label("Text prompts (one per line)",
                               style={"fontSize": "11px", "marginTop": "8px"}),
                    dcc.Textarea(
                        id="yolo-prompts",
                        value="sphere\nball",
                        style={"width": "100%", "height": "80px",
                               "fontSize": "12px", "fontFamily": "monospace"},
                    ),
                    html.Label("Confidence threshold",
                               style={"fontSize": "11px", "marginTop": "8px"}),
                    dcc.Slider(id="yolo-conf", min=0.01, max=0.5, step=0.01,
                               value=0.05,
                               marks={0.01: "0.01", 0.1: "0.1", 0.25: "0.25", 0.5: "0.5"},
                               tooltip={"always_visible": False, "placement": "bottom"}),
                    html.Div("Image preprocessing (applied to both the view and the detector)",
                             className="muted", style={"fontSize": "11px", "marginTop": "10px"}),
                    dcc.Checklist(
                        id="img-preproc",
                        options=[
                            {"label": " CLAHE", "value": "clahe"},
                            {"label": " Normalize", "value": "norm"},
                        ],
                        value=[], inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"fontSize": "11px", "marginRight": "12px"},
                    ),
                    html.Div(style={"display": "grid",
                                     "gridTemplateColumns": "1fr 1fr", "gap": "6px",
                                     "marginTop": "4px"}, children=[
                        html.Div([html.Label("CLAHE clip",
                                               style={"fontSize": "10px"}),
                                   dcc.Input(id="clahe-clip", type="number", value=2.0,
                                              min=0.5, max=10.0, step=0.25,
                                              style={"width": "100%", "fontSize": "11px"})]),
                        html.Div([html.Label("Gamma",
                                               style={"fontSize": "10px"}),
                                   dcc.Input(id="norm-gamma", type="number", value=1.2,
                                              min=0.5, max=3.0, step=0.1,
                                              style={"width": "100%", "fontSize": "11px"})]),
                    ]),
                    html.Div(style={"marginTop": "12px"}, children=[
                        html.Button("Run YOLO-World", id="btn-yolo",
                                    className="btn-run", n_clicks=0,
                                    disabled=not yolo_ok),
                    ]),
                    html.Div(id="yolo-status", className="muted",
                             style={"marginTop": "8px", "fontSize": "11px"}),
                ]),
                html.Div(className="panel", children=[
                    html.H6("2D detections"),
                    html.Div(id="yolo-det-list",
                             style={"maxHeight": "40vh", "overflowY": "auto"},
                             children=html.Div("No detector run yet.", className="muted")),
                ]),
                html.Div(className="panel", children=[
                    html.H6("How to use these"),
                    html.Div(
                        "Switch to the Localization tab and select preset "
                        "'D: 2D-guided ROIs + ICP' or 'E: 2D-guided spheres'. "
                        "Those pipelines lift each bbox to a 3D ROI via the "
                        "ordered grid and localize within it.",
                        className="muted", style={"fontSize": "11px"}),
                ]),
            ]),
            # --- Right: image figure ---
            dbc.Col(md=9, children=[
                html.Div(className="panel", style={"padding": "6px"}, children=[
                    html.Div(style={"display": "flex", "gap": "10px",
                                     "alignItems": "center", "marginBottom": "6px"},
                              children=[
                        html.Div("Layers:", style={"fontSize": "11px", "color": "var(--text-dim)"}),
                        dcc.Checklist(
                            id="img-layer-toggles",
                            options=[
                                {"label": " 2D boxes", "value": "boxes"},
                                {"label": " 3D back-projection", "value": "backproj"},
                                {"label": " masks", "value": "masks"},
                            ],
                            value=["boxes"], inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"fontSize": "11px", "marginRight": "12px"},
                        ),
                    ]),
                    dcc.Graph(id="image-fig", figure=empty_image_figure(),
                              style={"height": "80vh"},
                              config={"scrollZoom": True, "displaylogo": False}),
                ]),
            ]),
        ]),
        dcc.Store(id="yolo-ts", data=0),
        dcc.Interval(id="yolo-ticker", interval=400, disabled=True),
    ])


def layout_calibration() -> html.Div:
    return html.Div(children=[
        dbc.Row([
            dbc.Col(md=5, children=[
                html.Div(className="panel", children=[
                    html.H6("Pairwise distances"),
                    html.Div(id="calib-distances",
                             children=html.Div(
                                 "Run localization first. Detected instance centers "
                                 "will appear here as a pairwise distance matrix.",
                                 className="muted")),
                ]),
                html.Div(className="panel", children=[
                    html.H6("Ground-truth distances (optional)"),
                    html.Div("Enter known distances (mm), one pair per line: "
                             "`i,j,distance` (i < j)", className="muted"),
                    dcc.Textarea(
                        id="gt-distances",
                        style={"width": "100%", "height": "160px", "fontSize": "11px",
                               "fontFamily": "monospace"},
                        placeholder="0,1,152.4\n0,2,210.0\n1,2,145.3",
                    ),
                    html.Div(style={"marginTop": "6px"}, children=[
                        html.Button("Compute residuals", id="btn-calib",
                                    className="btn-run", n_clicks=0),
                    ]),
                ]),
            ]),
            dbc.Col(md=7, children=[
                html.Div(className="panel", children=[
                    html.H6("Residuals"),
                    # Residuals table gets its own scroll box so a long list of
                    # GT pairs doesn't squeeze the heatmap below it.
                    html.Div(id="calib-residuals",
                             style={"maxHeight": "30vh", "overflowY": "auto",
                                    "marginBottom": "8px"},
                             children=html.Div("Awaiting inputs.", className="muted")),
                    dcc.Graph(id="calib-heatmap", style={"height": "75vh"}),
                ]),
            ]),
        ]),
    ])


app.layout = html.Div(children=[
    html.Div(className="app-header", children=[
        html.H3([html.I(className="bi bi-bounding-box-circles", style={"marginRight": "6px"}),
                  "Localization Prototype"]),
        html.Div("3D part detection in point clouds — brainstorming sandbox",
                  className="sub"),
    ]),
    dbc.Tabs(id="main-tabs", active_tab="loc", children=[
        dbc.Tab(label="Localization", tab_id="loc", children=layout_localization()),
        dbc.Tab(label="2D RGB", tab_id="rgb2d", children=layout_2d_rgb()),
        dbc.Tab(label="Calibration / Precision", tab_id="cal", children=layout_calibration()),
    ]),
])


# ---------------------------------------------------------------- Callbacks --

# Upload scene
@app.callback(
    Output("scene-info", "children"),
    Output("scene-fig", "figure", allow_duplicate=True),
    Input("upload-scene", "contents"),
    State("upload-scene", "filename"),
    prevent_initial_call=True,
)
def on_scene_upload(contents, filename):
    if not contents:
        return no_update, no_update
    path = _save_upload(contents, filename, "scene")
    try:
        scene = load_scene(path)
        disp = build_display_copy(scene.points, scene.colors, max_points=cfg.DISPLAY_MAX_POINTS)
        with SESSION_LOCK:
            SESSION["scene"] = scene
            SESSION["display"] = disp
            SESSION["last_ctx"] = None
        ext = scene.points.max(0) - scene.points.min(0) if len(scene.points) else np.zeros(3)
        grid_txt = f" grid={scene.photoneo_grid[0]}×{scene.photoneo_grid[1]}" if scene.photoneo_grid else ""
        info = html.Div([
            html.Div(f"{Path(filename).name}", style={"fontWeight": 600}),
            html.Div(f"{len(scene.points):,} valid pts{grid_txt}", className="muted"),
            html.Div(f"bbox {ext[0]:.0f} × {ext[1]:.0f} × {ext[2]:.0f} mm", className="muted"),
            html.Div(f"display: {len(disp.points):,} pts (voxel {disp.voxel_mm:.2f} mm)",
                     className="muted"),
        ])
        fig = build_scene_figure(disp.points, disp.colors)
        return info, fig
    except Exception as e:
        return html.Div(f"error: {e}", style={"color": "#ff6b6b"}), no_update


# Upload model
@app.callback(
    Output("model-info", "children"),
    Input("upload-model", "contents"),
    State("upload-model", "filename"),
    prevent_initial_call=True,
)
def on_model_upload(contents, filename):
    if not contents:
        return no_update
    path = _save_upload(contents, filename, "model")
    try:
        with SESSION_LOCK:
            scene = SESSION.get("scene")
        bb = (scene.points.max(0) - scene.points.min(0)) if scene else None
        model = load_model(path, scene_bbox_mm=bb)
        with SESSION_LOCK:
            SESSION["model"] = model
        parts = [
            html.Div(f"{Path(filename).name}", style={"fontWeight": 600}),
            html.Div(f"{model.meta['triangles']} tri, {model.meta['vertices']} verts",
                     className="muted"),
            html.Div(f"extent {model.extents_mm[0]:.1f} × {model.extents_mm[1]:.1f} × "
                     f"{model.extents_mm[2]:.1f} mm", className="muted"),
        ]
        if model.auto_detected_scale:
            parts.append(html.Div(
                f"unit auto-detected: ×{model.auto_detected_scale:.0f} to mm",
                className="muted", style={"color": "#f0ca7f"}))
        if model.sphere_radius_mm is not None:
            parts.append(html.Div(
                f"sphere-like (sphericity {model.sphericity:.2f}, r={model.sphere_radius_mm:.2f} mm)",
                className="muted", style={"color": "#7ed492"}))
        return html.Div(parts)
    except Exception as e:
        return html.Div(f"error: {e}", style={"color": "#ff6b6b"})


# Load example button
@app.callback(
    Output("scene-info", "children", allow_duplicate=True),
    Output("model-info", "children", allow_duplicate=True),
    Output("scene-fig", "figure", allow_duplicate=True),
    Input("btn-load-example", "n_clicks"),
    prevent_initial_call=True,
)
def on_load_example(n):
    if not n:
        return no_update, no_update, no_update
    scene_path = "/home/gajdosech/Desktop/calib.ply"
    model_path = "/home/gajdosech/Desktop/sphere.stl"
    if not (os.path.exists(scene_path) and os.path.exists(model_path)):
        return html.Div("example files missing", style={"color": "#ff6b6b"}), no_update, no_update
    scene = load_scene(scene_path)
    disp = build_display_copy(scene.points, scene.colors, max_points=cfg.DISPLAY_MAX_POINTS)
    bb = scene.points.max(0) - scene.points.min(0)
    model = load_model(model_path, scene_bbox_mm=bb)
    with SESSION_LOCK:
        SESSION["scene"] = scene
        SESSION["display"] = disp
        SESSION["model"] = model
        SESSION["last_ctx"] = None
    fig = build_scene_figure(disp.points, disp.colors)
    scene_info = html.Div([
        html.Div("calib.ply", style={"fontWeight": 600}),
        html.Div(f"{len(scene.points):,} valid pts  grid={scene.photoneo_grid[0]}×{scene.photoneo_grid[1]}"
                 if scene.photoneo_grid else f"{len(scene.points):,} valid pts", className="muted"),
        html.Div(f"bbox {bb[0]:.0f} × {bb[1]:.0f} × {bb[2]:.0f} mm", className="muted"),
        html.Div(f"display: {len(disp.points):,} pts (voxel {disp.voxel_mm:.2f} mm)",
                 className="muted"),
    ])
    model_info = html.Div([
        html.Div("sphere.stl", style={"fontWeight": 600}),
        html.Div(f"{model.meta['triangles']} tri, {model.meta['vertices']} verts",
                 className="muted"),
        html.Div(f"extent {model.extents_mm[0]:.1f} × {model.extents_mm[1]:.1f} × "
                 f"{model.extents_mm[2]:.1f} mm", className="muted"),
        html.Div(f"sphere-like (sphericity {model.sphericity:.2f}, r={model.sphere_radius_mm:.2f} mm)"
                 if model.sphere_radius_mm is not None else "",
                 className="muted", style={"color": "#7ed492"}),
    ])
    return scene_info, model_info, fig


# Preset change -> reset stage cards
@app.callback(
    Output("pipeline-stages", "children"),
    Input("preset-selector", "value"),
    prevent_initial_call=True,
)
def on_preset_change(preset):
    return [
        _stage_card("preprocess", preset),
        _stage_card("background", preset),
        _stage_card("candidates", preset),
        _stage_card("refine", preset),
        _stage_card("scoring", preset),
    ]


# ---- Run pipeline: spawn an in-process thread, then tick to follow progress ----
@app.callback(
    Output("progress-ticker", "disabled", allow_duplicate=True),
    Output("progress-bar", "style", allow_duplicate=True),
    Output("progress-label", "children", allow_duplicate=True),
    Output("btn-run", "disabled", allow_duplicate=True),
    Input("btn-run", "n_clicks"),
    State("preset-selector", "value"),
    State({"type": "stage-method", "slot": dash.ALL}, "value"),
    State({"type": "stage-method", "slot": dash.ALL}, "id"),
    State({"type": "stage-params", "slot": dash.ALL}, "value"),
    prevent_initial_call=True,
)
def on_run(n_clicks, preset, method_values, method_ids, params_values):
    if not n_clicks:
        return no_update, no_update, no_update, no_update
    with SESSION_LOCK:
        scene = SESSION.get("scene")
        model = SESSION.get("model")
    if scene is None or model is None:
        return True, {"width": "0%"}, "upload scene and model first", False

    overrides = []
    for mid, mv, pv in zip(method_ids, method_values, params_values):
        try:
            params = json.loads(pv) if pv else {}
        except Exception:
            params = {}
        overrides.append((mid["slot"], mv, params))

    run_id = uuid.uuid4().hex
    with SESSION_LOCK:
        SESSION["run_id"] = run_id
        SESSION["run_done"] = False
        SESSION["run_error"] = None
        SESSION["run_progress"] = {"frac": 0.0, "label": "starting..."}

    t = threading.Thread(
        target=_run_pipeline_thread,
        args=(run_id, preset, overrides, scene, model),
        daemon=True,
    )
    t.start()
    # Enable the ticker so we start polling progress + completion.
    return False, {"width": "2%"}, "starting...", True


@app.callback(
    Output("progress-bar", "style", allow_duplicate=True),
    Output("progress-label", "children", allow_duplicate=True),
    Output("progress-ticker", "disabled", allow_duplicate=True),
    Output("btn-run", "disabled", allow_duplicate=True),
    Output("result-timestamp", "data"),
    Output("selected-instance-id", "data", allow_duplicate=True),
    Input("progress-ticker", "n_intervals"),
    Input("btn-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def tick_progress(n, cancel_clicks):
    trig = ctx.triggered_id
    if trig == "btn-cancel":
        with SESSION_LOCK:
            SESSION["run_id"] = None   # any running thread will see mismatch and stop writing
            SESSION["run_done"] = True
            SESSION["run_progress"] = {"frac": 0.0, "label": "cancelled"}
        return {"width": "0%"}, "cancelled", True, False, no_update, no_update

    with SESSION_LOCK:
        p = dict(SESSION.get("run_progress") or {"frac": 0.0, "label": "idle"})
        done = bool(SESSION.get("run_done"))
        ts = float(SESSION.get("run_result_ts") or 0.0)
        err = SESSION.get("run_error")
    pct = min(100, max(0, p["frac"] * 100))
    label = f"{p['label']}  ({pct:.0f}%)"
    bar_style = {"width": f"{pct:.0f}%"}
    if done:
        # finalize UI; trigger the refresh_ui callback via result-timestamp
        return bar_style, label, True, False, ts, None
    return bar_style, label, False, True, no_update, no_update


# ---- After run: refresh detections list, trace, figure ----
@app.callback(
    Output("detection-list", "children"),
    Output("pipeline-trace", "children"),
    Output("run-summary", "children"),
    Output("scene-fig", "figure", allow_duplicate=True),
    Input("result-timestamp", "data"),
    Input("color-mode", "value"),
    Input("layer-toggles", "value"),
    Input("selected-instance-id", "data"),
    Input("gripper-enable", "value"),
    Input("gripper-w", "value"),
    Input("gripper-h", "value"),
    Input("gripper-fl", "value"),
    Input("gripper-pd", "value"),
    prevent_initial_call=True,
)
def refresh_ui(ts, color_mode, layers, selected_id,
                grip_en, gw, gh, gfl, gpd):
    with SESSION_LOCK:
        ctx_ = SESSION.get("last_ctx")
        display = SESSION.get("display")
        scene = SESSION.get("scene")
        elapsed = SESSION.get("last_elapsed_s", 0.0)
        preset = SESSION.get("last_preset", "")
        model = SESSION.get("model")
        trace = SESSION.get("last_trace")

    if ctx_ is None or display is None:
        if display is not None:
            fig = build_scene_figure(display.points, display.colors)
            return no_update, no_update, no_update, fig
        return no_update, no_update, no_update, no_update

    # --- detections list ---
    det_children = []
    for d in ctx_.detections:
        sel = selected_id is not None and d.instance_id == int(selected_id)
        extras = []
        if "fitted_radius_mm" in d.extra:
            extras.append(f"r={d.extra['fitted_radius_mm']:.2f} mm")
        if "radius_error_mm" in d.extra:
            extras.append(f"Δr={d.extra['radius_error_mm']:.2f}")
        det_children.append(html.Div(
            className="det-row" + (" selected" if sel else ""),
            id={"type": "det-row", "id": int(d.instance_id)},
            n_clicks=0,
            children=[
                html.Div(f"#{d.instance_id}", className="rank"),
                html.Div(children=[
                    html.Div([
                        f"{d.method}  ",
                        html.Span(f"fit={d.fitness:.2f}", className="secondary")
                        if d.fitness > 0 else html.Span(""),
                    ], className="primary"),
                    html.Div(children=[
                        f"t = ({d.translation[0]:.0f}, {d.translation[1]:.0f}, {d.translation[2]:.0f})  ",
                        html.Span("  ".join(extras), className="secondary"),
                    ], className="secondary"),
                ]),
                html.Div(f"{d.confidence:.2f}", className="conf"),
            ],
        ))
    if not det_children:
        det_children = [html.Div("no detections passed scoring thresholds",
                                   className="muted")]

    # --- pipeline trace ---
    trace_children = []
    for r in (trace or []):
        trace_children.append(html.Div(className="trace-row", children=[
            html.Div(children=[
                html.Span(f"[{r.slot}] ", className="stage"),
                html.Span(f"{r.method} ", className="method"),
                html.Span(f"{r.duration_s*1000:.0f} ms", className="timing"),
                html.Span(f"  {r.n_in:,} → {r.n_out:,}", className="count"),
            ]),
            html.Div(
                " · ".join(f"{k}={_fmt_stat(v)}" for k, v in r.stats.items()
                            if not isinstance(v, (list, dict))),
                style={"color": "var(--text-mut)", "fontSize": "10px", "marginTop": "3px"},
            ),
        ]))

    # --- run summary ---
    n_dets = len(ctx_.detections)
    mean_conf = float(np.mean([d.confidence for d in ctx_.detections])) if ctx_.detections else 0.0
    mean_fit = float(np.mean([d.fitness for d in ctx_.detections])) if ctx_.detections else 0.0
    summary = html.Div(className="metric-grid", children=[
        html.Div("preset", className="k"), html.Div(preset, className="v"),
        html.Div("runtime", className="k"), html.Div(f"{elapsed:.2f} s", className="v"),
        html.Div("detections", className="k"), html.Div(f"{n_dets}", className="v"),
        html.Div("mean conf", className="k"), html.Div(f"{mean_conf:.2f}", className="v"),
        html.Div("mean fitness", className="k"), html.Div(f"{mean_fit:.2f}", className="v"),
        html.Div("scene pts (raw)", className="k"),
            html.Div(f"{len(scene.points):,}", className="v"),
        html.Div("display voxel", className="k"),
            html.Div(f"{display.voxel_mm:.2f} mm", className="v"),
    ])

    # --- figure ---
    # Compute display_points / colors source
    dp = display.points
    dc = display.colors

    # cluster labels / fit distances projected onto the display cloud via nearest neighbor
    cluster_labels = None
    fit_distances = None
    if color_mode == "segment" and ctx_.cluster_labels is not None:
        tree = cKDTree(ctx_.current_points)
        _, idx = tree.query(dp, k=1)
        lbl = ctx_.cluster_labels[idx]
        # points not near the working cloud -> -1
        dd, _ = tree.query(dp, k=1)
        lbl[dd > 8.0] = -1
        cluster_labels = lbl
    elif color_mode == "fit_quality" and ctx_.detections:
        # compute per-display-point distance to nearest model (transformed)
        all_model = []
        for det in ctx_.detections:
            pts = ctx_.model_points
            ones = np.ones((len(pts), 1))
            scene_pts = (np.hstack([pts, ones]) @ det.pose.T)[:, :3]
            all_model.append(scene_pts)
        if all_model:
            mp = np.vstack(all_model)
            tree = cKDTree(mp)
            d, _ = tree.query(dp, k=1)
            fit_distances = d

    # gripper planning
    gripper_traces = []
    show_gripper = (grip_en and 1 in grip_en) and ("gripper" in (layers or []))
    if show_gripper and ctx_.detections:
        spec = GripperSpec(
            width_mm=float(gw or 80), height_mm=float(gh or 30),
            finger_length_mm=float(gfl or 60), palm_depth_mm=float(gpd or 20),
        )
        # Use ctx_.current_points for the collision check (post-background)
        # Sample the full top-half hemisphere (cone_deg=90) so tight scenes
        # have the best chance of finding a non-colliding approach. 64 samples
        # is enough to reliably find any feasible approach wider than ~20°.
        # target_radius_mm tells the planner "points within this radius of
        # the grasp point are the target itself, not obstacles" — without
        # it every approach trivially collides with the object we're grasping.
        if model is not None and model.sphere_radius_mm is not None:
            target_r = float(model.sphere_radius_mm) + 3.0
        elif model is not None:
            target_r = float(0.5 * np.linalg.norm(model.extents_mm)) + 3.0
        else:
            target_r = 0.0
        for det in ctx_.detections[:20]:
            plan = plan_approach(ctx_.current_points, det.pose, spec,
                                  n_directions=64, cone_deg=90.0,
                                  safety_margin_mm=1.0,
                                  target_radius_mm=target_r)
            gripper_traces.extend(gripper_box_traces(plan, spec))

    fig = build_scene_figure(
        dp, dc,
        color_mode=color_mode,
        cluster_labels=cluster_labels,
        fit_distances=fit_distances,
        removed_points=ctx_.removed_points,
        model_points=ctx_.model_points if ("model" in (layers or [])) else None,
        detections=ctx_.detections,
        gripper_traces=gripper_traces,
        show_removed="bg" in (layers or []),
        show_model="model" in (layers or []),
        selected_instance_id=int(selected_id) if selected_id is not None else None,
    )
    return det_children, trace_children, summary, fig


def _fmt_stat(v):
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


# ---- Click a detection row -> update selected instance ----
@app.callback(
    Output("selected-instance-id", "data"),
    Input({"type": "det-row", "id": dash.ALL}, "n_clicks"),
    State({"type": "det-row", "id": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def on_detection_click(n_clicks_list, ids):
    if not ids or not any(n_clicks_list):
        return no_update
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and "id" in triggered:
        return triggered["id"]
    return no_update


# ---- Calibration tab ----
@app.callback(
    Output("calib-distances", "children"),
    Output("calib-residuals", "children"),
    Output("calib-heatmap", "figure"),
    Input("btn-calib", "n_clicks"),
    Input("main-tabs", "active_tab"),
    Input("result-timestamp", "data"),
    State("gt-distances", "value"),
    prevent_initial_call=False,
)
def on_calib(n_clicks, tab, ts, gt_text):
    with SESSION_LOCK:
        ctx_ = SESSION.get("last_ctx")
    if ctx_ is None or not ctx_.detections:
        msg = html.Div("No detections yet.", className="muted")
        return msg, msg, go.Figure()

    centers = np.array([d.translation for d in ctx_.detections])
    n = len(centers)
    D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)

    # table of distances
    rows = []
    for i in range(n):
        row = [html.Th(f"#{i}", style={"fontSize": "11px", "textAlign": "center"})]
        for j in range(n):
            if i == j:
                row.append(html.Td("—", style={"color": "#555", "textAlign": "center",
                                                 "fontSize": "11px"}))
            else:
                row.append(html.Td(f"{D[i, j]:.1f}",
                                    style={"textAlign": "center", "fontSize": "11px"}))
        rows.append(html.Tr(row))
    header = html.Tr([html.Th("")] +
                      [html.Th(f"#{j}", style={"fontSize": "11px", "textAlign": "center"})
                       for j in range(n)])
    table = html.Table([html.Thead(header), html.Tbody(rows)],
                        className="table table-dark table-sm")

    # heatmap
    hm = go.Figure(data=[go.Heatmap(
        z=D, x=[f"#{i}" for i in range(n)], y=[f"#{i}" for i in range(n)],
        colorscale="Viridis", colorbar=dict(title="mm"),
    )])
    hm.update_layout(
        paper_bgcolor="#1d2128", plot_bgcolor="#1d2128", font=dict(color="#e8ecf1"),
        # Top margin has to be big enough to actually contain the title text —
        # 10 px was clipping the title bar.
        margin=dict(l=30, r=10, t=40, b=30),
        title=dict(text="Pairwise distances (mm)", x=0.5, y=0.97,
                    xanchor="center", yanchor="top",
                    font=dict(size=12)),
    )

    # residuals
    residuals_ui = html.Div("Enter ground-truth distances and click Compute.",
                             className="muted")
    if gt_text and n_clicks:
        try:
            gt_pairs = []
            for line in gt_text.strip().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                i, j, d = [x.strip() for x in line.split(",")]
                gt_pairs.append((int(i), int(j), float(d)))
            rows_res = [html.Tr([
                html.Th("i"), html.Th("j"),
                html.Th("measured (mm)"), html.Th("GT (mm)"),
                html.Th("error (mm)"), html.Th("error %"),
            ])]
            errors = []
            for i, j, gt in gt_pairs:
                if i >= n or j >= n:
                    continue
                meas = D[i, j]
                err = meas - gt
                pct = (err / gt * 100.0) if gt > 0 else 0.0
                errors.append(err)
                rows_res.append(html.Tr([
                    html.Td(str(i)), html.Td(str(j)),
                    html.Td(f"{meas:.2f}"), html.Td(f"{gt:.2f}"),
                    html.Td(f"{err:+.2f}"), html.Td(f"{pct:+.2f}%"),
                ]))
            rms = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0
            mx = float(np.max(np.abs(errors))) if errors else 0.0
            residuals_ui = html.Div([
                html.Table(rows_res, className="table table-dark table-sm"),
                html.Div(f"RMS error: {rms:.3f} mm   ·   max |error|: {mx:.3f} mm",
                          style={"fontWeight": 600, "marginTop": "6px"}),
            ])
        except Exception as e:
            residuals_ui = html.Div(f"parse error: {e}", style={"color": "#ff6b6b"})

    return table, residuals_ui, hm


# ---- 2D detector callback ----
def _yolo_worker_thread(job_id: str, prompts: list[str], conf: float, rgb_hw, scene_snapshot_id):
    try:
        t0 = time.time()
        dets = yolo_detect(rgb_hw, prompts, conf=conf)
        dt = time.time() - t0
        with SESSION_LOCK:
            if SESSION.get("yolo_job_id") != job_id:
                return
            SESSION["twod_detections"] = dets
            SESSION["twod_status"] = f"{len(dets)} detections in {dt:.2f}s"
            SESSION["twod_ts"] = time.time()
            SESSION["yolo_running"] = False
    except Exception as e:
        traceback.print_exc()
        with SESSION_LOCK:
            SESSION["twod_status"] = f"ERROR: {e}"
            SESSION["yolo_running"] = False


@app.callback(
    Output("yolo-status", "children", allow_duplicate=True),
    Output("btn-yolo", "disabled"),
    Output("yolo-ticker", "disabled", allow_duplicate=True),
    Input("btn-yolo", "n_clicks"),
    State("yolo-prompts", "value"),
    State("yolo-conf", "value"),
    State("img-preproc", "value"),
    State("clahe-clip", "value"),
    State("norm-gamma", "value"),
    prevent_initial_call=True,
)
def on_yolo_click(n, prompts_text, conf, preproc, clahe_clip, gamma):
    if not n:
        return no_update, no_update, no_update
    with SESSION_LOCK:
        scene = SESSION.get("scene")
    if scene is None:
        return "upload a scene first", False, True
    if not scene.is_ordered:
        return "scene is not an ordered-grid PLY — can't run 2D detector", False, True
    prompts = [p.strip() for p in (prompts_text or "").splitlines() if p.strip()]
    if not prompts:
        return "enter at least one text prompt", False, True

    preproc = preproc or []
    rgb_input = preprocess_rgb(
        scene.rgb_hw,
        use_clahe=("clahe" in preproc),
        clahe_clip=float(clahe_clip or 2.0),
        use_normalize=("norm" in preproc),
        gamma=float(gamma or 1.2),
    )

    job_id = uuid.uuid4().hex
    with SESSION_LOCK:
        SESSION["yolo_job_id"] = job_id
        SESSION["yolo_running"] = True
        SESSION["twod_status"] = "running..."
        SESSION["rgb_preproc"] = {
            "use_clahe": "clahe" in preproc, "clahe_clip": float(clahe_clip or 2.0),
            "use_normalize": "norm" in preproc, "gamma": float(gamma or 1.2),
        }
    t = threading.Thread(
        target=_yolo_worker_thread,
        args=(job_id, prompts, float(conf), rgb_input, id(scene)),
        daemon=True,
    )
    t.start()
    pre_txt = "+".join([p for p in preproc]) or "raw"
    # Third return value enables the yolo-ticker, which periodically fires
    # refresh_2d_tab so the detections appear as soon as the worker thread
    # finishes (without the user having to switch tabs).
    return f"running YOLO-World on {pre_txt} image...", True, False


@app.callback(
    Output("yolo-status", "children"),
    Output("yolo-det-list", "children"),
    Output("image-fig", "figure"),
    Output("btn-yolo", "disabled", allow_duplicate=True),
    Output("yolo-ts", "data"),
    Output("yolo-ticker", "disabled"),
    Input("main-tabs", "active_tab"),
    Input("img-layer-toggles", "value"),
    Input("result-timestamp", "data"),
    Input("selected-instance-id", "data"),
    Input("yolo-ticker", "n_intervals"),
    Input("img-preproc", "value"),
    Input("clahe-clip", "value"),
    Input("norm-gamma", "value"),
    prevent_initial_call="initial_duplicate",
)
def refresh_2d_tab(active_tab, layers, result_ts, selected_id, _tick,
                   preproc, clahe_clip, gamma):
    with SESSION_LOCK:
        scene = SESSION.get("scene")
        dets_2d = list(SESSION.get("twod_detections") or [])
        status = SESSION.get("twod_status") or ""
        running = bool(SESSION.get("yolo_running"))
        last_ctx = SESSION.get("last_ctx")
    # Ticker is kept alive while the worker thread runs; we disable it as
    # soon as the thread flips yolo_running=False so we're not polling idly.
    ticker_disabled = not running

    if scene is None or not scene.is_ordered:
        fig = empty_image_figure(
            "Upload an ordered-grid scene (e.g. Photoneo PLY) to view its RGB texture."
            if scene is None else
            "Scene is not ordered — 2D view not available."
        )
        return status or "", html.Div("n/a", className="muted"), fig, running, no_update, ticker_disabled

    # Build detection list UI
    if dets_2d:
        rows = []
        for i, d in enumerate(dets_2d):
            rows.append(html.Div(className="det-row",
                id={"type": "yolo-det-row", "id": i}, n_clicks=0,
                children=[
                    html.Div(f"#{i}", className="rank"),
                    html.Div(children=[
                        html.Div(f"{d.class_name}", className="primary"),
                        html.Div(
                            f"bbox = ({d.bbox[0]:.0f},{d.bbox[1]:.0f})→"
                            f"({d.bbox[2]:.0f},{d.bbox[3]:.0f})",
                            className="secondary"),
                    ]),
                    html.Div(f"{d.confidence:.2f}", className="conf"),
                ]))
        det_list = rows
    else:
        det_list = html.Div("No 2D detections yet. Set prompts, click Run YOLO-World.",
                             className="muted")

    # Back-project 3D detections (from Localization tab) onto the image.
    bp_pix = None
    bp_labels = None
    if last_ctx is not None and last_ctx.detections:
        centers = np.array([d.translation for d in last_ctx.detections])
        tree = build_tree(scene.points)
        bp_pix = backproject_points_to_pixels(
            centers, scene.points, scene.pixel_of_point, tree=tree, max_dist_mm=30.0)
        bp_labels = [str(d.instance_id) for d in last_ctx.detections]

    preproc = preproc or []
    rgb_shown = preprocess_rgb(
        scene.rgb_hw,
        use_clahe=("clahe" in preproc),
        clahe_clip=float(clahe_clip or 2.0),
        use_normalize=("norm" in preproc),
        gamma=float(gamma or 1.2),
    )
    fig = build_image_figure(
        rgb_shown,
        twod_detections=dets_2d,
        backprojected_points=bp_pix,
        backprojected_labels=bp_labels,
        show_boxes="boxes" in (layers or []),
        show_backproj="backproj" in (layers or []),
        show_masks="masks" in (layers or []),
        selected_instance_id=(int(selected_id) if selected_id is not None else None),
    )
    return status, det_list, fig, running, time.time(), ticker_disabled


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
