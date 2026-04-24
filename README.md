# Localization Prototype

A web-based prototype for 3D object localization in point-cloud scans. Built as
a **brainstorming sandbox** for exploring bin-picking, calibration, and quality
inspection workflows — not a production tool.

Inputs: a `.ply` scene (point cloud, typically from a Photoneo scanner) and a
`.stl` / `.obj` model of the object(s) to detect. The app runs one of several
localization pipelines, shows per-instance poses (rotation + translation),
confidences, a 3D interactive visualization, and an optional gripper approach
vector per instance. It also has a calibration tab that turns detections into a
pairwise-distance precision report.

**Companion docs.** [`SUMMARY.md`](./SUMMARY.md) is the one-pager overview
(share it with stakeholders). [`BRAINSTORMING.md`](./BRAINSTORMING.md) is
the running log of empirical findings from real test scenes — open that
one during design meetings.

---

## 1. Layout

```
┌─── Localization Prototype ──────────────────────────────────────────────────┐
│ [ Localization ]  [ Calibration / Precision ]                                │
├────────────┬─────────────────────────────────────────────┬──────────────────┤
│ INPUTS     │  view: [RGB|Segment|Fit quality]            │  DETECTIONS       │
│  scene ply │  layers: bg ▢  model ☑  gripper ☑          │  #0  sphere 0.77  │
│  model stl │                                              │  #1  sphere 0.67  │
│ PIPELINE   │                                              │  #2  sphere 0.50  │
│  preset ▾  │                [  3D plotly scene ]          │  ...              │
│  ▸ preproc │                                              │  PIPELINE TRACE   │
│  ▸ bg      │                                              │  [preproc] 120ms  │
│  ▸ cand    │                                              │  [bg] 50ms 4 pl.  │
│  ▸ refine  │                                              │  [cand] 300ms     │
│  ▸ score   │                                              │  [refine] none    │
│ GRIPPER    │                                              │  [score] 1ms      │
│  W H FL PD │                                              │  SUMMARY          │
├────────────┼─────────────────────────────────────────────┴──────────────────┤
│            │  [Run] [Cancel]  ██████████░░  42%  FPFH features (420 ms)      │
└────────────┴───────────────────────────────────────────────────────────────────┘
```

- **Left column** — file uploads, pipeline builder (preset + per-slot method +
  raw JSON params), gripper config.
- **Center** — Plotly 3D scene with togglable layers (background, model overlay,
  gripper wireframe), RGB / segment / fit-quality colorization, plus a run bar
  with progress.
- **Right column** — detections list (click to focus), pipeline trace (per-stage
  method, timings, in/out point counts, stats), run summary.
- **Calibration tab** — pairwise distance matrix from detected instance centers
  with a ground-truth distances textarea to compute RMS / max residuals.

---

## 2. Pipelines shipped

The pipeline is a 5-slot stack. Each slot has swappable methods; changing the
preset repopulates sensible defaults, but you can mix-and-match.

| Slot          | Methods |
| ------------- | --------------------------------------------------- |
| `preprocess`  | voxel+outlier, voxel_only, outlier_only, passthrough |
| `background`  | plane_ransac, plane_ransac_multi, depth_cutoff, **twod_mask_foreground**, none |
| `candidates`  | dbscan, sphere_ransac_per_cluster, feature_ransac, cluster_centers, **twod_rois** |
| `refine`      | icp_p2p, icp_p2pl, icp_multiscale, **icp_colored**, none |
| `scoring`     | standard, pose_nms, passthrough |

Presets:

- **A: Spheres fast-path** — plane removal → DBSCAN → per-cluster RANSAC sphere
  fit using the model's inferred radius. Ideal for calibration spheres.
- **B: Features (FPFH) + ICP** — voxel-downsampled FPFH features +
  feature-matching RANSAC for global initial poses → point-to-plane ICP
  refinement. Generic bin-picking path. Note: fails on perfectly-symmetric
  shapes (e.g. spheres) because rotationally-symmetric features are
  indistinguishable — useful for demonstrating the limit.
- **C: Cluster-seeded ICP** — DBSCAN clusters → seed model pose at each cluster
  centroid with multi-start rotations → ICP. Simple, robust, slower than A.
- **D: 2D-guided ROIs + multi-start ICP** — open-vocab YOLO-World on the RGB
  texture → each 2D bbox lifted to a 3D ROI via the ordered grid. Inside each
  ROI: for sphere-like STLs we sphere-RANSAC the translation; for any other
  STL we PCA-align the model's principal axes to the ROI's (trying all four
  det-preserving flip ambiguities) to get a rotation seed. Then `icp_p2p`
  with 8 multi-start rotations around z refines. **General-purpose** path for
  asymmetric parts (bolts, cogs, cylinders); also usable on spheres with
  low-ish confidence.
- **E: 2D-guided sphere fit (RANSAC only)** — same 2D pre-filter as D, but
  sphere-specific: each 2D bbox becomes a RANSAC sphere fit with known radius,
  no ICP. Highest confidence numbers on calibration-sphere scenes; will
  misbehave on non-sphere STLs because it ignores rotation entirely.
- **F: 2D-guided + Features (FPFH) + ICP** — the strongest general path for
  asymmetric parts. For each 2D ROI: voxel-downsample, compute FPFH on both
  scene-ROI and STL sample, run feature-matching RANSAC for an initial pose
  that doesn't depend on any PCA heuristic, then ICP refines. **Fails by design
  on rotationally-symmetric parts** (spheres, cylinders) — use D or A there.

---

## 3. Running on localhost

### Option A — conda (development)

```bash
# Create the environment
cd localization-prototype
conda create -y -n loc-proto python=3.11
conda activate loc-proto
pip install -r requirements.txt

# Launch
python app.py --host 127.0.0.1 --port 8050
# Then open http://127.0.0.1:8050
```

Click **Load calib.ply + sphere.stl example** (if you're on the dev machine
with those files on `~/Desktop`) or upload your own PLY + STL.

### Option B — Docker (on-prem deployment)

```bash
cd localization-prototype
docker compose up -d --build
# Open http://localhost:8050 (or http://<on-prem-host>:8050)
```

The container binds `./uploads` and `./cache` from the host for persistence.
For a multi-user on-prem setup, put the service behind an nginx reverse proxy;
nothing in this app assumes a single user, but the algorithm is CPU-bound so
you should size accordingly (one Open3D pipeline run may pin multiple cores).

### Environment variables

| Var         | Default     | Meaning                   |
| ----------- | ----------- | ------------------------- |
| `DASH_HOST` | `127.0.0.1` | bind address in container |
| `DASH_PORT` | `8050`      | port in container         |

---

## 3b. 2D RGB tab (optional, open-vocab detector)

A third tab sits between Localization and Calibration. It shows the ordered
PLY's RGB texture (reshaped from per-vertex colors) and runs YOLO-World via
ultralytics against user-supplied text prompts:

1. Enter prompts, one per line (e.g. `sphere`, `metallic screw`, `green cog`).
2. Pick a confidence threshold (0.03–0.10 is typical for industrial / scanner
   textures).
3. Optional: **CLAHE** + **Normalize** checkboxes preprocess the RGB before
   detection. Photoneo scans often have mean brightness ~50/255 because the
   scanner optimises for depth, not photogrammetry; CLAHE (clip 2.0, 8×8 grid)
   + min-max + gamma 1.2 typically brings the mean to ~80/255. Useful but
   usually a small effect — see the "Pushing 2D recall" section below for the
   bigger levers. The view and the detector always see the same image so what
   you see is what the model sees.
4. Hit **Run YOLO-World**. First run downloads `yolov8s-world.pt` (~26 MB)
   and the CLIP ViT-B/32 text encoder (~340 MB) into `cache/` / `~/.cache/clip/`.
5. Bboxes overlay the image; each detection becomes an entry in the side list.
6. Switch to the Localization tab and pick preset **D**, **E**, or **F**.

**Pushing 2D recall when the detector under-fires** (e.g. the calibration
sphere scene only yields 3 detections, all at confidence < 0.2). The cause
is usually *not* image brightness — once CLAHE is on, brightness is rarely
the bottleneck. The real constraints, in decreasing order of impact:

1. **Prompts**. `yolov8s-world` is trained on natural images, so its
   vocabulary maps "ball" to sports balls and toys, not calibration
   reference spheres on metal stands under structured light. Try richer
   prompts — `white marker ball`, `metal reference sphere`, `studio prop
   ball` — and observe which phrasing lifts recall. Prompt swings often
   dwarf every other setting.
2. **Model capacity**. Swap the weight file in `detector/yolo_world.py`
   (`_DEFAULT_WEIGHT`) to `yolov8l-world-v2.pt` (~90 MB) or
   `yolov8x-world-v2.pt` (~140 MB). Larger variants have noticeably better
   recall on out-of-distribution classes, at 2–4× inference time.
3. **Confidence threshold**. The slider goes as low as 0.01. Lower catches
   weaker detections at the cost of more false positives; downstream
   geometric fitting in pipeline E is fairly tolerant of spurious ROIs
   since RANSAC sphere fit simply fails to produce a detection on a ROI
   that doesn't contain one.
4. **Image preprocessing** (CLAHE / normalize) — the smallest of the four
   levers on most scenes, but free and worth leaving on for dark textures.
5. **Small-object limit**. At > ~2 m depth, calibration spheres shrink below
   ~25 px. YOLO's default strides struggle at that scale. A future add
   could tile the image and detect per-tile; not implemented today.

After a 3D run, the tab also back-projects each 3D detection center to the
image (green markers) — a fast visual sanity check that the 3D poses
correspond to what a human sees in the RGB view. Back-projection uses the
ordered-grid pixel↔point index, not camera intrinsics, so it works even on
rectified or cropped scans as long as the grid layout is preserved.

**Which preset for what**: use **D** for generic parts (bolts, cogs,
asymmetric shapes), **E** when the STL is a sphere and you want the highest
confidence numbers, **F** for asymmetric parts where you want FPFH instead of
PCA-based rotation seeding (more robust on unusual geometries at the cost of
slightly more runtime). **F fails on symmetric parts** by design.

**Disable the 2D path on lean installs**: comment out the `ultralytics` line
in `requirements.txt` and the 2D tab's Run button will be disabled with a
helpful message. The rest of the app keeps working.

## 4. Key design choices

- **Framework**: Dash + Plotly. Multi-panel dashboards with long-running
  background callbacks + progress updates are a first-class Dash feature.
- **Display downsampling**: The browser renders a voxel-downsampled copy of
  the scene capped at ~50 k points (binary-searched voxel size). The algorithm
  *always* runs on the full cloud.
- **3D CV backend**: [Open3D](https://www.open3d.org/) 0.19 for ICP
  (point-to-point, point-to-plane, multiscale, colored), plane RANSAC,
  DBSCAN, FPFH, voxel ops; [trimesh](https://trimesh.org/) for STL loading
  and surface sampling; scipy cKDTree for gripper collision checks.
- **Open-vocab 2D detector**: [ultralytics](https://docs.ultralytics.com/) YOLO-World
  (`yolov8s-world.pt`) for text-prompt detection on the RGB texture. Used
  only when pipelines D/E are picked or the user visits the 2D RGB tab —
  import is lazy so the baseline remains lightweight.
- **Units**: scenes are assumed to be in millimeters (Photoneo default). For
  the model, we auto-detect if extents are sub-1 (likely meters) and upscale
  by 1000. All UI readouts are in mm.
- **Photoneo PLY**: the loader preserves `obj_info Width/Height/Ordered`
  metadata and builds a full (H, W, 3) RGB+XYZ grid plus pixel↔point index
  maps. Those maps make "2D detection → 3D ROI" a constant-time lookup,
  which is what pipelines D/E exploit.
- **Progress**: Dash `background=True` callback + `set_progress` streams
  (label, fraction) to the progress bar. The `ProgressReporter` wraps this
  and gives each stage a slice of the 0–100 % range.
- **State**: kept server-side in a process-global `SESSION` dict. Point clouds
  never round-trip to the browser as JSON; only the figure and small stats do.
  Adequate for a single-user prototype; swap to Flask session or Redis for
  multi-user.

---

## 5. File structure

```
localization-prototype/
├── app.py                    # Dash app: layout + callbacks
├── config.py                 # constants (paths, downsample caps)
├── requirements.txt
├── environment.yml
├── Dockerfile
├── docker-compose.yml
├── ioutil/
│   ├── scene_loader.py       # PLY load, Photoneo metadata, invalid-pt filter
│   ├── model_loader.py       # STL load, unit auto-detect, surface sampling
│   └── display.py            # voxel-downsample for Plotly rendering
├── pipelines/
│   ├── base.py               # Pipeline, Stage, Detection, PipelineContext, ProgressReporter
│   ├── presets.py            # shipped presets A–E
│   └── stages/
│       ├── preprocess.py     # voxel, outlier, normals
│       ├── background.py     # plane RANSAC (single/multi), depth cutoff, 2D-mask foreground
│       ├── candidates.py     # DBSCAN, sphere RANSAC, FPFH, 2D-detection ROIs
│       ├── refine.py         # ICP p2p / p2pl / multiscale / colored
│       └── scoring.py        # NMS + thresholding
├── detector/
│   └── yolo_world.py         # lazy YOLO-World wrapper (ultralytics)
├── viz/
│   ├── scene_fig.py          # main 3D Plotly figure
│   ├── image_fig.py          # 2D RGB + bbox/backproj overlays
│   ├── backproject.py        # 3D pose → pixel via ordered-grid nearest-neighbor
│   ├── color_modes.py        # fit-quality + segment colorizers
│   └── gripper_viz.py        # gripper wireframe + approach arrow traces
├── gripper/
│   └── planner.py            # collision-aware approach direction search
└── assets/
    └── custom.css            # dark theme + panel styling
```

---

## 6. Known limitations (intentional — these are the discussion starters)

- **Pipeline B on symmetric parts**: spheres, cylinders, other
  rotationally-symmetric parts will not be localized well by FPFH feature
  matching. This is a property of the feature, not a bug. Use Pipeline A / C
  for such parts, or explore PPF (Drost et al., 2010) as a next-step addition.
- **No PPF / template matching**: classical bin-picking often uses Point Pair
  Features. Open3D doesn't ship PPF; it's a natural third-party add.
- **No data-driven pipeline**: we stayed classical for this prototype. Natural
  additions: PointNet-based detection, SAM-2D into projected image + back-project,
  or MinkowskiNet instance segmentation.
- **Gripper model**: parallel-jaw box approximation with capsule-swept collision
  check. No fingertip geometry, no jaw-closing dynamics.
- **YOLO-World quirks on structured-light scenes**: recall is bounded by
  four things, roughly in order of impact: (1) **domain gap** — the model
  was trained on natural images, not industrial scanner textures, so matte
  calibration spheres on metal stands score low no matter how clean the
  RGB is; (2) **prompt vocabulary** — "sphere" / "ball" land near toys and
  sports balls in CLIP space, so phrases like "white marker ball" or
  "metal reference sphere" often shift recall more than any other setting;
  (3) **model size** — `-s` variant used by default; `-l` / `-x` have
  materially better OOD recall; (4) **small-object limit** at > ~2 m depth
  where spheres drop under ~25 px. Image preprocessing (CLAHE/gamma) helps
  but is typically the smallest lever; it's on the UI as an easy toggle
  because it's free to try.
- **2D tab requires ordered-grid PLY**: the lift "2D mask → 3D ROI" is free
  only because the PLY is a W×H grid. Unordered/merged clouds fall back to
  geometric-only pipelines gracefully (2D tab shows a note, presets D/E
  refuse to run).
- **Rendering**: Plotly slows down beyond ~100 k markers. For larger clouds
  consider Potree/three.js or custom WebGL.
- **State**: one active session per process. Not multi-user-safe.

---

## 7. Troubleshooting

- **"too few correspondences" warning (Pipeline B)**: expected on symmetric
  parts or when the scene model has too few salient surfaces. Try Pipeline A/C.
- **Pipeline A finds 0 detections**: check that auto-detected model scale is
  right (shown under the model upload). If the sphere is treated as 0.05 mm
  instead of 50 mm, every cluster will be rejected. Also widen
  `radius_tolerance_frac` or lower `sphere_min_inliers` in the `candidates`
  params textarea.
- **ICP very slow (Pipeline C, >60 s)**: reduce `n_rotations`, or switch to
  `multi_start=false`, or increase `voxel_mm` in preprocessing.
- **Docker build fails on `libGL`**: the Dockerfile installs `libgl1`; on
  older distros you may need `libgl1-mesa-glx` instead.
