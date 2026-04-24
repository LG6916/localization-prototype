# Localization Prototype

## Open-ended 3D object-detection sandbox for industrial scanning

> A working localhost-deployable web application for detecting any machinery
> part in a Photoneo (or any other) point-cloud scan and returning precise
> 6-DoF poses, confidence scores, and collision-aware robotic approach vectors
> — in under a minute on commodity hardware.

Built as an internal brainstorming and spec-discovery platform. Six
interchangeable pipelines, a modular five-stage runtime, and a live
dark-themed dashboard that surfaces every step of the algorithm so engineers,
product managers, and customers can **see** the trade-offs rather than argue
about them in abstract.

---

## Why it exists

Today's industrial bin-picking stack is a black box: scene in, poses out, no
visibility. When a customer reports a failure case, debugging means re-running
on their data offline and hoping the numbers match.

The Localization Prototype inverts this: every voxel downsample, every RANSAC
inlier count, every ICP iteration is visible, timed, and hot-swappable from
the browser.

---

## Core capabilities

### Six shipped pipeline presets

| preset | purpose | headline algorithm |
| --- | --- | --- |
| **A — Spheres fast-path** | calibration spheres, bearings | algebraic + iterative RANSAC sphere fit with radius prior |
| **B — Features + ICP** | generic asymmetric parts | FPFH descriptors + feature-matching RANSAC + point-to-plane ICP |
| **C — Cluster-seeded ICP** | conservative fallback | DBSCAN → per-cluster multi-start ICP |
| **D — 2D-guided ROIs + multi-start ICP** | cluttered scenes with textured or labelled parts | YOLO-World text-prompt 2D detection → ordered-grid 3D lift → PCA principal-axis seeding → multi-scale ICP |
| **E — 2D-guided sphere fit** | calibration scenes with RGB context | 2D detection → per-ROI sphere RANSAC |
| **F — 2D-guided + FPFH + ICP** | asymmetric parts in dense bins | per-ROI FPFH global registration + ICP refinement |

Each preset is a declarative list of five stages
(`preprocess / background / candidates / refine / scoring`). Every stage has
multiple swappable methods; users with deeper expertise open the **Advanced**
disclosure and edit method + JSON parameters per slot, live, without
restarting.

### Full 3D CV stack

Powered by **Open3D 0.19** with our own orchestration layer:

- **Preprocessing** — voxel downsampling, statistical outlier removal,
  oriented normal estimation via hybrid KD-tree radius / kNN search
- **Background removal** — iterative multi-plane RANSAC (up to _N_ planes),
  configurable depth cutoff
- **Candidate generation**
  - DBSCAN density clustering
  - RANSAC sphere fit with algebraic least-squares refinement and
    radius-prior gating
  - FPFH (Fast Point Feature Histograms) + Open3D's feature-matching RANSAC
    with correspondence-based edge-length and distance checkers
- **Pose refinement** — ICP point-to-point, point-to-plane, multi-scale
  coarse-to-fine, and colored ICP with geometric / photometric λ weighting;
  multi-start rotation seeding for pose-ambiguous geometries
- **Scoring** — pose-space non-maximum suppression using STL bbox diagonal as
  the default exclusion radius

### Open-vocabulary 2D detection

Bundles **YOLO-World (Ultralytics)** with the CLIP ViT-B/32 text encoder for
zero-shot, prompt-driven detection on the RGB texture:

- Text prompt → visual detection, no re-training
- Inference under 1 s on GPU, 2–4 s on CPU
- CLAHE (Contrast-Limited Adaptive Histogram Equalization) in Lab space +
  per-channel min-max + gamma pre-processing for Photoneo-dark textures
- Four documented recall levers — prompt engineering, model capacity (s/l/x
  variants), confidence threshold, image preprocessing — all user-tunable
  from the dashboard

### Free 2D ↔ 3D lifting via the ordered-grid format

Photoneo's ordered PLY embeds `Width × Height = N` with pixel-major layout.
The prototype preserves this grid end-to-end:

- **Lift 2D bbox → 3D ROI** is a constant-time indexing op, no camera
  intrinsics required
- **Back-project 3D pose → 2D pixel** via nearest-neighbor on the filtered
  cloud + stored `pixel_of_point` lookup
- Every 3D detection gets a marker overlaid on the RGB texture as a sanity
  check that 3D and 2D agree
- Graceful fallback: unordered / third-party PLYs still work, just without
  the 2D path

### Collision-aware gripper approach planning

- Parallel-jaw box model with configurable width, height, finger length,
  palm depth, approach offset
- **Fibonacci-sampled hemisphere** of 64 candidate approach directions
  around the scanner's view vector
- **Capsule-sweep collision check** against a SciPy `cKDTree` of the scene,
  with target-exclusion sphere so the object being grasped doesn't count as
  an obstacle
- Safety margin + explicit feasibility reporting — feasible plans render as
  a green wireframe + approach arrow; infeasible ones as a compact red
  marker with hover text explaining _why_ (e.g., _"tried 64, 0 feasible,
  shrink the gripper or widen the approach cone"_)

### Precision measurement tab

Turn any set of detections into a spatial-precision report:

- Pairwise inter-instance distance matrix (mm)
- Editable ground-truth input: `i,j,distance` lines
- Per-pair residuals, RMS error, max absolute error
- Viridis heatmap visualization

Ideal for side-by-side 3D-scanner comparison and factory-floor calibration
validation.

---

## Dashboard

- **Three tabs** — Localization (3D) · 2D RGB · Calibration / Precision
- **3D scene** (Plotly 3D, auto-oriented to the scanner's +Z view) — toggle
  model overlay, per-detection gripper wireframe + approach arrow,
  fit-quality heatmap colorization, segment-colored cluster view, RGB
  texture passthrough
- **2D scene** (Plotly with layout-cached PNG for 60 fps pan/zoom) — 2D
  detection bboxes, back-projected 3D pose markers, CLAHE / normalize
  toggles live-applied
- **Pipeline trace panel** — every stage's method, parameters, timing,
  input / output point counts, stage-specific stats (inlier fractions,
  plane models, cluster counts, ICP RMSE, FPFH feature radius, etc.)
- **Detection list** — ranked by confidence, click-to-focus, expandable
  4×4 pose matrices
- **Progress bar** — live % during runs via in-process worker thread +
  polling interval (no multiprocess state-sync complexity)
- **Run summary** — preset used, runtime, mean confidence, display voxel
  size, raw point count
- **Deliberate palette** — documented WCAG-AA-verified 4-shade dark palette
  with CSS custom-property overrides of Dash 4's design-system tokens

---

## Deployment

### Conda (development)

```bash
conda activate loc-proto
python app.py --port 8050
# → http://localhost:8050
```

### Docker (on-prem)

```bash
docker compose up -d --build
# → http://<on-prem-host>:8050
```

- Python 3.11 conda environment, `requirements.txt`-driven
- Dockerfile + compose file for on-prem deployment
- Stateless per session; point clouds never round-trip through the browser
  (display-only voxel downsample caps Plotly at ~50 k points)

---

## Technical stack

| layer | tech |
| --- | --- |
| Core 3D | Open3D 0.19, Trimesh 4, NumPy 2, SciPy 1.17, scikit-learn |
| Deep learning | Ultralytics YOLO-World, OpenAI CLIP, PyTorch 2.11 + CUDA 13 _(optional GPU)_ |
| Web frontend | Dash 4, Plotly 6, Dash Bootstrap Components, Pillow for fast PNG encoding |
| Infrastructure | Flask WSGI, Docker, docker-compose, diskcache |

---

## What's under the hood — algorithm highlights

- **Multi-plane RANSAC** to strip tote walls, conveyor, and bottom in one
  preprocessing pass
- **Algebraic LS sphere fit** (Pratt formulation) followed by RANSAC
  refinement over radius-constrained sampling — sub-mm center accuracy at
  2 mm voxel
- **FPFH with hybrid radius-kNN search**, correspondence mutual-filtering,
  edge-length + distance checkers for geometric consistency during RANSAC
  hypothesis generation
- **Multi-scale ICP** with descending voxel ladder (6 mm → 3 mm → 1.5 mm)
  for coarse-to-fine convergence
- **PCA principal-axis alignment** with four-way det-preserving flip
  enumeration as an ICP seed for asymmetric parts lifted from 2D ROIs
- **Capsule-swept gripper collision check** with _O(log n)_ KDTree ball
  queries, target-radius exclusion, and cone-constrained Fibonacci
  direction sampling

---

## What this prototype is NOT

It's a **brainstorming sandbox**, not a product.

- One session at a time
- CPU-bound runs (GPU used only for YOLO-World inference)
- No customer-facing robustness guarantees
- Deliberate limits are documented in the README (e.g., FPFH is
  pedagogically useless on spheres — kept for exactly that reason)

But: every algorithmic decision that would go into the next-generation
Localization offering can be prototyped, tested on real customer scans,
benchmarked, and demonstrated — in the same afternoon.

---

## Next steps — what we can discuss

- **Point-Pair Features** (Drost et al., 2010) as a seventh pipeline,
  filling the "symmetric asymmetric-part" gap between B and D
- **SAM 2** mask refinement on YOLO-World boxes for per-pixel 2D→3D ROIs
- **Image-to-image feature matching** (LightGlue) for painted / textured
  parts with STL texture
- **MinkowskiNet / PointNet++** neural candidate generation as a learnable
  alternative to DBSCAN
- **Multi-scan fusion** and time-of-flight accuracy modeling in the
  precision tab

---

_Source at `localization-prototype/`. See [README.md](./README.md) for setup,
file layout, and per-preset tuning guides._
