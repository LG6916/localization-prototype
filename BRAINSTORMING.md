# Brainstorming — Empirical findings from real test scenes

Running log of non-obvious insights discovered while exercising the prototype
on real scanner data. Each entry is something a brainstorming session should
chew on when specifying the next-generation Localization app. Opinionated;
challenge anything here.

---

## Test scenes used so far

| scene | model | geometry notes |
| --- | --- | --- |
| `calib.ply` | `sphere.stl` (r = 25 mm) | 7 reference spheres on a calibration rig at z ∈ [1070, 2500] mm |
| `flex.ply` | `part.stl` (120 × 40 × 40 mm bolt-like machined part) | bin of ~5 visible instances, PCA eigenvalue ratio 22.8 (strongly elongated) |

Add new scenes at the bottom of this doc as we exercise them. Each scene
should have: one-line description, which preset worked best, what was
surprising, what it tells us about the production spec.

---

## Finding 1 — PCA rotation seeding is load-bearing for generic bin-picking

**Observation.** Pipeline C's original implementation seeded every DBSCAN
cluster with **identity rotation** and relied on a 4-way multi-start around
the z-axis. On `flex.ply` (a bolt that lands at arbitrary 3D orientations in
the bin), this produced 4 detections with fitness 0.20–0.36 and RMSE 3.6 mm
— decent recall but mediocre alignment.

After adding PCA principal-axis alignment inside `cluster_centers` (cluster
covariance → eigen-decomp → align model's PCA frame onto cluster's, all
four det-preserving flip ambiguities scored by mean nearest-neighbor
distance), the same pipeline produces 4 detections at fitness 0.17–0.38,
RMSE 3.4 mm. **More importantly, every seed now starts within one flip
of the true orientation rather than from identity**, so ICP converges
consistently rather than depending on the initial Z-axis rotation matching.

**Why it matters.** Any part with a dominant axis — bolts, cylinders,
connecting rods, brackets, nozzles, shafts — falls into this same regime.
Identity-rotation + multi-start-around-z is a 4-sample cover of SO(3) which
is hopelessly sparse; PCA gives you SO(3) / (local flip group) for free.

**Recommendation for the spec.** Every cluster-seeded or proposal-based
pipeline in the next-gen app should include a PCA-seeding step before
handing to local optimisers. Cost is negligible (one eigh per cluster,
milliseconds). Fallback to identity only for sphere-like (sphericity >
0.95) models.

**Caveat.** PCA degenerates on rotationally-symmetric parts (spheres,
cylinders, toroids). The prototype auto-detects this via a radius inference
in `model_loader` and falls back to identity + multi-start for those.
Production should use a more robust symmetry check (e.g., eigenvalue
ratio < 1.5 on any two axes).

---

## Finding 2 — FPFH struggles on low-texture machined parts

**Observation.** Pipeline B (FPFH + RANSAC + ICP) on `flex.ply` returned
5 detections at RMSE 2.8 mm (acceptable) but emitted 8+ Open3D warnings
*"Too few correspondences after mutual filter, fall back to original
correspondences"*. FPFH descriptors on smooth machined surfaces have low
discriminability: most points look like cylindrical-shell neighborhoods to
each other, so mutual-NN feature matching falls apart and RANSAC has to fall
back to one-way correspondences.

Pipeline B on `calib.ply` returns **zero detections** by design —
rotationally-symmetric features are identical regardless of pose, the entire
FPFH premise fails.

**Recommendation for the spec.**
- FPFH is a **useful but narrow tool**. It shines on cogs, brackets, and
  anything with surface discontinuities. It fails on spheres, cylinders,
  smooth shafts.
- Pair it with a fallback: **at scan-load time, classify the STL by
  FPFH descriptor diversity** (run FPFH on the STL sample, look at
  the distribution of descriptors; low variance → FPFH will fail; do not
  route that part through Pipeline B).
- Consider **learned descriptors** (3DMatch, FCGF, GeoTransformer) as a
  drop-in alternative; they consistently outperform FPFH on low-texture
  industrial parts in the literature.

---

## Finding 3 — ICP fitness caps at ~0.3 for single-view scans

**Observation.** ICP's `fitness` metric is defined as *"fraction of source
(model) points with a scene correspondence within `max_correspondence_distance`"*.
For any part seen from a single scanner viewpoint, at least half of the
surface is self-occluded and has no possible scene correspondence.
Practical ceiling on fitness is therefore **roughly 0.3–0.5** even for a
pixel-perfect alignment. For small parts scanned at low angular resolution
(e.g., calibration spheres at z = 2 m), the ceiling drops further.

The original preset thresholds of `min_fitness = 0.15` were rejecting valid
detections wholesale on `flex.ply`. We had to drop to 0.08 for B and 0.10
for C to recover correct hits.

**Recommendation for the spec.**
- **Stop using raw fitness as the gatekeeping metric.** It's interpretable
  to a naive reader ("fraction matched") but it conflates "bad alignment"
  with "half the object was occluded, as expected".
- Candidate replacements:
  - **Fraction-of-visible-model**: estimate the visible subset of the STL
    (via ray-casting from the scanner position against the STL placed at
    the candidate pose) and score fitness relative to that subset.
  - **Per-inlier RMSE**: already computed by Open3D, is occlusion-invariant.
    RMSE < voxel_size + margin is a clean feasibility test.
  - **Surface-completeness score**: ratio of ICP inliers to expected visible
    surface area under the candidate pose, normalised by cluster size.
- Until we pick one, ship the tunable threshold in the UI as we do today
  and document per-scene-class sensible values in the README.

---

## Finding 4 — YOLO-World does not generalize to niche industrial geometry

**Observation.** Running YOLO-World on `flex.ply`'s RGB texture with every
prompt we tried (`bolt`, `screw`, `metallic part`, `cylinder`, `machined
part`, `shaft`) returned **zero detections**. The `yolov8s-world.pt` model
is trained on natural-image data; the combination of structured-light
illumination artifacts, low RGB contrast, unusual camera geometry, and a
part class that doesn't appear in COCO-descendent training data is too far
out-of-distribution.

Increasing CLAHE and gamma, lowering confidence to 0.01, switching to the
`-l` or `-x` variants, prompt engineering, etc. were all insufficient.

**Recommendation for the spec.**
- **Open-vocab 2D detection is a useful additive tool, not a reliable
  replacement for geometric pipelines.** Deploy it only when
  (a) parts are close to classes in natural imagery, (b) RGB quality is
  reasonable, and (c) you have a geometric-only fallback path.
- For arbitrary customer parts, the realistic options are:
  1. **Few-shot fine-tuning**: YOLO-World supports 1-5 shot prompting with
     reference image patches instead of pure text. Worth a serious test.
  2. **Dedicated detector trained per customer part**: standard pattern
     (`pho-inference-kit` or `PhoMMDetection` style). Higher effort,
     much higher recall.
  3. **Skip 2D entirely and rely on PCA-seeded C or FPFH-based B/F**.
     `flex.ply` demonstrates this path works with just geometry.
- **Do not sell the prototype as "text-prompt any part"** without those
  caveats. For spheres + good RGB it works; for niche industrial parts it
  frequently does not.

---

## Finding 5 — Symmetric parts need dedicated handling throughout the stack

**Observation.** Rotational symmetry breaks three separate assumptions
the stack makes:

1. **FPFH descriptors** — identical on all symmetric surface points →
   Pipeline B fails (verified on `calib.ply`).
2. **ICP** — zero gradient along symmetry axis → Pipeline C converges
   to arbitrary nearby poses (RMSE 3–4 mm even when alignment looks right).
3. **PCA seeding** — eigenvalue multiplicity on the symmetric axis →
   principal axes are ill-defined.

The prototype works around this by making Pipeline A (sphere-specific
algebraic fit with radius prior) the dedicated path for spheres. It gets
sub-mm center accuracy because the sphere fit is 4-parameter, not 6-DoF.

**Recommendation for the spec.**
- **Classify STL symmetry at load time** (sphere / cylinder / other) and
  route to dedicated pipelines per class.
- For cylinders: implement a 5-parameter cylinder RANSAC (axis + radius + position).
- For spheres: the current algebraic LS + radius-constrained RANSAC is good.
- For "other": apply the general pipelines (C, D, F) with PCA seeding.
- **Do not try to hide symmetry inside a generic pipeline** — the degrees
  of freedom are fundamentally different.

---

## Finding 6 — The gripper target-exclusion sphere is non-obvious but critical

**Observation.** The first draft of the gripper planner treated every scene
point as an obstacle. Every approach direction it tested "collided" with
the target object's own surface because the capsule-swept gripper volume
naturally intersects the target's bounding sphere when the jaws close around
it. Every plan came back `feasible=False` with clearance ≈ −35 mm.

The fix: exclude scene points within `target_radius_mm` of the grasp point
from the KDTree query, and shift the capsule start by the same radius. After
this, 27–32 of 32 tested directions are feasible for well-isolated targets;
only genuinely blocked targets (adjacent neighbors within gripper width)
return 0 feasible.

**Recommendation for the spec.**
- Any future collision-checker must **explicitly separate target surface
  from obstacle surface**. The "target" is whatever the pose's 6-DoF frame
  encloses within its STL geometry; everything else is obstacle.
- Better yet: **do the check in STL space**, not point-cloud space. Sweep
  the gripper's CAD model against the STL-placed-at-pose plus the scene
  point cloud with the STL's neighborhood removed. That's more work but
  eliminates the ad-hoc `target_radius_mm` parameter.

---

## Finding 7 — Ordered-grid PLY format is a quiet superpower

**Observation.** Photoneo's `obj_info Ordered` PLYs let us do 2D → 3D ROI
lifting in constant time without any camera-intrinsics arithmetic. The lift
is literally `xyz_hw[mask]`. We use this for:

- Pipeline D/E/F foreground masking from YOLO-World boxes
- 3D detection back-projection to 2D for visual sanity-checking
- (Potentially) per-pixel hover info in the 2D tab
- (Potentially) depth-image-based range-image segmentation that we haven't
  yet exploited

**Recommendation for the spec.**
- **Preserve the ordered-grid format** in all future scanner outputs and
  intermediate data products. Don't let downstream tools (e.g., cloud
  fusion, noise filters) silently drop it.
- **Document the grid contract** as part of the file-format spec —
  customers who build tooling around Photoneo PLYs should be able to rely
  on ordered layout when it's present.
- Consider a lightweight metadata flag that indicates grid validity
  (e.g., after crop/resample operations that preserve it vs. those that
  don't).

---

## TL;DR — best-tool-for-the-job table

| scene type | first try | why | second try |
| --- | --- | --- | --- |
| calibration spheres | **A** | algebraic sphere fit; sub-mm center accuracy | C (noisy but OK) |
| elongated machined parts (bolts, cylinders with flats, connecting rods) | **C** | PCA seeding + ICP → clean rotation | B (if surface has features) |
| cluttered bin, good RGB, common-looking parts | **D** or **F** | YOLO-World pre-filter + ICP | C, A if spherical |
| niche machinery, poor RGB | **C** | pure geometry; ignore 2D path | B, F |
| rotationally-symmetric parts (cylinders, pipes, tori) | **A** for spheres; need dedicated cylinder fit for others | ICP is ill-defined under symmetry | C with conservative thresholds |

---

## Open questions for the next brainstorming session

1. What **SLA for pose accuracy** should the next-gen app advertise per
   part class? (sub-mm for spheres, 2 mm for bolts, ??)
2. Do we ship **PPF (Drost et al.)** as a seventh pipeline? It addresses
   exactly the niche FPFH fails on (symmetric-ish parts with some
   asymmetry — think pipe fittings).
3. How do we surface the **"FPFH will probably fail on this STL"** warning
   to the user at load time?
4. Should the **gripper planner search cone** be auto-derived from
   scanner geometry (limit to approach directions the robot arm can reach
   given the scanner-arm mount) rather than a fixed hemisphere?
5. What's the path for **multi-scan fusion**? Does the next-gen app support
   360° part views, or does it assume a single overhead scan?

---

*Edited by hand after each test-scene session. Most recent session:
`flex.ply` + `part.stl` (bolt-like machined part) — motivated findings
1, 2, 3, 5.*
