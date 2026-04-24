"""Built-in pipeline presets. The UI can swap any slot's method."""
from __future__ import annotations

from .base import Pipeline, Stage
from .stages import (
    PreprocessStage,
    BackgroundStage,
    CandidatesStage,
    RefineStage,
    ScoringStage,
)

# Declarative definition — list of (slot, method, params) tuples per preset.
PRESETS: dict[str, list[tuple[str, str, dict]]] = {
    "A: Spheres fast-path": [
        ("preprocess", "voxel+outlier", {"voxel_mm": 2.0}),
        ("background", "plane_ransac_multi", {
            "distance_threshold_mm": 4.0, "min_plane_fraction": 0.10, "max_planes": 4,
        }),
        # Tightened for low false-positive rate. The three filters stack:
        #   - sphere_inlier_tol_mm 1.0 : RANSAC keeps only tight fits
        #   - radius_tolerance_frac 0.08 : rejects any cluster whose best
        #     sphere radius differs > 8% from the STL prior (for r=25 mm
        #     that's ±2 mm, enough slack for voxel noise)
        #   - sphere_min_inliers 80 : keeps far-field partial spheres that
        #     only have a small hemisphere visible
        # Final pass: scoring.min_fitness 0.7 rejects clusters whose best
        # sphere fit used less than 70% of the cluster's points.
        ("candidates", "sphere_ransac_per_cluster", {
            "dbscan_eps_mm": 8.0, "dbscan_min_points": 30,
            "sphere_inlier_tol_mm": 1.0, "sphere_ransac_iters": 400,
            "radius_tolerance_frac": 0.08, "sphere_min_inliers": 80,
        }),
        ("refine", "none", {}),
        ("scoring", "standard", {"min_fitness": 0.7, "nms_distance_mm": 60.0,
                                   "nms_use_model_diag": True}),
    ],
    "B: Features (FPFH) + ICP": [
        ("preprocess", "voxel+outlier", {"voxel_mm": 3.0}),
        ("background", "plane_ransac_multi", {
            "distance_threshold_mm": 4.0, "min_plane_fraction": 0.10, "max_planes": 3,
        }),
        ("candidates", "feature_ransac", {
            "voxel_scene_mm": 3.0, "voxel_model_mm": 2.0,
            "feature_radius_mul": 5.0, "max_candidates": 20,
        }),
        ("refine", "icp_p2pl", {"max_corr_dist_mm": 5.0, "max_iter": 60, "multi_start": False}),
        # min_fitness is deliberately low: ICP fitness = fraction of model
        # points with a scene correspondence, and for any single-view scan
        # at most ~half the surface is visible, so 0.25 is a practical ceiling
        # even for a pixel-perfect match. 0.08 admits partial/occluded hits
        # without letting random RANSAC noise through.
        ("scoring", "standard", {"min_fitness": 0.08, "nms_distance_mm": 40.0}),
    ],
    "C: Cluster-seeded ICP": [
        ("preprocess", "voxel+outlier", {"voxel_mm": 3.0}),
        ("background", "plane_ransac_multi", {
            "distance_threshold_mm": 4.0, "min_plane_fraction": 0.10, "max_planes": 3,
        }),
        # cluster_centers now PCA-aligns non-sphere models, so multi_start is
        # unnecessary for most parts — PCA gives a directly usable rotation
        # seed. Left on as a safety net for cases where PCA is degenerate.
        ("candidates", "cluster_centers", {"dbscan_eps_mm": 8.0, "dbscan_min_points": 30}),
        ("refine", "icp_p2pl", {"max_corr_dist_mm": 6.0, "max_iter": 50,
                                  "multi_start": True, "n_rotations": 4}),
        # 0.10 keeps noise controlled on rotationally-symmetric parts
        # (spheres, cylinders) where ICP is inherently ambiguous. Bolt-like
        # parts with PCA seeding comfortably clear this threshold.
        ("scoring", "standard", {"min_fitness": 0.10, "nms_distance_mm": 60.0}),
    ],
    "D: 2D-guided ROIs + multi-start ICP": [
        ("preprocess", "voxel+outlier", {"voxel_mm": 2.0}),
        ("background", "twod_mask_foreground", {"bbox_dilate_px": 10}),
        ("candidates", "twod_rois", {
            "roi_min_points": 40, "roi_bbox_dilate_px": 6,
            "roi_sphere_inlier_tol_mm": 1.5, "roi_sphere_ransac_iters": 300,
            "radius_tolerance_frac": 0.15,
        }),
        # p2p ICP (not p2pl) so spherical / rotationally-symmetric parts where
        # surface normals are degenerate still converge. Multi-start rotates
        # the seed around z to cover orientation uncertainty on PCA-symmetric parts.
        ("refine", "icp_p2p", {
            "max_corr_dist_mm": 4.0, "max_iter": 60,
            "multi_start": True, "n_rotations": 8,
        }),
        # ICP fitness = fraction of model points with a scene correspondence;
        # self-occlusion caps this at ~0.4–0.5 for one-sided views, so the
        # threshold is deliberately loose here. Raise it for asymmetric parts
        # where you expect nearly full surface alignment.
        ("scoring", "standard", {"min_fitness": 0.05, "nms_distance_mm": 40.0}),
    ],
    "E: 2D-guided sphere fit (RANSAC only)": [
        ("preprocess", "voxel+outlier", {"voxel_mm": 2.0}),
        ("background", "twod_mask_foreground", {"bbox_dilate_px": 10}),
        ("candidates", "twod_rois", {
            "roi_min_points": 40, "roi_bbox_dilate_px": 6,
            "roi_sphere_inlier_tol_mm": 1.2, "roi_sphere_ransac_iters": 400,
            "radius_tolerance_frac": 0.12,
        }),
        ("refine", "none", {}),
        ("scoring", "standard", {"min_fitness": 0.2, "nms_distance_mm": 40.0}),
    ],
    "F: 2D-guided + Features (FPFH) + ICP": [
        ("preprocess", "voxel+outlier", {"voxel_mm": 2.0}),
        ("background", "twod_mask_foreground", {"bbox_dilate_px": 10}),
        ("candidates", "twod_feature_rois", {
            "voxel_scene_mm": 2.0, "voxel_model_mm": 1.5,
            "feature_radius_mul": 5.0, "roi_min_points": 40,
            "roi_bbox_dilate_px": 6,
            "mutual_filter": True, "ransac_n": 3,
            "ransac_iters": 50000, "ransac_confidence": 0.999,
        }),
        ("refine", "icp_p2pl", {"max_corr_dist_mm": 3.0, "max_iter": 60,
                                  "multi_start": False}),
        ("scoring", "standard", {"min_fitness": 0.15, "nms_distance_mm": 30.0}),
    ],
}


_STAGE_CLS = {
    "preprocess": PreprocessStage,
    "background": BackgroundStage,
    "candidates": CandidatesStage,
    "refine": RefineStage,
    "scoring": ScoringStage,
}


def list_methods(slot: str) -> list[str]:
    return _STAGE_CLS[slot].METHODS


def build_pipeline(name: str, overrides: list[tuple[str, str, dict]] | None = None) -> Pipeline:
    """Build a Pipeline. `overrides` can replace the method and params of each slot."""
    spec = PRESETS[name] if overrides is None else overrides
    stages: list[Stage] = []
    for slot, method, params in spec:
        cls = _STAGE_CLS[slot]
        stages.append(cls(method=method, params=params))
    return Pipeline(name=name, stages=stages)
