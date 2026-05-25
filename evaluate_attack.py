"""
DTU-style 3D evaluation for VGGT predictions saved by `depth_attack.executed.ipynb`.

Reports the three metrics used by MVSNet / Depth Anything V3:

  Accuracy     = mean( dist(p, NN_GT(p))    for p in pred_in_ObsMask        )   [mm]
  Completeness = mean( dist(g, NN_pred(g))  for g in GT_above_Plane         )   [mm]
  Overall      = (Accuracy + Completeness) / 2                                  [mm]

VGGT predicts geometry up to a similarity (scale + SE(3)); the DTU ground-truth
point cloud is in millimetres in the DTU world frame. We therefore solve for a
7-DoF similarity (s, R, t) that maps pred -> GT before computing the metrics.

Alignment pipeline:
  1. Confidence-filter and random-subsample the predicted points (alignment
     uses a few × 10^4 points; the final metric uses the whole cloud).
  2. Coarse init: PCA matches the principal axes of pred and GT.  PCA has a
     sign ambiguity on each of three axes (constrained to det=+1), so we try
     all four valid sign flips and keep the lowest-RMSE start.
  3. ICP-with-scale: iterate (nearest neighbour -> Umeyama-with-scale) on the
     alignment-subsample, with point-pair gating to reject NN matches whose
     residual exceeds a robust threshold.
  4. The final (s, R, t) is applied to the full prediction.

Metric computation follows the published DTU MATLAB protocol
(`BaseEvalMain_web.m`, `ComputeStat_web.m`):
  - voxel-downsample both clouds at 0.2 mm
  - cap per-point distances at 20 mm (suppresses far-outlier influence)
  - Accuracy   uses pred points whose voxel falls inside the ObsMask region
  - Completeness uses GT points above the Plane cutoff
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from plyfile import PlyData
from scipy.io import loadmat
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_pred(npz_path: Path,
              conf_thresh: float,
              conf_field: str = "depth_conf") -> Tuple[np.ndarray, np.ndarray]:
    """Return (points [N,3], conf [N]) from a saved VGGT prediction.

    Always uses `world_pts` for the 3D points.  The confidence field used to
    filter them can be either `depth_conf` (depth-head confidence; appropriate
    when the attack targeted depth) or `world_pts_conf` (point-head confidence;
    appropriate when the attack targeted the point map).  They are not on the
    same scale, so the threshold has different meaning for each.
    """
    if conf_field not in ("depth_conf", "world_pts_conf"):
        raise ValueError(f"Unknown conf_field: {conf_field!r}")
    with np.load(npz_path) as d:
        pts = d["world_pts"].astype(np.float32)
        conf = d[conf_field].astype(np.float32)
    pts = pts.reshape(-1, 3)
    conf = conf.reshape(-1)
    keep = conf > conf_thresh
    return pts[keep], conf[keep]


def load_gt_ply(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)


@dataclass
class ObsMaskInfo:
    mask: np.ndarray      # 3-D uint8 voxel occupancy
    bb_min: np.ndarray    # (3,) lower corner in mm
    bb_max: np.ndarray    # (3,) upper corner in mm
    res: float            # voxel size in mm
    margin: int           # voxels of margin used when the mask was built

    def contains(self, pts_mm: np.ndarray) -> np.ndarray:
        """Boolean: True where pts (in DTU world mm) fall inside the mask."""
        idx = np.floor((pts_mm - self.bb_min) / self.res).astype(np.int64)
        nx, ny, nz = self.mask.shape
        inside = (
            (idx[:, 0] >= 0) & (idx[:, 0] < nx)
            & (idx[:, 1] >= 0) & (idx[:, 1] < ny)
            & (idx[:, 2] >= 0) & (idx[:, 2] < nz)
        )
        out = np.zeros(len(pts_mm), dtype=bool)
        ix, iy, iz = idx[inside].T
        out[inside] = self.mask[ix, iy, iz] > 0
        return out


def load_obsmask(mat_path: Path) -> ObsMaskInfo:
    m = loadmat(str(mat_path))
    bb = m["BB"].astype(np.float64)           # (2, 3)
    return ObsMaskInfo(
        mask=m["ObsMask"].astype(np.uint8),
        bb_min=bb[0],
        bb_max=bb[1],
        res=float(m["Res"][0, 0]),
        margin=int(m["Margin"][0, 0]),
    )


def load_plane(mat_path: Path) -> np.ndarray:
    """Return (4,) plane [a b c d] s.t. a*x+b*y+c*z+d > 0 means 'above table'."""
    return loadmat(str(mat_path))["P"].astype(np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# ETH3D ground truth (COLMAP-format SfM reconstruction)
# ---------------------------------------------------------------------------

def parse_eth3d_images_txt(images_txt: Path) -> dict:
    """Parse COLMAP `images.txt` → {image_relative_name: IMAGE_ID}.

    File layout (per image, two lines):
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      POINTS2D[] as (X, Y, POINT3D_ID) triplets

    We skip comments and read non-comment lines in pairs; only the first line
    of each pair is informative (the POINTS2D line is ignored).
    """
    out = {}
    with open(images_txt) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    # Even-indexed lines are headers, odd-indexed are POINTS2D lists.
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        name = parts[9]
        out[name] = image_id
    return out


def parse_eth3d_points3d_txt(points3d_txt: Path,
                             allowed_image_ids: Optional[set] = None
                             ) -> np.ndarray:
    """Parse COLMAP `points3D.txt` → (N, 3) point cloud.

    Each non-comment line: `POINT3D_ID X Y Z R G B ERROR (IMAGE_ID POINT2D_IDX)*`.
    If `allowed_image_ids` is given, keep only points whose track intersects
    it — i.e., points that are seen by at least one of our selected images.
    That filter is the ETH3D analog of DTU's ObsMask (it removes GT geometry
    no view in the attack actually sees, so completeness is not penalised by
    parts of the scene outside every camera frustum).
    """
    pts = []
    with open(points3d_txt) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
            if allowed_image_ids is not None:
                # Track starts at index 8: alternating (IMAGE_ID, POINT2D_IDX).
                track_ids = (int(parts[k]) for k in range(8, len(parts), 2))
                if not any(t in allowed_image_ids for t in track_ids):
                    continue
            pts.append(xyz)
    return np.asarray(pts, dtype=np.float32)


def load_eth3d_gt(scene_dir: Path,
                  image_rel_paths: list,
                  calib_subdir: str = "dslr_calibration_undistorted"
                  ) -> Tuple[np.ndarray, np.ndarray, set]:
    """Return (gt_visible, gt_all, allowed_ids) for an ETH3D scene.

    `gt_visible` is the SfM point cloud filtered to points whose track contains
    at least one of `image_rel_paths`; `gt_all` is the full SfM cloud (used
    only for the ICP alignment, where more reference points help).
    """
    calib = Path(scene_dir) / calib_subdir
    images_txt = calib / "images.txt"
    points3d_txt = calib / "points3D.txt"
    name_to_id = parse_eth3d_images_txt(images_txt)
    missing = [p for p in image_rel_paths if p not in name_to_id]
    if missing:
        raise RuntimeError(
            f"Image paths not found in {images_txt}: {missing[:3]}…"
        )
    allowed_ids = {name_to_id[p] for p in image_rel_paths}
    gt_visible = parse_eth3d_points3d_txt(points3d_txt, allowed_ids)
    gt_all = parse_eth3d_points3d_txt(points3d_txt, None)
    return gt_visible, gt_all, allowed_ids


# ---------------------------------------------------------------------------
# Alignment: Umeyama (similarity) + multi-start ICP
# ---------------------------------------------------------------------------

def umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Closed-form similarity (s, R, t) minimising || s R src + t - dst ||^2.

    Reference: Umeyama 1991.  Constrains det(R) = +1.
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    var_s = (sc ** 2).sum() / n
    cov = (dc.T @ sc) / n                       # (3,3)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / max(var_s, 1e-20))
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def apply_sim(pts: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * (pts @ R.T)) + t


def _pca_basis(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centroid, eigenvalues_desc, eigenvectors_desc) of pts covariance.

    Eigenvectors are columns. We force the basis to be right-handed (det=+1)
    so that the four sign-flip combinations in `_coarse_starts` all yield
    valid rotations.  The sign of each individual column is still arbitrary.
    """
    mu = pts.mean(axis=0)
    c = pts - mu
    cov = (c.T @ c) / max(len(c) - 1, 1)
    w, V = np.linalg.eigh(cov)               # ascending
    order = np.argsort(w)[::-1]
    V = V[:, order]
    if np.linalg.det(V) < 0:
        V[:, 2] = -V[:, 2]
    return mu, w[order], V


def _coarse_starts(pred: np.ndarray, gt: np.ndarray) -> list:
    """Build up to 4 (s, R, t) initial guesses via PCA + axis-sign flips.

    PCA leaves a sign ambiguity on each of the three principal axes; det(R)=+1
    means an even number of flips, giving 4 valid orientations.
    """
    mu_p, w_p, V_p = _pca_basis(pred)
    mu_g, w_g, V_g = _pca_basis(gt)
    # axis scales: stdev along each principal direction
    sigma_p = np.sqrt(np.clip(w_p, 1e-20, None))
    sigma_g = np.sqrt(np.clip(w_g, 1e-20, None))
    s_init = float((sigma_g / sigma_p).mean())
    starts = []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            sz = sx * sy                                  # forces det=+1
            S = np.diag([sx, sy, sz])
            R = V_g @ S @ V_p.T
            if np.linalg.det(R) < 0:
                continue
            t = mu_g - s_init * (R @ mu_p)
            starts.append((s_init, R, t))
    return starts


def icp_with_scale(pred: np.ndarray,
                  gt: np.ndarray,
                  *,
                  iters: int = 30,
                  inlier_quantile: float = 0.85,
                  tol: float = 1e-5,
                  scale_drift_max: float = 10.0,
                  verbose: bool = False) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Multi-start ICP-with-scale.  Returns the best (s, R, t, rmse).

    `scale_drift_max` clamps each start so its final scale stays within
    [s_init / scale_drift_max, s_init * scale_drift_max].  Without this, an
    adversarially corrupted pred cloud (which has no consistent rotation+
    scale to GT) can drag Umeyama into a degenerate s→0 solution that maps
    every pred point onto the GT centroid — yielding a misleadingly small
    capped RMSE.  The PCA-derived `s_init` is a strong physical prior on the
    overall pred-vs-GT scale ratio; an order of magnitude either side is
    plenty of slack for healthy fits while still catching collapses.
    """
    gt_tree = cKDTree(gt)
    best = None
    starts = _coarse_starts(pred, gt)
    for i_start, (s, R, t) in enumerate(starts):
        s_init = s
        s_lo = s_init / scale_drift_max
        s_hi = s_init * scale_drift_max
        prev_rmse = np.inf
        for it in range(iters):
            xform = apply_sim(pred, s, R, t)
            dist, idx = gt_tree.query(xform, k=1, workers=-1)
            thresh = max(np.quantile(dist, inlier_quantile), 1e-6)
            mask = dist <= thresh
            if mask.sum() < 12:                # need ≥ a handful of correspondences
                break
            s, R, t = umeyama(pred[mask], gt[idx[mask]])
            # Clamp scale into [s_init/X, s_init*X] to block degenerate fits.
            # When s is clipped, recompute t consistently from the new s so
            # the centroids still line up: t = mu_d - s R mu_s.
            if s < s_lo or s > s_hi:
                s_clamped = float(np.clip(s, s_lo, s_hi))
                mu_s = pred[mask].mean(axis=0)
                mu_d = gt[idx[mask]].mean(axis=0)
                t = mu_d - s_clamped * (R @ mu_s)
                s = s_clamped
            rmse = float(np.sqrt((dist[mask] ** 2).mean()))
            if abs(prev_rmse - rmse) < tol * max(prev_rmse, 1.0):
                break
            prev_rmse = rmse
        # Score on *all* correspondences, not just inliers, so a start that hits a
        # local minimum of a small inlier subset doesn't beat a real global fit.
        xform = apply_sim(pred, s, R, t)
        dist, _ = gt_tree.query(xform, k=1, workers=-1)
        capped = np.minimum(dist, np.quantile(dist, 0.9))
        score = float(np.sqrt((capped ** 2).mean()))
        if verbose:
            print(f"  start {i_start}: score={score:.3f}  s={s:.4g}  s_init={s_init:.4g}")
        if best is None or score < best[3]:
            best = (s, R, t, score)
    return best


# ---------------------------------------------------------------------------
# Metrics (DTU protocol)
# ---------------------------------------------------------------------------

def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Keep one mean-position point per occupied cubic voxel."""
    if voxel <= 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    # ordering on keys = lexsort
    order = np.lexsort(keys.T[::-1])
    keys = keys[order]
    pts = pts[order]
    diffs = np.any(keys[1:] != keys[:-1], axis=1)
    boundaries = np.concatenate(([0], np.flatnonzero(diffs) + 1, [len(keys)]))
    # mean within each run
    sums = np.add.reduceat(pts, boundaries[:-1], axis=0)
    counts = np.diff(boundaries).reshape(-1, 1)
    return (sums / counts).astype(np.float32)


def chamfer_metrics(pred_mm: np.ndarray,
                    gt_mm: np.ndarray,
                    obs: ObsMaskInfo,
                    plane: np.ndarray,
                    *,
                    voxel: float = 0.2,
                    max_dist: float = 20.0) -> dict:
    """Compute Acc / Comp / Overall in millimetres."""
    pred_ds = voxel_downsample(pred_mm, voxel)
    gt_ds = voxel_downsample(gt_mm, voxel)

    # Accuracy: predicted points inside the observable volume.
    pred_in_obs = obs.contains(pred_ds)
    pred_eval = pred_ds[pred_in_obs]

    # Completeness: GT points above the plane and inside ObsMask.
    gt_above = (gt_ds @ plane[:3] + plane[3]) > 0
    gt_in_obs = obs.contains(gt_ds)
    gt_eval = gt_ds[gt_above & gt_in_obs]

    if len(pred_eval) == 0 or len(gt_eval) == 0:
        return {
            "accuracy_mm": float("nan"),
            "completeness_mm": float("nan"),
            "overall_mm": float("nan"),
            "n_pred_total": int(len(pred_ds)),
            "n_pred_in_obs": int(len(pred_eval)),
            "n_gt_total": int(len(gt_ds)),
            "n_gt_eval": int(len(gt_eval)),
        }

    tree_gt = cKDTree(gt_ds)        # NN over full GT, not just gt_eval
    tree_pr = cKDTree(pred_ds)      # NN over full pred, not just pred_eval
    d_pred_to_gt, _ = tree_gt.query(pred_eval, k=1, workers=-1)
    d_gt_to_pred, _ = tree_pr.query(gt_eval, k=1, workers=-1)

    acc = float(np.minimum(d_pred_to_gt, max_dist).mean())
    comp = float(np.minimum(d_gt_to_pred, max_dist).mean())

    # Also report median + a coverage statistic (fraction within 2 mm) for
    # robustness against tail dominance.
    return {
        "accuracy_mm": acc,
        "completeness_mm": comp,
        "overall_mm": 0.5 * (acc + comp),
        "accuracy_median_mm": float(np.median(d_pred_to_gt)),
        "completeness_median_mm": float(np.median(d_gt_to_pred)),
        "frac_pred_within_2mm": float((d_pred_to_gt < 2.0).mean()),
        "frac_gt_within_2mm": float((d_gt_to_pred < 2.0).mean()),
        "n_pred_total": int(len(pred_ds)),
        "n_pred_in_obs": int(len(pred_eval)),
        "n_gt_total": int(len(gt_ds)),
        "n_gt_eval": int(len(gt_eval)),
        "voxel_size_mm": voxel,
        "max_dist_mm": max_dist,
    }


def chamfer_metrics_simple(pred: np.ndarray,
                           gt: np.ndarray,
                           *,
                           voxel: float,
                           max_dist: float,
                           taus: Tuple[float, ...] = (0.01, 0.02, 0.05),
                           ) -> dict:
    """Chamfer metrics without ObsMask/Plane filtering — for ETH3D, where the
    visibility filter has already been applied to `gt` upstream.

    Returns Acc/Comp/Overall in whatever distance units pred and gt share, plus
    F-scores at the supplied thresholds (ETH3D's published metric is the
    F-score at τ — defaults 1/2/5 cm match the public benchmark).
    """
    pred_ds = voxel_downsample(pred, voxel)
    gt_ds = voxel_downsample(gt, voxel)
    if len(pred_ds) == 0 or len(gt_ds) == 0:
        return {
            "accuracy": float("nan"),
            "completeness": float("nan"),
            "overall": float("nan"),
            "n_pred": int(len(pred_ds)),
            "n_gt": int(len(gt_ds)),
        }
    tree_gt = cKDTree(gt_ds)
    tree_pr = cKDTree(pred_ds)
    d_pred_to_gt, _ = tree_gt.query(pred_ds, k=1, workers=-1)
    d_gt_to_pred, _ = tree_pr.query(gt_ds, k=1, workers=-1)

    acc = float(np.minimum(d_pred_to_gt, max_dist).mean())
    comp = float(np.minimum(d_gt_to_pred, max_dist).mean())
    out = {
        "accuracy": acc,
        "completeness": comp,
        "overall": 0.5 * (acc + comp),
        "accuracy_median": float(np.median(d_pred_to_gt)),
        "completeness_median": float(np.median(d_gt_to_pred)),
        "n_pred": int(len(pred_ds)),
        "n_gt": int(len(gt_ds)),
        "voxel_size": voxel,
        "max_dist": max_dist,
    }
    # ETH3D F-score at τ: precision = frac(pred within τ of GT),
    # recall = frac(GT within τ of pred), F = 2 P R / (P + R).
    for tau in taus:
        prec = float((d_pred_to_gt < tau).mean())
        rec  = float((d_gt_to_pred < tau).mean())
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        out[f"precision@{tau}"] = prec
        out[f"recall@{tau}"]    = rec
        out[f"f1@{tau}"]        = f1
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def evaluate(pred_npz: Path,
             gt_ply: Path,
             obsmask_mat: Path,
             plane_mat: Path,
             *,
             conf_thresh: float = 1.5,
             conf_field: str = "depth_conf",
             align_subsample: int = 60_000,
             voxel: float = 0.2,
             max_dist: float = 20.0,
             seed: int = 0,
             verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)

    t0 = time.time()
    pred, _ = load_pred(pred_npz, conf_thresh, conf_field=conf_field)
    gt = load_gt_ply(gt_ply)
    obs = load_obsmask(obsmask_mat)
    plane = load_plane(plane_mat)
    if verbose:
        print(f"loaded pred={len(pred):,}  gt={len(gt):,}  obs={obs.mask.shape}"
              f"  ({time.time()-t0:.1f}s)")
    if len(pred) < 1000:
        raise RuntimeError(
            f"Only {len(pred)} predicted points survived conf_thresh={conf_thresh}; "
            f"the attack may have crushed all confidence — try a lower threshold.")

    # Random-subsample for ICP — full alignment over 3M GT points and 1.6M
    # pred points is unnecessary and slow.  We use the full clouds for the
    # final metric.
    sub_p = pred if len(pred) <= align_subsample else pred[
        rng.choice(len(pred), size=align_subsample, replace=False)]
    sub_g = gt if len(gt) <= align_subsample else gt[
        rng.choice(len(gt), size=align_subsample, replace=False)]

    t1 = time.time()
    s, R, t, align_score = icp_with_scale(sub_p, sub_g, verbose=verbose)
    if verbose:
        print(f"alignment: s={s:.4g}  align_score(capped RMSE)={align_score:.3f} mm"
              f"  ({time.time()-t1:.1f}s)")

    pred_mm = apply_sim(pred, s, R, t)

    t2 = time.time()
    metrics = chamfer_metrics(pred_mm, gt, obs, plane,
                              voxel=voxel, max_dist=max_dist)
    if verbose:
        print(f"metrics: acc={metrics['accuracy_mm']:.3f}  "
              f"comp={metrics['completeness_mm']:.3f}  "
              f"overall={metrics['overall_mm']:.3f}  ({time.time()-t2:.1f}s)")

    return {
        "pred_npz": str(pred_npz),
        "gt_ply": str(gt_ply),
        "conf_thresh": conf_thresh,
        "conf_field": conf_field,
        "alignment": {
            "scale": s,
            "rotation": R.tolist(),
            "translation": t.tolist(),
            "score_mm": align_score,
        },
        "metrics": metrics,
    }


def evaluate_eth3d(pred_npz: Path,
                   scene_dir: Optional[Path] = None,
                   *,
                   conf_thresh: float = 1.5,
                   conf_field: str = "world_pts_conf",
                   align_subsample: int = 60_000,
                   voxel: float = 0.005,
                   max_dist: float = 0.5,
                   seed: int = 0,
                   verbose: bool = True) -> dict:
    """Evaluate an ETH3D point-map attack.

    `pred_npz` must have been written by `point_map_attack.ipynb` (it carries
    `image_rel_paths` and `scene_dir`).  If `scene_dir` is None we read it
    from the npz; pass an override if the dataset has moved on disk.

    Units: ETH3D positions are in metres, so `voxel` and `max_dist` default
    to 5 mm and 50 cm respectively.
    """
    rng = np.random.default_rng(seed)

    t0 = time.time()
    pred, _ = load_pred(pred_npz, conf_thresh, conf_field=conf_field)
    with np.load(pred_npz) as d:
        image_rel_paths = [str(s) for s in d["image_rel_paths"].tolist()]
        npz_scene_dir = Path(str(d["scene_dir"]))
    scene_dir = Path(scene_dir) if scene_dir is not None else npz_scene_dir
    gt_vis, gt_full, allowed_ids = load_eth3d_gt(scene_dir, image_rel_paths)
    if verbose:
        print(f"loaded pred={len(pred):,}  gt_visible={len(gt_vis):,}/"
              f"{len(gt_full):,}  ({time.time()-t0:.1f}s)")
    if len(pred) < 1000:
        raise RuntimeError(
            f"Only {len(pred)} predicted points survived conf_thresh={conf_thresh}; "
            f"the attack may have crushed all confidence — try a lower threshold.")
    if len(gt_vis) < 100:
        raise RuntimeError(
            f"Only {len(gt_vis)} GT points fall in the tracks of the selected "
            f"images. Check that image_rel_paths match images.txt entries.")

    # Align pred → gt_visible. Aligning against gt_full pulls PCA toward the
    # SfM far-outliers (sky/distant facades extend to ±100 m, while the
    # courtyard footprint is ~20 m), which yields a poor rotation/scale.
    # Also clip the predicted cloud to the per-axis 1-99 percentile range so
    # the model's sky/edge artefacts don't bias PCA.
    def _percentile_clip(p: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
        bounds = np.stack([np.percentile(p, lo, axis=0),
                           np.percentile(p, hi, axis=0)], axis=0)
        m = np.all((p >= bounds[0]) & (p <= bounds[1]), axis=1)
        return p[m]

    pred_align = _percentile_clip(pred)
    sub_p = pred_align if len(pred_align) <= align_subsample else pred_align[
        rng.choice(len(pred_align), size=align_subsample, replace=False)]
    sub_g = gt_vis if len(gt_vis) <= align_subsample else gt_vis[
        rng.choice(len(gt_vis), size=align_subsample, replace=False)]

    t1 = time.time()
    s, R, t, align_score = icp_with_scale(sub_p, sub_g, verbose=verbose)
    if verbose:
        print(f"alignment: s={s:.4g}  align_score(capped RMSE)={align_score:.4f} m"
              f"  ({time.time()-t1:.1f}s)")

    pred_m = apply_sim(pred, s, R, t)

    t2 = time.time()
    metrics = chamfer_metrics_simple(pred_m, gt_vis,
                                     voxel=voxel, max_dist=max_dist)
    if verbose:
        print(f"metrics: acc={metrics['accuracy']:.4f}m  "
              f"comp={metrics['completeness']:.4f}m  "
              f"overall={metrics['overall']:.4f}m  "
              f"F1@2cm={metrics.get('f1@0.02', float('nan')):.3f}  "
              f"({time.time()-t2:.1f}s)")

    return {
        "pred_npz": str(pred_npz),
        "scene_dir": str(scene_dir),
        "image_rel_paths": image_rel_paths,
        "conf_thresh": conf_thresh,
        "conf_field": conf_field,
        "alignment": {
            "scale": s,
            "rotation": R.tolist(),
            "translation": t.tolist(),
            "score_m": align_score,
        },
        "metrics": metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eth3d", action="store_true",
                    help="Use ETH3D evaluation mode (COLMAP SfM ground truth, "
                         "track-visibility filter, metres-scale defaults). "
                         "When set, --gt/--obsmask/--plane are not required; "
                         "the scene directory is read from the npz unless "
                         "--scene-dir overrides it.")
    ap.add_argument("--pred", required=True, type=Path,
                    help="Path to a saved .npz from the attack notebook "
                         "(e.g. output/clean_depth/clean_depth.npz).")
    ap.add_argument("--gt", type=Path,
                    help="DTU ground-truth point cloud .ply "
                         "(e.g. stl006_total.ply). Required without --eth3d.")
    ap.add_argument("--obsmask", type=Path,
                    help="DTU ObsMaskN_10.mat. Required without --eth3d.")
    ap.add_argument("--plane", type=Path,
                    help="DTU PlaneN.mat. Required without --eth3d.")
    ap.add_argument("--scene-dir", type=Path, default=None,
                    help="ETH3D scene directory (containing "
                         "dslr_calibration_undistorted/). Defaults to the path "
                         "stored in the npz at attack time.")
    ap.add_argument("--conf-thresh", type=float, default=1.5,
                    help="Confidence threshold (default 1.5; adv. runs drop "
                         "confidence dramatically, so this can be lowered).")
    ap.add_argument("--conf-field", default="depth_conf",
                    choices=("depth_conf", "world_pts_conf"),
                    help="Which confidence field to filter the predicted points "
                         "by. Use depth_conf for depth-attack outputs and "
                         "world_pts_conf for point-map-attack outputs.")
    ap.add_argument("--voxel", type=float, default=None,
                    help="Voxel-downsample size. Default 0.2 (mm) for DTU, "
                         "0.005 (m) for ETH3D.")
    ap.add_argument("--max-dist", type=float, default=None,
                    help="Per-point distance cap. Default 20 (mm) for DTU, "
                         "0.5 (m) for ETH3D.")
    ap.add_argument("--align-subsample", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON output path.")
    args = ap.parse_args()

    if args.eth3d:
        voxel = 0.005 if args.voxel is None else args.voxel
        max_dist = 0.5 if args.max_dist is None else args.max_dist
        result = evaluate_eth3d(
            args.pred, scene_dir=args.scene_dir,
            conf_thresh=args.conf_thresh,
            conf_field=args.conf_field,
            align_subsample=args.align_subsample,
            voxel=voxel,
            max_dist=max_dist,
            seed=args.seed,
        )
    else:
        missing = [n for n, v in [("--gt", args.gt), ("--obsmask", args.obsmask),
                                  ("--plane", args.plane)] if v is None]
        if missing:
            ap.error(f"{', '.join(missing)} are required without --eth3d.")
        voxel = 0.2 if args.voxel is None else args.voxel
        max_dist = 20.0 if args.max_dist is None else args.max_dist
        result = evaluate(
            args.pred, args.gt, args.obsmask, args.plane,
            conf_thresh=args.conf_thresh,
            conf_field=args.conf_field,
            align_subsample=args.align_subsample,
            voxel=voxel,
            max_dist=max_dist,
            seed=args.seed,
        )
    print(json.dumps(result["metrics"], indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
