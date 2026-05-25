"""Sanity tests for evaluate_attack.py — purely synthetic, no I/O."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_attack import (
    umeyama, apply_sim, icp_with_scale, voxel_downsample, chamfer_metrics,
    ObsMaskInfo,
)


def random_sim(rng):
    # random rotation via QR
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    s = float(rng.uniform(0.3, 3.0))
    t = rng.normal(scale=10.0, size=3)
    return s, Q, t


def test_umeyama_closed_form():
    """Umeyama should recover an exact synthetic similarity."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(500, 3)) * np.array([1.0, 2.0, 0.5])
    s, R, t = random_sim(rng)
    dst = apply_sim(pts, s, R, t)
    s_h, R_h, t_h = umeyama(pts, dst)
    assert abs(s_h - s) < 1e-5, (s_h, s)
    assert np.allclose(R_h, R, atol=1e-5), R_h - R
    assert np.allclose(t_h, t, atol=1e-3), (t_h, t)
    print("[OK] umeyama recovers exact similarity")


def test_icp_recovers_known_similarity():
    """ICP-with-scale should recover a synthetic similarity even with
    50% gaussian noise dropouts and random subsampling."""
    rng = np.random.default_rng(1)
    # an asymmetric cloud so PCA has unambiguous axes
    n = 5000
    base = rng.normal(size=(n, 3)) * np.array([1.0, 3.0, 0.5])
    s_true, R_true, t_true = random_sim(rng)
    target = apply_sim(base, s_true, R_true, t_true)
    # Subsample target so it's not a 1-to-1 correspondence.
    target = target[rng.choice(n, size=n // 2, replace=False)]
    s, R, t, score = icp_with_scale(base, target, iters=40, verbose=False)
    # Test by composing: aligning pred -> target should give a low residual
    aligned = apply_sim(base, s, R, t)
    # mean NN distance from aligned to target should be ~small
    from scipy.spatial import cKDTree
    tree = cKDTree(target)
    d, _ = tree.query(aligned, k=1)
    mean_d = d.mean()
    extent = (target.max(0) - target.min(0)).max()
    rel = mean_d / extent
    assert rel < 0.02, f"ICP failed: rel error {rel:.4f}"
    print(f"[OK] icp_with_scale recovers similarity "
          f"(rel mean NN = {rel:.2%}, scale recovered {s:.3f} vs {s_true:.3f})")


def test_voxel_downsample_means():
    """voxel_downsample must return one mean-position point per cell."""
    pts = np.array([
        [0.1, 0.2, 0.3], [0.3, 0.4, 0.5],   # cell (0,0,0)
        [1.7, 2.1, 3.4],                    # cell (1,2,3)
        [1.6, 2.2, 3.3],                    # cell (1,2,3)
    ], dtype=np.float32)
    out = voxel_downsample(pts, voxel=1.0)
    assert len(out) == 2, out
    # sort for stable comparison
    out = out[np.lexsort(out.T)]
    expected = np.array([
        [0.2, 0.3, 0.4],          # mean of first two
        [1.65, 2.15, 3.35],       # mean of last two
    ], dtype=np.float32)
    expected = expected[np.lexsort(expected.T)]
    assert np.allclose(out, expected, atol=1e-5), (out, expected)
    print("[OK] voxel_downsample returns per-cell means")


def test_chamfer_identity_is_zero():
    """Pred == GT (after downsampling) ⇒ Acc = Comp = 0."""
    rng = np.random.default_rng(2)
    gt = rng.uniform(low=[-50, -50, 600], high=[50, 50, 700], size=(20_000, 3)).astype(np.float32)
    # ObsMask that fully contains the cloud
    bb_min = np.array([-100., -100., 500.])
    bb_max = np.array([100., 100., 800.])
    res = 1.0
    nx, ny, nz = ((bb_max - bb_min) / res).astype(int)
    mask = np.ones((nx, ny, nz), dtype=np.uint8)
    obs = ObsMaskInfo(mask=mask, bb_min=bb_min, bb_max=bb_max, res=res, margin=0)
    # plane below the cloud so everything passes
    plane = np.array([0.0, 0.0, 1.0, -400.0])
    out = chamfer_metrics(gt.copy(), gt, obs, plane, voxel=0.2, max_dist=20.0)
    assert out["accuracy_mm"] < 1e-3, out
    assert out["completeness_mm"] < 1e-3, out
    print(f"[OK] chamfer identity ≈ 0 (acc={out['accuracy_mm']:.2e}, "
          f"comp={out['completeness_mm']:.2e})")


def test_obsmask_filters_out_distant_points():
    """A predicted cloud entirely outside ObsMask should have 0 pred-eval pts."""
    rng = np.random.default_rng(3)
    pred = rng.uniform(low=[1000, 1000, 1000], high=[1100, 1100, 1100],
                       size=(2000, 3)).astype(np.float32)
    gt = rng.uniform(low=[-50, -50, 600], high=[50, 50, 700],
                     size=(2000, 3)).astype(np.float32)
    bb_min = np.array([-100., -100., 500.])
    bb_max = np.array([100., 100., 800.])
    res = 1.0
    nx, ny, nz = ((bb_max - bb_min) / res).astype(int)
    mask = np.ones((nx, ny, nz), dtype=np.uint8)
    obs = ObsMaskInfo(mask=mask, bb_min=bb_min, bb_max=bb_max, res=res, margin=0)
    plane = np.array([0.0, 0.0, 1.0, -400.0])
    out = chamfer_metrics(pred, gt, obs, plane, voxel=0.2, max_dist=20.0)
    assert out["n_pred_in_obs"] == 0, out
    print("[OK] obsmask filters out predicted points outside the volume")


if __name__ == "__main__":
    test_umeyama_closed_form()
    test_voxel_downsample_means()
    test_chamfer_identity_is_zero()
    test_obsmask_filters_out_distant_points()
    test_icp_recovers_known_similarity()
    print("\nAll sanity checks passed.")
