"""
AUC@30 evaluator for VGGT camera-pose adversarial attack outputs.

Consumes the npz files produced by `camera_attack.ipynb` and reports the
standard CO3Dv2 camera-pose-evaluation numbers (AUC@30 / @15 / @5 over all
pairs of relative poses; PoseDiffusion / VGGT evaluation-branch
convention). Run on any combination of:

    python evaluate_camera_attack.py output/camera_attack/clean.npz
    python evaluate_camera_attack.py output/camera_attack/clean.npz output/camera_attack/adv_A.npz output/camera_attack/adv_B.npz

Each input file must carry:
  * `extrinsics`     (1, S, 3, 4)  -- predicted OpenCV w2c
  * `gt_extrinsics`  (1, S, 3, 4)  -- ground-truth OpenCV w2c

This mirrors how `evaluate_attack.py` consumes the depth / point-map npz files
but uses camera-only quantities.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from pytorch3d.transforms import so3_relative_angle
from vggt.utils.geometry import closed_form_inverse_se3


def _se3(extr: torch.Tensor) -> torch.Tensor:
    """(*, 3, 4) -> (*, 4, 4)."""
    *batch, _, _ = extr.shape
    bottom = torch.zeros((*batch, 1, 4), device=extr.device, dtype=extr.dtype)
    bottom[..., 0, 3] = 1.0
    return torch.cat([extr, bottom], dim=-2)


def _pair_index(N: int):
    return torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)


def relative_pose_errors(pred_se3: torch.Tensor, gt_se3: torch.Tensor):
    """All-pairs (i<j) relative-pose rotation/translation errors in degrees.

    `pred_se3`, `gt_se3` are (N, 4, 4) world-to-camera SE(3). Returns
    (rel_R_deg, rel_t_deg), each shape (N*(N-1)/2,).
    """
    N = pred_se3.shape[0]
    i1, i2 = _pair_index(N)
    rel_gt   = gt_se3[i1]   @ closed_form_inverse_se3(gt_se3[i2])
    rel_pred = pred_se3[i1] @ closed_form_inverse_se3(pred_se3[i2])

    rel_R_deg = so3_relative_angle(rel_pred[:, :3, :3], rel_gt[:, :3, :3], eps=1e-4) * 180.0 / np.pi

    eps = 1e-15
    a = rel_pred[:, :3, 3]
    b = rel_gt[:, :3, 3]
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    cos_abs = (a * b).sum(-1).abs().clamp(0.0, 1.0)
    rel_t_deg = torch.acos(cos_abs) * 180.0 / np.pi
    rel_t_deg = torch.nan_to_num(rel_t_deg, nan=1e6, posinf=1e6, neginf=1e6)
    return rel_R_deg, rel_t_deg


def auc(rel_R_deg: torch.Tensor, rel_t_deg: torch.Tensor, thresh: int) -> float:
    err = torch.maximum(rel_R_deg, rel_t_deg)
    hist = torch.histc(err.float(), bins=thresh + 1, min=0, max=thresh)
    return (hist.cumsum(0) / err.numel()).mean().item()


def load_extrinsics(npz_path: Path):
    with np.load(npz_path, allow_pickle=True) as d:
        extr = torch.from_numpy(d["extrinsics"]).double().squeeze(0)
        gt   = torch.from_numpy(d["gt_extrinsics"]).double().squeeze(0)
    return extr, gt


def evaluate_one(npz_path: Path) -> dict:
    extr, gt = load_extrinsics(npz_path)
    pred_se3 = _se3(extr)
    gt_se3   = _se3(gt)
    rel_R, rel_t = relative_pose_errors(pred_se3, gt_se3)

    # The combined per-pair error used by the metric
    err = torch.maximum(rel_R, rel_t)
    # Per-degree histogram of max errors up to 30 deg, for diagnostics.
    hist = torch.histc(err.float(), bins=30, min=0, max=30).long().tolist()
    return {
        "path": str(npz_path),
        "num_frames": int(extr.shape[0]),
        "num_pairs": int(err.numel()),
        "AUC@30": auc(rel_R, rel_t, 30),
        "AUC@15": auc(rel_R, rel_t, 15),
        "AUC@5":  auc(rel_R, rel_t, 5),
        "rel_R_deg_mean": float(rel_R.mean()),
        "rel_R_deg_max":  float(rel_R.max()),
        "rel_t_deg_mean": float(rel_t.mean()),
        "rel_t_deg_max":  float(rel_t.max()),
        "max_err_hist_0_30_per_deg": hist,
    }


def format_row(label: str, r: dict) -> str:
    return (
        f"{label:<12s}  "
        f"AUC@30={r['AUC@30']:.4f}  AUC@15={r['AUC@15']:.4f}  AUC@5={r['AUC@5']:.4f}  "
        f"<R>={r['rel_R_deg_mean']:6.2f} deg  <t>={r['rel_t_deg_mean']:6.2f} deg"
    )


def main(argv: Sequence[str]):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", nargs="+", type=Path,
                   help="One or more .npz files from camera_attack.ipynb")
    p.add_argument("--json", action="store_true",
                   help="Dump full results dict as JSON (instead of the text table)")
    args = p.parse_args(argv)

    results: List[dict] = []
    for path in args.npz:
        if not path.exists():
            raise FileNotFoundError(path)
        results.append(evaluate_one(path))

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("=" * 80)
    for r in results:
        label = Path(r["path"]).stem  # clean / adv_A / adv_B
        print(format_row(label, r))
    print("-" * 80)
    print(f"{'frames':<12s}  {results[0]['num_frames']}  ({results[0]['num_pairs']} pairs)")
    # Side-by-side max-error histogram (0..29 deg bins, last bin includes >=30)
    print("\nper-degree max-error histogram (each row: # pairs with max err in [d, d+1) deg):")
    print(f"{'deg':>4s}  " + "  ".join(f"{Path(r['path']).stem:>8s}" for r in results))
    for d in range(30):
        cells = "  ".join(f"{r['max_err_hist_0_30_per_deg'][d]:>8d}" for r in results)
        print(f"{d:>4d}  " + cells)


if __name__ == "__main__":
    main(sys.argv[1:])
