# Adversarial Attacks on VGGT

This fork adds three PGD attack notebooks targeting VGGT's three prediction heads (depth, point map, camera pose), and evaluation scripts that measure the damage on standard benchmark metrics.

## Notebooks

| Notebook | Target head | Dataset | Default scene |
|---|---|---|---|
| [depth_attack.ipynb](depth_attack.ipynb) | `depth` | DTU MVS Sample Set | `scan6` |
| [point_map_attack.ipynb](point_map_attack.ipynb) | `world_points` | ETH3D high-res multi-view | `courtyard` |
| [camera_attack.ipynb](camera_attack.ipynb) | `pose_enc` (T, quat, FoV) | CO3Dv2 single-sequence subset | `apple` |

All three share the same threat model:

- L_inf-bounded image perturbation with `eps = 4/255` (the "barely visible" budget from image-classifier robustness work).
- 40 PGD steps, sign step `1/255`, random init inside the L_inf ball.
- One delta shared across all views (the realistic "one physical perturbation fools every camera frame" threat).
- Model weights frozen; only the perturbation is optimised.

Dataset paths are hardcoded in one cell near the top of each notebook (see the *Data* markdown cell for which variable). Edit `DTU_SCAN` / `ETH3D_SCENE` / `CO3D_DIR` to point at your local copy, then run-all.

## Per-attack losses

### Depth ([depth_attack.ipynb](depth_attack.ipynb))

Maximises a scale-invariant log-depth divergence weighted by the clean depth confidence:

```
L = E[w * d^2] - lambda * (E[w * d])^2
    where d = log(D_adv) - log(D_clean), w = detached clean depth_conf, lambda = 0.5
```

Why this form and not the paper's plain L1: maximising `|D_adv - D_clean|` lets PGD find a uniform multiplicative depth shift in a few steps, but the evaluator's Umeyama / ICP-with-scale alignment absorbs that shift and the Chamfer barely moves. SI-log subtracts off the global mean by construction, so PGD has to distort relative geometry. Detaching the confidence weight closes the "lower my own confidence on damaged pixels and let the filter cull them" escape hatch.

### Point map ([point_map_attack.ipynb](point_map_attack.ipynb))

Maximises a confidence-weighted L2 residual on `world_points`:

```
L = E_i [ Sigma_i^clean * || P_adv_i - P_clean_i ||_2 ]
```

This is the VGGT paper's `L_pmap` regression term with the ground truth replaced by the clean prediction. Sigma is the clean predicted precision, detached so the attacker cannot win by lowering its own confidence. The per-pixel gradient points each predicted point away from its clean value in 3D.

### Camera pose ([camera_attack.ipynb](camera_attack.ipynb))

Maximises the paper's Huber loss on the 9-D pose encoding, in two passes with different references:

```
Pass A (vs GT):    L = Huber(T_adv, T_gt)    + Huber(q_adv, q_gt)
Pass B (vs clean): L = Huber(T_adv, T_clean) + Huber(q_adv, q_clean) + 0.5 * Huber(fov_adv, fov_clean)
```

Pass A matches `training/loss.py::compute_camera_loss` and is the most direct attack on the AUC@30 metric. Pass B mirrors the "no GT needed" recipe of the depth and point-map attacks. Pass A drops the FoV term because the GT FoV after VGGT's resize would need a pixel-space conversion that AUC@30 does not depend on; Pass B keeps it at the training default of 0.5, since both sides are predictions on the same input.

## Metrics

Each evaluation script reads the `.npz` files written by the notebooks and reports the standard benchmark metric for that head.

### Depth: DTU Chamfer (`evaluate_attack.py`)

Acc / Comp / Overall Chamfer distance in millimetres, following the standard DTU pipeline:

1. Filter predicted points by `depth_conf >= conf_thresh` (default 1.5; lower it for harder attacks where confidence has collapsed).
2. Voxel downsample (default 0.2 mm).
3. Multi-start ICP-with-scale (Umeyama) to align pred to GT, with a per-start scale-drift clamp around the PCA-derived initial scale. The clamp stops an adversarial cloud from dragging the alignment into a degenerate `s -> 0` solution that maps every pred point onto the GT centroid.
4. Apply DTU's `ObsMaskN_10.mat` (visibility) and `PlaneN.mat` (background plane) to the GT.
5. Per-point distances capped at `max-dist` (default 20 mm).

`Overall = (Acc + Comp) / 2` is the DTU summary metric.

```bash
python evaluate_attack.py \
    --pred    output/adv_depth/adv_depth.npz \
    --gt      path/to/Points/stl/stl006_total.ply \
    --obsmask path/to/ObsMask/ObsMask6_10.mat \
    --plane   path/to/ObsMask/Plane6.mat
```

**Source.** Metric form is the DTU benchmark Acc / Comp / Overall reported by VGGT (and by every learning-based MVS paper since MVSNet). VGGT's author points to [GeoMVSNet](https://github.com/doubleZ0108/GeoMVSNet) as the MVS evaluation implementation VGGT itself used; GeoMVSNet runs the [official DTU MATLAB pipeline](https://github.com/doubleZ0108/GeoMVSNet/blob/main/scripts/dtu/matlab_quan_dtu.sh) (`ObsMaskN_10.mat` + `PlaneN.mat`), which is what this Python evaluator reproduces. The grid-correspondence + Umeyama alignment recipe is the one the VGGT author describes in [facebookresearch/vggt#45](https://github.com/facebookresearch/vggt/issues/45): predicted and GT depth share the same pixel grid, so per-pixel correspondences are free and a single Umeyama solve aligns the two clouds. The scale-drift clamp on top of Umeyama is specific to this repo: vanilla Umeyama can collapse to `s -> 0` on an adversarially corrupted prediction, which would give a misleadingly small RMSE.

### Point map: ETH3D Chamfer + F-score (`evaluate_attack.py --eth3d`)

Acc / Comp / Overall Chamfer in metres, plus F-score at 1, 2, and 5 cm (ETH3D's published headline metric is the F-score at tau):

1. Filter predicted points by `world_pts_conf >= conf_thresh`.
2. Voxel downsample (default 5 mm).
3. Load GT from the scene's COLMAP `points3D.txt` (full SfM cloud, no view-visibility filter).
4. Same Umeyama-with-scale-clamp alignment as the depth metric.

```bash
python evaluate_attack.py --eth3d \
    --pred       output/adv_pointmap/adv_pointmap.npz \
    --conf-field world_pts_conf
```

The ETH3D `--scene-dir` is read from the npz if not supplied.

**Source.** Metric form is the published ETH3D F-score @ τ (the dataset's headline metric on its public benchmark). The predicted point cloud is scored against the full SfM ground-truth cloud from the scene's `points3D.txt`, with no view-visibility filter, so absolute numbers are comparable to published ETH3D scores. Alignment follows the same grid-correspondence + Umeyama recipe as the depth metric: as the VGGT author notes in [facebookresearch/vggt#45](https://github.com/facebookresearch/vggt/issues/45), the same evaluation routine works for both heads since `world_points` is what you get by unprojecting depth with the camera parameters. Both the predicted and GT clouds are percentile-clipped before the alignment subsample so that ETH3D's SfM far-outliers (sky / distant facades) do not bias the PCA-seeded Umeyama; the final Chamfer / F-score is computed against the unclipped full GT.

### Camera pose: AUC@tau (`evaluate_camera_attack.py`)

AUC@30 / @15 / @5 over all unordered pairs of relative poses (PoseDiffusion convention, also reported by VGGT's evaluation branch). Per-pair error is `max(rotation_deg, translation_deg)`:

- Rotation: geodesic angle via `pytorch3d.so3_relative_angle` (eps=1e-4 for numerical stability).
- Translation: angular error between unit-normalised translation directions, folded into `[0, 90]` via `|cos|`. Camera-pose papers treat `t` and `-t` as equivalent because the metric absorbs scene scale.
- `AUC@tau` is the mean of the cumulative accuracy curve from 0 to `tau` degrees.

Higher AUC is better; the attack's job is to lower it.

```bash
python evaluate_camera_attack.py \
    output/camera_attack/clean.npz \
    output/camera_attack/adv_A.npz \
    output/camera_attack/adv_B.npz
```

**Source.** The evaluation routine is a trimmed-down rewrite of VGGT's [`evaluation` branch](https://github.com/facebookresearch/vggt/tree/evaluation), specifically [`evaluation/test_co3d.py`](https://github.com/facebookresearch/vggt/blob/evaluation/evaluation/test_co3d.py) (which defines `convert_pt3d_RT_to_opencv`, `build_pair_index`, `rotation_angle`, `translation_angle`, `calculate_auc_np`, `se3_to_relative_pose_error`). The PyTorch3D → OpenCV ground-truth conversion in `data_utils/co3d_loader.py::convert_pt3d_RT_to_opencv` is copied verbatim from that file. The "all `N*(N-1)/2` unordered pairs" convention is the one the VGGT author confirms in [facebookresearch/vggt#45](https://github.com/facebookresearch/vggt/issues/45) ("batched_all_pairs will generate N*(N-1) pairs ... follows the definition of RRE, RPE, and AUC"); the same protocol is used by PoseDiffusion and VGGSfM, which the maintainer points to as reference implementations for camera-pose evaluation.

## Output layout

Each notebook writes:

- `output/<attack>/clean*.npz` and `output/<attack>/adv*.npz`: predictions, the delta tensor, the adversarial images, and the loss history.
- `output/<attack>_metadata.json` (or `output/camera_attack/metadata.json`): seed, hyperparameters, summary stats, and environment (Python / Torch / CUDA / cuDNN / GPU).
- `images/<attack>/`: PNGs of the adversarial views for visual inspection.

The clean and adversarial outputs both include extrinsics and intrinsics decoded from `pose_enc`, so the evaluation scripts never have to touch the pose encoding directly.

## Reproducibility notes

- All notebooks set `seed = 0` and `torch.backends.cudnn.deterministic = True`. cuDNN determinism is hardware-conditional, so the GPU and CUDA / cuDNN versions are recorded in the metadata file for each run.
- The list of sampled image files is recorded in metadata as `selected_images`. To re-run on the exact same images without re-shuffling, paste those filenames into the load cell directly.
- The attack runs in fp32. VGGT's prediction heads already force fp32 internally; the aggregator runs in whatever precision the model is loaded with.
