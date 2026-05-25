"""Pick one CO3Dv2 sequence + GT cameras for the camera-pose attack.

We bypass CO3D's own dataset class and read ``frame_annotations.jgz`` /
``sequence_annotations.jgz`` directly. The PyTorch3D-NDC -> OpenCV
conversion is copied verbatim from VGGT's evaluation branch
(``evaluation/test_co3d.py::convert_pt3d_RT_to_opencv``) so the GT
extrinsics line up with what the AUC@30 metric there computes.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np


def convert_pt3d_RT_to_opencv(R, T) -> np.ndarray:
    """PyTorch3D world-to-camera (R, T) -> OpenCV [R|t] 3x4.

    Copied from ``vggt/evaluation`` branch:
        T[:2] *= -1; R[:, :2] *= -1; R = R.T
    """
    R = np.asarray(R, dtype=np.float64).copy()
    T = np.asarray(T, dtype=np.float64).copy()
    T[:2] *= -1
    R[:, :2] *= -1
    R = R.T
    return np.hstack((R, T[:, None]))


@dataclass
class Co3dSequence:
    category: str
    sequence_name: str
    co3d_dir: Path
    image_paths: List[str]           # absolute paths, length S
    frame_numbers: List[int]         # length S
    extrinsics: np.ndarray           # (S, 3, 4) OpenCV w2c [R|t]
    image_hw: List[List[int]]        # per-frame [H, W] (annotation order)
    viewpoint_quality_score: float


def load_co3d_sequence(
    co3d_dir: Union[str, Path],
    category: str = "apple",
    num_frames: int = 10,
    min_quality: float = 0.5,
    prefer_landscape: bool = True,
) -> Co3dSequence:
    """Pick one sequence and return ``num_frames`` evenly-spaced GT frames.

    Sequences are filtered by ``viewpoint_quality_score >= min_quality`` (the
    Dust3R / VGGT-eval threshold). When ``prefer_landscape`` is set we skip
    sequences whose images are taller than they are wide, because VGGT's
    default ``load_and_preprocess_images`` center-crops portrait inputs to
    518x518 and loses content from top/bottom.
    """
    co3d_dir = Path(co3d_dir)
    cat_dir = co3d_dir / category

    with gzip.open(cat_dir / "frame_annotations.jgz") as f:
        frames = json.loads(f.read())
    with gzip.open(cat_dir / "sequence_annotations.jgz") as f:
        sequences = json.loads(f.read())

    quality = {s["sequence_name"]: s.get("viewpoint_quality_score", 0.0) for s in sequences}
    good_seqs = sorted(
        s["sequence_name"] for s in sequences
        if s.get("viewpoint_quality_score", 0.0) >= min_quality
    )
    if not good_seqs:
        raise RuntimeError(
            f"No sequences with viewpoint_quality_score >= {min_quality} in {cat_dir}"
        )

    by_seq: dict[str, list] = {}
    for fr in frames:
        by_seq.setdefault(fr["sequence_name"], []).append(fr)

    chosen = None
    for s in good_seqs:
        if s not in by_seq:
            continue
        H, W = by_seq[s][0]["image"]["size"]  # annotation is [H, W]
        if prefer_landscape and W < H:
            continue
        chosen = s
        break
    if chosen is None:
        chosen = good_seqs[0]

    seq_frames = sorted(by_seq[chosen], key=lambda x: x["frame_number"])
    if len(seq_frames) < num_frames:
        raise RuntimeError(
            f"Sequence {chosen} has only {len(seq_frames)} frames < {num_frames}"
        )

    idxs = np.linspace(0, len(seq_frames) - 1, num_frames).round().astype(int).tolist()
    selected = [seq_frames[i] for i in idxs]

    image_paths = [str(co3d_dir / fr["image"]["path"]) for fr in selected]
    frame_numbers = [fr["frame_number"] for fr in selected]
    extrinsics = np.stack(
        [convert_pt3d_RT_to_opencv(fr["viewpoint"]["R"], fr["viewpoint"]["T"])
         for fr in selected],
        axis=0,
    ).astype(np.float32)
    image_hw = [list(fr["image"]["size"]) for fr in selected]

    return Co3dSequence(
        category=category,
        sequence_name=chosen,
        co3d_dir=co3d_dir,
        image_paths=image_paths,
        frame_numbers=frame_numbers,
        extrinsics=extrinsics,
        image_hw=image_hw,
        viewpoint_quality_score=float(quality[chosen]),
    )
