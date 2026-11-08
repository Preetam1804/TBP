"""Focal-length estimation from vanishing points of orthogonal scene directions.

Forward-flight drone video cannot constrain the focal length through bundle
adjustment (see ``focal_search.py``), but man-made scenes contain orthogonal
families of parallel 3D lines (building edges, roads, garden paths). Two finite
vanishing points ``v1`` and ``v2`` of orthogonal directions satisfy, for a
camera with square pixels, zero skew and a known principal point ``c``::

    (v1 - c) . (v2 - c) = -f^2

so each orthogonal pair yields a focal length. Pairs are collected over several
frames and the focal is taken from the dominant mode; if no clear mode exists the
estimator reports ``None`` rather than guessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("uav3d.sfm.vp_focal")

_WORK_WIDTH = 960  # lines are detected on frames resized to this width


@dataclass
class VpFocalEstimate:
    """Focal length (in original-image pixels) estimated from vanishing points."""

    focal_px: float
    confidence: float  # share of total pair weight inside the winning mode (0..1)
    num_frames: int  # frames that contributed a pair inside the winning mode
    num_pairs: int  # pairs inside the winning mode
    spread: float = 0.0  # std of log(focal) inside the winning mode (relative scatter)
    all_candidates: list[float] = field(default_factory=list)

    def is_reliable(self, min_frames: int = 4, min_pairs: int = 4, max_spread: float = 0.08) -> bool:
        """Accept only a tight mode supported by several frames (guards against coincidental pairs)."""
        return self.num_frames >= min_frames and self.num_pairs >= min_pairs and self.spread <= max_spread

    def as_dict(self) -> dict:
        return {
            "focal_px": round(self.focal_px, 2),
            "confidence": round(self.confidence, 3),
            "num_frames": self.num_frames,
            "num_pairs": self.num_pairs,
            "spread": round(self.spread, 4),
            "num_candidates": len(self.all_candidates),
        }


def _detect_segments(gray: np.ndarray, min_len: float) -> np.ndarray:
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4))
    seg = lines.reshape(-1, 4).astype(np.float64)
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    return seg[length >= min_len]


def _residual_deg(mid: np.ndarray, direc: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle between each segment and the direction from its midpoint to vanishing point v."""
    w = v[None, :] - mid
    wn = np.linalg.norm(w, axis=1, keepdims=True)
    w = w / np.maximum(wn, 1e-9)
    cross = np.abs(direc[:, 0] * w[:, 1] - direc[:, 1] * w[:, 0])
    return np.degrees(np.arcsin(np.clip(cross, 0.0, 1.0)))


def _find_vanishing_points(
    seg: np.ndarray,
    max_vps: int,
    thresh_deg: float,
    rng: np.random.Generator,
    min_support_frac: float,
    far: float,
) -> list[tuple[np.ndarray, float]]:
    """Greedy RANSAC extraction of up to ``max_vps`` vanishing points (point, summed length support)."""
    if len(seg) < 8:
        return []
    mid = 0.5 * (seg[:, :2] + seg[:, 2:])
    vec = seg[:, 2:] - seg[:, :2]
    length = np.linalg.norm(vec, axis=1)
    direc = vec / length[:, None]
    homo = np.cross(np.c_[seg[:, :2], np.ones(len(seg))], np.c_[seg[:, 2:], np.ones(len(seg))])
    total_len = float(length.sum())

    remaining = np.arange(len(seg))
    found: list[tuple[np.ndarray, float]] = []
    for _ in range(max_vps):
        if len(remaining) < 6:
            break
        p = length[remaining] / length[remaining].sum()
        best_score, best_v = 0.0, None
        for _ in range(600):
            i, j = rng.choice(len(remaining), 2, replace=False, p=p)
            a, b = remaining[i], remaining[j]
            h = np.cross(homo[a], homo[b])
            if abs(h[2]) < 1e-9:
                continue
            v = h[:2] / h[2]
            if np.linalg.norm(v) > far:
                continue
            res = _residual_deg(mid[remaining], direc[remaining], v)
            score = float(length[remaining][res < thresh_deg].sum())
            if score > best_score:
                best_score, best_v = score, v
        if best_v is None or best_score < min_support_frac * total_len:
            break
        res = _residual_deg(mid[remaining], direc[remaining], best_v)
        inl = res < thresh_deg
        # refine by least squares on inlier lines (intersection of their infinite lines)
        L = homo[remaining][inl]
        L = L / np.linalg.norm(L[:, :2], axis=1, keepdims=True)
        W = length[remaining][inl][:, None]
        _, _, vt = np.linalg.svd(L * W)
        h = vt[-1]
        if abs(h[2]) > 1e-9 and np.linalg.norm(h[:2] / h[2]) < far:
            best_v = h[:2] / h[2]
            res = _residual_deg(mid[remaining], direc[remaining], best_v)
            inl = res < thresh_deg
        found.append((best_v, float(length[remaining][inl].sum())))
        remaining = remaining[~inl]
    return found


def estimate_focal_from_vanishing_points(
    image_paths: list[Path],
    max_frames: int = 12,
    thresh_deg: float = 1.5,
    min_support_frac: float = 0.06,
    seed: int = 0,
) -> VpFocalEstimate | None:
    """Estimate the focal length (px of the original image) from orthogonal vanishing-point pairs."""
    if not image_paths:
        return None
    idx = np.unique(np.linspace(0, len(image_paths) - 1, min(max_frames, len(image_paths))).round().astype(int))
    rng = np.random.default_rng(seed)

    pairs: list[tuple[float, float, int]] = []  # (focal_orig, weight, frame_index)
    for fi, k in enumerate(idx):
        img = cv2.imread(str(image_paths[k]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h0, w0 = img.shape
        scale = _WORK_WIDTH / max(w0, h0) if max(w0, h0) > _WORK_WIDTH else 1.0
        gray = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else img
        h, w = gray.shape
        diag = float(np.hypot(w, h))
        seg = _detect_segments(gray, min_len=0.04 * diag)
        c = np.array([w / 2.0, h / 2.0])
        vps = _find_vanishing_points(seg, 4, thresh_deg, rng, min_support_frac, far=30.0 * diag)
        for a in range(len(vps)):
            for b in range(a + 1, len(vps)):
                va, sa = vps[a]
                vb, sb = vps[b]
                f2 = -float(np.dot(va - c, vb - c))
                if f2 <= 0:
                    continue
                f = np.sqrt(f2) / scale
                pairs.append((f, min(sa, sb), int(fi)))

    if len(pairs) < 3:
        logger.info(f"VP focal: only {len(pairs)} orthogonal pair candidates; no estimate")
        return None

    f_arr = np.array([p[0] for p in pairs])
    w_arr = np.array([p[1] for p in pairs])
    w_arr = w_arr / w_arr.sum()
    logf = np.log(f_arr)
    # weighted mode: densest +/-12% window
    best_mass, best_center = 0.0, None
    for center in np.log(np.geomspace(0.3, 6.0, 240) * max(w0, h0)):
        mass = float(w_arr[np.abs(logf - center) < 0.12].sum())
        if mass > best_mass:
            best_mass, best_center = mass, center
    assert best_center is not None
    sel = np.abs(logf - best_center) < 0.12
    order = np.argsort(f_arr[sel])
    cw = np.cumsum(w_arr[sel][order])
    focal = float(f_arr[sel][order][np.searchsorted(cw, 0.5 * cw[-1])])
    frames = {pairs[i][2] for i in np.flatnonzero(sel)}
    return VpFocalEstimate(
        focal_px=focal,
        confidence=best_mass,
        num_frames=len(frames),
        num_pairs=int(sel.sum()),
        spread=float(np.std(logf[sel])),
        all_candidates=[float(x) for x in np.sort(f_arr)],
    )
