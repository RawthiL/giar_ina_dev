"""
Perceptual (LPIPS) matching of legacy 200x200 manual crops to cell boxes in the full FOV image.

The legacy crops were cut from the full images as bbox crops squished to a square. A candidate
box is therefore rendered the same ways a legacy crop may have been made (squished bbox, and the
aspect-preserving square window) and compared with LPIPS; the smaller distance wins. No pixel is
ever synthesised outside the image: a candidate whose square window leaves the image only has its
squished view.

Used by ``rescan_legacy.py`` (which writes ``legacy_matches.json``) and, for the shared border
rule, by ``crop_dataset.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

LEGACY_SIZE = 200  # edge of the legacy crops


# --------------------------------------------------------------------------- geometry


def square_window(bbox: Sequence[float], margin: float = 0.0) -> tuple[int, int, int, int]:
    """Square (left, top, right, bottom) centred on the bbox, side max(w, h)*(1+margin)."""
    x, y, w, h = bbox
    side = max(2, int(round(max(w, h) * (1.0 + margin))))
    cx, cy = x + w / 2.0, y + h / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    return left, top, left + side, top + side


def window_inside(window: tuple[int, int, int, int], width: int, height: int) -> bool:
    """True when the whole window lies within a ``width`` x ``height`` image."""
    left, top, right, bottom = window
    return left >= 0 and top >= 0 and right <= width and bottom <= height


def is_border_cell(bbox: Sequence[float], width: int, height: int, margin: float = 0.0) -> bool:
    """A cell is a border cell when its square crop window is not fully inside the image."""
    return not window_inside(square_window(bbox, margin), width, height)


def iou_xywh(a: Sequence[float], b: np.ndarray) -> np.ndarray:
    """IoU of one box ``a`` (x, y, w, h) against an (N, 4) array of boxes."""
    if len(b) == 0:
        return np.zeros(0)
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    union = a[2] * a[3] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


# --------------------------------------------------------------------------- views


def _squish(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Image.Image | None:
    height, width = arr.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return Image.fromarray(arr[y0:y1, x0:x1]).resize((LEGACY_SIZE, LEGACY_SIZE), Image.LANCZOS)


def candidate_views(
    arr: np.ndarray,
    bbox: Sequence[float],
    margin: float = 0.0,
    scales: Sequence[float] = (1.0, 1.3),
) -> list[Image.Image]:
    """Legacy-style renderings of ``bbox`` from the full image ``arr`` (H, W, 3), 200x200.

    Per scale (legacy crops were often framed looser than the annotation box): the bbox grown
    about its centre and squished to a square (clipped to the image, never padded). Scale 1 also
    gets the aspect-preserving square window when it lies fully inside the image.
    """
    height, width = arr.shape[:2]
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    views: list[Image.Image] = []
    for sc in scales:
        v = _squish(
            arr,
            int(cx - w * sc / 2.0),
            int(cy - h * sc / 2.0),
            int(cx + w * sc / 2.0),
            int(cy + h * sc / 2.0),
        )
        if v is not None:
            views.append(v)
    win = square_window(bbox, margin)
    if window_inside(win, width, height):
        left, top, right, bottom = win
        views.append(_squish(arr, left, top, right, bottom))  # type: ignore[arg-type]
    return views


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> grayscale replicated to (3, H, W) float in [-1, 1].

    Many legacy crops are grayscale while the FOV is colour; comparing in luminance keeps the
    colour gap from dominating the perceptual distance.
    """
    t = torch.from_numpy(np.asarray(img.convert("L").convert("RGB"), dtype=np.float32).copy())
    return t.permute(2, 0, 1) / 127.5 - 1.0


# --------------------------------------------------------------------------- LPIPS


@dataclass
class LpipsScorer:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    chunk: int = 256

    def __post_init__(self) -> None:
        import lpips  # imported lazily: heavy, and only matching needs it

        self.net = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()

    @torch.no_grad()
    def matrix(self, legacy: list[Image.Image], cands: list[list[Image.Image]]) -> np.ndarray:
        """(n_legacy, n_cands) min-over-views LPIPS distance (inf for candidates with no view)."""
        n, m = len(legacy), len(cands)
        out = np.full((n, m), np.inf, dtype=np.float32)
        if n == 0 or m == 0:
            return out
        flat = [(j, v) for j, vs in enumerate(cands) for v in vs]
        if not flat:
            return out
        cidx = np.array([j for j, _ in flat])
        ctens = torch.stack([to_tensor(v) for _, v in flat])
        ltens = torch.stack([to_tensor(v) for v in legacy])
        for i in range(n):
            dists = []
            for s in range(0, len(flat), self.chunk):
                c = ctens[s : s + self.chunk].to(self.device)
                ref = ltens[i : i + 1].to(self.device).expand(len(c), -1, -1, -1)
                dists.append(self.net(ref, c).flatten().cpu())
            d = torch.cat(dists).numpy()
            np.minimum.at(out[i], cidx, d)
        return out


def mutual_best(
    dist: np.ndarray,
    threshold: float,
    max_ratio: float = 1.0,
    boxes: np.ndarray | None = None,
    overlap_iou: float = 0.3,
) -> list[tuple[int, int, float, float]]:
    """(legacy_idx, cand_idx, distance, ratio) for confident mutual-best pairs.

    A pair is accepted when the candidate is the legacy crop's closest one and vice versa, the
    distance is <= ``threshold`` and the best distance is <= ``max_ratio`` x the runner-up's
    (the match must be distinctive among the image's candidates). With ``boxes`` (N, 4) xywh the
    runner-up ignores candidates overlapping the best one (IoU >= ``overlap_iou``): duplicate
    boxes around the same cell must not make a correct match look ambiguous.
    """
    if dist.size == 0:
        return []
    best_c = dist.argmin(axis=1)
    best_l = dist.argmin(axis=0)
    pairs = []
    for i, j in enumerate(best_c):
        d = float(dist[i, j])
        if best_l[j] != i or d > threshold:
            continue
        keep = np.ones(dist.shape[1], dtype=bool)
        keep[j] = False
        if boxes is not None:
            keep &= iou_xywh(boxes[j], boxes) < overlap_iou
        row = dist[i][keep]
        second = float(row.min()) if row.size else np.inf
        ratio = d / second if np.isfinite(second) and second > 0 else 0.0
        if ratio <= max_ratio:
            pairs.append((i, int(j), d, ratio))
    return pairs
