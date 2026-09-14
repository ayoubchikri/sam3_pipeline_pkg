"""
pick.py — reduce a multi-object segmentation to the single organoid.

Automatic methods (Cellpose-SAM, Fiji/threshold) output many objects. We pick the
one at the shared dark-blob point; if nothing sits exactly on the point, we fall
back to the nearest object, then to the largest.
"""

import numpy as np
import cv2


def _labels_from_masks(masks, shape):
    """Accept either a label image (H×W int) or a list/stack of boolean masks -> label image."""
    if isinstance(masks, np.ndarray) and masks.ndim == 2 and masks.dtype != bool:
        return masks.astype(np.int32)
    lab = np.zeros(shape, dtype=np.int32)
    for i, m in enumerate(masks, start=1):
        lab[np.asarray(m, dtype=bool)] = i
    return lab


def pick_instance_at_point(masks, point, shape=None):
    """
    masks : label image (H×W ints, 0=bg) OR iterable of boolean masks.
    point : [x, y].
    Returns a boolean H×W mask of the selected object (empty mask if none).
    """
    if shape is None:
        shape = (masks.shape if isinstance(masks, np.ndarray) and masks.ndim == 2
                 else np.asarray(next(iter(masks))).shape)
    lab = _labels_from_masks(masks, shape)
    x, y = int(point[0]), int(point[1])
    h, w = lab.shape

    # 1) object exactly under the point
    if 0 <= y < h and 0 <= x < w and lab[y, x] > 0:
        return lab == lab[y, x]

    ids = [i for i in np.unique(lab) if i != 0]
    if not ids:
        return np.zeros(shape, dtype=bool)

    # 2) nearest object centroid to the point
    best_id, best_d = None, None
    for i in ids:
        ys, xs = np.where(lab == i)
        d = (xs.mean() - x) ** 2 + (ys.mean() - y) ** 2
        if best_d is None or d < best_d:
            best_id, best_d = i, d
    return lab == best_id


def largest_component(mask):
    """Largest connected component of a boolean mask (empty if none)."""
    m = np.asarray(mask, dtype=np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros(mask.shape, dtype=bool)
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return labels == best


def fill_holes(mask):
    """Fill interior holes via external-contour fill."""
    m = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)
