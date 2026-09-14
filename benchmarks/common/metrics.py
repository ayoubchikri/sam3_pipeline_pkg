"""
metrics.py — segmentation metrics shared by all benchmark methods.

  compute_iou(pred, gt)   -> float
  compute_dice(pred, gt)  -> float
  compute_bf1(pred, gt)   -> float   (boundary F1, tolerance tau px)
  compute_all(pred, gt)   -> {"iou":..., "dice":..., "bf1":...}

pred, gt : 2D binary numpy masks (bool or 0/1).
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def compute_iou(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return 1.0 if union == 0 else float(inter / union)


def compute_dice(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2.0 * inter / denom)


def compute_bf1(pred, gt, tolerance=2):
    """Boundary F1: contours (mask XOR erosion) dilated by `tolerance` px, then F1 on boundary pixels."""
    pred = pred.astype(bool); gt = gt.astype(bool)
    struct = np.ones((3, 3), dtype=bool)

    def boundary(m):
        return m & ~binary_erosion(m)

    b_pred = boundary(pred)
    b_gt   = boundary(gt)
    b_pred_d = binary_dilation(b_pred, structure=struct, iterations=tolerance)
    b_gt_d   = binary_dilation(b_gt,   structure=struct, iterations=tolerance)

    tp_p = (b_pred & b_gt_d).sum()
    tp_r = (b_gt   & b_pred_d).sum()
    precision = float(tp_p / b_pred.sum()) if b_pred.sum() > 0 else 1.0
    recall    = float(tp_r / b_gt.sum())   if b_gt.sum()   > 0 else 1.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_all(pred, gt, bf1_tolerance=2):
    return {
        "iou":  compute_iou(pred, gt),
        "dice": compute_dice(pred, gt),
        "bf1":  compute_bf1(pred, gt, tolerance=bf1_tolerance),
    }
