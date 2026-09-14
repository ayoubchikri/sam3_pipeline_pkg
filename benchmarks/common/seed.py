"""
seed.py — shared "dark blob point inside the well" seed, used across methods.

Every prompt-based method (SAM2, µSAM, SAM3) is seeded with the SAME point: the
darkest blob inside the detected well. The automatic methods (Cellpose-SAM, Fiji)
use this same point to *select* which output object is "the organoid".

Only needs numpy + opencv, so it imports cleanly in every method's conda env.
"""

import numpy as np
import cv2


def detect_well(frame_gray):
    """Detect the well disc (cx, cy, r) by Hough circle; fall back to a centred disc."""
    h, w = frame_gray.shape
    blurred = cv2.GaussianBlur(frame_gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=min(h, w) // 2, param1=80, param2=30,
                               minRadius=int(min(h, w) * 0.3),
                               maxRadius=int(min(h, w) * 0.6))
    if circles is not None:
        cx, cy, r = np.round(circles[0][0]).astype(int)
        return int(cx), int(cy), int(r)
    return w // 2, h // 2, int(min(h, w) * 0.4)


def well_mask(frame_gray, margin=1.0):
    """Boolean disc mask of the well interior (optionally shrunk by `margin`)."""
    h, w = frame_gray.shape
    cx, cy, r = detect_well(frame_gray)
    r_in = int(r * margin)
    Y, X = np.ogrid[:h, :w]
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r_in ** 2


def dark_point_in_well(frame_gray, margin=0.98):
    """
    Return [x, y] of the darkest blob's centroid *inside the well* (never outside).

    Same logic as the production _dark_centroid_in_mask, but the search region is
    the well disc rather than an AMG mask — so no AMG is needed and the point is
    guaranteed to stay in the well.
    """
    gray = frame_gray
    disc = well_mask(gray, margin=margin).astype(np.float32)

    local_bg = cv2.GaussianBlur(gray.astype(np.float32), (71, 71), 0)
    dark = np.clip(local_bg - gray.astype(np.float32), 0, None) * disc
    dark_smooth = cv2.GaussianBlur(dark, (31, 31), 0)
    _, peak_val, _, peak_loc = cv2.minMaxLoc(dark_smooth)
    if peak_val > 0:
        blob = (dark_smooth > 0.35 * peak_val).astype(np.uint8)
        _, labels, _, centroids = cv2.connectedComponentsWithStats(blob)
        label = labels[peak_loc[1], peak_loc[0]]
        if label > 0:
            return [int(round(centroids[label][0])), int(round(centroids[label][1]))]
    # fallback: centroid of the well disc
    ys, xs = np.where(disc > 0)
    return [int(xs.mean()), int(ys.mean())]
