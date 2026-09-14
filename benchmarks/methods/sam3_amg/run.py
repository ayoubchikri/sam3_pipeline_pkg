"""
SAM3 AMG — STRICTEMENT identique à sam2_amg (mêmes seuils AMG, AMG dans la bbox du
puits, sélection par la MÊME score_mask : moyenne pondérée aire / predicted_iou /
contraste au bord, meilleur masque > 0.35). Seule différence : le générateur est
le pipeline `mask-generation` de SAM3 au lieu de SAM2AutomaticMaskGenerator.

  python run.py --input frame.npy --output mask.npy   (via common.batch)

Env: bench312 (GPU) + transformers (Sam3) + scikit-image + scipy + cv2. HF_HOME configuré.
"""

import sys, argparse
from pathlib import Path
import numpy as np
import cv2
from skimage.measure import regionprops
from scipy.ndimage import binary_erosion, binary_dilation

BENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH_ROOT))
from common.seed import detect_well, well_mask

# --- IDENTIQUE à sam2_amg ---
AVG_ORGANOID_AREA_PX = 4384.58
AREA_SIGMA_PX = 2440.98
SCORE_MIN = 0.35
# paramètres AMG identiques à sam2_amg (points_per_side=24, pred_iou=0.7, stability=0.8)
POINTS_PER_SIDE = 24
PRED_IOU_THRESH = 0.7
STABILITY_THRESH = 0.8
POINTS_PER_BATCH = 64
# garde-fous d'aire : un organoïde plausible est dans [200, 20000] px (plus gros GT = 8596)
MIN_AREA_PX = 200
MAX_AREA_PX = 20000
# fallback : si 0 masque exploitable dans le puits ET dans [200,20000], on rebaisse
# stability_score_thresh de 0.05 et on régénère, jusqu'au plancher STAB_MIN.
STAB_START = STABILITY_THRESH
STAB_STEP = 0.05
STAB_MIN = 0.35


def score_mask(mask, image_gray, predicted_iou):
    """Copie fidèle de score_mask() de sam2_amg : moyenne pondérée area/iou/edge."""
    props_list = regionprops(mask.astype(int))
    if not props_list:
        return 0.0
    A = props_list[0].area
    area_score = float(np.exp(-((A - AVG_ORGANOID_AREA_PX) ** 2) / (2 * AREA_SIGMA_PX ** 2)))
    iou_score = float(np.clip(predicted_iou, 0, 1))

    gray_f = image_gray.astype(np.float32)
    sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    inner = mask & ~binary_erosion(mask, iterations=3)
    outer = binary_dilation(mask, iterations=3) & ~mask
    boundary = inner | outer
    edge_score = float(np.clip(np.mean(grad[boundary]) / 255, 0, 1)) if boundary.any() else 0.0

    return 0.10 * area_score + 0.70 * iou_score + 0.20 * edge_score


def find_best_amg_mask(image_gray, generator):
    """AMG (SAM3) sur la bbox du puits, garde le masque de meilleur score (> 0.35).
    Garde-fous d'aire [200,20000] px ; si 0 masque exploitable, on rebaisse
    stability_score_thresh (−0.05, plancher STAB_MIN) et on régénère."""
    from PIL import Image
    h, w = image_gray.shape
    cx, cy, r = detect_well(image_gray)
    y0, y1 = max(0, cy - r), min(h, cy + r)
    x0, x1 = max(0, cx - r), min(w, cx + r)
    crop = image_gray[y0:y1, x0:x1]
    img = Image.fromarray(np.stack([crop] * 3, -1))

    disc = well_mask(image_gray, margin=1.0)          # disque du puits (rejet des masques hors puits)

    stab = STAB_START
    best_score, best_seg, n_masks = -1, None, 0
    while True:
        out = generator(img, points_per_batch=POINTS_PER_BATCH, points_per_side=POINTS_PER_SIDE,
                        pred_iou_thresh=PRED_IOU_THRESH, stability_score_thresh=stab)
        cmasks = out.get("masks", []) if hasattr(out, "get") else []
        scores = out.get("scores", [1.0] * len(cmasks)) if hasattr(out, "get") else []
        n_masks = len(cmasks)
        best_score, best_seg, n_ok = -1, None, 0
        for cm, piou in zip(cmasks, scores):
            cm = np.asarray(cm).astype(bool)
            if cm.shape != crop.shape:
                continue
            seg = np.zeros((h, w), dtype=bool)
            seg[y0:y1, x0:x1] = cm
            a = int(seg.sum())
            if a < MIN_AREA_PX or a > MAX_AREA_PX:            # hors [200,20000] -> pas de scoring
                continue
            if int((seg & disc).sum()) < 0.8 * a:            # >=80% dans le puits, sinon rejet
                continue
            n_ok += 1
            s = score_mask(seg, image_gray, float(np.asarray(piou).ravel()[0]))
            if s > best_score:
                best_score, best_seg = s, seg
        if n_ok > 0 or stab <= STAB_MIN + 1e-9:              # masque exploitable OU plancher atteint
            break
        stab = round(stab - STAB_STEP, 2)
        print(f"[sam3_amg] 0 masque exploitable -> stability_score_thresh={stab}", flush=True)
    print(f"[sam3_amg] {n_masks} masques AMG, best_score={best_score:.3f}, stab={stab}", flush=True)
    return best_seg if best_score > SCORE_MIN else None


def load():
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    print(f"[sam3_amg] pipeline mask-generation | device={device}", flush=True)
    return pipeline("mask-generation", model="facebook/sam3", device=device)


def segment(generator, frame_gray):
    seg = find_best_amg_mask(frame_gray, generator)
    if seg is None:
        seg = np.zeros(frame_gray.shape, dtype=bool)
    return seg, None


if __name__ == "__main__":
    from common.batch import cli
    cli("sam3_amg", load, segment)
