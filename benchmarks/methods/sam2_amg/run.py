"""
SAM2 AMG pur — SAM2AutomaticMaskGenerator + la fonction de score ACTUELLE du
pipeline (score_mask : moyenne pondérée area / predicted_iou SAM2 / contraste au
bord), SANS pointeur. C'est l'ancien pipeline "AMG choisit tout seul le meilleur
masque". AMG tourne dans la bbox du puits, on garde le masque de meilleur score
(> 0.35), sinon rien.

  python run.py --input frame.npy --output mask.npy

Env: sam2_env / ayoub (repo SAM2 + checkpoint). Reproduit score_mask/find_best_amg_mask
de sam2_segmentation_save_roi.py pour rester autonome (le fichier racine diffère Mac/DCE).
"""

import os, sys, argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from skimage.measure import regionprops
from scipy.ndimage import binary_erosion, binary_dilation

BENCH_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BENCH_ROOT.parent
REPO = PROJECT_ROOT / "repo_sam2"
sys.path.insert(0, str(BENCH_ROOT))
from common.seed import detect_well, well_mask

# constantes empiriques du pipeline (mesurées sur les GT)
AVG_ORGANOID_AREA_PX = 4384.58
AREA_SIGMA_PX = 2440.98
SCORE_MIN = 0.35
# garde-fous d'aire : un organoïde plausible est dans [200, 20000] px (plus gros GT = 8596)
MIN_AREA_PX = 200
MAX_AREA_PX = 20000
# fallback : si 0 masque exploitable dans le puits ET dans [200,20000], on rebaisse
# stability_score_thresh de 0.05 et on régénère, jusqu'au plancher STAB_MIN.
STAB_START = 0.8
STAB_STEP = 0.05
STAB_MIN = 0.35


def score_mask(mask, image_gray, predicted_iou):
    """Copie fidèle de score_mask() du pipeline : moyenne pondérée area/iou/edge."""
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

    # poids fittés sur les PGR (score_fit) : predicted_iou domine
    return 0.10 * area_score + 0.70 * iou_score + 0.20 * edge_score


def find_best_amg_mask(image_gray, mask_gen):
    """AMG sur la bbox du puits, garde le masque de meilleur score (> 0.35).
    Garde-fous d'aire [200,20000] px ; si 0 masque exploitable, on rebaisse
    stability_score_thresh (−0.05, plancher STAB_MIN) et on régénère."""
    h, w = image_gray.shape
    cx, cy, r = detect_well(image_gray)
    y0, y1 = max(0, cy - r), min(h, cy + r)
    x0, x1 = max(0, cx - r), min(w, cx + r)
    crop = image_gray[y0:y1, x0:x1]
    rgb = np.stack([crop] * 3, axis=-1)

    disc = well_mask(image_gray, margin=1.0)          # disque du puits (rejet des masques hors puits)

    stab = STAB_START
    best_score, best_seg, n_masks = -1, None, 0
    while True:
        mask_gen.stability_score_thresh = stab        # remis à STAB_START à chaque frame (générateur persistant)
        masks = mask_gen.generate(rgb)
        n_masks = len(masks)
        best_score, best_seg, n_ok = -1, None, 0
        for m in masks:
            seg = np.zeros((h, w), dtype=bool)
            seg[y0:y1, x0:x1] = m["segmentation"]
            a = int(seg.sum())
            if a < MIN_AREA_PX or a > MAX_AREA_PX:            # hors [200,20000] -> pas de scoring
                continue
            if int((seg & disc).sum()) < 0.8 * a:            # >=80% dans le puits, sinon rejet
                continue
            n_ok += 1
            s = score_mask(seg, image_gray, m["predicted_iou"])
            if s > best_score:
                best_score, best_seg = s, seg
        if n_ok > 0 or stab <= STAB_MIN + 1e-9:              # masque exploitable OU plancher atteint
            break
        stab = round(stab - STAB_STEP, 2)
        print(f"[sam2_amg] 0 masque exploitable -> stability_score_thresh={stab}", flush=True)
    print(f"[sam2_amg] {n_masks} masques AMG, best_score={best_score:.3f}, stab={stab}", flush=True)
    return best_seg if best_score > SCORE_MIN else None


def build_mask_gen():
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = str(REPO / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt")
    sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", ckpt, device=device)
    return SAM2AutomaticMaskGenerator(
        sam2_model, points_per_side=24, pred_iou_thresh=0.7,
        stability_score_thresh=STAB_START, use_m2m=True)


def load():
    os.chdir(REPO)
    return build_mask_gen()


def segment(mask_gen, frame_gray):
    with torch.inference_mode():
        seg = find_best_amg_mask(frame_gray, mask_gen)
    if seg is None:
        seg = np.zeros(frame_gray.shape, dtype=bool)
    return seg, None


if __name__ == "__main__":
    from common.batch import cli
    cli("sam2_amg", load, segment)
