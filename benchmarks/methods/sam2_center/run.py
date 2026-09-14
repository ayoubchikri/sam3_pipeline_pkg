"""
SAM2 pointeur au centre — SAM2 image predictor, UN point positif au centre
géométrique de l'image (W/2, H/2), et rien d'autre : pas de détection de puits,
pas de point sombre, pas de post-traitement. On prend le meilleur masque renvoyé
par SAM2 (argmax de ses scores). Baseline "pointeur naïf".

  python run.py --input frame.npy --output mask.npy

Env: sam2_env / ayoub (repo SAM2 + checkpoint).
"""

import os, sys, argparse
from pathlib import Path
import numpy as np
import cv2
import torch

BENCH_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BENCH_ROOT.parent
REPO = PROJECT_ROOT / "repo_sam2"
sys.path.insert(0, str(BENCH_ROOT))


def build_image_predictor():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = str(REPO / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt")
    sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", ckpt, device=device)
    return SAM2ImagePredictor(sam2_model)


def load():
    os.chdir(REPO)
    return build_image_predictor()


def segment(image_predictor, frame_gray):
    h, w = frame_gray.shape
    center = [w // 2, h // 2]                       # point au centre de l'image
    image_predictor.set_image(np.stack([frame_gray] * 3, axis=-1))
    with torch.inference_mode():
        masks, scores, _ = image_predictor.predict(
            point_coords=np.array([center], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
    mask = masks[int(np.argmax(scores))].astype(bool)   # meilleur masque SAM2, sans post-traitement
    return mask, center


if __name__ == "__main__":
    from common.batch import cli
    cli("sam2_center", load, segment)
