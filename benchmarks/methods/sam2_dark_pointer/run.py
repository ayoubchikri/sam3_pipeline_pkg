"""
SAM2 method (ours) — NO AMG.

well detection -> dark-blob point INSIDE the well -> single positive point prompt
-> SAM2 ImagePredictor -> best mask. (No AMG, no scoring.)

  python run.py --input frame.npy --output mask.npy

Env: sam2_env (needs the SAM2 repo + checkpoint at the project root).
"""

import os, sys, argparse
from pathlib import Path
import numpy as np
import cv2
import torch

BENCH_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BENCH_ROOT.parent               # /Volumes/Ultra Touch/SAM2_segmentation
REPO = PROJECT_ROOT / "repo_sam2"
sys.path.insert(0, str(BENCH_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
from common.seed import dark_point_in_well
from common.pick import largest_component, fill_holes


def segment(image_predictor, frame_gray):
    img_rgb = np.stack([frame_gray] * 3, axis=-1)
    seed_xy = dark_point_in_well(frame_gray)          # point stays inside the well

    image_predictor.set_image(img_rgb)
    with torch.inference_mode():
        masks, scores, _ = image_predictor.predict(
            point_coords=np.array([seed_xy], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
    mask = masks[int(np.argmax(scores))].astype(bool)
    mask = largest_component(mask)
    return fill_holes(mask), seed_xy


def build_image_predictor():
    """Construit directement le SAM2 image-predictor (autonome, ne dépend pas de
    sam2_segmentation_save_roi.py qui peut différer entre Mac et DCE)."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = str(REPO / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"   # relatif à REPO (cwd)
    sam2_model = build_sam2(model_cfg, ckpt, device=device)
    return SAM2ImagePredictor(sam2_model)


def load():
    os.chdir(REPO)  # SAM2 needs its configs on the relative path
    return build_image_predictor()


if __name__ == "__main__":
    from common.batch import cli
    cli("sam2", load, segment)
