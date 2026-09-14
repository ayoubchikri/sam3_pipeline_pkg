"""
µSAM method — micro-sam (Archit et al., the paper that introduces AIS).

Used here in INTERACTIVE point-prompt mode with the shared dark-blob seed (same
seeding as our SAM2), NOT its automatic AIS — we want the single organoid at the
point. Uses the LM generalist model (vit_b_lm).

  python run.py --input frame.npy --output mask.npy

Env: microsam env (conda install -c conda-forge micro_sam). Downloads vit_b_lm weights on first run.
"""

import sys, argparse
from pathlib import Path
import numpy as np
import cv2

BENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH_ROOT))
from common.seed import dark_point_in_well
from common.pick import largest_component, fill_holes

MODEL_TYPE = "vit_b_lm"   # micro-sam LM generalist


def load():
    import os, torch
    from micro_sam.util import get_sam_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[microsam] torch {torch.__version__} | cuda_available={torch.cuda.is_available()} "
          f"| device={device} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)
    return get_sam_model(model_type=MODEL_TYPE, device=device)     # force le GPU si dispo


def segment(predictor, frame_gray):
    from micro_sam.prompt_based_segmentation import segment_from_points
    img_rgb = np.stack([frame_gray] * 3, axis=-1)
    seed_xy = dark_point_in_well(frame_gray)
    predictor.set_image(img_rgb)
    # micro-sam expects points as (y, x) arrays and labels 1=fg
    points = np.array([[seed_xy[1], seed_xy[0]]])
    labels = np.array([1])
    mask = np.asarray(segment_from_points(predictor, points, labels)).astype(bool)
    if mask.ndim > 2:
        mask = mask.squeeze()
    return fill_holes(largest_component(mask)), seed_xy


if __name__ == "__main__":
    from common.batch import cli
    cli("microsam", load, segment)
