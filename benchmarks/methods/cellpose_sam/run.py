"""
Cellpose-SAM method — cellpose v4 'cpsam' model (SAM encoder -> Cellpose flows,
prompt encoder + mask decoder removed). It's automatic (no prompt), so it returns
a label image of all organoids; we then SELECT the instance at the shared
dark-blob point.

  python run.py --input frame.npy --output mask.npy

Env: cellpose env (pip install cellpose>=4). Downloads the cpsam weights on first run.
"""

import sys, argparse
from pathlib import Path
import numpy as np
import cv2

BENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH_ROOT))
from common.seed import dark_point_in_well, well_mask
from common.pick import pick_instance_at_point, fill_holes


def load():
    import torch
    from cellpose import models
    return models.CellposeModel(gpu=torch.cuda.is_available())   # v4 default = Cellpose-SAM (cpsam)


def segment(model, frame_gray):
    img = np.stack([frame_gray] * 3, axis=-1)
    out = model.eval(img)                            # (masks, flows, styles)
    labels = np.asarray(out[0]).astype(np.int32)

    disc = well_mask(frame_gray, margin=1.0)         # objets dans le puits, puis point sombre
    labels[~disc] = 0
    if labels.max() == 0:
        return np.zeros(frame_gray.shape, dtype=bool), None
    pt = dark_point_in_well(frame_gray)
    return fill_holes(pick_instance_at_point(labels, pt, shape=frame_gray.shape)), pt


if __name__ == "__main__":
    from common.batch import cli
    cli("cellpose_sam", load, segment)
