"""
Cellpose (classique, PAS Cellpose-SAM) — modèle cyto3 (U-Net + flows), non basé
sur SAM. Automatique : renvoie un label image ; on garde l'instance au point sombre
du puits (même règle single-organoïde que les autres).

  python run.py --input frame.npy --output mask.npy

Env: cellpose env (pip install cellpose). Télécharge les poids cyto3 au 1er run.
"""

import sys, argparse
from pathlib import Path
import numpy as np
import cv2

BENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH_ROOT))
from common.seed import dark_point_in_well, well_mask
from common.pick import pick_instance_at_point, fill_holes

MODEL_TYPE = "cyto3"     # cellpose classique (pas cpsam)


def load():
    import torch
    from cellpose import models
    gpu = torch.cuda.is_available()
    try:
        model = models.CellposeModel(gpu=gpu, model_type=MODEL_TYPE)
    except TypeError:                                   # API cellpose v4
        model = models.CellposeModel(gpu=gpu, pretrained_model=MODEL_TYPE)
    print(f"[cellpose] model={MODEL_TYPE} gpu={gpu}", flush=True)
    return model


def segment(model, frame_gray):
    img = np.stack([frame_gray] * 3, axis=-1)
    out = model.eval(img, diameter=None, channels=[0, 0])   # diamètre auto, niveaux de gris
    labels = np.asarray(out[0]).astype(np.int32)
    disc = well_mask(frame_gray, margin=1.0)
    labels[~disc] = 0
    if labels.max() == 0:
        return np.zeros(frame_gray.shape, dtype=bool), None
    pt = dark_point_in_well(frame_gray)
    return fill_holes(pick_instance_at_point(labels, pt, shape=frame_gray.shape)), pt


if __name__ == "__main__":
    from common.batch import cli
    cli("cellpose", load, segment)
