"""
SAM3 pointeur — SAM3 en mode PVS (Sam3Tracker, via transformers), prompté par un
POINT au point sombre du puits, exactement comme SAM2/µSAM : le point désigne
l'OBJET à segmenter à cet endroit (segmentation d'instance, pas d'exemplaire, pas
de propagation vidéo). On garde le meilleur masque (iou_scores).

NB : c'est une interface DIFFÉRENTE du `sam3` (texte) qui, lui, utilise le package
facebookresearch `Sam3Processor` (concept). Ici on passe par `transformers`.

  python run.py --input frame.npy --output mask.npy   (via common.batch)

Env: bench312 (GPU) + transformers (Sam3Tracker) + scipy 1.13.x (numpy<2). HF_HOME configuré.
"""

import sys
from pathlib import Path
import numpy as np

BENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH_ROOT))
from common.seed import dark_point_in_well
from common.pick import largest_component, fill_holes

MODEL_ID = "facebook/sam3"


def load():
    import torch
    from transformers import Sam3TrackerProcessor, Sam3TrackerModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sam3_pointer] Sam3Tracker (transformers) | device={device}", flush=True)
    processor = Sam3TrackerProcessor.from_pretrained(MODEL_ID)
    model = Sam3TrackerModel.from_pretrained(MODEL_ID).to(device)
    return {"processor": processor, "model": model, "device": device}


def segment(ctx, frame_gray):
    import torch
    from PIL import Image
    processor, model, device = ctx["processor"], ctx["model"], ctx["device"]

    seed_xy = dark_point_in_well(frame_gray)                 # (x, y) point sombre du puits
    sx, sy = int(seed_xy[0]), int(seed_xy[1])
    raw = Image.fromarray(np.stack([frame_gray] * 3, -1))
    inputs = processor(images=raw,
                       input_points=[[[[sx, sy]]]],          # [batch][obj][point][xy]
                       input_labels=[[[1]]],                 # 1 = positif
                       return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

    arr = (masks.numpy() if hasattr(masks, "numpy") else np.asarray(masks)).astype(bool)
    while arr.ndim > 3:                                       # -> (k, H, W)
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[None]
    try:                                                     # meilleur des k masques (multimask)
        sc = outputs.iou_scores.detach().cpu().numpy().reshape(-1)
        idx = int(np.argmax(sc[:arr.shape[0]]))
    except Exception:
        idx = 0
    mask = arr[idx]
    if not mask.any():
        return np.zeros(frame_gray.shape, dtype=bool), seed_xy
    return fill_holes(largest_component(mask)), seed_xy


if __name__ == "__main__":
    from common.batch import cli
    cli("sam3_pointer", load, segment)
