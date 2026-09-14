"""
SAM3 texte + point — mode CONCEPT combiné (Sam3Model + Sam3Processor, transformers) :
on donne un CONCEPT texte ("organoid") ET un prompt visuel positif localisé au
point sombre du puits. Le prompt visuel de SAM3-concept est une BOÎTE (input_boxes),
donc on met une petite boîte (±BOX_HALF) autour du point sombre, label 1 (inclure).
On garde ensuite l'instance située au point sombre.

  python run.py --input frame.npy --output mask.npy   (via common.batch)

Env: bench312 (GPU) + transformers (Sam3) + scipy 1.13.x (numpy<2). HF_HOME configuré.
"""

import sys
from pathlib import Path
import numpy as np

BENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH_ROOT))
from common.seed import dark_point_in_well
from common.pick import pick_instance_at_point, fill_holes

MODEL_ID = "facebook/sam3"
TEXT_PROMPT = "organoid"
BOX_HALF = 50            # demi-taille (px) de la boîte positive autour du point sombre


def _mask_list(masks_out):
    if masks_out is None:
        return []
    arr = masks_out.cpu().numpy() if hasattr(masks_out, "cpu") else np.asarray(masks_out)
    if arr.size == 0:
        return []
    if arr.ndim == 2:
        arr = arr[None]
    return [m.astype(bool) for m in arr]


def load():
    import torch
    from transformers import Sam3Processor, Sam3Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sam3_text_point] Sam3 concept (texte+boîte) | device={device} | prompt={TEXT_PROMPT!r}", flush=True)
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model = Sam3Model.from_pretrained(MODEL_ID).to(device)
    return {"processor": processor, "model": model, "device": device}


def segment(ctx, frame_gray):
    import torch
    from PIL import Image
    processor, model, device = ctx["processor"], ctx["model"], ctx["device"]

    h, w = frame_gray.shape
    seed_xy = dark_point_in_well(frame_gray)
    sx, sy = float(seed_xy[0]), float(seed_xy[1])
    box = [max(0.0, sx - BOX_HALF), max(0.0, sy - BOX_HALF),
           min(w - 1.0, sx + BOX_HALF), min(h - 1.0, sy + BOX_HALF)]

    raw = Image.fromarray(np.stack([frame_gray] * 3, -1))
    inputs = processor(images=raw, text=TEXT_PROMPT,
                       input_boxes=[[box]], input_boxes_labels=[[1]],   # 1 = positif (inclure)
                       return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_instance_segmentation(
        outputs, threshold=0.5, mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist())[0]

    masks = _mask_list(res.get("masks") if isinstance(res, dict) else None)
    if not masks:
        return np.zeros(frame_gray.shape, dtype=bool), seed_xy
    return fill_holes(pick_instance_at_point(masks, seed_xy, shape=frame_gray.shape)), seed_xy


if __name__ == "__main__":
    from common.batch import cli
    cli("sam3_text_point", load, segment)
