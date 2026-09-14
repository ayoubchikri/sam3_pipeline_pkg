"""
SAM3 method — Meta Segment Anything Model 3 (facebookresearch/sam3), mode TEXTE
(concept segmentation).

SAM3 segmente par PROMPT ÉCRIT : on lui donne un concept ("organoid") et il renvoie
toutes les instances correspondantes. Comme la frame ne contient qu'un organoïde
(dans une cavité), on sélectionne l'instance située au point sombre du puits — même
règle single-organoïde que pour cellpose.

  python run.py --input frame.npy --output mask.npy

Env: bench312 (Python>=3.12, GPU). Modèle facebook/sam3 gated → il faut être
authentifié (hf auth login) et HF_HOME pointant vers un dossier avec de la place.

Le prompt texte est éditable ci-dessous (TEXT_PROMPT).
"""

import sys, argparse
from pathlib import Path
import numpy as np
import cv2

BENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH_ROOT))
from common.seed import dark_point_in_well
from common.pick import pick_instance_at_point, fill_holes

TEXT_PROMPT = "organoid"        # <-- le prompt écrit passé à SAM3
CONF_THRESHOLD = 0.15           # bas exprès : on filtre nous-mêmes / on inspecte les scores


def _to_mask_list(masks, shape):
    """state['masks'] -> liste de masques booléens H×W (gère tensor/array, (N,H,W) ou (N,1,H,W))."""
    if masks is None:
        return []
    if hasattr(masks, "deetach"):        # torch tensor (bfloat16 -> float avant numpy)
        masks = masks.detach().float().cpu().numpy()
    masks = np.asarray(masks)
    if masks.size == 0:
        return []
    if masks.ndim == 4:                 # (N,1,H,W)
        masks = masks[:, 0]
    if masks.ndim == 2:                 # (H,W) -> une seule instance
        masks = masks[None]
    return [np.asarray(m).astype(bool) for m in masks]


def load():
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sam3] torch {torch.__version__} | device={device} | prompt={TEXT_PROMPT!r}", flush=True)
    model = build_sam3_image_model(device=device, load_from_HF=True, enable_segmentation=True)
    processor = Sam3Processor(model, device=device, confidence_threshold=CONF_THRESHOLD)
    return {"processor": processor, "device": device}


def segment(ctx, frame_gray):
    import contextlib, torch
    from PIL import Image
    processor, device = ctx["processor"], ctx["device"]
    img = Image.fromarray(np.stack([frame_gray] * 3, axis=-1))
    seed_xy = dark_point_in_well(frame_gray)
    # SAM3 tourne en bfloat16 -> autocast pour éviter le mismatch de dtype.
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device == "cuda" else contextlib.nullcontext())
    with torch.inference_mode(), autocast:
        state = processor.set_image(img)
        state = processor.set_text_prompt(TEXT_PROMPT, state)
    masks = _to_mask_list(state.get("masks"), frame_gray.shape)
    if not masks:
        return np.zeros(frame_gray.shape, dtype=bool), seed_xy
    return fill_holes(pick_instance_at_point(masks, seed_xy, shape=frame_gray.shape)), seed_xy


if __name__ == "__main__":
    from common.batch import cli
    cli("sam3", load, segment)
