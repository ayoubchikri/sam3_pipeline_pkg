"""
make_visual.py — génère l'image comparative des masques d'un run du benchmark.

Lit un dossier results/<frame>/ (input_frame.npy + *_mask.npy + scores.json) et
le GT correspondant dans annotations/, puis sort masks_overview.png :
  frame | GT | fiji | sam2 | cellpose_sam | microsam | ...  (chaque masque superposé)

Autonome : ne dépend que de numpy + matplotlib (donc tourne dans l'env `ayoub`).

Usage :
  python make_visual.py                          # frame par défaut (CGR0073 framelast)
  python make_visual.py results/<frame_dir>      # un dossier de résultats précis
"""

import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH   = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BENCH)
ANN     = os.path.join(PROJECT, "annotations")

DEFAULT_DIR = os.path.join(
    BENCH, "results", "99f4e1b9e6170cc570748a2c7108a3b2_1008280_70_framelast")

PALETTE = [[1, 0.3, 0.3], [0.3, 0.55, 1], [1, 0.8, 0.2], [0.75, 0.3, 1], [0.3, 0.9, 0.9]]


def overlay(frame, mask, color, alpha=0.55):
    rgb = np.stack([frame] * 3, -1).astype(np.float32) / 255.0
    m = np.asarray(mask, bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(color)
    return np.clip(rgb, 0, 1)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    d = os.path.abspath(d)
    name = os.path.basename(d.rstrip("/"))          # <tifname>_<framelabel>

    frame = np.load(os.path.join(d, "input_frame.npy"))

    gt_path = os.path.join(ANN, f"{name}_gt.npy")
    gt = np.load(gt_path).astype(bool) if os.path.exists(gt_path) else None

    scores = {}
    sp = os.path.join(d, "scores.json")
    if os.path.exists(sp):
        scores = json.load(open(sp))

    masks = {}
    for f in sorted(glob.glob(os.path.join(d, "*_mask.npy"))):
        method = os.path.basename(f)[:-len("_mask.npy")]
        masks[method] = np.load(f).astype(bool)

    panels = [("frame", None)]
    if gt is not None:
        panels.append(("GT", gt))
    for method in masks:
        panels.append((method, masks[method]))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.2))
    if n == 1:
        axes = [axes]

    for i, (nm, mask) in enumerate(panels):
        ax = axes[i]
        if nm == "frame":
            ax.imshow(frame, cmap="gray"); ax.set_title("frame", fontsize=9)
        else:
            color = [0.2, 1, 0.2] if nm == "GT" else PALETTE[(i - 2) % len(PALETTE)]
            ax.imshow(overlay(frame, mask, color))
            px = int(mask.sum())
            iou = scores.get(nm, {}).get("iou")
            ax.set_title(f"{nm}\n{px}px" if iou is None else f"{nm}\nIoU={iou:.2f} px={px}",
                         fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Masks — {name}", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(d, "masks_overview.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
