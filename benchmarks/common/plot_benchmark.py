"""
plot_benchmark.py — grouped bar chart from a scores.json.

x-axis = method, 3 bars per method (IoU, Dice, BF1), y-axis = score [0,1].

  python plot_benchmark.py <scores.json> <out.png> [frame_name]
"""

import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["iou", "dice", "bf1"]
METRIC_LABELS = {"iou": "IoU", "dice": "Dice", "bf1": "BF1"}
METRIC_COLORS = {"iou": "#4C8BF5", "dice": "#34A853", "bf1": "#F4B400"}
# noms affichés dans les plots (clé interne -> label lisible)
DISPLAY_NAMES = {"fiji": "Otsu", "sam2": "SAM2", "sam2_amg": "SAM2-AMG",
                 "sam2_center": "SAM2-center", "sam3": "SAM3-text",
                 "sam3_pointer": "SAM3-point", "sam3_text_point": "SAM3-text+point",
                 "sam3_amg": "SAM3-AMG",
                 "cellpose_sam": "Cellpose-SAM",
                 "cellpose": "Cellpose", "microsam": "µSAM", "organoid": "OrganoID"}


def plot(scores, out_path, frame_name=""):
    methods = list(scores.keys())
    n_m = len(methods)
    x = np.arange(n_m)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * n_m), 5))
    for k, metric in enumerate(METRICS):
        vals = [scores[m].get(metric, np.nan) for m in methods]
        bars = ax.bar(x + (k - 1) * width, vals, width,
                      label=METRIC_LABELS[metric], color=METRIC_COLORS[metric])
        for b, v in zip(bars, vals):
            if v == v:  # not NaN
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in methods], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Segmentation benchmark{(' — ' + frame_name) if frame_name else ''}")
    ax.legend(title="Metric")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Bar chart -> {out_path}")


if __name__ == "__main__":
    scores = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    frame = sys.argv[3] if len(sys.argv) > 3 else ""
    plot(scores, out, frame)
