"""
run_benchmark.py — benchmark des méthodes de segmentation.

Deux modes :
  1) une frame (défaut)  : python run_benchmark.py [--tif <t> --frame framelast|frame0|<int>]
  2) TOUTES les annotations : python run_benchmark.py --all

En mode --all, pour chaque GT de annotations/ (frame0 + framelast de chaque tif) :
  - retrouve le tif dans bench_tifs/, extrait la frame
  - lance chaque méthode (dans son env conda) -> <method>_mask.npy
  - score vs GT (IoU / Dice / BF1)
  - génère l'overlay des masques (make_visual.py)
et à la fin agrège en 3 bar charts (toutes frames / frame0 / framelast) + summary.json,
le tout dans results_summary/.

  python run_benchmark.py --all
  python run_benchmark.py --all --methods fiji,sam2,microsam
"""

import os, sys, json, glob, shutil, argparse, subprocess
from pathlib import Path
import numpy as np
import tifffile
import cv2

BENCH = Path(__file__).resolve().parent
PROJECT = BENCH.parent
ANN_DIR = PROJECT / "annotations"
TIF_DIR = BENCH / "bench_tifs"                    # 58 tifs annotés (copiés depuis TIF_DATASET)
sys.path.insert(0, str(BENCH))
from common.metrics import compute_all
from common.plot_benchmark import plot

# Conda env python par méthode (DCE = 3 envs). Override via variables d'env.
ENV_AYOUB = os.environ.get("BENCH_PY_AYOUB",
    "python")
ENV_MSAM  = os.environ.get("BENCH_PY_MSAM",
    "python")
ENV_SAM3  = os.environ.get("BENCH_PY_SAM3",
    "python")
ENV_ORG   = os.environ.get("BENCH_PY_ORG",
    "python")
ORGANOID_REPO = os.environ.get("ORGANOID_REPO",
    "/path/to/OrganoID")

WEIGHTS = os.environ.get("BENCH_WEIGHTS",
    "/path/to/model_weights")

METHODS = {
    "fiji":         {"dir": "methods/fiji",              "python": ENV_AYOUB},
    "sam2":         {"dir": "methods/sam2_dark_pointer", "python": ENV_AYOUB},
    "sam2_amg":     {"dir": "methods/sam2_amg",          "python": ENV_AYOUB},
    "sam2_center":  {"dir": "methods/sam2_center",       "python": ENV_AYOUB},
    "cellpose_sam": {"dir": "methods/cellpose_sam",      "python": ENV_AYOUB,
                     "env": {"CELLPOSE_LOCAL_MODELS_PATH": f"{WEIGHTS}/cellpose"}},
    "cellpose":     {"dir": "methods/cellpose",          "python": ENV_AYOUB,
                     "env": {"CELLPOSE_LOCAL_MODELS_PATH": f"{WEIGHTS}/cellpose"}},
    "microsam":     {"dir": "methods/microsam",          "python": ENV_MSAM,
                     "env": {"XDG_CACHE_HOME": f"{WEIGHTS}/microsam_cache"}},
    "sam3":         {"dir": "methods/sam3",              "python": ENV_SAM3,
                     "env": {"TORCH_HOME": f"{WEIGHTS}/torch"}},
    "sam3_pointer": {"dir": "methods/sam3_pointer",      "python": ENV_SAM3,
                     "env": {"TORCH_HOME": f"{WEIGHTS}/torch"}},
    "sam3_text_point": {"dir": "methods/sam3_text_point", "python": ENV_SAM3,
                        "env": {"TORCH_HOME": f"{WEIGHTS}/torch"}},
    "sam3_amg":     {"dir": "methods/sam3_amg",          "python": ENV_SAM3,
                     "env": {"TORCH_HOME": f"{WEIGHTS}/torch"}},
    "organoid":     {"dir": "methods/organoid",          "python": ENV_ORG,
                     "env": {"ORGANOID_REPO": ORGANOID_REPO,
                             "ORGANOID_MODEL": f"{ORGANOID_REPO}/TrainableModel",
                             "TF_USE_LEGACY_KERAS": "1"}},   # Keras 2 -> charge le SavedModel
}

DEFAULT_TIF = str(BENCH / "test_tif" / "99f4e1b9e6170cc570748a2c7108a3b2_1008280_70.tif")
DEFAULT_FRAME = "framelast"
DEFAULT_PAIRS = str(BENCH / "well_pairs")        # GT par cavité d'Émilie = jeu par défaut


def tif_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def load_frame_uint8(arr, idx):
    f = arr[idx]
    if f.dtype != np.uint8:
        lo, hi = f.min(), f.max()
        f = ((f - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    if f.ndim == 3:
        f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    return f


def frame_index(arr, frame):
    if frame == "frame0":
        return 0, "frame0"
    if frame == "framelast":
        return arr.shape[0] - 1, "framelast"
    return int(frame), f"frame{int(frame)}"


def run_frame(tif_path, frame, selected):
    """Extrait la frame, lance chaque méthode, score vs GT.
    Retourne (frame_key, out_dir, scores) ou (frame_key, None, None) si pas de GT."""
    name = tif_name(tif_path)
    arr = tifffile.imread(tif_path)
    idx, flabel = frame_index(arr, frame)
    frame_key = f"{name}_{flabel}"
    out_dir = BENCH / "results" / frame_key
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = ANN_DIR / f"{frame_key}_gt.npy"
    if not gt_path.exists():
        print(f"  ⚠ pas de GT: {gt_path.name} — skip"); return frame_key, None, None
    gt = np.load(gt_path).astype(bool)

    gray = load_frame_uint8(arr, idx)
    frame_path = out_dir / "input_frame.npy"
    np.save(frame_path, gray)
    print(f"Frame: {frame_key}  shape={gray.shape}")

    scores = score_methods(frame_path, gt, out_dir, selected)
    return frame_key, out_dir, scores


def score_methods(frame_path, gt, out_dir, selected):
    """Lance chaque méthode (dans son env) sur frame_path, score vs gt. Échec = 0."""
    scores = {}
    for m in selected:
        cfg = METHODS[m]
        py = cfg["python"]
        run_py = BENCH / cfg["dir"] / "run.py"
        mask_path = out_dir / f"{m}_mask.npy"
        if not os.path.exists(py):
            print(f"  [{m}] SKIP — env introuvable: {py}"); continue
        sub_env = dict(os.environ)
        sub_env.update(cfg.get("env", {}))
        for v in cfg.get("env", {}).values():
            os.makedirs(v, exist_ok=True)
        r = subprocess.run([py, str(run_py), "--input", str(frame_path),
                            "--output", str(mask_path)],
                           capture_output=True, text=True, env=sub_env)
        if r.stdout.strip():
            print("      " + r.stdout.strip().replace("\n", "\n      "))
        if r.returncode != 0 or not mask_path.exists():
            print(f"  [{m}] FAILED (compté 0):\n{r.stderr[-1200:]}")
            scores[m] = {"iou": 0.0, "dice": 0.0, "bf1": 0.0}   # échec = 0
            continue
        pred = np.load(mask_path).astype(bool)
        scores[m] = compute_all(pred, gt)
        print(f"  [{m}] IoU={scores[m]['iou']:.3f} Dice={scores[m]['dice']:.3f} BF1={scores[m]['bf1']:.3f}")
    with open(out_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    return scores


def make_overlay(out_dir):
    """Génère l'overlay des masques via make_visual.py (même env que cet orchestrateur)."""
    subprocess.run([sys.executable, str(BENCH / "make_visual.py"), str(out_dir)],
                   capture_output=True, text=True)
    return out_dir / "masks_overview.png"


def find_tif(name):
    p = TIF_DIR / f"{name}.tif"
    return p if p.exists() else None


def mean_scores(per_frame, keys):
    """{frame_key: {method: {iou,dice,bf1}}} restreint à `keys` -> {method: {metric: moyenne}}."""
    agg = {}
    for k in keys:
        for m, sc in per_frame.get(k, {}).items():
            agg.setdefault(m, {"iou": [], "dice": [], "bf1": []})
            for metric in ("iou", "dice", "bf1"):
                agg[m][metric].append(sc[metric])
    return {m: {metric: float(np.mean(v)) for metric, v in d.items()}
            for m, d in agg.items()}


def run_all(selected):
    gts = sorted(glob.glob(str(ANN_DIR / "*_gt.npy")))
    # frame_key -> (tif_name, flabel)
    jobs = []
    missing = []
    for g in gts:
        b = os.path.basename(g)[:-len("_gt.npy")]      # <name>_<flabel>
        for flabel in ("frame0", "framelast"):
            if b.endswith("_" + flabel):
                name = b[:-len("_" + flabel)]
                tif = find_tif(name)
                if tif is None:
                    missing.append(name)
                else:
                    jobs.append((tif, flabel))
                break
    if missing:
        print(f"⚠ {len(missing)} tif(s) absents de bench_tifs/ (skip): {missing[:5]}...")
    print(f"== {len(jobs)} frames à traiter, méthodes={selected} ==\n")

    summary_dir = BENCH / "results_summary"
    visuals_dir = summary_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    per_frame = {}
    for i, (tif, flabel) in enumerate(jobs, 1):
        print(f"--- [{i}/{len(jobs)}] {tif_name(tif)} {flabel} ---")
        key, out_dir, scores = run_frame(tif, flabel, selected)
        if out_dir is None:
            continue
        per_frame[key] = scores
        png = make_overlay(out_dir)
        if png.exists():
            shutil.copy2(png, visuals_dir / f"{key}.png")

    # agrégats : toutes frames / frame0 / framelast
    all_keys = list(per_frame)
    subsets = {
        "all":       all_keys,
        "frame0":    [k for k in all_keys if k.endswith("_frame0")],
        "framelast": [k for k in all_keys if k.endswith("_framelast")],
    }
    summary = {"n_frames": len(all_keys), "per_frame": per_frame, "means": {}}
    for sub, keys in subsets.items():
        if not keys:
            continue
        means = mean_scores(per_frame, keys)
        summary["means"][sub] = {"n": len(keys), "scores": means}
        plot(means, str(summary_dir / f"benchmark_{sub}.png"),
             f"Benchmark {sub} (moyenne sur {len(keys)} frames)")
        print(f"\n[{sub}] n={len(keys)}")
        for m, sc in sorted(means.items(), key=lambda x: -x[1]["iou"]):
            print(f"    {m:14} IoU={sc['iou']:.3f} Dice={sc['dice']:.3f} BF1={sc['bf1']:.3f}")

    with open(summary_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nrésumé -> {summary_dir}/  (benchmark_all.png, benchmark_frame0.png, "
          f"benchmark_framelast.png, visuals/, summary.json)")


def pair_overlay(frame, gt, masks, scores, out_png):
    """Overlay simple frame | GT | méthodes pour une paire cavité."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def ov(f, m, col):
        rgb = np.stack([f] * 3, -1).astype(np.float32) / 255.0
        rgb[m] = 0.5 * rgb[m] + 0.5 * np.array(col)
        return np.clip(rgb, 0, 1)
    panels = [("frame", None), ("GT", gt)] + [(m, masks[m]) for m in masks]
    fig, ax = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    for i, (nm, mk) in enumerate(panels):
        if nm == "frame":
            ax[i].imshow(frame, cmap="gray"); ax[i].set_title("frame", fontsize=8)
        else:
            col = [0.2, 1, 0.2] if nm == "GT" else [1, 0.3, 0.3]
            ax[i].imshow(ov(frame, mk, col))
            iou = scores.get(nm, {}).get("iou")
            ax[i].set_title(nm if iou is None else f"{nm}\nIoU={iou:.2f}", fontsize=8)
        ax[i].axis("off")
    plt.tight_layout(); plt.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)


def run_pairs(pairs_dir, selected, out_tag=""):
    """Benchmark sur les paires (image, GT) par cavité — MODE BATCH : chaque méthode
    charge son modèle une seule fois et traite toutes les frames (manifest).
    out_tag != "" -> tout isolé dans results_pairs_<tag>/ et results_<tag>/ (pas d'écrasement)."""
    pairs_dir = Path(pairs_dir)
    imgs = sorted(p for p in pairs_dir.rglob("*_img.npy") if not p.name.startswith("._"))
    meta = {}
    mpath = pairs_dir / "manifest.json"
    if mpath.exists():
        for x in json.load(open(mpath)):
            meta[x["key"]] = (x.get("pid", x["key"].split("_")[0]), x.get("dose", "unknown"))
    suffix = f"_{out_tag}" if out_tag else ""
    summary_dir = BENCH / f"results_pairs{suffix}"
    mask_base = BENCH / f"results{suffix}"
    visuals_dir = summary_dir / "visuals"
    man_dir = summary_dir / "_manifests"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    # liste des jobs (key, img, gt, out_dir) + patient/dose
    jobs, key_meta = [], {}
    for img_p in imgs:
        key = img_p.name[:-len("_img.npy")]
        gt_p = img_p.parent / f"{key}_gt.npy"
        if not gt_p.exists():
            continue
        key_meta[key] = meta.get(key, (img_p.parent.parent.name, img_p.parent.name))
        out_dir = mask_base / f"pair_{key}"
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs.append((key, img_p, gt_p, out_dir))
    print(f"== {len(jobs)} paires × {len(selected)} méthodes (mode batch) ==\n")

    # 1) chaque méthode UNE fois sur toutes les frames (modèle chargé une fois)
    for m in selected:
        cfg = METHODS[m]
        py = cfg["python"]
        run_py = BENCH / cfg["dir"] / "run.py"
        if not os.path.exists(py):
            print(f"  [{m}] SKIP — env introuvable: {py}"); continue
        manifest = [[str(img_p), str(out_dir / f"{m}_mask.npy")]
                    for (key, img_p, gt_p, out_dir) in jobs]
        man_file = man_dir / f"{m}.json"
        json.dump(manifest, open(man_file, "w"))
        sub_env = dict(os.environ)
        sub_env.update(cfg.get("env", {}))
        for v in cfg.get("env", {}).values():
            os.makedirs(v, exist_ok=True)
        print(f"  [{m}] batch sur {len(manifest)} frames ...", flush=True)
        r = subprocess.run([py, str(run_py), "--manifest", str(man_file)],
                           capture_output=True, text=True, env=sub_env)
        if r.stdout.strip():
            print("      " + r.stdout.strip()[-600:].replace("\n", "\n      "))
        if r.returncode != 0:
            print(f"  [{m}] FAILED:\n{r.stderr[-1500:]}")

    # 2) scoring de toutes les frames + overlays
    per = {}
    for key, img_p, gt_p, out_dir in jobs:
        gt = np.load(gt_p).astype(bool)
        scores = {}
        for m in selected:
            mp = out_dir / f"{m}_mask.npy"
            scores[m] = (compute_all(np.load(mp).astype(bool), gt) if mp.exists()
                         else {"iou": 0.0, "dice": 0.0, "bf1": 0.0})   # méthode ratée = 0
        per[key] = scores
        with open(out_dir / "scores.json", "w") as f:
            json.dump(scores, f, indent=2)
        frame = np.load(img_p)
        masks = {m: np.load(out_dir / f"{m}_mask.npy").astype(bool)
                 for m in selected if (out_dir / f"{m}_mask.npy").exists()}
        pair_overlay(frame, gt, masks, scores, str(visuals_dir / f"{key}.png"))

    # agrégats : global + par patient + par type (CGR/PGR) + par type×dose
    all_keys = list(per)
    ptype = lambda k: key_meta[k][0][:3]          # type patient = 3 premières lettres (CGR/PGR)
    subsets = {"all": all_keys}
    for pid in sorted({key_meta[k][0] for k in all_keys}):          # par patient
        subsets[pid] = [k for k in all_keys if key_meta[k][0] == pid]
    for t in sorted({ptype(k) for k in all_keys}):                  # par type de patient
        subsets[f"type-{t}"] = [k for k in all_keys if ptype(k) == t]
    for t, dose in sorted({(ptype(k), key_meta[k][1]) for k in all_keys}):   # par type × dose
        subsets[f"{t}-{dose}"] = [k for k in all_keys if ptype(k) == t and key_meta[k][1] == dose]
    summary = {"n_pairs": len(all_keys), "meta": {k: key_meta[k] for k in all_keys},
               "per_pair": per, "means": {}}
    for sub, keys in subsets.items():
        if not keys:
            continue
        means = mean_scores(per, keys)
        summary["means"][sub] = {"n": len(keys), "scores": means}
        plot(means, str(summary_dir / f"benchmark_pairs_{sub}.png"),
             f"Benchmark cavités {sub} (moyenne sur {len(keys)})")
        print(f"\n[{sub}] n={len(keys)}")
        for m, sc in sorted(means.items(), key=lambda x: -x[1]["iou"]):
            print(f"    {m:14} IoU={sc['iou']:.3f} Dice={sc['dice']:.3f} BF1={sc['bf1']:.3f}")

    with open(summary_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nrésumé -> {summary_dir}/  (benchmark_pairs_all.png, benchmark_pairs_<PID>.png, "
          f"visuals/, summary.json)")


def main():
    ap = argparse.ArgumentParser()
    # DÉFAUT = benchmark sur les frames d'Émilie (well_pairs). La Source 1 (--all,
    # tes 116 GT annotations/) reste dispo mais n'est plus lancée par défaut.
    ap.add_argument("--pairs", nargs="?", const=DEFAULT_PAIRS, default=None,
                    help=f"benchmark GT par cavité d'Émilie (défaut: {DEFAULT_PAIRS})")
    ap.add_argument("--all", action="store_true", help="[Source 1] tes 116 GT annotations/ + agrégats")
    ap.add_argument("--tif", default=None)
    ap.add_argument("--frame", default=DEFAULT_FRAME, help="framelast | frame0 | <int>")
    ap.add_argument("--methods", default=None, help="liste séparée par virgules; défaut = toutes")
    ap.add_argument("--out", default="", help="suffixe de dossier de sortie -> results_pairs_<out>/ (isole le run)")
    args = ap.parse_args()

    selected = args.methods.split(",") if args.methods else list(METHODS)
    for m in selected:
        if m not in METHODS:
            print(f"méthode inconnue: {m}"); sys.exit(1)

    if args.all:                                   # Source 1, sur demande explicite
        run_all(selected)
        return

    if args.tif:                                   # une frame précise (debug), sur demande explicite
        key, out_dir, scores = run_frame(args.tif, args.frame, selected)
        if out_dir is None:
            sys.exit(1)
        if scores:
            plot(scores, str(out_dir / "benchmark_barchart.png"), key)
            make_overlay(out_dir)
        else:
            print("Aucune méthode n'a produit de masque.")
        return

    # DÉFAUT : frames d'Émilie
    run_pairs(args.pairs or DEFAULT_PAIRS, selected, out_tag=args.out)


if __name__ == "__main__":
    main()
