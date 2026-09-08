#!/usr/bin/env python3
"""
Filtre de netteté (focus gate) POST-segmentation, INDÉPENDANT de run_segmentation.
Netteté ORGANOÏDE = variance du Laplacien sur (masque PRÉDIT dilaté de 2*TAU px).
Un film est marqué FLOU si >= K de ses frames sont sous le seuil.

Deux étapes (pour pouvoir changer le seuil sans tout recalculer) :
  1) CALCUL (coûteux, relit les frames)  -> <puits>_sharp.npy = {cavité: netteté par frame}
  2) CLASSIFICATION (instantané)         -> <puits>_focus.npy = {cavité: {blurry, n_below}}
                                          + CSV récap  focus_thr<seuil>_k<K>.csv

Le calcul n'est fait qu'une fois (réutilisé si présent). Rejoue avec un autre
--thresh / --k : seule la classification (rapide) est refaite.

  python focus_filter.py                          # seuil 280, K=15
  python focus_filter.py --thresh 300 --k 20      # autre point de fonctionnement (rapide)
  python focus_filter.py --recompute              # force le recalcul des nettetés

CPU pur (cv2 + Pillow + scipy). À lancer sur le DCE (masques + frames y sont).
"""
import os, sys, re, glob, argparse, csv
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import binary_dilation

DATASET_ROOT = "/usr/users/projets_p15_igr/chikri_ayo/stage_segmentation/processed_files_ALL_frames"
MASKS_ROOT = str(Path(DATASET_ROOT).parent / f"{Path(DATASET_ROOT).name}_masks_SAM3")


def load_gray_frames(tif_path):
    from PIL import Image, ImageSequence
    im = Image.open(tif_path)
    frames = []
    for p in ImageSequence.Iterator(im):
        a = np.asarray(p)
        if a.ndim > 2:
            a = a[..., 0] if a.shape[-1] <= 4 else a[0]
        if a.dtype != np.uint8:
            lo, hi = float(a.min()), float(a.max())
            a = ((a - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
        frames.append(a)
    return frames


def sharp_of_film(frames_gray, mask_stack, dil):
    """Netteté organoïde par frame (var. Laplacien sur masque prédit + 2*TAU px). NaN si pas de masque."""
    n = min(len(frames_gray), len(mask_stack))
    sh = np.full(n, np.nan, np.float32)
    for t in range(n):
        g, m = frames_gray[t], mask_stack[t]
        if m is not None and np.any(m):
            reg = binary_dilation(np.asarray(m, bool), iterations=dil)
            sh[t] = float(cv2.Laplacian(g, cv2.CV_64F)[reg].var())
    return sh


def compute_sharpness(masks_root, dataset_root, dil, recompute):
    """Étape 1 : pour chaque <puits>.npy de masques, calcule <puits>_sharp.npy."""
    masks_root, dataset_root = Path(masks_root), Path(dataset_root)
    npys = [p for p in glob.glob(str(masks_root / "**" / "*.npy"), recursive=True)
            if not os.path.basename(p).startswith("._")
            and not p.endswith("_sharp.npy") and not p.endswith("_focus.npy")]
    print(f"[calcul] {len(npys)} puits segmentés", flush=True)
    for i, mp in enumerate(npys):
        sp = mp[:-4] + "_sharp.npy"
        if os.path.exists(sp) and not recompute:
            continue
        rel = Path(mp).relative_to(masks_root).with_suffix("")     # chemin du puits (sans .npy)
        well_dir = dataset_root / rel
        try:
            masks = np.load(mp, allow_pickle=True).item()
        except Exception as e:
            print(f"  ERR load {mp}: {e}", flush=True); continue
        sharp = {}
        for cav, stack in masks.items():
            tif = well_dir / f"{cav}.tif"
            if stack is None or not tif.exists():
                sharp[cav] = None; continue
            try:
                frames = load_gray_frames(tif)
                sharp[cav] = sharp_of_film(frames, stack, dil)
            except Exception as e:
                print(f"    ERR {cav}: {e}", flush=True); sharp[cav] = None
        np.save(sp, np.array(sharp, dtype=object), allow_pickle=True)
        if i % 20 == 0:
            print(f"  {i}/{len(npys)} puits", flush=True)


def classify(masks_root, thresh, k):
    """Étape 2 : applique (seuil, K) aux <puits>_sharp.npy -> <puits>_focus.npy + CSV."""
    masks_root = Path(masks_root)
    sps = [p for p in glob.glob(str(masks_root / "**" / "*_sharp.npy"), recursive=True)
           if not os.path.basename(p).startswith("._")]
    csv_path = masks_root / f"focus_thr{thresh}_k{k}.csv"
    n_films = n_blur = 0
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["well", "cavity", "n_frames", "n_below", "min_sharp", "blurry"])
        for sp in sps:
            sharp = np.load(sp, allow_pickle=True).item()
            focus, rel = {}, os.path.relpath(sp, masks_root)[:-len("_sharp.npy")]
            for cav, sh in sharp.items():
                if sh is None or len(sh) == 0:
                    focus[cav] = {"blurry": False, "n_below": 0}; continue
                sh = np.asarray(sh, float)
                n_below = int(np.sum(sh < thresh))                 # NaN -> non compté
                blurry = n_below >= k
                focus[cav] = {"blurry": bool(blurry), "n_below": n_below}
                n_films += 1; n_blur += int(blurry)
                mn = np.nanmin(sh) if np.isfinite(sh).any() else float("nan")
                w.writerow([rel, cav, len(sh), n_below, f"{mn:.0f}", int(blurry)])
            np.save(sp[:-len("_sharp.npy")] + "_focus.npy", np.array(focus, dtype=object), allow_pickle=True)
    print(f"[classif] seuil={thresh} K={k} | {n_blur}/{n_films} films flous ({100*n_blur/max(n_films,1):.1f}%)")
    print(f"          récap -> {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks-root", default=MASKS_ROOT)
    ap.add_argument("--dataset-root", default=DATASET_ROOT)
    ap.add_argument("--thresh", type=float, default=280, help="seuil de netteté organoïde")
    ap.add_argument("--k", type=int, default=15, help="film flou si >= K frames sous le seuil")
    ap.add_argument("--tau", type=int, default=2, help="dilatation du masque = 2*tau px")
    ap.add_argument("--recompute", action="store_true", help="force le recalcul des nettetés")
    a = ap.parse_args()
    compute_sharpness(a.masks_root, a.dataset_root, 2 * a.tau, a.recompute)
    classify(a.masks_root, a.thresh, a.k)


if __name__ == "__main__":
    main()
