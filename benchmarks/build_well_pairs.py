"""
build_well_pairs.py — assemble les paires (image, GT) par cavité pour le benchmark,
à partir des dossiers de correspondance (gt_masks/ produits par gt_from_correspondance.py).

Pour chaque cavité annotée : entrée = la frame de correspondance du film dataset
(`..._wellid_N.tif`, même frame que celle alignée à la GT), GT = gt_masks/cav_N.npy.

Sortie (portable, à scp sur le DCE) : <out>/<key>_img.npy (uint8 H×W) + <key>_gt.npy (bool).
  key = <PID>_<welltag>_<wellid>_cav<N>

  python build_well_pairs.py --root "<racine correspondance>" --out well_pairs
"""

import os, re, glob, json, argparse
import numpy as np
import tifffile
import cv2


def parse_dose(name):
    """dose lue dans le nom (ctrl/control -> ctrl, high, low) ; 'unknown' sinon."""
    s = name.lower()
    for kw, norm in [("control", "ctrl"), ("ctrl", "ctrl"), ("high", "high"), ("low", "low")]:
        if kw in s:
            return norm
    return "unknown"


def to_gray_u8(img):
    if img.ndim > 2:
        img = img[..., 0] if img.shape[-1] <= 4 else img[0]
    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    return img


def frame_idx_from_name(name, n):
    name = name.lower()
    if "lastframe" in name:
        return n - 1
    m = re.search(r"frame\s*0*(\d+)", name)
    return max(0, min(n - 1, (int(m.group(1)) - 1) if m else n // 2))


def find_film_dir(folder):
    for d in glob.glob(os.path.join(folder, "*")):
        if not os.path.isdir(d) or d.endswith("gt_masks") or d.endswith("gt_overlays"):
            continue
        s = [t for t in glob.glob(os.path.join(d, "*.tif")) if "._" not in os.path.basename(t)]
        if s and re.search(r"_\d+\.tif$", os.path.basename(s[0])):
            return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="well_pairs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest = []

    for gm in sorted(glob.glob(os.path.join(args.root, "*", "*", "gt_masks"))):
        folder = os.path.dirname(gm)
        pid = folder.split(os.sep)[-2].split("_")[0]
        film_dir = find_film_dir(folder)
        raws = [t for t in glob.glob(os.path.join(folder, "*.tif")) if "_tpt" in os.path.basename(t)]
        if film_dir is None or not raws:
            print(f"[skip] {folder}"); continue
        toks = os.path.basename(raws[0]).split("_")
        welltag, wellid = toks[0], toks[1]
        dose = parse_dose(os.path.basename(raws[0]))
        dest = os.path.join(args.out, pid, dose)              # imbriqué patient/dose
        os.makedirs(dest, exist_ok=True)

        films = {int(os.path.basename(t).split("_")[-1].split(".")[0]): t
                 for t in glob.glob(os.path.join(film_dir, "*.tif")) if "._" not in os.path.basename(t)}
        sample = tifffile.imread(next(iter(films.values())))
        fidx = frame_idx_from_name(os.path.basename(folder), len(sample) if sample.ndim == 3 else 1)

        n = 0
        for npy in sorted(glob.glob(os.path.join(gm, "cav_*.npy"))):
            N = int(re.search(r"cav_(\d+)", os.path.basename(npy)).group(1))
            if N not in films:
                continue
            a = tifffile.imread(films[N])
            if a.ndim != 3 or len(a) <= fidx:
                continue
            frame = to_gray_u8(a[fidx])
            gt = np.load(npy).astype(bool)
            if gt.shape != frame.shape:          # sécurité : recale si besoin
                gt = cv2.resize(gt.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            key = f"{pid}_{welltag}_{wellid}_cav{N}"
            np.save(os.path.join(dest, f"{key}_img.npy"), frame)
            np.save(os.path.join(dest, f"{key}_gt.npy"), gt)
            manifest.append({"key": key, "pid": pid, "dose": dose, "well": welltag, "wellid": wellid,
                             "cavity": N, "frame_idx": fidx, "film": films[N],
                             "shape": list(frame.shape), "gt_px": int(gt.sum())})
            n += 1
        print(f"{pid}_{welltag}_{wellid} [{dose}]: {n} paires (frame idx {fidx})")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{len(manifest)} paires -> {args.out}/")


if __name__ == "__main__":
    main()
