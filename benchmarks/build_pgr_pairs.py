"""
build_pgr_pairs.py — ajoute les annotations PGR (frame0, dossier "à annoter") au
jeu de GT du benchmark, dans la MÊME structure que build_well_pairs.py :
  well_pairs/<patient>/<dose>/<key>_img.npy + <key>_gt.npy
et fusionne dans well_pairs/manifest.json.

Source : à annoter/<patient>/<hash_wellid_..._dose_date>/<hash_wellid_N.tif>
         + <puits>/annotations/<hash_wellid_N>_frame0.npy  (ton masque)
frame0 = arr[0]. Les masques vides (inannotables, 0 px) sont ignorés.
key = <patient>_<wellid>_cav<N>.

  python build_pgr_pairs.py            (env avec numpy, cv2, tifffile)
"""

import os, re, glob, json
from pathlib import Path
import numpy as np
import cv2
import tifffile

BENCH = Path(__file__).resolve().parent
PROJECT = BENCH.parent
ANNOTER = os.environ.get("ANNOTER_ROOT", str(PROJECT / "à annoter"))
OUT = str(BENCH / "well_pairs")


def parse_dose(name):
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


def main():
    manifest = []
    mpath = os.path.join(OUT, "manifest.json")
    if os.path.exists(mpath):
        manifest = json.load(open(mpath))
    have = {x["key"] for x in manifest}

    added = 0
    for npy in glob.glob(os.path.join(ANNOTER, "*", "*", "annotations", "*_frame0.npy")):
        if os.path.basename(npy).startswith("._"):
            continue
        gt = np.load(npy).astype(bool)
        if gt.sum() == 0:                      # inannotable -> pas de paire benchmark
            continue
        base = os.path.basename(npy)[:-len("_frame0.npy")]   # hash_wellid_N
        well_dir = os.path.dirname(os.path.dirname(npy))
        patient = Path(well_dir).parent.name
        dose = parse_dose(os.path.basename(well_dir))
        m = re.search(r"_(\d+)$", base)
        N = m.group(1) if m else "?"
        wellid = base.split("_")[-2] if len(base.split("_")) >= 2 else "?"
        tif = os.path.join(well_dir, base + ".tif")
        if not os.path.exists(tif):
            continue
        arr = tifffile.imread(tif)
        frame = to_gray_u8(arr[0] if arr.ndim >= 3 else arr)
        if gt.shape != frame.shape:
            gt = cv2.resize(gt.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        key = f"{patient}_{wellid}_cav{N}"
        if key in have:
            continue
        dest = os.path.join(OUT, patient, dose)
        os.makedirs(dest, exist_ok=True)
        np.save(os.path.join(dest, f"{key}_img.npy"), frame)
        np.save(os.path.join(dest, f"{key}_gt.npy"), gt)
        manifest.append({"key": key, "pid": patient, "dose": dose, "wellid": wellid,
                         "cavity": int(N) if N.isdigit() else N, "frame_idx": 0,
                         "source": "pgr", "tif": tif, "shape": list(frame.shape),
                         "gt_px": int(gt.sum())})
        have.add(key); added += 1

    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"{added} paires PGR ajoutées -> {OUT}/<patient>/<dose>/  (manifest: {len(manifest)} au total)")


if __name__ == "__main__":
    main()
