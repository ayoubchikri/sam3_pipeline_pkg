"""
Fiji method — VRAI ImageJ/Fiji (pas une approximation).

Émilie segmente sous Fiji. On pilote donc le vrai ImageJ via pyimagej (JVM +
Fiji headless) : c'est ImageJ qui calcule l'auto-threshold et binarise l'image
(exactement setAutoThreshold + Convert to Mask). On applique ensuite NOTRE
sélection au point sombre du puits pour ne garder qu'un organoïde — la même
règle single-organoïde que les autres méthodes du benchmark.

  python run.py --input frame.npy --output mask.npy

Env: env avec pyimagej + Java (openjdk) + maven. Première exécution : pyimagej
télécharge Fiji (~qq centaines de Mo) dans le cache maven (~/.m2 / IMAGEJ_DIR).
Voir les instructions d'install données à part.

Réglages ImageJ (éditables) :
  AUTO_THRESHOLD_METHOD  = "Otsu"      # méthode d'auto-threshold ImageJ
  DARK_OBJECTS           = True        # organoïdes plus sombres que le fond
"""

import os, sys, argparse
from pathlib import Path
import numpy as np
import cv2

BENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH_ROOT))
from common.seed import well_mask, dark_point_in_well
from common.pick import pick_instance_at_point, fill_holes

AUTO_THRESHOLD_METHOD = "Otsu"     # ImageJ auto-threshold (Default, Otsu, Huang, ...)
DARK_OBJECTS = True                # objets sombres sur fond clair -> "<method> dark"

# Coordonnées Fiji : soit fixées par $FIJI_APP (une install Fiji locale), soit
# pyimagej télécharge Fiji via maven ("sc.fiji:fiji").
FIJI_ENDPOINT = os.environ.get("FIJI_APP", "sc.fiji:fiji")


def imagej_binarize(ctx, frame_gray):
    """Seuillage ImageJ RÉEL -> masque binaire booléen (foreground = objets seuillés).
    ctx = {'ij', 'IJ'} chargé une seule fois par load()."""
    ij, IJ = ctx["ij"], ctx["IJ"]
    imp = ij.py.to_imageplus(frame_gray.astype(np.uint8))
    # setAutoThreshold("<Method> dark") puis Convert to Mask = le workflow Fiji.
    dark = " dark" if DARK_OBJECTS else ""
    IJ.setAutoThreshold(imp, f"{AUTO_THRESHOLD_METHOD}{dark}")
    IJ.run(imp, "Convert to Mask", "")
    arr = np.asarray(ij.py.from_java(imp)).astype(np.float32)
    if arr.ndim > 2:
        arr = arr.squeeze()
    return arr > 127                       # ImageJ Mask : foreground = 255


def load():
    import imagej
    from scyjava import jimport
    print(f"[fiji] init ImageJ ({FIJI_ENDPOINT}) headless ...", flush=True)
    ij = imagej.init(FIJI_ENDPOINT, mode="headless")
    print(f"[fiji] ImageJ version = {ij.getVersion()}", flush=True)
    return {"ij": ij, "IJ": jimport("ij.IJ")}


def segment(ctx, frame_gray):
    binm = imagej_binarize(ctx, frame_gray)

    # polarité : le point sombre est SUR l'organoïde -> il doit être foreground.
    seed_xy = dark_point_in_well(frame_gray)        # (x, y)
    sx, sy = int(seed_xy[0]), int(seed_xy[1])
    if 0 <= sy < binm.shape[0] and 0 <= sx < binm.shape[1] and not binm[sy, sx]:
        binm = ~binm                                # masque inversé -> on remet à l'endroit

    # restreint au puits (comme le pipeline) puis Analyze-Particles-like selection
    disc = well_mask(frame_gray, margin=0.95)
    fg = (binm & disc).astype(np.uint8)
    if fg.sum() == 0:
        return np.zeros(frame_gray.shape, dtype=bool), seed_xy

    n, labels, _, _ = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return np.zeros(frame_gray.shape, dtype=bool), seed_xy

    mask = pick_instance_at_point(labels, seed_xy, shape=frame_gray.shape)
    return fill_holes(mask), seed_xy


if __name__ == "__main__":
    from common.batch import cli
    cli("fiji", load, segment)
