"""
batch.py — CLI partagée des méthodes, avec chargement du modèle UNE SEULE FOIS.

Chaque methods/<m>/run.py expose :
    load()              -> ctx        # charge le modèle (coûteux) une fois
    segment(ctx, gray)  -> mask  OU  (mask, seed)

et finit par :  from common.batch import cli; cli("<tag>", load, segment)

Modes :
  --input X --output Y     # une frame
  --manifest pairs.json    # [[in, out], ...] -> modèle chargé une fois, toutes traitées
"""

import argparse, json
import numpy as np
import cv2


def _gray(path):
    g = np.load(path)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    return g.astype(np.uint8)


def cli(tag, load, segment):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--manifest", help="JSON [[input, output], ...] -> modèle chargé une seule fois")
    args = ap.parse_args()

    pairs = json.load(open(args.manifest)) if args.manifest else [[args.input, args.output]]
    ctx = load()                                       # <-- une seule fois
    for i, (inp, out) in enumerate(pairs, 1):
        try:
            res = segment(ctx, _gray(inp))
            mask = res[0] if isinstance(res, tuple) else res
            mask = np.asarray(mask).astype(bool)
        except Exception as e:
            print(f"[{tag}] ERROR {inp}: {e}", flush=True)
            continue
        np.save(out, mask)
        print(f"[{tag}] [{i}/{len(pairs)}] px={int(mask.sum())} -> {out}", flush=True)
