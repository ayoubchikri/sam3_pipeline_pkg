#!/usr/bin/env python3
"""
Segmentation SAM3 du dataset : masque initial par sam3_amg (IDENTIQUE au benchmark,
garde-fous [200,20000] + fallback stability), puis propagation vidéo native SAM3
(Sam3TrackerVideo) à partir de ce masque.

Sortie = miroir de l'arborescence du dataset, SAUF que chaque dossier de PUITS
(le dossier contenant les .tif) devient UN SEUL .npy :
    <...>/<puits>/                 (dossier de tifs, dans le dataset)
    <...>/<puits>.npy              (1 npy = dict {nom_cavité: organoïde_bright_field [n_frames,H,W]})

Le contenu n'est PAS le masque binaire mais l'organoïde en BRIGHT-FIELD sur fond
noir (frame * masque : pixels originaux dans l'organoïde, 0 ailleurs) — le dataset
final directement exploitable. Un .tif grille de contrôle (overlay rouge) est aussi écrit.

Le masque initial sam3_amg est cherché sur la DERNIÈRE frame de chaque cavité
(comme SAM2 avant), puis propagé en arrière sur toutes les frames.

Env: bench312 (GPU) + transformers (Sam3 + Sam3TrackerVideo) + scikit-image + cv2.
     HF_HOME configuré vers les poids gated.

  python run_segmentation.py --in <racine_dataset> --out <racine_sortie> [--overwrite]
"""

import os, sys, re, math
from pathlib import Path
import numpy as np
import cv2
import tifffile
from tqdm import tqdm

# --- viz grille ---
CELL_PX = 140          # taille (px) d'une cavité dans la grille
PAD = 6                # marge autour de chaque cellule
LABEL_H = 22           # bande sous la cellule pour le numéro

HERE = Path(__file__).resolve().parent
BENCH = HERE / "benchmarks"
# on réutilise EXACTEMENT la logique AMG du benchmark (find_best_amg_mask + score_mask + seuils)
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "methods" / "sam3_amg"))
from run import find_best_amg_mask                       # noqa: E402  (methods/sam3_amg/run.py)

MODEL_ID = "facebook/sam3"
# dataset à segmenter (racine à parcourir) — modifie ici si besoin
DATASET_ROOT = os.environ.get("DATASET_ROOT", "/path/to/processed_files_ALL_frames")


# ----------------------------------------------------------------------------- I/O tif
def _to_uint8_gray(frame):
    if frame.ndim == 3:
        frame = frame[..., 0] if frame.shape[-1] <= 4 else frame[0]
    if frame.dtype != np.uint8:
        lo, hi = float(frame.min()), float(frame.max())
        frame = ((frame - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    return frame


def load_frames(tif_path):
    """Retourne (frames_rgb [liste de HxWx3 uint8], H, W).
    Lecture via tifffile (rapide, ~5x PIL) ; PIL en secours."""
    try:
        arr = np.asarray(tifffile.imread(tif_path))
    except Exception:
        from PIL import Image, ImageSequence
        arr = np.stack([np.asarray(p) for p in ImageSequence.Iterator(Image.open(tif_path))])
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 4:                                     # (n, H, W, C) -> canal 0
        arr = arr[..., 0]
    frames = []
    for f in arr:
        g = _to_uint8_gray(f)
        frames.append(np.stack([g] * 3, axis=-1))
    h, w = frames[0].shape[:2]
    return frames, h, w


# ----------------------------------------------------------------------------- SAM3
def load_models():
    import torch
    from transformers import pipeline, Sam3TrackerVideoProcessor, Sam3TrackerVideoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_idx = 0 if device == "cuda" else -1
    print(f"[sam3] chargement AMG + TrackerVideo | device={device}", flush=True)
    amg = pipeline("mask-generation", model=MODEL_ID, device=dev_idx)
    processor = Sam3TrackerVideoProcessor.from_pretrained(MODEL_ID)
    model = Sam3TrackerVideoModel.from_pretrained(MODEL_ID).to(device)
    return {"amg": amg, "processor": processor, "model": model, "device": device}


def segment_cavity(ctx, frames, h, w):
    """sam3_amg sur la dernière frame -> propagation vidéo. Retourne masks [n_frames,H,W] bool."""
    import torch
    processor, model, device = ctx["processor"], ctx["model"], ctx["device"]
    n = len(frames)
    prompt_idx = n - 1
    gray_prompt = frames[prompt_idx][..., 0]                       # HxW uint8

    best_seg = find_best_amg_mask(gray_prompt, ctx["amg"])         # IDENTIQUE au benchmark
    stack = np.zeros((n, h, w), dtype=bool)
    if best_seg is None:
        return stack

    session = processor.init_video_session(video=frames, inference_device=device)
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=prompt_idx,
        obj_ids=1,
        input_masks=best_seg[None].astype(np.float32),            # (1, H, W)
        original_size=(h, w),
    )
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(session, start_frame_idx=prompt_idx, reverse=True):
            m = processor.post_process_masks(
                [out.pred_masks], original_sizes=[[h, w]], binarize=True)[0]
            if hasattr(m, "detach"):                                # tensor (souvent CUDA)
                m = m.detach().cpu().numpy()
            arr = np.asarray(m).astype(bool)
            while arr.ndim > 2:                                    # -> HxW (1 objet)
                arr = arr[0]
            stack[out.frame_idx] = arr
    return stack


# ----------------------------------------------------------------------------- viz grille
def cavity_number(stem):
    """Numéro de la cavité = derniers chiffres du nom du tif (ex: ..._46 -> '46')."""
    m = re.search(r"(\d+)\s*$", stem)
    return m.group(1) if m else stem


def _overlay_cell(gray, mask):
    """HxW gris + masque rouge -> cellule RGB uint8 redimensionnée à CELL_PX."""
    rgb = np.stack([gray] * 3, -1).astype(np.float32)
    if mask is not None and mask.any():
        rgb[mask] = 0.5 * rgb[mask] + 0.5 * np.array([255, 0, 0], np.float32)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.resize(rgb, (CELL_PX, CELL_PX), interpolation=cv2.INTER_AREA)


def make_grid_tif(cells, out_tif):
    """cells = liste de (frames_gris, mask [n,H,W] bool, label str).
    Vidéo-grille : chaque cavité, seg en rouge, numéro dessous,
    toutes jouées en même temps (cavités plus courtes gelées sur leur dernière frame)."""
    n = len(cells)
    if n == 0:
        return
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    T = max(len(c[0]) for c in cells)
    tileW, tileH = CELL_PX + 2 * PAD, CELL_PX + 2 * PAD + LABEL_H
    gridH, gridW = rows * tileH, cols * tileW
    video = np.zeros((T, gridH, gridW, 3), np.uint8)

    for i, (frames, mask, label) in enumerate(cells):
        r, c = divmod(i, cols)
        y0, x0 = r * tileH + PAD, c * tileW + PAD
        nf = len(frames)
        for t in range(T):
            ft = min(t, nf - 1)                                    # gèle sur la dernière frame
            g = frames[ft]
            m = mask[ft] if (mask is not None and ft < len(mask)) else None
            video[t, y0:y0 + CELL_PX, x0:x0 + CELL_PX] = _overlay_cell(g, m)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = x0 + max(0, (CELL_PX - tw) // 2)
        ty = y0 + CELL_PX + LABEL_H - 6
        for t in range(T):
            cv2.putText(video[t], label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
    tifffile.imwrite(str(out_tif), video)                          # (T, H, W, 3) uint8


# ----------------------------------------------------------------------------- walk dataset
def process_well(ctx, well_dir, out_npy, out_tif):
    tifs = sorted((p for p in Path(well_dir).iterdir()
                   if p.suffix.lower() in (".tif", ".tiff") and not p.name.startswith("._")),
                  key=lambda p: int(re.search(r"(\d+)\s*$", p.stem).group(1)) if re.search(r"(\d+)\s*$", p.stem) else 0)
    out_data, cells = {}, []
    for tif in tifs:
        try:
            frames, h, w = load_frames(tif)
            stack = segment_cavity(ctx, frames, h, w)              # masque booléen [n,H,W]
            gray = [f[..., 0] for f in frames]                     # bright-field (niveaux de gris)
            # dataset final = organoïde en bright-field sur fond noir (frame * masque)
            n = min(len(gray), stack.shape[0])
            bf = np.zeros((n, h, w), dtype=gray[0].dtype)
            for t in range(n):
                if stack[t].any():
                    bf[t][stack[t]] = gray[t][stack[t]]
            out_data[tif.stem] = bf
            cells.append((gray, stack, cavity_number(tif.stem)))   # grille de contrôle = overlay rouge
        except Exception as e:
            import traceback
            print(f"    [{tif.name}] ERREUR: {e}", flush=True)
            traceback.print_exc()
            out_data[tif.stem] = None
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, np.array(out_data, dtype=object), allow_pickle=True)
    make_grid_tif(cells, out_tif)
    print(f"  -> {out_npy.name} + {out_tif.name}  ({len(out_data)} cavités)", flush=True)


def _patient_id(well, root_in):
    return well.relative_to(root_in).parts[0].split("_")[0]   # ex: CGR0108_0_215 -> CGR0108


def run_dataset(root_in, root_out, skip_existing=True, shard=0, nshards=1, only=None, patient_index=None):
    root_in, root_out = Path(root_in), Path(root_out)
    wells = []
    for cur, dirs, files in os.walk(root_in):
        if any(f.lower().endswith((".tif", ".tiff")) and not f.startswith("._") for f in files):
            wells.append(Path(cur))
    wells.sort()                                            # ordre stable
    if only:                                                # ne garde que les patients d'un type (ex: PGR)
        wells = [w for w in wells if _patient_id(w, root_in).startswith(only)]
    patients = sorted({_patient_id(w, root_in) for w in wells})
    if patient_index is not None:                          # 1 job = 1 patient (job array)
        if patient_index >= len(patients):
            print(f"== patient-index {patient_index} >= {len(patients)} patients -> rien à faire ==", flush=True)
            return
        pid = patients[patient_index]
        wells = [w for w in wells if _patient_id(w, root_in) == pid]
        print(f"== patient {pid} : {len(wells)} puits ==", flush=True)
    elif nshards > 1:                                       # sinon : sharding par modulo
        wells = [w for i, w in enumerate(wells) if i % nshards == shard]
    print(f"== {len(wells)} puits à traiter (shard {shard}/{nshards}) sous {root_in} ==", flush=True)

    ctx = load_models()
    for well_dir in tqdm(wells, desc="puits"):
        rel = well_dir.relative_to(root_in)
        out_npy = root_out / rel.parent / f"{rel.name}.npy"        # dossier puits -> <puits>.npy
        out_tif = root_out / rel.parent / f"{rel.name}.tif"        # + vidéo-grille
        if skip_existing and out_npy.exists() and out_tif.exists():
            continue
        print(f"[puits] {rel}", flush=True)
        process_well(ctx, well_dir, out_npy, out_tif)


STALE_SECONDS = 3 * 3600        # claim sans sortie plus vieux que ça (worker tué) -> repris


def _outputs(root_out, rel):
    base = root_out / rel.parent
    return base / f"{rel.name}.npy", base / f"{rel.name}.tif"


def run_queue(root_in, root_out, only=None, worker=0):
    """File d'attente partagée : chaque worker prend le prochain puits non fait et non
    réservé (claim ATOMIQUE via os.mkdir), le traite, puis passe au suivant.
      - fini       = npy+tif présents -> sauté (reprise sûre)
      - en cours   = .lock récent -> sauté par les autres (pas de collision)
      - abandonné  = .lock > STALE sans sortie (worker tué) -> nettoyé et repris
    Resoumets le sbatch autant de fois que nécessaire jusqu'à ce que tout soit fait."""
    import time
    root_in, root_out = Path(root_in), Path(root_out)
    wells = []
    for cur, dirs, files in os.walk(root_in):
        if any(f.lower().endswith((".tif", ".tiff")) and not f.startswith("._") for f in files):
            wells.append(Path(cur))
    wells.sort()
    if only:
        wells = [w for w in wells if _patient_id(w, root_in).startswith(only)]
    total = len(wells)
    claims = root_out / "_claims"; claims.mkdir(parents=True, exist_ok=True)

    ctx, processed = None, 0
    for well in wells:
        rel = well.relative_to(root_in)
        out_npy, out_tif = _outputs(root_out, rel)
        if out_npy.exists() and out_tif.exists():
            continue                                           # déjà fait
        lock = claims / (str(rel).replace("/", "__") + ".lock")
        if lock.exists():                                      # nettoyage d'un claim périmé
            try:
                if time.time() - lock.stat().st_mtime > STALE_SECONDS:
                    os.rmdir(lock)
            except OSError:
                pass
        try:
            os.mkdir(lock)                                     # réservation ATOMIQUE
        except FileExistsError:
            continue                                           # un autre worker s'en occupe
        try:
            if ctx is None:
                ctx = load_models()                            # modèle chargé au 1er puits pris
            done_cnt = sum(1 for _ in root_out.glob("**/*.tif"))
            print(f"[w{worker}] START {rel}   (global {done_cnt}/{total} faits)", flush=True)
            process_well(ctx, well, out_npy, out_tif)
            processed += 1
        except Exception as e:
            import traceback
            print(f"[w{worker}] ERREUR {rel}: {e}", flush=True); traceback.print_exc()
        finally:
            try:
                os.rmdir(lock)                                 # libère (fini ou échoué)
            except OSError:
                pass
    done_cnt = sum(1 for _ in root_out.glob("**/*.tif"))
    print(f"[w{worker}] FIN — ce worker a traité {processed} puits | global {done_cnt}/{total}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="re-traiter même si le .npy existe")
    ap.add_argument("--shard", default="0/1", help="i/N : sharding par modulo (si pas de --patient-index)")
    ap.add_argument("--only", default=None, help="ne traiter que les patients d'un type (ex: PGR, CGR)")
    ap.add_argument("--patient-index", type=int, default=None,
                    help="1 job = 1 patient : traite le K-ème patient (trié) parmi ceux filtrés")
    ap.add_argument("--list-patients", action="store_true", help="liste les patients (avec --only) et sort")
    ap.add_argument("--queue", action="store_true", help="mode file d'attente partagée (4 workers)")
    ap.add_argument("--worker-id", type=int, default=0, help="id du worker (SLURM_ARRAY_TASK_ID)")
    a = ap.parse_args()
    if not os.path.isdir(DATASET_ROOT):
        print(f"ERREUR: dossier introuvable: {DATASET_ROOT}"); sys.exit(1)
    root_out = str(HERE.parent / f"{Path(DATASET_ROOT).name}_brightfield_SAM3")

    if a.queue:                                            # file d'attente : chaque worker prend le prochain puits
        print(f"[dataset] {DATASET_ROOT}\n[sortie]  {root_out}\n[filtre]  {a.only or 'tous'}"
              f"  [worker]  {a.worker_id}", flush=True)
        run_queue(DATASET_ROOT, root_out, only=a.only, worker=a.worker_id)
        return

    if a.list_patients:                                    # utilitaire : combien de patients / l'array
        R = Path(DATASET_ROOT); ws = []
        for cur, _, files in os.walk(R):
            if any(f.lower().endswith((".tif", ".tiff")) and not f.startswith("._") for f in files):
                ws.append(Path(cur))
        pats = sorted({_patient_id(w, R) for w in ws if (not a.only) or _patient_id(w, R).startswith(a.only)})
        print(f"{len(pats)} patients ({a.only or 'tous'}) -> array 0-{len(pats)-1} :\n" + " ".join(pats))
        return

    shard, nshards = (int(x) for x in a.shard.split("/"))
    print(f"[dataset] {DATASET_ROOT}\n[sortie]  {root_out}\n[filtre]  {a.only or 'tous'}"
          f"  [patient-index]  {a.patient_index}", flush=True)
    run_dataset(DATASET_ROOT, root_out, skip_existing=not a.overwrite,
                shard=shard, nshards=nshards, only=a.only, patient_index=a.patient_index)


if __name__ == "__main__":
    main()
