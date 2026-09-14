# SAM3 Organoid Segmentation

Segmentation of patient-derived organoids in bright-field time-lapse microscopy
with **SAM 3**, plus the full **benchmark** that compares SAM3-AMG against 11
other approaches.

The repository has two independent parts:

- **Part A — Production run** ([below](#part-a--production-run)): segment a whole
  dataset with the selected model (SAM3-AMG + video propagation), as a resumable
  SLURM work queue. This is what you run to produce masks.
- **Part B — Benchmark** ([below](#part-b--benchmark)): reproduce the model
  comparison (Otsu, SAM2/SAM3 variants, µSAM, Cellpose, OrganoID…) on a set of
  expert-annotated organoids.

> **No data or weights are included.** Model weights are downloaded from their
> sources; the dataset and the evaluation annotations are yours to provide.
> Every machine-specific path is read from an environment variable (see the
> *What to modify* tables).

---

## Repository layout

```
run_segmentation.py              # Part A: production pipeline (AMG + video propagation)
focus_filter.py                  # optional post-hoc focus gate (CPU)
run.sh                           # SLURM job: 4 workers, resumable
requirements.txt                 # main "sam3" environment
benchmarks/
  run_benchmark.py               # Part B: benchmark driver (runs each method in its env)
  build_well_pairs.py            # build the evaluation set (image+GT pairs) from your data
  build_pgr_pairs.py
  make_visual.py                 # per-case overlay visuals
  common/                        # metrics, plots, well detection, dark-blob seed
  methods/                       # one run.py per benchmarked method
    fiji/            (Otsu baseline)
    sam2_dark_pointer/ (SAM2, point prompt)
    sam2_center/     (SAM2, well-centre prompt)
    sam2_amg/        (SAM2 automatic mask generation)
    sam3/            (SAM3, text prompt)
    sam3_pointer/    (SAM3, point prompt)
    sam3_text_point/ (SAM3, text + box)
    sam3_amg/        (SAM3 automatic mask generation)  <-- selected model
    microsam/        (µSAM)
    cellpose/        (Cellpose, cyto3)
    cellpose_sam/    (Cellpose-SAM)
    organoid/        (OrganoID)
```

---

## Part A — Production run

### 1. Environment

```bash
conda create -n sam3 python=3.12 && conda activate sam3
# install PyTorch matching your CUDA first: https://pytorch.org
pip install -r requirements.txt
```

### 2. Weights

`facebook/sam3` is **gated** on Hugging Face. Request access on the model page,
then:

```bash
huggingface-cli login                 # paste your HF token
export HF_HOME=~/.cache/huggingface
huggingface-cli download facebook/sam3
```

### 3. Dataset layout

One time-lapse `.tif` per cavity, organised as:

```
DATASET_ROOT/
  <patient>/<well>/<cavity>.tif
```

### 4. What to modify

| Variable            | Where            | Meaning                                              |
|---------------------|------------------|------------------------------------------------------|
| `DATASET_ROOT` (env)| shell / `run.sh` | root of the dataset to segment                       |
| `HF_HOME` (env)     | shell / `run.sh` | Hugging Face cache holding the SAM 3 weights         |
| `CONDA_ENV`         | `run.sh`         | name of the conda env (`sam3`)                       |

Nothing else is hard-coded (`run_segmentation.py` reads `DATASET_ROOT` from the
environment; the model id `facebook/sam3` is resolved through `HF_HOME`).

### 5. Run

```bash
export DATASET_ROOT=/path/to/processed_files_ALL_frames
export HF_HOME=~/.cache/huggingface

# one worker per process; run several in parallel on separate GPUs
python -u run_segmentation.py --queue --worker-id 0
```

On SLURM, launch 4 workers as a resumable array (edit `DATASET_ROOT` and
`CONDA_ENV` in `run.sh` first):

```bash
sbatch run.sh          # resubmit to resume after a walltime kill
```

Workers claim wells atomically, so they run concurrently without collisions and
the job is fully resumable — re-submitting picks up only the wells not done yet.

### 6. Output

Written to `<parent of DATASET_ROOT>/<dataset name>_brightfield_SAM3/`, mirroring
the input tree, per well:

- `<well>.npy` — dict `{cavity: organoid [n_frames, H, W]}`, the organoid in
  **bright field on a black background** (`frame × mask`: original pixels inside
  the organoid, 0 elsewhere) — the ready-to-use final dataset, not a binary mask.
- `<well>.tif` — control grid video (red mask overlay per cavity)

### 7. Focus filter (optional)

Run after segmentation to flag out-of-focus films. CPU only, and re-runnable
with different thresholds without recomputing any mask:

```bash
export DATASET_ROOT=/path/to/processed_files_ALL_frames
python focus_filter.py --thresh 280 --k 15
```

---

## Part B — Benchmark

Compares every method on the same expert-annotated organoids and writes IoU /
Dice / BF1 scores plus per-method plots.

### 1. Methods and their environments

Methods are grouped by the conda environment they need (each has different,
often conflicting dependencies), selected via `run_benchmark.py`'s `--methods`:

| `--methods` name  | Paper method     | Conda env (var)        | Weights / extra                                   |
|-------------------|------------------|------------------------|---------------------------------------------------|
| `fiji`            | Otsu             | `BENCH_PY_AYOUB`       | none                                              |
| `sam2`            | SAM2 (point)     | `BENCH_PY_AYOUB`       | SAM 2 repo + checkpoint in `repo_sam2/`           |
| `sam2_center`     | SAM2-center      | `BENCH_PY_AYOUB`       | idem                                              |
| `sam2_amg`        | SAM2-AMG         | `BENCH_PY_AYOUB`       | idem                                              |
| `cellpose`        | Cellpose (cyto3) | `BENCH_PY_AYOUB`       | `pip install cellpose>=4` (auto-downloads cyto3)  |
| `cellpose_sam`    | Cellpose-SAM     | `BENCH_PY_AYOUB`       | idem (auto-downloads cpsam)                        |
| `microsam`        | µSAM             | `BENCH_PY_MSAM`        | `conda install -c conda-forge micro_sam`          |
| `sam3`            | SAM3-text        | `BENCH_PY_SAM3`        | `facebook/sam3` (HF)                              |
| `sam3_pointer`    | SAM3-point       | `BENCH_PY_SAM3`        | idem                                              |
| `sam3_text_point` | SAM3-text+box    | `BENCH_PY_SAM3`        | idem                                              |
| `sam3_amg`        | **SAM3-AMG**     | `BENCH_PY_SAM3`        | idem                                              |
| `organoid`        | OrganoID         | `BENCH_PY_ORG`         | OrganoID repo (`ORGANOID_REPO`)                   |

The four environments (create the ones you need for the methods you want to run):

- **sam3** — `pip install -r requirements.txt` (used above; serves all `sam3_*`).
- **ayoub** — SAM 2 (clone the SAM 2 repo into `repo_sam2/` + place
  `sam2.1_hiera_large.pt` under `repo_sam2/sam2/checkpoints/`) and `cellpose>=4`.
- **msam** — `micro_sam`.
- **organoid** — the OrganoID repo and its dependencies.

### 2. What to modify (environment variables)

`run_benchmark.py` reads every path from the environment (defaults are
placeholders). Set only what the methods you run need:

| Variable          | Points to                                             |
|-------------------|-------------------------------------------------------|
| `BENCH_PY_AYOUB`  | python of the SAM2/Cellpose/Otsu env                  |
| `BENCH_PY_MSAM`   | python of the µSAM env                                |
| `BENCH_PY_SAM3`   | python of the SAM 3 env                               |
| `BENCH_PY_ORG`    | python of the OrganoID env                            |
| `BENCH_WEIGHTS`   | folder holding cellpose / µSAM / torch caches         |
| `ORGANOID_REPO`   | path to the cloned OrganoID repository                |

Example:

```bash
export BENCH_PY_SAM3=$(conda run -n sam3 which python)
export BENCH_PY_AYOUB=$(conda run -n ayoub which python)
export BENCH_WEIGHTS=/path/to/model_weights
```

### 3. Build the evaluation set

The benchmark scores each method against expert masks. Build the image+GT pairs
from your annotated data into `benchmarks/well_pairs/`:

```bash
cd benchmarks
python build_well_pairs.py       # edit the input paths inside to point at your GT
```

Each pair is `benchmarks/well_pairs/<...>/<key>_img.npy` + `<key>_gt.npy`.

### 4. Run

All methods on the evaluation set:

```bash
cd benchmarks
python run_benchmark.py --pairs
```

**One model at a time** (only that method's env is needed):

```bash
# SAM3-AMG (the selected model)
BENCH_PY_SAM3=$(conda run -n sam3 which python) \
  python run_benchmark.py --pairs --methods sam3_amg

# SAM2-AMG
BENCH_PY_AYOUB=$(conda run -n ayoub which python) \
  python run_benchmark.py --pairs --methods sam2_amg

# a subset
python run_benchmark.py --pairs --methods sam3_amg,sam2_amg,fiji

# isolate a run in its own output folder (no overwrite): results_pairs_<tag>/
python run_benchmark.py --pairs --methods sam3_amg --out mytest
```

### 5. Output

Written under `benchmarks/results_pairs<suffix>/`:

- `summary.json` — per-organoid IoU/Dice/BF1 for every method
- `benchmark_pairs_*.png` — per-method mean-score bar charts
- per-case predicted masks under `benchmarks/results<suffix>/pair_<key>/`

---

## Notes

- The SAM2 methods require the external SAM 2 repository and its checkpoint
  (`repo_sam2/`); they are only needed to reproduce the SAM2 rows of the
  benchmark, not for the production run.
- `sam3_amg` (Part B) and `run_segmentation.py` (Part A) share the exact same
  AMG selection logic (`benchmarks/methods/sam3_amg/run.py`), so the deployed
  model matches the benchmarked one.
