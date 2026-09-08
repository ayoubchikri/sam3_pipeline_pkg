# SAM3-AMG organoid segmentation pipeline

Automatic mask generation on the last frame, followed by backward SAM3 video propagation. Multiple workers share a queue of wells.

## Contents

| File | Purpose |
| --- | --- |
| [`SAM2_segmentation/run_segmentation.py`](SAM2_segmentation/run_segmentation.py) | Main pipeline: SAM3-AMG initialization and video propagation |
| [`SAM2_segmentation/benchmarks/`](SAM2_segmentation/benchmarks/) | Internal dependencies: well detection, AMG mask selection, and batch utility |
| [`SAM2_segmentation/run.sh`](SAM2_segmentation/run.sh) | SLURM launch script to adapt to your cluster |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| `model_weights/hf/` | Model cache to download or provide separately |

This package contains the segmentation pipeline and its internal dependencies. It does not include the full comparative benchmark, `study_pixel.py`, datasets, or model weights. Original multipage TIFF images are required to run segmentation.

## Before you start

- Set `DATASET_ROOT` to the absolute path of your input dataset.
- Set `HF_HOME` to the absolute path of your Hugging Face model cache.
- Adapt the paths, conda environment, and SLURM settings in `run.sh` if using it.
- Check write permissions and available storage in the package directory. On the reference dataset, masks require approximately 470 MB per well, plus the preview TIFFs.

## Installation

From the root of `sam3_pipeline_pkg`, create an environment:

```bash
conda create -n sam3 python=3.12
conda activate sam3
```

Install PyTorch for your server's GPU and CUDA configuration first. PyTorch is not installed by `requirements.txt`. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Model weights

The model is `facebook/sam3`, a gated Hugging Face repository. Downloading requires a Hugging Face account that has accepted the model license.

From the package root:

```bash
export HF_HOME="$PWD/model_weights/hf"
huggingface-cli login
huggingface-cli download facebook/sam3
```

Alternatively, copy an existing Hugging Face cache into `model_weights/hf/`.

## Configure input paths

Run these commands from the package root, replacing the dataset path:

```bash
export HF_HOME="$PWD/model_weights/hf"
export DATASET_ROOT="/PATH/TO/processed_files_ALL_frames"
```

Expected input layout:

```text
processed_files_ALL_frames/
└── patient/
    └── well/
        ├── cavity_1.tif
        └── cavity_2.tif
```

Each cavity file is a multipage TIFF movie. Without `DATASET_ROOT`, the script uses a placeholder path and exits.

## Run segmentation

One process runs one worker. For four parallel workers, allocate four separate GPUs through your scheduler.

```bash
cd SAM2_segmentation
python -u run_segmentation.py --queue --worker-id 0
```

Start additional workers with IDs `1`, `2`, and `3` on separate GPUs. `--worker-id` identifies log messages; it does not select a GPU. Assign GPUs through SLURM or `CUDA_VISIBLE_DEVICES`.

All patients are processed by default. Add `--only CGR` or `--only PGR` to restrict processing to a patient group.

## Adapt `run.sh` for SLURM

The supplied script retains the original DCE cluster configuration. **Adapt it before submitting it on another server.**

| Setting | Required adjustment |
| --- | --- |
| `#SBATCH --partition` | Select your cluster's GPU partition. |
| `#SBATCH --exclude` | Remove or replace the DCE node list. |
| GPU allocation | Add your cluster's request for one GPU per task, such as `#SBATCH --gres=gpu:1` where supported. |
| `--time`, `--cpus-per-task`, `--array=0-3` | Match your permitted resources and worker count. |
| `--output`, `--error` | Replace the log paths and create their directory before submission. Use `%A_%a` to retain logs from separate submissions. |
| Conda activation | Use your conda installation path and environment name. |
| `HF_HOME` | Replace the original path with your absolute model cache path. |
| `DATASET_ROOT` | Add an export pointing to your input dataset. |
| `cd` | Replace the original path with the absolute path to your `SAM2_segmentation` directory. |

Replace the current conda activation with explicit initialization, adapting both values:

```bash
source "/PATH/TO/CONDA/etc/profile.d/conda.sh"
conda activate sam3
```

Add your dataset path to the script:

```bash
export DATASET_ROOT="/PATH/TO/processed_files_ALL_frames"
```

To process only PGR patients, add `--only PGR` to the Python command. The job name `sam3PGR` does not restrict patient selection. The pipeline does not explicitly enable FP16.

## Outputs

Outputs are written under the **package root**, preserving the patient directory structure:

```text
sam3_pipeline_pkg/
└── <dataset_name>_masks_SAM3/
    ├── _claims/
    └── patient/
        ├── well.npy
        └── well.tif
```

- `well.npy`: a dictionary mapping cavity names to boolean mask arrays with shape `[n_frames, H, W]`, at the original image resolution.
- `well.tif`: a preview grid with red mask overlays, resized to 140 pixels per cavity.

The masks are binary arrays. To produce grayscale organoids on a black background, apply them to the original frames. Keep the source TIFFs for this purpose.

The output directory is inside the package, not beside the input dataset. To change it, edit the `root_out` assignment in `main()` in `run_segmentation.py`. The `--in` and `--out` options mentioned in that script's header are not implemented; configure the input through `DATASET_ROOT`.

## Resuming and checking results

Restarting workers resumes wells without both an output `.npy` and `.tif`. Completion is checked by file existence, not file integrity.

- A failed cavity is saved as `None`. If both well output files exist, that well is skipped even when some cavities failed.
- An interruption during writing can leave an incomplete file. Inspect logs and outputs for interrupted wells before restarting.
- A claim becomes reclaimable after three hours without checking whether its worker is still running. If a well can take longer, increase `STALE_SECONDS` in `run_segmentation.py` to avoid concurrent processing of the same well. Only remove a claim manually after confirming that its worker has stopped.
- `--overwrite` has no effect with `--queue`. To retry a failed well, stop the relevant workers, move that well's outputs outside the results directory, and restart the queue.
