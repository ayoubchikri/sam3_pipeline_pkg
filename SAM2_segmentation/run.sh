#!/bin/bash
#SBATCH --job-name=sam3PGR
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --exclude=sh00,sh[10-19]
#SBATCH --array=0-3
#SBATCH --output=/usr/users/projets_p15_igr/chikri_ayo/stage_segmentation/logs/worker_%a.out
#SBATCH --error=/usr/users/projets_p15_igr/chikri_ayo/stage_segmentation/logs/worker_%a.err

# 4 workers en file d'attente partagée : chacun prend le prochain puits PGR non fait
# (claim atomique), le traite (SAM3-AMG fp16 + propagation + focus gate), passe au suivant.
# QOS = 4 jobs simultanés. Reprise sûre : resoumets ce script en boucle jusqu'à la fin.
# .out/.err à nom fixe (worker_0..3) -> écrasés à chaque resoumission (avancement clair).

source ~/.bashrc
conda activate bench312
export HF_HOME=/usr/users/projets_p15_igr/chikri_ayo/stage_segmentation/model_weights/hf
export OMP_NUM_THREADS=4

cd /usr/users/projets_p15_igr/chikri_ayo/stage_segmentation/SAM2_segmentation

python -u run_segmentation.py --queue --worker-id ${SLURM_ARRAY_TASK_ID}
