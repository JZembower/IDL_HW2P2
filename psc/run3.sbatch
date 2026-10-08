#!/bin/bash
# HW2P2 run three (SE-ResNet-34 + ArcFace + EMA, 70 epochs) as a batch job: no Jupyter or open terminal needed.
# Submit from your home dir:   sbatch psc/run3.sbatch
# Watch:                       squeue -u $USER      tail -f run3-<jobid>.out      (and wandb)
# If it stops before 70 epochs, submit again: it resumes from checkpoints/run3_seresnet34_ema/last.pth
#SBATCH -J run3
#SBATCH -A cis250019p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 8:00:00
#SBATCH -o run3-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /ocean/projects/cis250019p/mzhang23/TA/envs/IDLF26
export PYTHONNOUSERSITE=1
export TQDM_MININTERVAL=60   # one progress line per minute instead of thousands in the log

nvidia-smi
echo "LOCAL=$LOCAL"
python -u psc/train.py
