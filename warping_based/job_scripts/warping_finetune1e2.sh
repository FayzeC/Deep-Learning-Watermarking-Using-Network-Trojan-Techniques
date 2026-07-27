#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --time=360
#SBATCH --job-name=warping_finetune1e2
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate wanet

PROJECT_ROOT="$HOME/FYP"

cd "$PROJECT_ROOT"

python -m warping_based.warping_finetune \
    --lr 1e-2 \
    --epochs 100 \
    --out-dir ./warping_based/run/finetuning/lr1e-2 \
    --device cuda \
    --batch-size 128 \
    --num-workers 4