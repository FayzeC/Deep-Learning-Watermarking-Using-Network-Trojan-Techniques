#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --time=360
#SBATCH --job-name=patch_train
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate wanet

PROJECT_ROOT="$HOME/FYP"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

python -m patch_trigger.patch_training \
    --data-root "$PROJECT_ROOT/data" \
    --out-dir "$PROJECT_ROOT/patch_trigger/run/training" \
    --split-file "$PROJECT_ROOT/splits/cifar10_train_val_seed42.pt" \
    --split-seed 42 \
    --val-per-class 500 \
    --epochs 100 \
    --batch-size 128 \
    --num-workers 4 \
    --lr 0.1 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --poison-rate 0.05 \
    --target-label 0 \
    --patch-size 6 \
    --alpha 0.35 \
    --validation-trigger-seed 999 \
    --seed 42 \
    --device cuda
