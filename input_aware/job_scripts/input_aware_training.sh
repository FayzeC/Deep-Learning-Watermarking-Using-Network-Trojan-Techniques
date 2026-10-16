#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --time=360
#SBATCH --job-name=iadb_train
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate wanet

PROJECT_ROOT="$HOME/FYP"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

python -m input_aware.input_aware_training \
    --data-root "$PROJECT_ROOT/data" \
    --out-dir "$PROJECT_ROOT/input_aware/run/training" \
    --split-file "$PROJECT_ROOT/splits/cifar10_train_val_seed42.pt" \
    --split-seed 42 \
    --val-per-class 500 \
    --epochs 100 \
    --batch-size 128 \
    --num-workers 4 \
    --lr 0.1 \
    --gen-lr 0.01 \
    --mask-lr 0.01 \
    --target-label 0 \
    --p-attack 0.10 \
    --p-cross 0.10 \
    --lambda-div 1.0 \
    --lambda-mask-norm 100.0 \
    --mask-density 0.032 \
    --mask-warmup-epochs 5 \
    --seed 42 \
    --device cuda
