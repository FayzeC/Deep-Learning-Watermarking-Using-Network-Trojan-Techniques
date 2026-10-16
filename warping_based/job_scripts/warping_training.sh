#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --time=360
#SBATCH --job-name=warp_train
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate wanet

PROJECT_ROOT="$HOME/FYP"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

python -m warping_based.warping_training \
    --data-dir "$PROJECT_ROOT/data" \
    --save-dir "$PROJECT_ROOT/warping_based/run/training" \
    --split-file "$PROJECT_ROOT/splits/cifar10_train_val_seed42.pt" \
    --split-seed 42 \
    --val-per-class 500 \
    --epochs 100 \
    --batch-size 128 \
    --num-workers 4 \
    --lr 0.1 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --target-label 0 \
    --poison-rate 0.10 \
    --noise-rate 0.20 \
    --grid-k 4 \
    --grid-s 0.5 \
    --noise-s 0.5 \
    --seed 42 \
    --device cuda
