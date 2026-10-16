#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=60
#SBATCH --job-name=stealthiness
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate wanet

PROJECT_ROOT="$HOME/FYP"
cd "$PROJECT_ROOT"

python stealthiness_psnr_ssim.py \
    --patch-training-dir "$PROJECT_ROOT/patch_trigger/run/training" \
    --warping-training-dir "$PROJECT_ROOT/warping_based/run/training" \
    --input-aware-training-dir "$PROJECT_ROOT/input_aware/run/training" \
    --data-root "$PROJECT_ROOT/data" \
    --verification-index-file "$PROJECT_ROOT/verification_sets/cifar10_test_non_target_class0.pt" \
    --out-dir "$PROJECT_ROOT/stealthiness" \
    --target-label 0 \
    --batch-size 128 \
    --num-workers 4 \
    --deterministic-trigger-seed 999 \
    --device cuda
