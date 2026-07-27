#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --time=360
#SBATCH --job-name=patch_finetune1e3
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate wanet

PROJECT_ROOT="$HOME/FYP"
cd "$PROJECT_ROOT"

python -m patch_trigger.patch_finetune \
    --ckpt-dir "$PROJECT_ROOT/patch_trigger/run/training" \
    --out-dir "$PROJECT_ROOT/patch_trigger/run/finetuning/lr1e-3" \
    --data-root "$PROJECT_ROOT/data" \
    --verification-index-file "$PROJECT_ROOT/verification_sets/cifar10_test_non_target_class0.pt" \
    --epochs 100 \
    --lr 1e-3 \
    --momentum 0.9 \
    --weight-decay 0 \
    --batch-size 128 \
    --num-workers 4 \
    --device cuda