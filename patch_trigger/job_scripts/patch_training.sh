#! /bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --time=360
#SBATCH --job-name=patch_training
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda
source activate /tc1apps/2_conda_env/CZ4042_v6
python pacth_training.py --device cuda --epochs 100