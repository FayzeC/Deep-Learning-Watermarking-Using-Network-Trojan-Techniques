# Deep Learning Watermarking Using Network Trojan Techniques

> **Status:** ⚠️ **Work in Progress (WIP)**  
> *This repository is actively under development. Experimental results, code modules and evaluation scripts are subject to ongoing updates and refinements.*

---

## 📌 Project Overview

This project explores the adaptation of Network Trojan (Backdoor) techniques as a mechanism for Deep Learning Watermarking. By embedding backdoor triggers into deep neural networks during training, model owners can verify ownership by demonstrating that the watermarked model behaves in a pre-defined, deterministic manner when presented with trigger-embedded inputs.

The repository implements, evaluates and compares multiple watermarking trigger designs to determine effectiveness, functionality preservation and robustness against fine-tuning, pruning, transfer learning and input transformations attacks.

### Implemented Watermarks

1. **Patch-Based Triggers:**
   - Static Visible Patch
   - Alpha-Blended Patch
   - Random Shape Patch
   - Random Location Patch
2. **Warping-Based Triggers:**
   - Spatial transformation-based backdoors (e.g., WaNet warping fields).
3. **Input-Aware Dynamic Triggers:**
   - Dynamic trigger patterns generated conditionally on specific input features.

## 📁 Repository & Directory Structure

The project follows a consistent architecture across all watermark trigger designs. Each trigger family directory (`patch_trigger/`, `warping_based/`, and `input_aware/`) mirrors the same functional workflow.

```text
Deep Learning Watermarking Using Network Trojan Techniques/
│
├── README.md                           # Main repository documentation
├── create_verification_set.py          # Script to generate shared evaluation subset
├── common_verification.py              # Shared verification logic & metrics
├── Main Summary of Evaluations.xlsx    # Master evaluation summary sheet
├── generate_plots.ipynb                # Jupyter Notebook for report graph generation
│
├── verification_sets/
│   └── cifar10_test_non_target_class0.pt  # Pre-computed verification set indices
│
├── data/                               # Dataset storage
│
├── patch_trigger/                      # Patch-based watermarking suite
├── warping_based/                      # Warping-based watermarking suite
└── input_aware/                        # Input-aware dynamic watermarking suite
```

Each watermark directory (`patch_trigger/`, `warping_based/`, `input_aware/`) adheres to the standardised structure detailed below:

```text
<trigger_directory>/
├── <prefix>_utilities.py               # Trigger generation, data loaders, helper functions
├── <prefix>_training.py                # Model training script with trigger embedding
├── <prefix>_eval.py                    # Evaluation script for Clean Accuracy and Trigger Success Rate
├── <prefix>_finetune.py                # Robustness test: Fine-tuning attack evaluation
├── <prefix>_pruning.py                 # Robustness test: Pruning attack
├── <prefix>_transfer_cifar100.py       # Robustness test: Transfer learning on CIFAR-100
├── <prefix>_input_transformation.py    # Robustness test: Various input transformations
├── <prefix>_visualise_triggers.py      # Utility script to inspect and save trigger visuals
│
├── run/                                # Output artifacts directory
│   ├── training/                       # Saved model checkpoints & training logs
│   ├── evaluation/                     # Evaluation results and metric CSVs
│   ├── finetuning/                     # Post-fine-tuning evaluation outputs
│   ├── pruning/                        # Post-pruning evaluation outputs
│   ├── transfer_learning/              # Transfer learning evaluation outputs
│   └── input_transformation/           # Transformation test outputs
│
├── trigger_visuals/                    # Output directory for visual inspection images
└── job_scripts/                        # SLURM job submission scripts
```

> *Note: `<prefix>` refers to the module naming convention (`patch`, `warping`, or `input_aware`).*  
> *The `input_aware/` folder includes an additional file: `input_aware_cross_trigger.py` for evaluating cross-trigger specificity*

---
## 🚀 Getting Started & Execution Workflow

All training, evaluation and robustness experiments are executed using SLURM batch job scripts located inside the `job_scripts/` subdirectory of each trigger family.

### 1. Initialize Verification Set
Before submitting any job scripts, generate the standard verification indices once locally:
```bash
python create_verification_set.py
```

### 2. Prerequisites & Environment Setup
Experiments are run inside a Conda environment containing all necessary dependencies such as PyTorch, Torchvision, NumPy, Pandas, etc. 

By default, the provided job scripts load a Conda environment named **`wanet`**. However, **you can set up your own Conda environment under any name you prefer.** If you use a custom environment name other than wanet, simply update the conda activate <env_name> line inside the batch scripts before submitting them.

### 3. Job Script Execution
To run an experiment, navigate to the target trigger's `job_scripts/` directory and submit the relevant script:
```bash
# Example: Executing Patch-Based Trigger Workflow
cd patch_trigger/job_scripts/

# Submit training job
sbatch patch_training.sh

# Submit evaluation job
sbatch patch_eval.sh

# Submit robustness test jobs (Fine-tuning, Pruning, Transformations)
sbatch patch_finetune.sh
sbatch patch_pruning.sh
sbatch patch_input_transformation.sh
```
⚠️ Configuring Job Scripts Before Running:

Before running or submitting any job scripts, make sure to open and edit them to match your system setup:
1. Conda Environment Name: Ensure conda activate <your_env_name> matches the Conda environment you created.
2. Directory Paths: Update path variables (e.g., PROJECT_ROOT="$HOME/FYP") and flag parameters (--data-root, --verification-index-file, etc.) to point to your actual local or cluster file locations.