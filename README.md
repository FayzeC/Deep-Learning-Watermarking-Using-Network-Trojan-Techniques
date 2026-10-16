# Deep Learning Watermarking Using Network Trojan Techniques

This repository contains the completed implementation and evaluation framework for the Final Year Project **Deep Learning Watermarking Using Network Trojan Techniques**.

## 📌 Project Overview

This project explores the adaptation of network Trojan techniques as deep learning watermarks for black-box model ownership verification. Watermark triggers are embedded during training so that the model owner can later verify ownership through a predefined trigger-response behaviour.

The repository implements and compares six watermarking designs across four evaluation dimensions:

- **Effectiveness** — Trigger Success Rate (TSR), trigger-set loss and false-positive baseline.
- **Functionality preservation** — clean classification accuracy and clean loss.
- **Robustness** — fine-tuning, pruning, transfer learning and input transformation attacks.
- **Stealthiness** — PSNR, SSIM.

### Implemented Watermarks

1. **Patch-Based Triggers**
   - Static Visible Patch
   - Alpha Blended Patch
   - Random Location Patch
   - Random Shape Patch

2. **Warping-Based Trigger**

3. **Input-Aware Dynamic Trigger**
   - Input-dependent trigger and mask generation
   - Cross-trigger specificity evaluation

---

## 📁 Repository & Directory Structure

```text
Deep-Learning-Watermarking-Using-Network-Trojan-Techniques/
│
├── false_positive_baseline/             # Results from the vanilla false-positive baseline
├── input_aware/                         # Input-aware dynamic watermark implementation
├── job_scripts/                         # Project-level SLURM jobs
│   ├── false_positive_baseline.sh        # False-positive baseline job script
│   ├── stealthiness_psnr_ssim.sh         # Stealthiness evaluation job script
├── patch_trigger/                       # Patch-based watermark implementations
├── splits/                              # Fixed train/validation split metadata
│   └── cifar10_train_val_seed42.pt
├── stealthiness/                        # PSNR/SSIM evaluation results
├── verification_sets/                   # Shared final verification-set metadata
│   └── cifar10_test_non_target_class0.pt
├── warping_based/                       # Warping-based watermark implementation
│
├── Main Summary of Evaluations.xlsx     # Summary of experimental results
├── README.md
├── common_verification.py               # Shared verification utilities and metrics
├── create_verification_set.py           # Generates the shared 9,000-image verifier indices
├── false_positive_baseline.py           # Vanilla-model false-positive evaluation
├── generate_plots.ipynb                 # Plot and figure generation for report
├── stealthiness_psnr_ssim.py            # PSNR / SSIM evaluation
└── validation_split.py                  # Fixed stratified CIFAR-10 train/validation split
```

Each watermark family follows the same general workflow. The exact filenames may differ slightly between implementations, but the functional structure is:

```text
<trigger_directory>/
│
├── <prefix>_utilities.py                # Trigger generation and shared helper functions
├── <prefix>_training.py                 # Watermark training
├── <prefix>_eval.py                     # Clean accuracy and watermark evaluation
├── <prefix>_finetune.py                 # Fine-tuning robustness evaluation
├── <prefix>_pruning.py                  # Pruning robustness evaluation
├── <prefix>_transfer_cifar100.py        # CIFAR-100 transfer-learning evaluation
├── <prefix>_input_transformation.py     # Input-transformation robustness evaluation
│
├── run/
│   ├── training/                        # Training logs and local model checkpoints
│   ├── evaluation/                      # Final evaluation results
│   ├── finetuning/                      # Fine-tuning outputs
│   ├── pruning/                         # Pruning outputs
│   ├── transfer_learning/               # Transfer-learning outputs
│   └── input_transformation/            # Input-transformation outputs
│
└── job_scripts/
    ├── ...                              # SLURM job submission scripts
    └── output/                          # SLURM stdout/stderr outputs
```

> `<prefix>` refers to the naming convention used by each watermark implementation, such as `patch`, `warping`, or `input_aware`.

The `input_aware/` directory additionally contains `input_aware_cross_trigger.py`, which evaluates input-specific cross-trigger behaviour.

---

## 🧪 Experimental Protocol

The primary experiments use CIFAR-10 with a ResNet-18 adapted for 32×32 images.

- Official CIFAR-10 training set: **50,000 images**
- Fixed stratified split using seed **42**:
  - **45,000 training images**
  - **5,000 validation images** (500 per class)
- Official CIFAR-10 test set: **10,000 images**, reserved for final evaluation
- Watermark target class: **class 0 (`airplane`)**
- Final watermark verifier: **9,000 non-target CIFAR-10 test images**
- Watermarked checkpoints are selected using the harmonic mean of validation clean accuracy and validation TSR.
- The vanilla checkpoint is selected using validation clean accuracy only.
- CIFAR-100 is used for the transfer-learning evaluation.

The same split and final verification population are shared across all watermarking methods.

---

## 🚀 Getting Started

### 1. Environment

The experiments were developed using:

- Python 3.10.20
- PyTorch 2.1.2 + CUDA 11.8
- torchvision 0.16.2 + CUDA 11.8
- NumPy 1.26.4
- Pillow 12.1.1

The experiments were executed on the NTU College of Computing and Data Science GPU Cluster (TC1), but the scripts can be adapted to another CUDA-enabled environment.

### 2. Dataset

CIFAR-10 is used for training and final evaluation, while CIFAR-100 is used for transfer learning. Dataset files are not stored in this repository.

### 3. Validation Split

The project uses the fixed split stored in:

```text
splits/cifar10_train_val_seed42.pt
```

This ensures that every watermarking method uses the same training and validation populations.

### 4. Verification Set

The final ownership-verification population contains all 9,000 CIFAR-10 test images whose original label is not class 0.

To regenerate the verification index file:

```bash
python create_verification_set.py
```

The generated file is stored under:

```text
verification_sets/cifar10_test_non_target_class0.pt
```

### 5. Running Watermark Experiments

Each watermark directory contains its own training, evaluation, robustness and SLURM submission scripts.

Before running a job script, update:

1. **Conda environment name**
2. **Project root and dataset paths**
3. **SLURM resource configuration**, if required by your computing environment

Example workflow:

```bash
cd patch_trigger/job_scripts/

# Submit the relevant training job
sbatch <training_job_script>

# Submit final evaluation
sbatch <evaluation_job_script>

# Submit robustness experiments
sbatch <finetuning_job_script>
sbatch <pruning_job_script>
sbatch <transfer_job_script>
sbatch <input_transformation_job_script>
```

### 6. Project-Level Evaluations

The repository also contains project-level evaluations shared across watermark types:

```bash
python false_positive_baseline.py
python stealthiness_psnr_ssim.py
```

Equivalent SLURM submission scripts are located in the top-level `job_scripts/` directory.

Results are written to:

```text
false_positive_baseline/
stealthiness/
```

---

## 📦 Model Checkpoints

Trained model checkpoints are **not included directly in the GitHub repository** because the `.pt` files exceed GitHub's normal per-file size limit.

Small `.pt` metadata files used for reproducibility, such as the fixed split and verification-set indices, remain included under `splits/` and `verification_sets/`.

For full reproducibility, the trained model checkpoints can be downloaded separately from:

**[Google Drive – Model Checkpoints](https://drive.google.com/drive/folders/1gDuCQz00Wybl37v_aLg1XRKAgw5MqMiu?usp=drive_link)**

After downloading, place the checkpoints back into the corresponding watermark directory's `run/training/` folder.

---

## 📊 Results

The consolidated experimental results are available in:

```text
Main Summary of Evaluations.xlsx
```

This workbook contains the main effectiveness, functionality-preservation and robustness results used in the final report.

Additional experiment-specific outputs are stored within each watermark directory's `run/` subdirectories and in the top-level:

```text
false_positive_baseline/
stealthiness/
```

The plotting notebook used to generate report figures is:

```text
generate_plots.ipynb
```

---

## 🔁 Reproducibility Notes

- Main training seed: **42**
- Train/validation split: fixed and shared across all methods
- Final watermark verification population: fixed 9,000-image non-target test subset
- Test data is used only after checkpoint selection
- Randomised verification triggers use deterministic evaluation settings where required
- Robustness attacks are applied independently from the original retained checkpoint rather than sequentially
