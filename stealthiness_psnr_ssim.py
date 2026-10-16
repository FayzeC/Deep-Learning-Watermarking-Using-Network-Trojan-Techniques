"""Quantitative trigger stealthiness evaluation using PSNR and SSIM.

Compares each clean CIFAR-10 verification image with its triggered version on
exactly the same 9,000 non-target test images used by the ownership-verification
protocol.

Metrics are computed in raw image space [0,1], BEFORE classifier normalisation.
For random-location and random-shape patch triggers, the same deterministic
trigger seed used by the standardised evaluation should be supplied (default 999).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from common_verification import (
    VerificationBaseSet,
    make_cifar10_test_dataset,
    make_data_loader,
    write_csv,
)
from patch_trigger.patch_utilities import (
    discover_patch_checkpoints,
    load_patch_checkpoint,
    make_patch_trigger,
    patch_parameters_from_checkpoint,
)
from warping_based.warping_utilities import (
    apply_wanet_trigger,
    load_wanet_grid,
)
from input_aware.input_aware_utilities import (
    MaskGenerator,
    inject_trigger,
    load_best_input_aware_model,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_INDEX_FILE = (
    PROJECT_ROOT / "verification_sets" / "cifar10_test_non_target_class0.pt"
)
DEFAULT_PATCH_TRAINING_DIR = PROJECT_ROOT / "patch_trigger" / "run" / "training"
DEFAULT_WARPING_TRAINING_DIR = PROJECT_ROOT / "warping_based" / "run" / "training"
DEFAULT_INPUT_AWARE_TRAINING_DIR = PROJECT_ROOT / "input_aware" / "run" / "training"
DEFAULT_OUT_DIR = PROJECT_ROOT / "stealthiness"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patch-training-dir",
        type=Path,
        default=DEFAULT_PATCH_TRAINING_DIR,
    )
    parser.add_argument(
        "--warping-training-dir",
        type=Path,
        default=DEFAULT_WARPING_TRAINING_DIR,
    )
    parser.add_argument(
        "--input-aware-training-dir",
        type=Path,
        default=DEFAULT_INPUT_AWARE_TRAINING_DIR,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--verification-index-file",
        type=Path,
        default=DEFAULT_INDEX_FILE,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--deterministic-trigger-seed",
        type=int,
        default=999,
        help=(
            "Seed controlling deterministic random-location/random-shape "
            "verification triggers. Keep this equal to the standardised "
            "evaluation seed."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def psnr_per_image(clean: torch.Tensor, triggered: torch.Tensor) -> torch.Tensor:
    """PSNR in dB for [N,C,H,W] images with data range [0,1]."""
    mse = (clean - triggered).pow(2).mean(dim=(1, 2, 3))
    # Exact equality would produce +inf, which is mathematically correct.
    return torch.where(
        mse == 0,
        torch.full_like(mse, float("inf")),
        10.0 * torch.log10(1.0 / mse),
    )


def _gaussian_window(
    channels: int,
    *,
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords = coords - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_per_image(
    clean: torch.Tensor,
    triggered: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Standard local SSIM with an 11x11 Gaussian window (sigma=1.5).

    Returns one SSIM value per image, averaged across channels and spatial
    positions. Inputs must be raw [0,1] tensors.
    """
    if clean.shape != triggered.shape:
        raise ValueError(
            f"SSIM inputs must have identical shapes, got {clean.shape} and "
            f"{triggered.shape}."
        )
    if clean.ndim != 4:
        raise ValueError("Expected [N,C,H,W] tensors for SSIM.")

    channels = clean.shape[1]
    window = _gaussian_window(
        channels,
        window_size=window_size,
        sigma=sigma,
        device=clean.device,
        dtype=clean.dtype,
    )

    pad = window_size // 2
    clean_pad = F.pad(clean, (pad, pad, pad, pad), mode="reflect")
    trig_pad = F.pad(triggered, (pad, pad, pad, pad), mode="reflect")

    mu_x = F.conv2d(clean_pad, window, groups=channels)
    mu_y = F.conv2d(trig_pad, window, groups=channels)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(clean_pad * clean_pad, window, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(trig_pad * trig_pad, window, groups=channels) - mu_y2
    sigma_xy = F.conv2d(clean_pad * trig_pad, window, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / denominator.clamp_min(1e-12)

    return ssim_map.mean(dim=(1, 2, 3))


def summarise(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().double().cpu()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {
            "mean": float("inf"),
            "std": 0.0,
            "median": float("inf"),
            "min": float("inf"),
            "max": float("inf"),
        }
    return {
        "mean": finite.mean().item(),
        "std": finite.std(unbiased=False).item(),
        "median": finite.median().item(),
        "min": finite.min().item(),
        "max": finite.max().item(),
    }


def append_metrics(
    *,
    method: str,
    clean: torch.Tensor,
    triggered: torch.Tensor,
    base_indices: torch.Tensor,
    per_image_rows: list[dict[str, object]],
    psnr_all: list[torch.Tensor],
    ssim_all: list[torch.Tensor],
) -> None:
    clean = clean.clamp(0.0, 1.0)
    triggered = triggered.clamp(0.0, 1.0)

    psnr = psnr_per_image(clean, triggered)
    ssim = ssim_per_image(clean, triggered)

    psnr_all.append(psnr.detach().cpu())
    ssim_all.append(ssim.detach().cpu())

    for idx, p, s in zip(
        base_indices.detach().cpu().tolist(),
        psnr.detach().cpu().tolist(),
        ssim.detach().cpu().tolist(),
    ):
        per_image_rows.append(
            {
                "trigger_type": method,
                "cifar10_test_index": int(idx),
                "psnr_db": float(p),
                "ssim": float(s),
            }
        )


def evaluate_trigger_batches(
    *,
    method: str,
    loader,
    device: torch.device,
    trigger_batch_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    per_image_rows: list[dict[str, object]],
) -> dict[str, object]:
    psnr_all: list[torch.Tensor] = []
    ssim_all: list[torch.Tensor] = []
    sample_count = 0

    for images, _original_labels, _target_labels, base_indices in loader:
        images = images.to(device, non_blocking=True)
        # Keep indices on CPU: patch placement only needs integer indices, and
        # avoiding GPU .item() calls prevents unnecessary synchronisation.
        triggered = trigger_batch_fn(images, base_indices)

        append_metrics(
            method=method,
            clean=images,
            triggered=triggered,
            base_indices=base_indices,
            per_image_rows=per_image_rows,
            psnr_all=psnr_all,
            ssim_all=ssim_all,
        )
        sample_count += images.size(0)

    psnr_values = torch.cat(psnr_all)
    ssim_values = torch.cat(ssim_all)
    psnr_stats = summarise(psnr_values)
    ssim_stats = summarise(ssim_values)

    return {
        "trigger_type": method,
        "sample_count": sample_count,
        "psnr_mean_db": f"{psnr_stats['mean']:.6f}",
        "psnr_std_db": f"{psnr_stats['std']:.6f}",
        "psnr_median_db": f"{psnr_stats['median']:.6f}",
        "psnr_min_db": f"{psnr_stats['min']:.6f}",
        "psnr_max_db": f"{psnr_stats['max']:.6f}",
        "ssim_mean": f"{ssim_stats['mean']:.6f}",
        "ssim_std": f"{ssim_stats['std']:.6f}",
        "ssim_median": f"{ssim_stats['median']:.6f}",
        "ssim_min": f"{ssim_stats['min']:.6f}",
        "ssim_max": f"{ssim_stats['max']:.6f}",
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    test_set = make_cifar10_test_dataset(args.data_root)
    verification_set = VerificationBaseSet(
        test_set,
        args.verification_index_file,
        target_label=args.target_label,
    )
    if len(verification_set) != 9000:
        raise ValueError(
            f"Expected 9,000 non-target verification images, got {len(verification_set)}."
        )

    verification_loader = make_data_loader(
        verification_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    summary_rows: list[dict[str, object]] = []
    per_image_rows: list[dict[str, object]] = []

    # --------------------------------------------------------------
    # Patch triggers
    # --------------------------------------------------------------
    patch_checkpoints = discover_patch_checkpoints(
        args.patch_training_dir,
        include_vanilla=False,
    )

    for checkpoint_path in patch_checkpoints:
        # Load only to recover the retained checkpoint metadata/configuration.
        unused_model, metadata = load_patch_checkpoint(checkpoint_path, "cpu")
        parameters = patch_parameters_from_checkpoint(metadata, checkpoint_path)
        del unused_model

        if parameters.trigger_type in {"vanilla", "unknown"}:
            continue

        trigger_fn = make_patch_trigger(
            parameters.trigger_type,
            parameters.patch_size,
            parameters.alpha,
            args.deterministic_trigger_seed,
        )

        def patch_batch_fn(
            images: torch.Tensor,
            base_indices: torch.Tensor,
            trigger_fn=trigger_fn,
        ) -> torch.Tensor:
            # The utility is defined per image because random placement depends
            # on the original CIFAR-10 index.
            triggered = [
                trigger_fn(image, int(base_index.item()))
                for image, base_index in zip(images, base_indices)
            ]
            return torch.stack(triggered, dim=0)

        row = evaluate_trigger_batches(
            method=parameters.trigger_type,
            loader=verification_loader,
            device=device,
            trigger_batch_fn=patch_batch_fn,
            per_image_rows=per_image_rows,
        )
        row["trigger_source"] = str(checkpoint_path.resolve())
        row["metric_space"] = "raw_[0,1]_before_normalisation"
        summary_rows.append(row)

    # --------------------------------------------------------------
    # WaNet trigger
    # --------------------------------------------------------------
    warping_grid, grid_metadata, grid_path = load_wanet_grid(
        args.warping_training_dir,
        device,
    )
    grid_target = int(grid_metadata.get("target_label", args.target_label))
    if grid_target != args.target_label:
        raise ValueError(
            f"WaNet grid target label is {grid_target}, expected {args.target_label}."
        )

    def wanet_batch_fn(images: torch.Tensor, _base_indices: torch.Tensor) -> torch.Tensor:
        return apply_wanet_trigger(images, warping_grid)

    row = evaluate_trigger_batches(
        method="wanet_warping",
        loader=verification_loader,
        device=device,
        trigger_batch_fn=wanet_batch_fn,
        per_image_rows=per_image_rows,
    )
    row["trigger_source"] = str(grid_path.resolve())
    row["metric_space"] = "raw_[0,1]_before_normalisation"
    summary_rows.append(row)

    # --------------------------------------------------------------
    # Input-aware dynamic trigger
    # --------------------------------------------------------------
    (
        unused_classifier,
        trigger_generator,
        mask_generator,
        iadb_artifacts,
    ) = load_best_input_aware_model(
        args.input_aware_training_dir,
        device,
        expected_target_label=args.target_label,
    )
    del unused_classifier
    trigger_generator.eval()
    mask_generator.eval()

    @torch.no_grad()
    def iadb_batch_fn(images: torch.Tensor, _base_indices: torch.Tensor) -> torch.Tensor:
        patterns = trigger_generator(images)
        masks = MaskGenerator.sharpen(mask_generator(images))
        return inject_trigger(images, patterns, masks)

    row = evaluate_trigger_batches(
        method="input_aware_dynamic",
        loader=verification_loader,
        device=device,
        trigger_batch_fn=iadb_batch_fn,
        per_image_rows=per_image_rows,
    )
    row["trigger_source"] = str(iadb_artifacts.checkpoint_path.resolve())
    row["metric_space"] = "raw_[0,1]_before_normalisation"
    summary_rows.append(row)

    # --------------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------------
    summary_fields = [
        "trigger_type",
        "sample_count",
        "psnr_mean_db",
        "psnr_std_db",
        "psnr_median_db",
        "psnr_min_db",
        "psnr_max_db",
        "ssim_mean",
        "ssim_std",
        "ssim_median",
        "ssim_min",
        "ssim_max",
        "trigger_source",
        "metric_space",
    ]
    summary_path = args.out_dir / "stealthiness_summary.csv"
    write_csv(summary_path, summary_fields, summary_rows)

    per_image_path = args.out_dir / "stealthiness_per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "trigger_type",
                "cifar10_test_index",
                "psnr_db",
                "ssim",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    print("\nStealthiness evaluation complete")
    print(f"Verification images per method: {len(verification_set)}")
    print("Metrics computed in raw [0,1] image space before normalisation.\n")

    for result in summary_rows:
        print(
            f"{result['trigger_type']}: "
            f"PSNR={float(result['psnr_mean_db']):.2f} ± "
            f"{float(result['psnr_std_db']):.2f} dB | "
            f"SSIM={float(result['ssim_mean']):.4f} ± "
            f"{float(result['ssim_std']):.4f}"
        )

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved per-image metrics: {per_image_path}")


if __name__ == "__main__":
    main()
