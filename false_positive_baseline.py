"""False-positive baseline for watermark ownership verification.

Evaluates the validation-selected vanilla ResNet-18 on the same 9,000-image
non-target CIFAR-10 verification set used by the normal watermark evaluation.

For each trigger family, the script reports:
  * vanilla target-class activation rate (false-positive baseline),
  * corresponding watermarked-model TSR,
  * verification margin = watermarked TSR - vanilla activation rate.

It also reports the vanilla model's natural target-class activation rate on the
clean 9,000 non-target images before any trigger is applied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset

from common_verification import (
    TriggeredVerificationSet,
    VerificationBaseSet,
    evaluate_pair_loader,
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
    WarpingVerificationSet,
    apply_wanet_trigger,
    load_wanet_artifacts,
    make_cifar10_normalize,
    make_trigger_forward as make_wanet_trigger_forward,
)
from input_aware.input_aware_utilities import (
    InputAwareVerificationSet,
    load_best_input_aware_model,
    make_trigger_forward as make_iadb_trigger_forward,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_INDEX_FILE = (
    PROJECT_ROOT / "verification_sets" / "cifar10_test_non_target_class0.pt"
)
DEFAULT_PATCH_TRAINING_DIR = PROJECT_ROOT / "patch_trigger" / "run" / "training"
DEFAULT_WARPING_TRAINING_DIR = PROJECT_ROOT / "warping_based" / "run" / "training"
DEFAULT_INPUT_AWARE_TRAINING_DIR = PROJECT_ROOT / "input_aware" / "run" / "training"
DEFAULT_OUT_DIR = PROJECT_ROOT / "false_positive_baseline"


class TargetLabelView(Dataset):
    """Return clean verification images with the watermark target as the label.
    Accuracy on this view is therefore the natural target-class activation rate,
    not ordinary clean classification accuracy.
    """

    def __init__(self, verification_set: VerificationBaseSet) -> None:
        self.verification_set = verification_set

    def __len__(self) -> int:
        return len(self.verification_set)

    def __getitem__(self, item: int):
        image, _original_label, target_label, _base_index = self.verification_set[item]
        return image, target_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patch-training-dir", type=Path, default=DEFAULT_PATCH_TRAINING_DIR
    )
    parser.add_argument(
        "--warping-training-dir", type=Path, default=DEFAULT_WARPING_TRAINING_DIR
    )
    parser.add_argument(
        "--input-aware-training-dir",
        type=Path,
        default=DEFAULT_INPUT_AWARE_TRAINING_DIR,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--verification-index-file", type=Path, default=DEFAULT_INDEX_FILE
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
            "Use the same value as patch_eval.py so random-location and "
            "random-shape verification inputs are identical."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def find_vanilla_checkpoint(training_dir: Path) -> Path:
    checkpoints = discover_patch_checkpoints(training_dir, include_vanilla=True)
    vanilla = [path for path in checkpoints if "vanilla" in path.name.lower()]
    if len(vanilla) != 1:
        raise RuntimeError(
            f"Expected exactly one vanilla checkpoint in {training_dir}, "
            f"found {len(vanilla)}: {[p.name for p in vanilla]}"
        )
    return vanilla[0]


def make_row(
    *,
    trigger_type: str,
    vanilla_activation: float,
    vanilla_target_loss: float,
    natural_activation: float,
    watermarked_tsr: float | None,
    watermarked_trigger_loss: float | None,
    sample_count: int,
    target_label: int,
    vanilla_checkpoint: Path,
    trigger_source: str,
    note: str,
) -> dict[str, object]:
    margin = (
        None if watermarked_tsr is None else watermarked_tsr - vanilla_activation
    )
    return {
        "trigger_type": trigger_type,
        "vanilla_target_activation_percent": f"{vanilla_activation:.6f}",
        "vanilla_target_loss": f"{vanilla_target_loss:.6f}",
        "natural_vanilla_target_activation_percent": f"{natural_activation:.6f}",
        "trigger_induced_increase_pp": f"{vanilla_activation - natural_activation:.6f}",
        "watermarked_tsr_percent": (
            "" if watermarked_tsr is None else f"{watermarked_tsr:.6f}"
        ),
        "watermarked_trigger_loss": (
            ""
            if watermarked_trigger_loss is None
            else f"{watermarked_trigger_loss:.6f}"
        ),
        "verification_margin_pp": "" if margin is None else f"{margin:.6f}",
        "sample_count": sample_count,
        "target_label": target_label,
        "vanilla_checkpoint": str(vanilla_checkpoint.resolve()),
        "trigger_source": trigger_source,
        "note": note,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Common official CIFAR-10 test verification population.
    test_set = make_cifar10_test_dataset(args.data_root)
    verification_base = VerificationBaseSet(
        test_set,
        args.verification_index_file,
        target_label=args.target_label,
    )
    if len(verification_base) != 9000:
        raise ValueError(
            f"Expected 9,000 non-target verification images, got {len(verification_base)}."
        )

    # Load the new validation-selected vanilla checkpoint.
    vanilla_checkpoint = find_vanilla_checkpoint(args.patch_training_dir)
    vanilla_model, _ = load_patch_checkpoint(vanilla_checkpoint, device)
    vanilla_model.eval()

    results: list[dict[str, object]] = []

    # Natural target activation on CLEAN non-target inputs.
    clean_target_loader = make_data_loader(
        TargetLabelView(verification_base),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    natural = evaluate_pair_loader(vanilla_model, clean_target_loader, device)
    natural_activation = natural.accuracy

    results.append(
        make_row(
            trigger_type="none_clean_non_target",
            vanilla_activation=natural_activation,
            vanilla_target_loss=natural.loss,
            natural_activation=natural_activation,
            watermarked_tsr=None,
            watermarked_trigger_loss=None,
            sample_count=natural.sample_count,
            target_label=args.target_label,
            vanilla_checkpoint=vanilla_checkpoint,
            trigger_source="none",
            note=(
                "Natural target-class activation on the clean 9,000-image "
                "non-target verification set."
            ),
        )
    )

    # ------------------------------------------------------------------
    # Patch triggers
    # ------------------------------------------------------------------
    patch_checkpoints = discover_patch_checkpoints(
        args.patch_training_dir,
        include_vanilla=False,
    )

    for checkpoint_path in patch_checkpoints:
        watermarked_model, metadata = load_patch_checkpoint(checkpoint_path, device)
        params = patch_parameters_from_checkpoint(metadata, checkpoint_path)

        if params.trigger_type in {"vanilla", "unknown"}:
            continue
        if params.target_label != args.target_label:
            raise ValueError(
                f"{checkpoint_path.name} uses target {params.target_label}, "
                f"expected {args.target_label}."
            )

        trigger_fn = make_patch_trigger(
            params.trigger_type,
            params.patch_size,
            params.alpha,
            args.deterministic_trigger_seed,
        )
        trigger_loader = make_data_loader(
            TriggeredVerificationSet(verification_base, trigger_fn),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

        vanilla_metrics = evaluate_pair_loader(vanilla_model, trigger_loader, device)
        watermarked_metrics = evaluate_pair_loader(
            watermarked_model, trigger_loader, device
        )

        results.append(
            make_row(
                trigger_type=params.trigger_type,
                vanilla_activation=vanilla_metrics.accuracy,
                vanilla_target_loss=vanilla_metrics.loss,
                natural_activation=natural_activation,
                watermarked_tsr=watermarked_metrics.accuracy,
                watermarked_trigger_loss=watermarked_metrics.loss,
                sample_count=vanilla_metrics.sample_count,
                target_label=args.target_label,
                vanilla_checkpoint=vanilla_checkpoint,
                trigger_source=str(checkpoint_path.resolve()),
                note=(
                    "Patch from patch_utilities.py; same deterministic verifier "
                    "construction is used for vanilla and watermarked models."
                ),
            )
        )

        del watermarked_model

    # ------------------------------------------------------------------
    # WaNet trigger
    # ------------------------------------------------------------------
    wanet_model, warping_grid, wanet_artifacts = load_wanet_artifacts(
        args.warping_training_dir,
        device,
        expected_target_label=args.target_label,
    )
    wanet_loader = make_data_loader(
        WarpingVerificationSet(verification_base),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # Apply only the image-space warp to the vanilla classifier. Do NOT apply
    # WaNet classifier normalisation to vanilla, because vanilla was trained on
    # raw [0,1] tensors.
    vanilla_wanet = evaluate_pair_loader(
        vanilla_model,
        wanet_loader,
        device,
        forward_fn=lambda model, images: model(
            apply_wanet_trigger(images, warping_grid)
        ),
    )

    wanet_normalize = make_cifar10_normalize().to(device)
    watermarked_wanet = evaluate_pair_loader(
        wanet_model,
        wanet_loader,
        device,
        forward_fn=make_wanet_trigger_forward(
            warping_grid,
            wanet_normalize,
        ),
    )

    results.append(
        make_row(
            trigger_type="wanet_warping",
            vanilla_activation=vanilla_wanet.accuracy,
            vanilla_target_loss=vanilla_wanet.loss,
            natural_activation=natural_activation,
            watermarked_tsr=watermarked_wanet.accuracy,
            watermarked_trigger_loss=watermarked_wanet.loss,
            sample_count=vanilla_wanet.sample_count,
            target_label=args.target_label,
            vanilla_checkpoint=vanilla_checkpoint,
            trigger_source=str(wanet_artifacts.grid_path.resolve()),
            note=(
                "Exact saved WaNet grid. Vanilla keeps its original no-normalisation "
                "pipeline; WaNet uses its CIFAR-10 normalisation."
            ),
        )
    )

    del wanet_model

    # ------------------------------------------------------------------
    # Input-aware trigger
    # ------------------------------------------------------------------
    (
        iadb_classifier,
        trigger_generator,
        mask_generator,
        iadb_artifacts,
    ) = load_best_input_aware_model(
        args.input_aware_training_dir,
        device,
        expected_target_label=args.target_label,
    )

    iadb_loader = make_data_loader(
        InputAwareVerificationSet(verification_base),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    iadb_forward = make_iadb_trigger_forward(trigger_generator, mask_generator)

    vanilla_iadb = evaluate_pair_loader(
        vanilla_model,
        iadb_loader,
        device,
        forward_fn=iadb_forward,
    )
    watermarked_iadb = evaluate_pair_loader(
        iadb_classifier,
        iadb_loader,
        device,
        forward_fn=iadb_forward,
    )

    results.append(
        make_row(
            trigger_type="input_aware_dynamic",
            vanilla_activation=vanilla_iadb.accuracy,
            vanilla_target_loss=vanilla_iadb.loss,
            natural_activation=natural_activation,
            watermarked_tsr=watermarked_iadb.accuracy,
            watermarked_trigger_loss=watermarked_iadb.loss,
            sample_count=vanilla_iadb.sample_count,
            target_label=args.target_label,
            vanilla_checkpoint=vanilla_checkpoint,
            trigger_source=str(iadb_artifacts.checkpoint_path.resolve()),
            note=(
                "G/M learned with the input-aware watermarked model are applied "
                "to the common vanilla classifier. Interpret this as a conditional "
                "classifier-level false-positive baseline."
            ),
        )
    )

    fields = [
        "trigger_type",
        "vanilla_target_activation_percent",
        "vanilla_target_loss",
        "natural_vanilla_target_activation_percent",
        "trigger_induced_increase_pp",
        "watermarked_tsr_percent",
        "watermarked_trigger_loss",
        "verification_margin_pp",
        "sample_count",
        "target_label",
        "vanilla_checkpoint",
        "trigger_source",
        "note",
    ]

    output_csv = args.out_dir / "false_positive_summary.csv"
    write_csv(output_csv, fields, results)

    print("\nFalse-positive baseline complete")
    print(f"Vanilla checkpoint: {vanilla_checkpoint}")
    print(f"Verification images: {len(verification_base)}")
    print(f"Natural vanilla target activation: {natural_activation:.2f}%\n")

    for result in results[1:]:
        print(
            f"{result['trigger_type']}: "
            f"vanilla activation={float(result['vanilla_target_activation_percent']):.2f}% | "
            f"watermarked TSR={float(result['watermarked_tsr_percent']):.2f}% | "
            f"margin={float(result['verification_margin_pp']):.2f} pp"
        )

    print(f"\nSaved summary: {output_csv}")


if __name__ == "__main__":
    main()
