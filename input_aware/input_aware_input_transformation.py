"""Input-transformation attack for the retained input-aware watermark model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common_verification import (
    VerificationBaseSet,
    build_standard_input_transformations,
    evaluate_pair_loader,
    make_cifar10_test_dataset,
    make_data_loader,
    write_csv,
)
from input_aware.input_aware_utilities import (
    InputAwareVerificationSet,
    load_best_input_aware_model,
    make_clean_forward,
    make_trigger_forward,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_DIR = Path(__file__).resolve().parent / "run" / "training"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_INDEX_FILE = (
    PROJECT_ROOT
    / "verification_sets"
    / "cifar10_test_non_target_class0.pt"
)
DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parent
    / "run"
    / "input_transformation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-dir",
        "--training_dir",
        type=Path,
        default=DEFAULT_TRAINING_DIR,
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--verification-index-file",
        type=Path,
        default=DEFAULT_INDEX_FILE,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-label", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    classifier, trigger_generator, mask_generator, artifacts = (
        load_best_input_aware_model(
            args.training_dir,
            args.device,
            expected_target_label=args.target_label,
        )
    )

    test_set = make_cifar10_test_dataset(args.data_root)
    clean_loader = make_data_loader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    verification_base = VerificationBaseSet(
        test_set,
        args.verification_index_file,
        target_label=artifacts.target_label,
    )
    trigger_set = InputAwareVerificationSet(verification_base)
    trigger_loader = make_data_loader(
        trigger_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    clean = evaluate_pair_loader(
        classifier,
        clean_loader,
        args.device,
        forward_fn=make_clean_forward(),
    )

    rows: list[dict[str, object]] = []
    baseline_tsr: float | None = None
    baseline_loss: float | None = None

    for transformation in build_standard_input_transformations():
        metrics = evaluate_pair_loader(
            classifier,
            trigger_loader,
            args.device,
            forward_fn=make_trigger_forward(
                trigger_generator,
                mask_generator,
                post_trigger_transform=transformation.transform,
            ),
        )

        if transformation.name == "none":
            baseline_tsr = metrics.accuracy
            baseline_loss = metrics.loss

        if baseline_tsr is None or baseline_loss is None:
            raise RuntimeError(
                "The no-transformation baseline must be evaluated first."
            )

        row = {
            "model_file": artifacts.checkpoint_path.name,
            "trigger_type": "input_aware_dynamic",
            "target_label": artifacts.target_label,
            "transform_name": transformation.name,
            "transform_family": transformation.family,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "clean_acc_percent": f"{clean.accuracy:.6f}",
            "clean_loss": f"{clean.loss:.6f}",
            "baseline_tsr_no_transform_percent": (
                f"{baseline_tsr:.6f}"
            ),
            "baseline_tsr_loss_no_transform": (
                f"{baseline_loss:.6f}"
            ),
            "tsr_after_transform_percent": (
                f"{metrics.accuracy:.6f}"
            ),
            "tsr_loss_after_transform": f"{metrics.loss:.6f}",
            "tsr_drop_from_no_transform_pp": (
                f"{baseline_tsr - metrics.accuracy:.6f}"
            ),
            "verification_size": metrics.sample_count,
            "checkpoint_path": str(artifacts.checkpoint_path),
            "verification_index_file": str(
                args.verification_index_file.resolve()
            ),
        }
        rows.append(row)

        print(
            f"{transformation.name:20s} "
            f"TSR={metrics.accuracy:.2f}% "
            f"drop={baseline_tsr - metrics.accuracy:.2f} pp"
        )

    output_path = args.out_dir / "input_transform_summary.csv"
    write_csv(output_path, rows[0].keys(), rows)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
