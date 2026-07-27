"""Input-transformation attack for patch-based watermarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common_verification import (
    TriggeredVerificationSet,
    VerificationBaseSet,
    build_standard_input_transformations,
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_INDEX_FILE = (
    PROJECT_ROOT / "verification_sets" / "cifar10_test_non_target_class0.pt"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "run" / "input_transformation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", "--ckpt_dir", type=Path, required=True)
    parser.add_argument("--out-dir", "--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-root", "--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--verification-index-file", type=Path, default=DEFAULT_INDEX_FILE
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--deterministic-trigger-seed", type=int, default=999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    test_set = make_cifar10_test_dataset(args.data_root)
    clean_loader = make_data_loader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    transformations = build_standard_input_transformations()
    checkpoints = discover_patch_checkpoints(
        args.ckpt_dir,
        include_vanilla=False,
    )

    summary_rows: list[dict[str, object]] = []

    for checkpoint_path in checkpoints:
        model, metadata = load_patch_checkpoint(checkpoint_path, args.device)
        parameters = patch_parameters_from_checkpoint(metadata, checkpoint_path)
        if parameters.trigger_type in {"vanilla", "unknown"}:
            continue

        clean = evaluate_pair_loader(model, clean_loader, args.device)
        verification_base = VerificationBaseSet(
            test_set,
            args.verification_index_file,
            target_label=parameters.target_label,
        )
        trigger = make_patch_trigger(
            parameters.trigger_type,
            parameters.patch_size,
            parameters.alpha,
            args.deterministic_trigger_seed,
        )

        model_rows: list[dict[str, object]] = []
        baseline_tsr: float | None = None
        baseline_tsr_loss: float | None = None

        for transformation in transformations:
            transformed_set = TriggeredVerificationSet(
                verification_base,
                trigger,
                post_trigger_transform=lambda image, _index, t=transformation: t(image),
            )
            transformed_loader = make_data_loader(
                transformed_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
            )
            metrics = evaluate_pair_loader(model, transformed_loader, args.device)

            if transformation.name == "none":
                baseline_tsr = metrics.accuracy
                baseline_tsr_loss = metrics.loss

            if baseline_tsr is None or baseline_tsr_loss is None:
                raise RuntimeError("The no-transformation baseline must be evaluated first.")

            row = {
                "model_file": checkpoint_path.name,
                "trigger_type": parameters.trigger_type,
                "target_label": parameters.target_label,
                "patch_size": parameters.patch_size,
                "alpha": parameters.alpha,
                "verification_size": metrics.sample_count,
                "transform_name": transformation.name,
                "transform_family": transformation.family,
                "clean_acc_percent": f"{clean.accuracy:.6f}",
                "clean_loss": f"{clean.loss:.6f}",
                "baseline_tsr_no_transform_percent": f"{baseline_tsr:.6f}",
                "baseline_tsr_loss_no_transform": f"{baseline_tsr_loss:.6f}",
                "tsr_after_transform_percent": f"{metrics.accuracy:.6f}",
                "tsr_loss_after_transform": f"{metrics.loss:.6f}",
                "tsr_drop_pp": f"{baseline_tsr - metrics.accuracy:.6f}",
                "verification_index_file": str(args.verification_index_file.resolve()),
            }
            model_rows.append(row)
            summary_rows.append(row)

            print(
                f"[{checkpoint_path.stem}] {transformation.name:18s} "
                f"TSR={metrics.accuracy:.2f}% "
                f"drop={baseline_tsr - metrics.accuracy:.2f} pp"
            )

        per_model_path = args.out_dir / f"input_transform_{checkpoint_path.stem}.csv"
        write_csv(per_model_path, model_rows[0].keys(), model_rows)

    if not summary_rows:
        raise RuntimeError("No watermarked patch checkpoints were evaluated.")

    summary_path = args.out_dir / "input_transform_summary.csv"
    write_csv(summary_path, summary_rows[0].keys(), summary_rows)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
