"""Global unstructured L1 pruning attack for the retained WaNet model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common_verification import (
    VerificationBaseSet,
    actual_sparsity_percent,
    apply_global_l1_pruning,
    evaluate_pair_loader,
    finalize_pruning,
    make_cifar10_test_dataset,
    make_data_loader,
    parse_float_list,
    write_csv,
)
from warping_based.warping_utilities import (
    WarpingVerificationSet,
    load_best_wanet_model,
    load_wanet_artifacts,
    make_cifar10_normalize,
    make_clean_forward,
    make_trigger_forward,
    save_wanet_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_DIR = Path(__file__).resolve().parent / "run" / "training"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_INDEX_FILE = (
    PROJECT_ROOT / "verification_sets" / "cifar10_test_non_target_class0.pt"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "run" / "pruning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-dir", "--training_dir", type=Path, default=DEFAULT_TRAINING_DIR
    )
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument(
        "--sparsities",
        type=str,
        default="0.10,0.30,0.50,0.70",
    )
    parser.add_argument("--save-pruned-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sparsities = parse_float_list(args.sparsities)
    if any(value < 0.0 or value > 1.0 for value in sparsities):
        raise ValueError("Every sparsity must be in [0, 1].")

    baseline_model, warping_grid, artifacts = load_wanet_artifacts(
        args.training_dir,
        args.device,
        expected_target_label=args.target_label,
    )
    normalize = make_cifar10_normalize().to(args.device)
    clean_forward = make_clean_forward(normalize)
    trigger_forward = make_trigger_forward(warping_grid, normalize)

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
    trigger_set = WarpingVerificationSet(verification_base)
    trigger_loader = make_data_loader(
        trigger_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    clean_before = evaluate_pair_loader(
        baseline_model, clean_loader, args.device, forward_fn=clean_forward
    )
    trigger_before = evaluate_pair_loader(
        baseline_model, trigger_loader, args.device, forward_fn=trigger_forward
    )

    rows: list[dict[str, object]] = []
    for sparsity in sparsities:
        # Each sparsity point is an independent attack on best_model.pt.
        model, _metadata, checkpoint_path = load_best_wanet_model(
            args.training_dir,
            args.device,
        )
        apply_global_l1_pruning(model, sparsity)
        finalize_pruning(model)

        actual = actual_sparsity_percent(model)
        clean_after = evaluate_pair_loader(
            model, clean_loader, args.device, forward_fn=clean_forward
        )
        trigger_after = evaluate_pair_loader(
            model, trigger_loader, args.device, forward_fn=trigger_forward
        )

        row = {
            "model_file": checkpoint_path.name,
            "grid_file": artifacts.grid_path.name,
            "trigger_type": "wanet_warping",
            "pruning_method": "global_unstructured_l1_conv_linear_weights",
            "target_label": artifacts.target_label,
            "sparsity_percent": f"{100.0 * sparsity:.6f}",
            "actual_sparsity_percent": f"{actual:.6f}",
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "verification_size": trigger_after.sample_count,
            "clean_acc_before_percent": f"{clean_before.accuracy:.6f}",
            "clean_loss_before": f"{clean_before.loss:.6f}",
            "tsr_before_percent": f"{trigger_before.accuracy:.6f}",
            "tsr_loss_before": f"{trigger_before.loss:.6f}",
            "clean_acc_after_percent": f"{clean_after.accuracy:.6f}",
            "clean_loss_after": f"{clean_after.loss:.6f}",
            "tsr_after_percent": f"{trigger_after.accuracy:.6f}",
            "tsr_loss_after": f"{trigger_after.loss:.6f}",
            "clean_acc_drop_pp": f"{clean_before.accuracy - clean_after.accuracy:.6f}",
            "tsr_drop_pp": f"{trigger_before.accuracy - trigger_after.accuracy:.6f}",
            "checkpoint_path": str(checkpoint_path),
            "grid_path": str(artifacts.grid_path),
            "verification_index_file": str(args.verification_index_file.resolve()),
        }
        rows.append(row)

        print(
            f"sparsity={100*sparsity:.0f}% actual={actual:.2f}% "
            f"clean_acc={clean_after.accuracy:.2f}% "
            f"TSR={trigger_after.accuracy:.2f}%"
        )

        if args.save_pruned_checkpoints:
            save_wanet_model(
                args.out_dir
                / "checkpoints"
                / f"best_model_pruned_{int(round(100*sparsity))}pct.pt",
                model,
                source_checkpoint=checkpoint_path,
                attack="global_unstructured_l1_pruning",
                metrics=row,
            )

    output_path = args.out_dir / "pruning_summary.csv"
    write_csv(output_path, rows[0].keys(), rows)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
