"""Global unstructured L1 pruning attack for patch-based watermarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common_verification import (
    TriggeredVerificationSet,
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
from patch_trigger.patch_utilities import (
    discover_patch_checkpoints,
    load_patch_checkpoint,
    make_patch_trigger,
    patch_parameters_from_checkpoint,
    save_patch_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_INDEX_FILE = (
    PROJECT_ROOT / "verification_sets" / "cifar10_test_non_target_class0.pt"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "run" / "pruning"


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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--deterministic-trigger-seed", type=int, default=999)
    parser.add_argument(
        "--sparsities",
        type=str,
        default="0.10,0.30,0.50,0.70",
        help="Comma-separated fractions, for example 0.10,0.30,0.50,0.70.",
    )
    parser.add_argument("--save-pruned-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sparsities = parse_float_list(args.sparsities)
    if any(value < 0.0 or value > 1.0 for value in sparsities):
        raise ValueError("Every sparsity must be in [0, 1].")

    test_set = make_cifar10_test_dataset(args.data_root)
    clean_loader = make_data_loader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    checkpoints = discover_patch_checkpoints(
        args.ckpt_dir,
        include_vanilla=False,
    )

    summary_rows: list[dict[str, object]] = []

    for checkpoint_path in checkpoints:
        baseline_model, metadata = load_patch_checkpoint(checkpoint_path, args.device)
        parameters = patch_parameters_from_checkpoint(metadata, checkpoint_path)
        if parameters.trigger_type in {"vanilla", "unknown"}:
            continue

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
        trigger_set = TriggeredVerificationSet(verification_base, trigger)
        trigger_loader = make_data_loader(
            trigger_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

        clean_before = evaluate_pair_loader(baseline_model, clean_loader, args.device)
        trigger_before = evaluate_pair_loader(
            baseline_model, trigger_loader, args.device
        )

        model_rows: list[dict[str, object]] = []
        for sparsity in sparsities:
            # Each point starts from the original retained checkpoint.
            model, _ = load_patch_checkpoint(checkpoint_path, args.device)
            apply_global_l1_pruning(model, sparsity)
            finalize_pruning(model)

            actual_sparsity = actual_sparsity_percent(model)
            clean_after = evaluate_pair_loader(model, clean_loader, args.device)
            trigger_after = evaluate_pair_loader(model, trigger_loader, args.device)

            row = {
                "model_file": checkpoint_path.name,
                "trigger_type": parameters.trigger_type,
                "pruning_method": "global_unstructured_l1_conv_linear_weights",
                "target_label": parameters.target_label,
                "patch_size": parameters.patch_size,
                "alpha": parameters.alpha,
                "verification_size": trigger_after.sample_count,
                "sparsity_percent": f"{100.0 * sparsity:.6f}",
                "actual_sparsity_percent": f"{actual_sparsity:.6f}",
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
                "verification_index_file": str(args.verification_index_file.resolve()),
            }
            model_rows.append(row)
            summary_rows.append(row)

            print(
                f"[{checkpoint_path.stem}] sparsity={100*sparsity:.0f}% "
                f"actual={actual_sparsity:.2f}% "
                f"clean_acc={clean_after.accuracy:.2f}% "
                f"TSR={trigger_after.accuracy:.2f}%"
            )

            if args.save_pruned_checkpoints:
                save_patch_model(
                    args.out_dir
                    / "checkpoints"
                    / f"{checkpoint_path.stem}_pruned_{int(round(100*sparsity))}pct.pt",
                    model,
                    metadata,
                    extra={
                        "attack": "global_unstructured_l1_pruning",
                        "sparsity_percent": 100.0 * sparsity,
                        "actual_sparsity_percent": actual_sparsity,
                    },
                )

        per_model_path = args.out_dir / f"pruning_{checkpoint_path.stem}.csv"
        write_csv(per_model_path, model_rows[0].keys(), model_rows)

    if not summary_rows:
        raise RuntimeError("No watermarked patch checkpoints were evaluated.")

    summary_path = args.out_dir / "pruning_summary.csv"
    write_csv(summary_path, summary_rows[0].keys(), summary_rows)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
