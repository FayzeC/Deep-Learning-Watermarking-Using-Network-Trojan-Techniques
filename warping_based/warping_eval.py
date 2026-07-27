"""Standard evaluation for the retained WaNet CIFAR-10 watermark model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common_verification import (
    VerificationBaseSet,
    evaluate_pair_loader,
    make_cifar10_test_dataset,
    make_data_loader,
    write_csv,
)
from warping_based.warping_utilities import (
    WarpingVerificationSet,
    load_wanet_artifacts,
    make_cifar10_normalize,
    make_clean_forward,
    make_trigger_forward,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_DIR = Path(__file__).resolve().parent / "run" / "training"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_INDEX_FILE = (
    PROJECT_ROOT / "verification_sets" / "cifar10_test_non_target_class0.pt"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "run" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-dir",
        "--training_dir",
        type=Path,
        default=DEFAULT_TRAINING_DIR,
        help="Directory containing best_model.pt and wanet_grid.pt.",
    )
    parser.add_argument("--out-dir", "--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-root", "--data_root", type=Path, default=DEFAULT_DATA_ROOT)
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
    parser.add_argument("--tsr-threshold", type=float, default=90.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, warping_grid, artifacts = load_wanet_artifacts(
        args.training_dir,
        args.device,
        expected_target_label=args.target_label,
    )
    normalize = make_cifar10_normalize().to(args.device)

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

    clean = evaluate_pair_loader(
        model,
        clean_loader,
        args.device,
        forward_fn=make_clean_forward(normalize),
    )
    trigger = evaluate_pair_loader(
        model,
        trigger_loader,
        args.device,
        forward_fn=make_trigger_forward(warping_grid, normalize),
    )

    row = {
        "model_file": artifacts.checkpoint_path.name,
        "grid_file": artifacts.grid_path.name,
        "trigger_type": "wanet_warping",
        "target_label": artifacts.target_label,
        "clean_acc_percent": f"{clean.accuracy:.6f}",
        "clean_loss": f"{clean.loss:.6f}",
        "tsr_percent": f"{trigger.accuracy:.6f}",
        "tsr_loss": f"{trigger.loss:.6f}",
        "verified": trigger.accuracy >= args.tsr_threshold,
        "tsr_threshold_percent": args.tsr_threshold,
        "clean_test_size": clean.sample_count,
        "verification_size": trigger.sample_count,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "checkpoint_path": str(artifacts.checkpoint_path),
        "grid_path": str(artifacts.grid_path),
        "verification_index_file": str(args.verification_index_file.resolve()),
    }

    output_path = args.out_dir / "metrics_summary.csv"
    write_csv(output_path, row.keys(), [row])

    print(f"Checkpoint: {artifacts.checkpoint_path}")
    print(f"Grid: {artifacts.grid_path}")
    print(f"Clean accuracy: {clean.accuracy:.2f}%")
    print(f"TSR: {trigger.accuracy:.2f}%")
    print(f"Verification samples: {trigger.sample_count}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
