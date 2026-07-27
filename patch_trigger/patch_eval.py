"""Standard evaluation for patch-based CIFAR-10 watermark checkpoints."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_INDEX_FILE = (
    PROJECT_ROOT / "verification_sets" / "cifar10_test_non_target_class0.pt"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "run" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-dir",
        "--ckpt_dir",
        type=Path,
        required=True,
        help="Directory containing vanilla and patch-watermarked .pt checkpoints.",
    )
    parser.add_argument("--out-dir", "--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-root", "--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--verification-index-file",
        type=Path,
        default=DEFAULT_INDEX_FILE,
        help="Saved indices for all 9,000 non-target CIFAR-10 test images.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--deterministic-trigger-seed", type=int, default=999)
    parser.add_argument(
        "--tsr-threshold",
        type=float,
        default=90.0,
        help="Verification threshold expressed as a percentage.",
    )
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

    checkpoints = discover_patch_checkpoints(
        args.ckpt_dir,
        include_vanilla=True,
    )

    rows: list[dict[str, object]] = []
    for checkpoint_path in checkpoints:
        model, metadata = load_patch_checkpoint(checkpoint_path, args.device)
        parameters = patch_parameters_from_checkpoint(metadata, checkpoint_path)

        clean = evaluate_pair_loader(model, clean_loader, args.device)

        tsr = math.nan
        tsr_loss = math.nan
        verification_size = 0
        verified: bool | str = "not_applicable"

        if parameters.trigger_type != "vanilla":
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
            trigger_metrics = evaluate_pair_loader(model, trigger_loader, args.device)
            tsr = trigger_metrics.accuracy
            tsr_loss = trigger_metrics.loss
            verification_size = trigger_metrics.sample_count
            verified = tsr >= args.tsr_threshold

        rows.append(
            {
                "model_file": checkpoint_path.name,
                "trigger_type": parameters.trigger_type,
                "target_label": parameters.target_label,
                "patch_size": parameters.patch_size,
                "alpha": parameters.alpha,
                "clean_acc_percent": f"{clean.accuracy:.6f}",
                "clean_loss": f"{clean.loss:.6f}",
                "tsr_percent": "nan" if math.isnan(tsr) else f"{tsr:.6f}",
                "tsr_loss": "nan" if math.isnan(tsr_loss) else f"{tsr_loss:.6f}",
                "verified": verified,
                "tsr_threshold_percent": args.tsr_threshold,
                "verification_size": verification_size,
                "verification_index_file": str(args.verification_index_file.resolve()),
            }
        )

        print(
            f"[{checkpoint_path.name}] trigger={parameters.trigger_type} "
            f"clean_acc={clean.accuracy:.2f}% "
            f"TSR={'n/a' if math.isnan(tsr) else f'{tsr:.2f}%'}"
        )

    output_path = args.out_dir / "metrics_summary.csv"
    write_csv(output_path, rows[0].keys(), rows)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
