"""Full-layer clean-data fine-tuning attack for the retained input-aware model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.optim as optim

from common_verification import (
    VerificationBaseSet,
    append_csv_row,
    evaluate_pair_loader,
    make_cifar10_test_dataset,
    make_cifar10_train_dataset,
    make_data_loader,
    set_seed,
    train_one_epoch,
    write_csv,
)
from input_aware.input_aware_utilities import (
    InputAwareVerificationSet,
    freeze_trigger_modules,
    load_best_input_aware_model,
    make_clean_forward,
    make_trigger_forward,
    save_input_aware_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_DIR = Path(__file__).resolve().parent / "run" / "training"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_INDEX_FILE = (
    PROJECT_ROOT
    / "verification_sets"
    / "cifar10_test_non_target_class0.pt"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "run" / "finetuning"


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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Use 1e-3 for gentle FTAL or 1e-2 for aggressive FTAL.",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--save-final-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lr_tag = f"{args.lr:.0e}".replace("+", "")

    set_seed(args.seed)

    train_set = make_cifar10_train_dataset(args.data_root)
    train_loader = make_data_loader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        generator_seed=args.seed,
    )

    test_set = make_cifar10_test_dataset(args.data_root)
    clean_loader = make_data_loader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    classifier, trigger_generator, mask_generator, artifacts = (
        load_best_input_aware_model(
            args.training_dir,
            args.device,
            expected_target_label=args.target_label,
        )
    )
    freeze_trigger_modules(trigger_generator, mask_generator)

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

    clean_forward = make_clean_forward()
    trigger_forward = make_trigger_forward(
        trigger_generator,
        mask_generator,
    )

    optimizer = optim.SGD(
        classifier.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    detail_fields = [
        "epoch",
        "phase",
        "train_loss",
        "clean_acc_percent",
        "clean_loss",
        "tsr_percent",
        "tsr_loss",
        "lr",
        "verification_size",
    ]
    detail_path = args.out_dir / f"ftal_best_model_lr{lr_tag}.csv"
    if detail_path.exists():
        detail_path.unlink()

    clean_before = evaluate_pair_loader(
        classifier,
        clean_loader,
        args.device,
        forward_fn=clean_forward,
    )
    trigger_before = evaluate_pair_loader(
        classifier,
        trigger_loader,
        args.device,
        forward_fn=trigger_forward,
    )

    append_csv_row(
        detail_path,
        detail_fields,
        {
            "epoch": 0,
            "phase": "before_attack",
            "train_loss": "",
            "clean_acc_percent": f"{clean_before.accuracy:.6f}",
            "clean_loss": f"{clean_before.loss:.6f}",
            "tsr_percent": f"{trigger_before.accuracy:.6f}",
            "tsr_loss": f"{trigger_before.loss:.6f}",
            "lr": f"{args.lr:.8f}",
            "verification_size": trigger_before.sample_count,
        },
    )

    clean_after = clean_before
    trigger_after = trigger_before

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            classifier,
            train_loader,
            optimizer,
            args.device,
            forward_fn=clean_forward,
        )
        clean_after = evaluate_pair_loader(
            classifier,
            clean_loader,
            args.device,
            forward_fn=clean_forward,
        )
        trigger_after = evaluate_pair_loader(
            classifier,
            trigger_loader,
            args.device,
            forward_fn=trigger_forward,
        )

        append_csv_row(
            detail_path,
            detail_fields,
            {
                "epoch": epoch,
                "phase": "ftal",
                "train_loss": f"{train_loss:.6f}",
                "clean_acc_percent": f"{clean_after.accuracy:.6f}",
                "clean_loss": f"{clean_after.loss:.6f}",
                "tsr_percent": f"{trigger_after.accuracy:.6f}",
                "tsr_loss": f"{trigger_after.loss:.6f}",
                "lr": f"{args.lr:.8f}",
                "verification_size": trigger_after.sample_count,
            },
        )

        if epoch in {1, 5, 10, 25, 50, 75, 100, args.epochs}:
            print(
                f"epoch={epoch:03d} "
                f"clean_acc={clean_after.accuracy:.2f}% "
                f"TSR={trigger_after.accuracy:.2f}%"
            )

    summary = {
        "model_file": artifacts.checkpoint_path.name,
        "trigger_type": "input_aware_dynamic",
        "target_label": artifacts.target_label,
        "epochs": args.epochs,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "verification_size": trigger_before.sample_count,
        "clean_acc_before_percent": f"{clean_before.accuracy:.6f}",
        "clean_loss_before": f"{clean_before.loss:.6f}",
        "tsr_before_percent": f"{trigger_before.accuracy:.6f}",
        "tsr_loss_before": f"{trigger_before.loss:.6f}",
        "clean_acc_after_percent": f"{clean_after.accuracy:.6f}",
        "clean_loss_after": f"{clean_after.loss:.6f}",
        "tsr_after_percent": f"{trigger_after.accuracy:.6f}",
        "tsr_loss_after": f"{trigger_after.loss:.6f}",
        "clean_acc_drop_pp": (
            f"{clean_before.accuracy - clean_after.accuracy:.6f}"
        ),
        "tsr_drop_pp": (
            f"{trigger_before.accuracy - trigger_after.accuracy:.6f}"
        ),
        "checkpoint_path": str(artifacts.checkpoint_path),
        "verification_index_file": str(
            args.verification_index_file.resolve()
        ),
    }
    summary_path = args.out_dir / f"ftal_summary_lr{lr_tag}.csv"
    write_csv(summary_path, summary.keys(), [summary])

    if args.save_final_checkpoint:
        save_input_aware_checkpoint(
            args.out_dir
            / "checkpoints"
            / f"best_model_ftal_lr{lr_tag}_ep{args.epochs}.pt",
            classifier,
            trigger_generator,
            mask_generator,
            source_checkpoint=artifacts.checkpoint_path,
            attack="full_layer_clean_finetuning",
            metrics=summary,
        )

    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
