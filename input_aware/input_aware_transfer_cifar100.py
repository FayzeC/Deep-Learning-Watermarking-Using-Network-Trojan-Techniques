"""Full-model CIFAR-100 transfer-learning attack for the input-aware model."""

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
    make_cifar100_datasets,
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
    replace_classifier_head,
    save_input_aware_checkpoint,
    set_cifar10_normalization,
    set_cifar100_normalization,
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
    / "transfer_learning"
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def evaluate_cifar10_and_watermark(
    classifier: torch.nn.Module,
    trigger_generator: torch.nn.Module,
    mask_generator: torch.nn.Module,
    clean_loader,
    trigger_loader,
    device: str,
):
    """Evaluate using the original CIFAR-10 preprocessing policy."""

    set_cifar10_normalization(classifier)

    clean_metrics = evaluate_pair_loader(
        classifier,
        clean_loader,
        device,
        forward_fn=make_clean_forward(),
    )
    trigger_metrics = evaluate_pair_loader(
        classifier,
        trigger_loader,
        device,
        forward_fn=make_trigger_forward(
            trigger_generator,
            mask_generator,
        ),
    )
    return clean_metrics, trigger_metrics


def evaluate_cifar100(
    classifier: torch.nn.Module,
    loader,
    device: str,
):
    """Evaluate CIFAR-100 using CIFAR-100 channel statistics."""

    set_cifar100_normalization(classifier)
    return evaluate_pair_loader(
        classifier,
        loader,
        device,
        forward_fn=make_clean_forward(),
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    cifar10_test = make_cifar10_test_dataset(args.data_root)
    cifar100_train, cifar100_test = make_cifar100_datasets(
        args.data_root
    )

    c10_clean_loader = make_data_loader(
        cifar10_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    c100_test_loader = make_data_loader(
        cifar100_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    c100_train_loader = make_data_loader(
        cifar100_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        generator_seed=args.seed,
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
        cifar10_test,
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

    detail_fields = [
        "epoch",
        "phase",
        "c100_train_loss",
        "c100_acc_percent",
        "c100_loss",
        "c10_clean_acc_percent",
        "c10_clean_loss",
        "tsr_percent",
        "tsr_loss",
        "verification_size",
    ]
    detail_path = args.out_dir / "transfer_best_model.csv"
    if detail_path.exists():
        detail_path.unlink()

    # Original 10-class checkpoint before any architectural modification.
    c10_before, trigger_before = evaluate_cifar10_and_watermark(
        classifier,
        trigger_generator,
        mask_generator,
        c10_clean_loader,
        trigger_loader,
        args.device,
    )
    append_csv_row(
        detail_path,
        detail_fields,
        {
            "epoch": -1,
            "phase": "before_transfer",
            "c100_train_loss": "",
            "c100_acc_percent": "",
            "c100_loss": "",
            "c10_clean_acc_percent": f"{c10_before.accuracy:.6f}",
            "c10_clean_loss": f"{c10_before.loss:.6f}",
            "tsr_percent": f"{trigger_before.accuracy:.6f}",
            "tsr_loss": f"{trigger_before.loss:.6f}",
            "verification_size": trigger_before.sample_count,
        },
    )

    # Standardised full transfer: reset the seed, replace the 10-class head with
    # a fresh 100-class head, and keep every classifier layer trainable.
    set_seed(args.seed)
    classifier = replace_classifier_head(
        classifier,
        100,
        args.device,
    )
    optimizer = optim.SGD(
        classifier.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Recompute every metric after the head replacement. CIFAR-100 uses
    # CIFAR-100 statistics, while original watermark verification switches back
    # to the model's original CIFAR-10 statistics.
    c100_epoch0 = evaluate_cifar100(
        classifier,
        c100_test_loader,
        args.device,
    )
    c10_epoch0, trigger_epoch0 = evaluate_cifar10_and_watermark(
        classifier,
        trigger_generator,
        mask_generator,
        c10_clean_loader,
        trigger_loader,
        args.device,
    )
    append_csv_row(
        detail_path,
        detail_fields,
        {
            "epoch": 0,
            "phase": "after_head_swap_epoch0",
            "c100_train_loss": "",
            "c100_acc_percent": f"{c100_epoch0.accuracy:.6f}",
            "c100_loss": f"{c100_epoch0.loss:.6f}",
            "c10_clean_acc_percent": f"{c10_epoch0.accuracy:.6f}",
            "c10_clean_loss": f"{c10_epoch0.loss:.6f}",
            "tsr_percent": f"{trigger_epoch0.accuracy:.6f}",
            "tsr_loss": f"{trigger_epoch0.loss:.6f}",
            "verification_size": trigger_epoch0.sample_count,
        },
    )

    best_c100_acc = c100_epoch0.accuracy
    c100_final = c100_epoch0
    c10_final = c10_epoch0
    trigger_final = trigger_epoch0

    if args.save_checkpoints:
        set_cifar100_normalization(classifier)
        save_input_aware_checkpoint(
            args.out_dir / "checkpoints" / "best_model_best_c100.pt",
            classifier,
            trigger_generator,
            mask_generator,
            source_checkpoint=artifacts.checkpoint_path,
            attack="transfer_learning_cifar100_full",
            metrics={
                "transfer_epoch": 0,
                "c100_acc_percent": best_c100_acc,
                "tsr_percent": trigger_epoch0.accuracy,
            },
        )

    for epoch in range(1, args.epochs + 1):
        # New-task training and evaluation always use CIFAR-100 statistics.
        set_cifar100_normalization(classifier)
        c100_train_loss = train_one_epoch(
            classifier,
            c100_train_loader,
            optimizer,
            args.device,
            forward_fn=make_clean_forward(),
        )
        c100_final = evaluate_cifar100(
            classifier,
            c100_test_loader,
            args.device,
        )

        # Original watermark verification always uses CIFAR-10 statistics.
        c10_final, trigger_final = evaluate_cifar10_and_watermark(
            classifier,
            trigger_generator,
            mask_generator,
            c10_clean_loader,
            trigger_loader,
            args.device,
        )

        append_csv_row(
            detail_path,
            detail_fields,
            {
                "epoch": epoch,
                "phase": "transfer_training",
                "c100_train_loss": f"{c100_train_loss:.6f}",
                "c100_acc_percent": f"{c100_final.accuracy:.6f}",
                "c100_loss": f"{c100_final.loss:.6f}",
                "c10_clean_acc_percent": f"{c10_final.accuracy:.6f}",
                "c10_clean_loss": f"{c10_final.loss:.6f}",
                "tsr_percent": f"{trigger_final.accuracy:.6f}",
                "tsr_loss": f"{trigger_final.loss:.6f}",
                "verification_size": trigger_final.sample_count,
            },
        )

        if c100_final.accuracy > best_c100_acc:
            best_c100_acc = c100_final.accuracy
            if args.save_checkpoints:
                set_cifar100_normalization(classifier)
                save_input_aware_checkpoint(
                    args.out_dir
                    / "checkpoints"
                    / "best_model_best_c100.pt",
                    classifier,
                    trigger_generator,
                    mask_generator,
                    source_checkpoint=artifacts.checkpoint_path,
                    attack="transfer_learning_cifar100_full",
                    metrics={
                        "transfer_epoch": epoch,
                        "c100_acc_percent": best_c100_acc,
                        "tsr_percent": trigger_final.accuracy,
                    },
                )

        if epoch in {1, 5, 10, 25, 50, args.epochs}:
            print(
                f"epoch={epoch:03d} "
                f"C100_acc={c100_final.accuracy:.2f}% "
                f"TSR={trigger_final.accuracy:.2f}%"
            )

    summary = {
        "model_file": artifacts.checkpoint_path.name,
        "trigger_type": "input_aware_dynamic",
        "target_label": artifacts.target_label,
        "mode": "full_all_layers",
        "epochs": args.epochs,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "verification_size": trigger_before.sample_count,
        "cifar100_preprocessing": (
            "internal CIFAR-100 channel normalisation"
        ),
        "cifar10_verification_preprocessing": (
            "original internal CIFAR-10 channel normalisation"
        ),
        "c10_clean_acc_before_transfer_percent": (
            f"{c10_before.accuracy:.6f}"
        ),
        "c10_clean_loss_before_transfer": f"{c10_before.loss:.6f}",
        "tsr_before_transfer_percent": (
            f"{trigger_before.accuracy:.6f}"
        ),
        "tsr_loss_before_transfer": f"{trigger_before.loss:.6f}",
        "c100_acc_after_head_swap_epoch0_percent": (
            f"{c100_epoch0.accuracy:.6f}"
        ),
        "c100_loss_after_head_swap_epoch0": (
            f"{c100_epoch0.loss:.6f}"
        ),
        "c10_clean_acc_after_head_swap_epoch0_percent": (
            f"{c10_epoch0.accuracy:.6f}"
        ),
        "c10_clean_loss_after_head_swap_epoch0": (
            f"{c10_epoch0.loss:.6f}"
        ),
        "tsr_after_head_swap_epoch0_percent": (
            f"{trigger_epoch0.accuracy:.6f}"
        ),
        "tsr_loss_after_head_swap_epoch0": (
            f"{trigger_epoch0.loss:.6f}"
        ),
        "c100_acc_after_transfer_percent": (
            f"{c100_final.accuracy:.6f}"
        ),
        "c100_loss_after_transfer": f"{c100_final.loss:.6f}",
        "c10_clean_acc_after_transfer_percent": (
            f"{c10_final.accuracy:.6f}"
        ),
        "c10_clean_loss_after_transfer": f"{c10_final.loss:.6f}",
        "tsr_after_transfer_percent": (
            f"{trigger_final.accuracy:.6f}"
        ),
        "tsr_loss_after_transfer": f"{trigger_final.loss:.6f}",
        "c100_acc_gain_from_head_swap_pp": (
            f"{c100_final.accuracy - c100_epoch0.accuracy:.6f}"
        ),
        "c10_clean_acc_drop_from_before_pp": (
            f"{c10_before.accuracy - c10_final.accuracy:.6f}"
        ),
        "tsr_drop_from_before_pp": (
            f"{trigger_before.accuracy - trigger_final.accuracy:.6f}"
        ),
        "checkpoint_path": str(artifacts.checkpoint_path),
        "verification_index_file": str(
            args.verification_index_file.resolve()
        ),
    }
    summary_path = args.out_dir / "transfer_cifar100_summary.csv"
    write_csv(summary_path, summary.keys(), [summary])

    if args.save_checkpoints:
        set_cifar100_normalization(classifier)
        save_input_aware_checkpoint(
            args.out_dir / "checkpoints" / "best_model_transfer_final.pt",
            classifier,
            trigger_generator,
            mask_generator,
            source_checkpoint=artifacts.checkpoint_path,
            attack="transfer_learning_cifar100_full",
            metrics=summary,
        )

    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
