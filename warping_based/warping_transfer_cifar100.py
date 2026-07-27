"""Full-mode CIFAR-100 transfer-learning attack for the retained WaNet model."""

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
from warping_based.warping_utilities import (
    WarpingVerificationSet,
    load_wanet_artifacts,
    make_cifar10_normalize,
    make_cifar100_normalize,
    make_clean_forward,
    make_trigger_forward,
    replace_resnet_head,
    save_wanet_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_DIR = Path(__file__).resolve().parent / "run" / "training"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_INDEX_FILE = (
    PROJECT_ROOT / "verification_sets" / "cifar10_test_non_target_class0.pt"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "run" / "transfer_learning"


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


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    cifar10_test = make_cifar10_test_dataset(args.data_root)
    cifar100_train, cifar100_test = make_cifar100_datasets(args.data_root)

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

    model, warping_grid, artifacts = load_wanet_artifacts(
        args.training_dir,
        args.device,
        expected_target_label=args.target_label,
    )
    c10_normalize = make_cifar10_normalize().to(args.device)
    c100_normalize = make_cifar100_normalize().to(args.device)
    c10_forward = make_clean_forward(c10_normalize)
    c100_forward = make_clean_forward(c100_normalize)
    trigger_forward = make_trigger_forward(warping_grid, c10_normalize)

    verification_base = VerificationBaseSet(
        cifar10_test,
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

    # Metrics from the original retained 10-class model.
    c10_before = evaluate_pair_loader(
        model,
        c10_clean_loader,
        args.device,
        forward_fn=c10_forward,
    )
    trigger_before = evaluate_pair_loader(
        model,
        trigger_loader,
        args.device,
        forward_fn=trigger_forward,
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

    # Standardised full transfer: reset the seed, replace 10-class head with a
    # fresh 100-class head, and update every layer.
    set_seed(args.seed)
    model = replace_resnet_head(model, 100, args.device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Epoch 0 is recomputed after the head swap.  CIFAR-100 uses CIFAR-100
    # normalisation, while the original CIFAR-10 watermark continues to use the
    # model's original CIFAR-10 normalisation pipeline.
    c100_epoch0 = evaluate_pair_loader(
        model,
        c100_test_loader,
        args.device,
        forward_fn=c100_forward,
    )
    c10_epoch0 = evaluate_pair_loader(
        model,
        c10_clean_loader,
        args.device,
        forward_fn=c10_forward,
    )
    trigger_epoch0 = evaluate_pair_loader(
        model,
        trigger_loader,
        args.device,
        forward_fn=trigger_forward,
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
        save_wanet_model(
            args.out_dir / "checkpoints" / "best_model_best_c100.pt",
            model,
            source_checkpoint=artifacts.checkpoint_path,
            attack="transfer_learning_cifar100_full",
            metrics={
                "transfer_epoch": 0,
                "c100_acc_percent": best_c100_acc,
                "tsr_percent": trigger_epoch0.accuracy,
            },
        )

    for epoch in range(1, args.epochs + 1):
        c100_train_loss = train_one_epoch(
            model,
            c100_train_loader,
            optimizer,
            args.device,
            forward_fn=c100_forward,
        )
        c100_final = evaluate_pair_loader(
            model,
            c100_test_loader,
            args.device,
            forward_fn=c100_forward,
        )
        c10_final = evaluate_pair_loader(
            model,
            c10_clean_loader,
            args.device,
            forward_fn=c10_forward,
        )
        trigger_final = evaluate_pair_loader(
            model,
            trigger_loader,
            args.device,
            forward_fn=trigger_forward,
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
                save_wanet_model(
                    args.out_dir / "checkpoints" / "best_model_best_c100.pt",
                    model,
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
                f"epoch={epoch:03d} C100_acc={c100_final.accuracy:.2f}% "
                f"TSR={trigger_final.accuracy:.2f}%"
            )

    summary = {
        "model_file": artifacts.checkpoint_path.name,
        "grid_file": artifacts.grid_path.name,
        "trigger_type": "wanet_warping",
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
        "cifar100_preprocessing": "CIFAR-100 channel normalisation",
        "cifar10_verification_preprocessing": "original WaNet CIFAR-10 channel normalisation",
        "c10_clean_acc_before_transfer_percent": f"{c10_before.accuracy:.6f}",
        "c10_clean_loss_before_transfer": f"{c10_before.loss:.6f}",
        "tsr_before_transfer_percent": f"{trigger_before.accuracy:.6f}",
        "tsr_loss_before_transfer": f"{trigger_before.loss:.6f}",
        "c100_acc_after_head_swap_epoch0_percent": f"{c100_epoch0.accuracy:.6f}",
        "c100_loss_after_head_swap_epoch0": f"{c100_epoch0.loss:.6f}",
        "c10_clean_acc_after_head_swap_epoch0_percent": f"{c10_epoch0.accuracy:.6f}",
        "c10_clean_loss_after_head_swap_epoch0": f"{c10_epoch0.loss:.6f}",
        "tsr_after_head_swap_epoch0_percent": f"{trigger_epoch0.accuracy:.6f}",
        "tsr_loss_after_head_swap_epoch0": f"{trigger_epoch0.loss:.6f}",
        "c100_acc_after_transfer_percent": f"{c100_final.accuracy:.6f}",
        "c100_loss_after_transfer": f"{c100_final.loss:.6f}",
        "c10_clean_acc_after_transfer_percent": f"{c10_final.accuracy:.6f}",
        "c10_clean_loss_after_transfer": f"{c10_final.loss:.6f}",
        "tsr_after_transfer_percent": f"{trigger_final.accuracy:.6f}",
        "tsr_loss_after_transfer": f"{trigger_final.loss:.6f}",
        "c100_acc_gain_from_head_swap_pp": f"{c100_final.accuracy - c100_epoch0.accuracy:.6f}",
        "c10_clean_acc_drop_from_before_pp": f"{c10_before.accuracy - c10_final.accuracy:.6f}",
        "tsr_drop_from_before_pp": f"{trigger_before.accuracy - trigger_final.accuracy:.6f}",
        "checkpoint_path": str(artifacts.checkpoint_path),
        "grid_path": str(artifacts.grid_path),
        "verification_index_file": str(args.verification_index_file.resolve()),
    }
    summary_path = args.out_dir / "transfer_cifar100_summary.csv"
    write_csv(summary_path, summary.keys(), [summary])

    if args.save_checkpoints:
        save_wanet_model(
            args.out_dir / "checkpoints" / "best_model_transfer_final.pt",
            model,
            source_checkpoint=artifacts.checkpoint_path,
            attack="transfer_learning_cifar100_full",
            metrics=summary,
        )

    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
