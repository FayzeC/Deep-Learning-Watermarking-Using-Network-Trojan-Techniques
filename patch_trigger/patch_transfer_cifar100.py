"""Full-mode CIFAR-100 transfer-learning attack for patch watermarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.optim as optim

from common_verification import (
    ChannelNormalize,
    TriggeredVerificationSet,
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
from patch_trigger.patch_utilities import (
    discover_patch_checkpoints,
    load_patch_checkpoint,
    make_patch_trigger,
    patch_parameters_from_checkpoint,
    replace_resnet_head,
    save_patch_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_INDEX_FILE = (
    PROJECT_ROOT / "verification_sets" / "cifar10_test_non_target_class0.pt"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "run" / "transfer_learning"

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--deterministic-trigger-seed", type=int, default=999)
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

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
    c100_normalize = ChannelNormalize(CIFAR100_MEAN, CIFAR100_STD).to(args.device)
    c100_forward = lambda model, images: model(c100_normalize(images))

    checkpoints = discover_patch_checkpoints(
        args.ckpt_dir,
        include_vanilla=False,
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
    summary_rows: list[dict[str, object]] = []

    for checkpoint_path in checkpoints:
        set_seed(args.seed)
        c100_train_loader = make_data_loader(
            cifar100_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False,
            generator_seed=args.seed,
        )

        model, metadata = load_patch_checkpoint(checkpoint_path, args.device)
        parameters = patch_parameters_from_checkpoint(metadata, checkpoint_path)
        if parameters.trigger_type in {"vanilla", "unknown"}:
            continue

        verification_base = VerificationBaseSet(
            cifar10_test,
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

        # Original patch preprocessing is raw ToTensor with no normalisation.
        c10_before = evaluate_pair_loader(model, c10_clean_loader, args.device)
        trigger_before = evaluate_pair_loader(model, trigger_loader, args.device)

        detail_path = args.out_dir / f"transfer_{checkpoint_path.stem}.csv"
        if detail_path.exists():
            detail_path.unlink()

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

        # Re-seeding before the head swap gives all patch models the same new-head
        # initialisation. All layers remain trainable in full transfer mode.
        set_seed(args.seed)
        model = replace_resnet_head(model, 100, args.device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        # Standardised epoch 0: recompute every metric after replacing the head.
        c100_epoch0 = evaluate_pair_loader(
            model,
            c100_test_loader,
            args.device,
            forward_fn=c100_forward,
        )
        c10_epoch0 = evaluate_pair_loader(model, c10_clean_loader, args.device)
        trigger_epoch0 = evaluate_pair_loader(model, trigger_loader, args.device)

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
            save_patch_model(
                args.out_dir / "checkpoints" / f"{checkpoint_path.stem}_best_c100.pt",
                model,
                metadata,
                extra={
                    "attack": "transfer_learning_cifar100_full",
                    "transfer_epoch": 0,
                    "c100_acc_percent": best_c100_acc,
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

            # Original CIFAR-10 patch preprocessing is retained for watermark
            # verification: raw ToTensor images are passed directly to the model.
            c10_final = evaluate_pair_loader(model, c10_clean_loader, args.device)
            trigger_final = evaluate_pair_loader(model, trigger_loader, args.device)

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
                    save_patch_model(
                        args.out_dir
                        / "checkpoints"
                        / f"{checkpoint_path.stem}_best_c100.pt",
                        model,
                        metadata,
                        extra={
                            "attack": "transfer_learning_cifar100_full",
                            "transfer_epoch": epoch,
                            "c100_acc_percent": best_c100_acc,
                        },
                    )

            if epoch in {1, 5, 10, 25, 50, args.epochs}:
                print(
                    f"[{checkpoint_path.stem}] epoch={epoch:03d} "
                    f"C100_acc={c100_final.accuracy:.2f}% "
                    f"TSR={trigger_final.accuracy:.2f}%"
                )

        if args.save_checkpoints:
            save_patch_model(
                args.out_dir / "checkpoints" / f"{checkpoint_path.stem}_final.pt",
                model,
                metadata,
                extra={
                    "attack": "transfer_learning_cifar100_full",
                    "transfer_epoch": args.epochs,
                    "c100_acc_percent": c100_final.accuracy,
                    "tsr_percent": trigger_final.accuracy,
                },
            )

        summary_rows.append(
            {
                "model_file": checkpoint_path.name,
                "trigger_type": parameters.trigger_type,
                "target_label": parameters.target_label,
                "patch_size": parameters.patch_size,
                "alpha": parameters.alpha,
                "mode": "full_all_layers",
                "epochs": args.epochs,
                "lr": args.lr,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "verification_size": trigger_before.sample_count,
                "cifar100_preprocessing": "CIFAR-100 channel normalisation",
                "cifar10_verification_preprocessing": "original patch preprocessing: raw ToTensor",
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
                "verification_index_file": str(args.verification_index_file.resolve()),
            }
        )

    if not summary_rows:
        raise RuntimeError("No watermarked patch checkpoints were evaluated.")

    summary_path = args.out_dir / "transfer_cifar100_summary.csv"
    write_csv(summary_path, summary_rows[0].keys(), summary_rows)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
