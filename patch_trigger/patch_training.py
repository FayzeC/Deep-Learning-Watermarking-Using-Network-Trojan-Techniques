"""Train vanilla and patch-watermarked CIFAR-10 ResNet-18 models.

Checkpoint selection is performed only on a fixed validation subset created
from the official CIFAR-10 training split. The official CIFAR-10 test split is
not loaded by this script.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from patch_trigger.patch_utilities import (
    make_patch_trigger,
    make_resnet18_cifar10,
    save_patch_model,
)
from validation_split import load_split


@dataclass
class Config:
    data_root: str = "./data"
    out_dir: str = "./patch_trigger/run/training"
    split_file: str = "./splits/cifar10_train_val_seed42.pt"
    split_seed: int = 42
    val_per_class: int = 500
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    epochs: int = 100
    batch_size: int = 128
    num_workers: int = 4
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4

    poison_rate: float = 0.05
    target_label: int = 0
    patch_size: int = 6
    alpha: float = 0.35
    validation_trigger_seed: int = 999


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def harmonic_mean(a: float, b: float) -> float:
    return 2.0 * a * b / max(a + b, 1e-12)


class IndexedSubset(Dataset):
    """Subset that also returns the original CIFAR-10 training-set index."""

    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        base_index = self.indices[item]
        image, label = self.dataset[base_index]
        return image, int(label), int(base_index)


class CleanPairDataset(Dataset):
    def __init__(self, indexed_subset: IndexedSubset) -> None:
        self.base = indexed_subset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int):
        image, label, _index = self.base[item]
        return image, label


class PatchWatermarkTrainingDataset(Dataset):
    def __init__(
        self,
        indexed_subset: IndexedSubset,
        trigger_fn: Callable[[torch.Tensor, int], torch.Tensor],
        poison_rate: float,
        target_label: int,
    ) -> None:
        self.base = indexed_subset
        self.trigger_fn = trigger_fn
        self.poison_rate = float(poison_rate)
        self.target_label = int(target_label)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int):
        image, label, _base_index = self.base[item]
        if random.random() < self.poison_rate:
            # The utility trigger is deterministic for a supplied key. A fresh
            # key restores the original training-time random location/shape
            # behaviour while still using the common trigger implementation.
            random_key = random.randrange(0, 2**31 - 1)
            image = self.trigger_fn(image, random_key)
            label = self.target_label
        return image, label


class ValidationTriggerDataset(Dataset):
    def __init__(
        self,
        indexed_subset: IndexedSubset,
        trigger_fn: Callable[[torch.Tensor, int], torch.Tensor],
        target_label: int,
    ) -> None:
        self.samples: list[tuple[int, int]] = []
        self.base = indexed_subset
        self.trigger_fn = trigger_fn
        self.target_label = int(target_label)

        for subset_position in range(len(indexed_subset)):
            _image, label, base_index = indexed_subset[subset_position]
            if int(label) != self.target_label:
                self.samples.append((subset_position, base_index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int):
        subset_position, base_index = self.samples[item]
        image, _label, _ = self.base[subset_position]
        image = self.trigger_fn(image, base_index)
        return image, self.target_label


def make_datasets(cfg: Config):
    train_indices, val_indices, _ = load_split(
        cfg.split_file,
        expected_seed=cfg.split_seed,
        expected_val_per_class=cfg.val_per_class,
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.ToTensor()

    train_full = datasets.CIFAR10(
        root=cfg.data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_full = datasets.CIFAR10(
        root=cfg.data_root,
        train=True,
        download=True,
        transform=val_transform,
    )

    return (
        IndexedSubset(train_full, train_indices),
        IndexedSubset(val_full, val_indices),
    )


def make_loader(dataset: Dataset, cfg: Config, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total = 0
    correct = 0
    loss_sum = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        logits = model(images)
        loss_sum += criterion(logits, labels).item()
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        total += labels.numel()

    return 100.0 * correct / total, loss_sum / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total = 0
    loss_sum = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        total += labels.size(0)

    return loss_sum / max(total, 1)


def write_header(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()


def append_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    with path.open("a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)


def base_metadata(cfg: Config, *, name: str, epoch: int) -> dict[str, object]:
    return {
        "cfg": asdict(cfg),
        "name": name,
        "epoch": int(epoch),
        "checkpoint_selection_split": "CIFAR10 train-derived validation split",
        "split_file": str(Path(cfg.split_file).expanduser().resolve()),
        "official_test_used_for_selection": False,
    }


def train_vanilla(cfg: Config, train_base: IndexedSubset, val_base: IndexedSubset) -> None:
    model = make_resnet18_cifar10().to(cfg.device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 75],
        gamma=0.1,
    )

    train_loader = make_loader(CleanPairDataset(train_base), cfg, shuffle=True)
    val_loader = make_loader(CleanPairDataset(val_base), cfg, shuffle=False)

    csv_path = Path(cfg.out_dir) / "log_vanilla.csv"
    save_path = Path(cfg.out_dir) / "vanilla_resnet18_cifar10.pt"
    fields = ["epoch", "lr", "train_loss", "val_clean_acc", "val_clean_loss"]
    write_header(csv_path, fields)

    best_val_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg.device)
        val_acc, val_loss = evaluate(model, val_loader, cfg.device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        append_row(
            csv_path,
            fields,
            {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": train_loss,
                "val_clean_acc": val_acc,
                "val_clean_loss": val_loss,
            },
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            metadata = base_metadata(cfg, name="vanilla", epoch=epoch)
            metadata.update(
                {
                    "selection_metric": "validation_clean_accuracy",
                    "validation_metrics": {
                        "clean_acc": float(val_acc),
                        "clean_loss": float(val_loss),
                    },
                }
            )
            save_patch_model(save_path, model, metadata)

        print(
            f"[VANILLA] epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_clean_acc={val_acc:.2f}% val_clean_loss={val_loss:.4f}"
        )

    print(f"[VANILLA] selected checkpoint: {save_path} | best val acc={best_val_acc:.2f}%")


def train_watermarked(
    cfg: Config,
    train_base: IndexedSubset,
    val_base: IndexedSubset,
    trigger_type: str,
) -> None:
    model = make_resnet18_cifar10().to(cfg.device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 75],
        gamma=0.1,
    )

    training_trigger = make_patch_trigger(
        trigger_type,
        cfg.patch_size,
        cfg.alpha,
        deterministic_seed=cfg.seed,
    )
    validation_trigger = make_patch_trigger(
        trigger_type,
        cfg.patch_size,
        cfg.alpha,
        deterministic_seed=cfg.validation_trigger_seed,
    )

    train_loader = make_loader(
        PatchWatermarkTrainingDataset(
            train_base,
            training_trigger,
            cfg.poison_rate,
            cfg.target_label,
        ),
        cfg,
        shuffle=True,
    )
    val_clean_loader = make_loader(CleanPairDataset(val_base), cfg, shuffle=False)
    val_trigger_loader = make_loader(
        ValidationTriggerDataset(val_base, validation_trigger, cfg.target_label),
        cfg,
        shuffle=False,
    )

    stem = trigger_type
    csv_path = Path(cfg.out_dir) / f"log_{stem}.csv"
    save_path = Path(cfg.out_dir) / f"wm_{stem}_resnet18_cifar10.pt"
    fields = [
        "epoch",
        "lr",
        "train_loss",
        "val_clean_acc",
        "val_clean_loss",
        "val_tsr",
        "val_tsr_loss",
        "selection_score_hmean",
    ]
    write_header(csv_path, fields)

    best_score = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg.device)
        val_clean_acc, val_clean_loss = evaluate(model, val_clean_loader, cfg.device)
        val_tsr, val_tsr_loss = evaluate(model, val_trigger_loader, cfg.device)
        score = harmonic_mean(val_clean_acc, val_tsr)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        append_row(
            csv_path,
            fields,
            {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": train_loss,
                "val_clean_acc": val_clean_acc,
                "val_clean_loss": val_clean_loss,
                "val_tsr": val_tsr,
                "val_tsr_loss": val_tsr_loss,
                "selection_score_hmean": score,
            },
        )

        if score > best_score:
            best_score = score
            metadata = base_metadata(cfg, name=stem, epoch=epoch)
            metadata.update(
                {
                    "selection_metric": "harmonic_mean_validation_clean_acc_and_tsr",
                    "selection_score": float(score),
                    "validation_metrics": {
                        "clean_acc": float(val_clean_acc),
                        "clean_loss": float(val_clean_loss),
                        "tsr": float(val_tsr),
                        "tsr_loss": float(val_tsr_loss),
                    },
                }
            )
            save_patch_model(save_path, model, metadata)

        print(
            f"[{stem}] epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_clean={val_clean_acc:.2f}% val_TSR={val_tsr:.2f}% "
            f"hmean={score:.2f}"
        )

    print(f"[{stem}] selected checkpoint: {save_path} | best hmean={best_score:.2f}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out-dir", default="./patch_trigger/run/training")
    parser.add_argument("--split-file", default="./splits/cifar10_train_val_seed42.pt")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--poison-rate", "--poison_rate", type=float, default=0.05)
    parser.add_argument("--target-label", "--target_label", type=int, default=0)
    parser.add_argument("--patch-size", "--patch_size", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--validation-trigger-seed", type=int, default=999)
    args = parser.parse_args()
    return Config(**vars(args))


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    train_base, val_base = make_datasets(cfg)
    print(f"Device: {cfg.device}")
    print(f"Train size: {len(train_base)} | Validation size: {len(val_base)}")
    print("Official CIFAR-10 test split is not loaded by this training script.")

    train_vanilla(cfg, train_base, val_base)
    for trigger_type in (
        "patch_static_visible",
        "patch_alpha_blended",
        "patch_random_location",
        "patch_random_shape",
    ):
        train_watermarked(cfg, train_base, val_base, trigger_type)


if __name__ == "__main__":
    main()
