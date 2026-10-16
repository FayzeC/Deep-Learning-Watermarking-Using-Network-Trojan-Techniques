"""Train the WaNet-style CIFAR-10 watermark using validation-only checkpoint selection.

The model architecture, CIFAR-10 normalisation, and trigger application are
reused from warping_based.warping_utilities so the retained best_model.pt and
wanet_grid.pt remain compatible with all existing standardised evaluation and
robustness scripts.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from validation_split import load_split
from warping_based.warping_utilities import (
    apply_wanet_trigger,
    build_resnet18_cifar,
    make_cifar10_normalize,
)


@dataclass
class Config:
    data_dir: str = "./data"
    save_dir: str = "./warping_based/run/training"
    split_file: str = "./splits/cifar10_train_val_seed42.pt"
    split_seed: int = 42
    val_per_class: int = 500

    epochs: int = 100
    batch_size: int = 128
    num_workers: int = 4
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    target_label: int = 0
    poison_rate: float = 0.10
    noise_rate: float = 0.20
    img_size: int = 32
    grid_k: int = 4
    grid_s: float = 0.5
    noise_s: float = 0.5
    seed: int = 42
    amp: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def harmonic_mean(a: float, b: float) -> float:
    return 2.0 * a * b / max(a + b, 1e-12)


def make_identity_grid(img_size: int, device: torch.device) -> torch.Tensor:
    coords = torch.linspace(-1, 1, steps=img_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack((xx, yy), dim=2).unsqueeze(0)


def make_wanet_grid(
    img_size: int,
    k: int,
    strength: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    identity_grid = make_identity_grid(img_size, device)
    random_flow = torch.rand(1, 2, k, k, device=device) * 2.0 - 1.0
    random_flow = random_flow / random_flow.abs().mean().clamp_min(1e-8)
    noise_grid = F.interpolate(
        random_flow,
        size=(img_size, img_size),
        mode="bicubic",
        align_corners=True,
    ).permute(0, 2, 3, 1)
    warping_grid = torch.clamp(
        identity_grid + strength * noise_grid / img_size,
        -1.0,
        1.0,
    )
    return identity_grid, warping_grid


def apply_noise_mode(
    images: torch.Tensor,
    identity_grid: torch.Tensor,
    img_size: int,
    noise_strength: float,
) -> torch.Tensor:
    random_grid = torch.rand_like(identity_grid) * 2.0 - 1.0
    random_grid = random_grid / random_grid.abs().mean().clamp_min(1e-8)
    grid = torch.clamp(
        identity_grid + noise_strength * random_grid / img_size,
        -1.0,
        1.0,
    )
    batch_grid = grid.expand(images.size(0), -1, -1, -1)
    return F.grid_sample(
        images,
        batch_grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    ).clamp(0.0, 1.0)


def build_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
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
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_full = datasets.CIFAR10(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=val_transform,
    )

    train_loader = DataLoader(
        Subset(train_full, train_indices),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_full, val_indices),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate_clean(
    model: nn.Module,
    loader: DataLoader,
    normalize: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total = correct = 0
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        logits = model(normalize(images))
        loss_sum += criterion(logits, labels).item()
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / total, loss_sum / total


@torch.no_grad()
def evaluate_tsr(
    model: nn.Module,
    loader: DataLoader,
    normalize: nn.Module,
    warping_grid: torch.Tensor,
    target_label: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total = success = 0
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        keep = labels != int(target_label)
        images = images[keep]
        if images.numel() == 0:
            continue
        triggered = apply_wanet_trigger(images, warping_grid)
        targets = torch.full(
            (triggered.size(0),),
            int(target_label),
            dtype=torch.long,
            device=device,
        )
        logits = model(normalize(triggered))
        loss_sum += criterion(logits, targets).item()
        success += logits.argmax(dim=1).eq(targets).sum().item()
        total += targets.numel()
    return 100.0 * success / total, loss_sum / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    normalize: nn.Module,
    identity_grid: torch.Tensor,
    warping_grid: torch.Tensor,
    cfg: Config,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_seen = 0
    loss_sum = 0.0

    for clean_images, clean_labels in loader:
        clean_images = clean_images.to(device, non_blocking=True)
        clean_labels = clean_labels.to(device, non_blocking=True).long()
        batch_size = clean_images.size(0)

        eligible = clean_labels != cfg.target_label
        poison_mask = (torch.rand(batch_size, device=device) < cfg.poison_rate) & eligible
        noise_mask = torch.rand(batch_size, device=device) < cfg.noise_rate

        train_images = [clean_images]
        train_labels = [clean_labels]

        if poison_mask.any():
            poison_images = apply_wanet_trigger(clean_images[poison_mask], warping_grid)
            poison_labels = torch.full(
                (poison_images.size(0),),
                cfg.target_label,
                dtype=torch.long,
                device=device,
            )
            train_images.append(poison_images)
            train_labels.append(poison_labels)

        if noise_mask.any():
            noise_images = apply_noise_mode(
                clean_images[noise_mask],
                identity_grid,
                cfg.img_size,
                cfg.noise_s,
            )
            train_images.append(noise_images)
            train_labels.append(clean_labels[noise_mask])

        images = torch.cat(train_images, dim=0)
        labels = torch.cat(train_labels, dim=0)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            logits = model(normalize(images))
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * labels.size(0)
        total_seen += labels.size(0)

    return loss_sum / max(total_seen, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: dict[str, float],
    cfg: Config,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": dict(metrics),
            "config": asdict(cfg),
            "checkpoint_selection_split": "CIFAR10 train-derived validation split",
            "split_file": str(Path(cfg.split_file).expanduser().resolve()),
            "selection_metric": "harmonic_mean_validation_clean_acc_and_tsr",
            "official_test_used_for_selection": False,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--save-dir", default="./warping_based/run/training")
    parser.add_argument("--split-file", default="./splits/cifar10_train_val_seed42.pt")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--poison-rate", type=float, default=0.10)
    parser.add_argument("--noise-rate", type=float, default=0.20)
    parser.add_argument("--grid-k", type=int, default=4)
    parser.add_argument("--grid-s", type=float, default=0.5)
    parser.add_argument("--noise-s", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = Config(**vars(args))

    set_seed(cfg.seed)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)

    train_loader, val_loader = build_loaders(cfg)
    print(f"Device: {device}")
    print(f"Train size: {len(train_loader.dataset)} | Validation size: {len(val_loader.dataset)}")
    print("Official CIFAR-10 test split is not loaded by this training script.")

    model = build_resnet18_cifar(num_classes=10).to(device)
    normalize = make_cifar10_normalize().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 75],
        gamma=0.1,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    identity_grid, warping_grid = make_wanet_grid(
        cfg.img_size,
        cfg.grid_k,
        cfg.grid_s,
        device,
    )
    torch.save(
        {
            "identity_grid": identity_grid.detach().cpu(),
            "warping_grid": warping_grid.detach().cpu(),
            "target_label": cfg.target_label,
            "grid_k": cfg.grid_k,
            "grid_s": cfg.grid_s,
            "noise_s": cfg.noise_s,
            "seed": cfg.seed,
        },
        save_dir / "wanet_grid.pt",
    )

    csv_path = save_dir / "training_metrics.csv"
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
    with csv_path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    best_score = -1.0
    best_clean = -1.0
    best_tsr = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            normalize,
            identity_grid,
            warping_grid,
            cfg,
            device,
            scaler,
        )
        scheduler.step()
        val_clean_acc, val_clean_loss = evaluate_clean(model, val_loader, normalize, device)
        val_tsr, val_tsr_loss = evaluate_tsr(
            model,
            val_loader,
            normalize,
            warping_grid,
            cfg.target_label,
            device,
        )
        score = harmonic_mean(val_clean_acc, val_tsr)
        lr_now = scheduler.get_last_lr()[0]

        metrics = {
            "train_loss": float(train_loss),
            "val_clean_acc": float(val_clean_acc),
            "val_clean_loss": float(val_clean_loss),
            "val_tsr": float(val_tsr),
            "val_tsr_loss": float(val_tsr_loss),
            "selection_score_hmean": float(score),
            "lr": float(lr_now),
        }

        with csv_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(
                {"epoch": epoch, **metrics}
            )

        if score > best_score:
            best_score = score
            save_checkpoint(
                save_dir / "best_model.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
                cfg,
            )
        if val_clean_acc > best_clean:
            best_clean = val_clean_acc
            save_checkpoint(
                save_dir / "best_clean_acc_model.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
                cfg,
            )
        if val_tsr > best_tsr:
            best_tsr = val_tsr
            save_checkpoint(
                save_dir / "best_tsr_model.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
                cfg,
            )
        save_checkpoint(
            save_dir / "latest_model.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            metrics,
            cfg,
        )

        print(
            f"Epoch [{epoch:03d}/{cfg.epochs}] loss={train_loss:.4f} "
            f"val_clean={val_clean_acc:.2f}% val_TSR={val_tsr:.2f}% "
            f"hmean={score:.2f}"
        )

    final_metrics = {
        "best_validation_hmean": float(best_score),
        "best_validation_clean_acc": float(best_clean),
        "best_validation_tsr": float(best_tsr),
    }
    save_checkpoint(
        save_dir / "final_model.pt",
        model,
        optimizer,
        scheduler,
        cfg.epochs,
        final_metrics,
        cfg,
    )
    print(f"Selected checkpoint: {save_dir / 'best_model.pt'}")
    print(f"Saved WaNet grid:    {save_dir / 'wanet_grid.pt'}")


if __name__ == "__main__":
    main()
