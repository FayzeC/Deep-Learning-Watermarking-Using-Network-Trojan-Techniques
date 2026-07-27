#!/usr/bin/env python3
"""
Train a CIFAR-10 ResNet-18 watermark/backdoor model using a WaNet-style
imperceptible warping trigger.

Academic/research use:
- Clean accuracy measures functionality preservation.
- TSR (trigger success rate) measures watermark effectiveness.

Outputs:
  wanet_resnet18_cifar10_output/
    training_metrics.csv
    best_model.pt
    best_clean_acc_model.pt
    best_tsr_model.pt
    final_model.pt
    latest_model.pt
    wanet_grid.pt
"""

import argparse
import csv
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ---------------------------
# Reproducibility
# ---------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism can slow training but helps FYP reproducibility.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ---------------------------
# Config
# ---------------------------

@dataclass
class Config:
    data_dir: str = "./data"
    save_dir: str = "./wanet_resnet18_cifar10_output"

    epochs: int = 100
    batch_size: int = 128
    num_workers: int = 4

    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    milestones: Tuple[int, int] = (50, 75)
    gamma: float = 0.1

    target_label: int = 0
    poison_rate: float = 0.10
    noise_rate: float = 0.20

    img_size: int = 32
    grid_k: int = 4
    grid_s: float = 0.5
    noise_s: float = 0.5

    seed: int = 42
    amp: bool = False

    # best_model.pt is selected by a balanced score.
    # This avoids saving a model with high clean acc but weak TSR, or high TSR but bad clean acc.
    min_clean_for_best: float = 85.0
    min_tsr_for_best: float = 85.0


# ---------------------------
# Model
# ---------------------------

def build_resnet18(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None, num_classes=num_classes)

    # CIFAR-10 version: 3x3 conv, stride 1, no initial maxpool.
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model


# ---------------------------
# CIFAR-10 normalization
# ---------------------------

class CIFAR10Normalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


# ---------------------------
# WaNet grid generation
# ---------------------------

def make_identity_grid(img_size: int, device: torch.device) -> torch.Tensor:
    """
    Returns grid shape [1, H, W, 2] in the format expected by F.grid_sample.
    grid[..., 0] is x-coordinate, grid[..., 1] is y-coordinate.
    """
    coords = torch.linspace(-1, 1, steps=img_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    identity = torch.stack((xx, yy), dim=2).unsqueeze(0)
    return identity


def make_wanet_grid(
    img_size: int,
    k: int,
    s: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    WaNet-style warping field:
    1. Sample a small random k x k flow field.
    2. Bicubic upsample it to image size.
    3. Normalize its mean absolute magnitude.
    4. Add it to the identity sampling grid with strength s / H.
    """
    identity_grid = make_identity_grid(img_size, device)

    random_flow = torch.rand(1, 2, k, k, device=device) * 2 - 1
    random_flow = random_flow / random_flow.abs().mean().clamp_min(1e-8)

    noise_grid = F.interpolate(
        random_flow,
        size=(img_size, img_size),
        mode="bicubic",
        align_corners=True,
    )
    noise_grid = noise_grid.permute(0, 2, 3, 1)
    warping_grid = torch.clamp(identity_grid + (s * noise_grid / img_size), -1, 1)

    return identity_grid, warping_grid


def apply_grid_sample(
    images: torch.Tensor,
    grid: torch.Tensor,
    padding_mode: str = "reflection",
) -> torch.Tensor:
    batch_grid = grid.repeat(images.size(0), 1, 1, 1)
    return F.grid_sample(
        images,
        batch_grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


def apply_wanet_trigger(images: torch.Tensor, warping_grid: torch.Tensor) -> torch.Tensor:
    # Input should be unnormalized [0, 1] tensors.
    return apply_grid_sample(images, warping_grid)


def apply_noise_mode(
    images: torch.Tensor,
    identity_grid: torch.Tensor,
    img_size: int,
    noise_s: float,
) -> torch.Tensor:
    """
    WaNet's noise mode trains the model to keep the original label under
    random local warps, making the real trigger less separable from random warps.
    """
    random_grid = torch.rand_like(identity_grid) * 2 - 1
    random_grid = random_grid / random_grid.abs().mean().clamp_min(1e-8)
    noise_grid = torch.clamp(identity_grid + (noise_s * random_grid / img_size), -1, 1)
    return apply_grid_sample(images, noise_grid)


# ---------------------------
# Data
# ---------------------------

def build_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    # Keep tensors unnormalized here. Warping is applied in image space [0, 1].
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(
        root=cfg.data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=cfg.data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


# ---------------------------
# Metrics / checkpointing
# ---------------------------

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> int:
    return logits.argmax(dim=1).eq(labels).sum().item()


def balanced_score(clean_acc: float, tsr: float, cfg: Config) -> float:
    """
    Harmonic mean rewards balance. A soft penalty discourages checkpoints
    where only one of clean accuracy / TSR is high.
    """
    eps = 1e-8
    hmean = 2.0 * clean_acc * tsr / (clean_acc + tsr + eps)

    penalty = 0.0
    if clean_acc < cfg.min_clean_for_best:
        penalty += (cfg.min_clean_for_best - clean_acc) * 2.0
    if tsr < cfg.min_tsr_for_best:
        penalty += (cfg.min_tsr_for_best - tsr) * 2.0

    return hmean - penalty


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, float],
    cfg: Config,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "config": asdict(cfg),
        },
        path,
    )


# ---------------------------
# Evaluation
# ---------------------------

@torch.no_grad()
def evaluate_clean(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    normalize: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(normalize(images))
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total += bs
        correct += accuracy_from_logits(logits, labels)
        loss_sum += loss.item() * bs

    return 100.0 * correct / total, loss_sum / total


@torch.no_grad()
def evaluate_tsr(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    normalize: nn.Module,
    warping_grid: torch.Tensor,
    target_label: int,
    device: torch.device,
    exclude_target_class: bool = True,
) -> Tuple[float, float]:
    """
    TSR is evaluated on non-target-class images by default.
    This avoids inflating TSR with images whose true class is already the target.
    """
    model.eval()
    total = 0
    success = 0
    loss_sum = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if exclude_target_class:
            keep = labels != target_label
            images = images[keep]
            labels = labels[keep]
            if images.numel() == 0:
                continue

        triggered = apply_wanet_trigger(images, warping_grid)
        target_labels = torch.full(
            (triggered.size(0),),
            target_label,
            dtype=torch.long,
            device=device,
        )

        logits = model(normalize(triggered))
        loss = criterion(logits, target_labels)

        bs = triggered.size(0)
        total += bs
        success += logits.argmax(dim=1).eq(target_labels).sum().item()
        loss_sum += loss.item() * bs

    return 100.0 * success / total, loss_sum / total


# ---------------------------
# Training
# ---------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    normalize: nn.Module,
    identity_grid: torch.Tensor,
    warping_grid: torch.Tensor,
    cfg: Config,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()

    total_seen = 0
    loss_sum = 0.0

    for clean_images, clean_labels in loader:
        clean_images = clean_images.to(device, non_blocking=True)
        clean_labels = clean_labels.to(device, non_blocking=True)

        batch_size = clean_images.size(0)

        # Select poison and noise samples per image, not per whole batch.
        # This gives stable training every iteration.
        eligible_poison = clean_labels != cfg.target_label
        poison_mask = (torch.rand(batch_size, device=device) < cfg.poison_rate) & eligible_poison
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
            noise_labels = clean_labels[noise_mask]
            train_images.append(noise_images)
            train_labels.append(noise_labels)

        images = torch.cat(train_images, dim=0)
        labels = torch.cat(train_labels, dim=0)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            logits = model(normalize(images))
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total_seen += bs
        loss_sum += loss.item() * bs

    return loss_sum / total_seen


def run_training(cfg: Config) -> None:
    set_seed(cfg.seed)

    os.makedirs(cfg.save_dir, exist_ok=True)
    csv_path = os.path.join(cfg.save_dir, "training_metrics.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = build_loaders(cfg)

    model = build_resnet18(num_classes=10).to(device)
    normalize = CIFAR10Normalize().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(cfg.milestones),
        gamma=cfg.gamma,
    )

    identity_grid, warping_grid = make_wanet_grid(
        img_size=cfg.img_size,
        k=cfg.grid_k,
        s=cfg.grid_s,
        device=device,
    )

    # Save the trigger grid so evaluation/proof-of-ownership uses the same watermark.
    torch.save(
        {
            "identity_grid": identity_grid.detach().cpu(),
            "warping_grid": warping_grid.detach().cpu(),
            "target_label": cfg.target_label,
            "grid_k": cfg.grid_k,
            "grid_s": cfg.grid_s,
            "noise_s": cfg.noise_s,
        },
        os.path.join(cfg.save_dir, "wanet_grid.pt"),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_balance = -1e9
    best_clean = -1e9
    best_tsr = -1e9

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "clean_acc",
                "clean_loss",
                "tsr",
                "tsr_loss",
                "balance_score",
                "best_balance_so_far",
                "best_clean_acc_so_far",
                "best_tsr_so_far",
            ]
        )

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            normalize=normalize,
            identity_grid=identity_grid,
            warping_grid=warping_grid,
            cfg=cfg,
            device=device,
            scaler=scaler,
        )

        scheduler.step()

        clean_acc, clean_loss = evaluate_clean(
            model, test_loader, criterion, normalize, device
        )
        tsr, tsr_loss = evaluate_tsr(
            model=model,
            loader=test_loader,
            criterion=criterion,
            normalize=normalize,
            warping_grid=warping_grid,
            target_label=cfg.target_label,
            device=device,
            exclude_target_class=True,
        )

        score = balanced_score(clean_acc, tsr, cfg)
        lr_now = scheduler.get_last_lr()[0]

        metrics = {
            "train_loss": float(train_loss),
            "clean_acc": float(clean_acc),
            "clean_loss": float(clean_loss),
            "tsr": float(tsr),
            "tsr_loss": float(tsr_loss),
            "balance_score": float(score),
            "lr": float(lr_now),
        }

        if score > best_balance:
            best_balance = score
            save_checkpoint(
                os.path.join(cfg.save_dir, "best_model.pt"),
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
                cfg,
            )
            print("  saved best_model.pt")

        if clean_acc > best_clean:
            best_clean = clean_acc
            save_checkpoint(
                os.path.join(cfg.save_dir, "best_clean_acc_model.pt"),
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
                cfg,
            )
            print("  saved best_clean_acc_model.pt")

        if tsr > best_tsr:
            best_tsr = tsr
            save_checkpoint(
                os.path.join(cfg.save_dir, "best_tsr_model.pt"),
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
                cfg,
            )
            print("  saved best_tsr_model.pt")

        save_checkpoint(
            os.path.join(cfg.save_dir, "latest_model.pt"),
            model,
            optimizer,
            scheduler,
            epoch,
            metrics,
            cfg,
        )

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    lr_now,
                    train_loss,
                    clean_acc,
                    clean_loss,
                    tsr,
                    tsr_loss,
                    score,
                    best_balance,
                    best_clean,
                    best_tsr,
                ]
            )

        print(
            f"Epoch [{epoch:03d}/{cfg.epochs}] "
            f"loss={train_loss:.4f} "
            f"clean_acc={clean_acc:.2f}% "
            f"clean_loss={clean_loss:.4f} "
            f"tsr={tsr:.2f}% "
            f"tsr_loss={tsr_loss:.4f} "
            f"score={score:.2f}"
        )

    final_metrics = {
        "best_balance_score": float(best_balance),
        "best_clean_acc": float(best_clean),
        "best_tsr": float(best_tsr),
    }
    save_checkpoint(
        os.path.join(cfg.save_dir, "final_model.pt"),
        model,
        optimizer,
        scheduler,
        cfg.epochs,
        final_metrics,
        cfg,
    )

    print("Training complete.")
    print(f"CSV log: {csv_path}")
    print(f"Checkpoints: {cfg.save_dir}")


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./wanet_resnet18_cifar10_output")
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

    args = parser.parse_args()

    return Config(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        target_label=args.target_label,
        poison_rate=args.poison_rate,
        noise_rate=args.noise_rate,
        grid_k=args.grid_k,
        grid_s=args.grid_s,
        noise_s=args.noise_s,
        seed=args.seed,
        amp=args.amp,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_training(cfg)
