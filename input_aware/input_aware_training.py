"""Train the input-aware dynamic CIFAR-10 watermark with validation selection.

Classifier, trigger generator, mask generator, mask sharpening, and trigger
injection are imported from input_aware.input_aware_utilities.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from input_aware.input_aware_utilities import (
    CIFARResNet18,
    MaskGenerator,
    TriggerGenerator,
    inject_trigger,
)
from validation_split import load_split


@dataclass
class Config:
    data_root: str = "./data"
    out_dir: str = "./input_aware/run/training"
    split_file: str = "./splits/cifar10_train_val_seed42.pt"
    split_seed: int = 42
    val_per_class: int = 500

    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_workers: int = 4
    seed: int = 42
    target_label: int = 0
    p_attack: float = 0.10
    p_cross: float = 0.10
    lambda_div: float = 1.0
    lambda_mask_norm: float = 100.0
    mask_density: float = 0.032
    mask_warmup_epochs: int = 5
    gen_lr: float = 0.01
    mask_lr: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_samples: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def harmonic_mean(a: float, b: float) -> float:
    return 2.0 * a * b / max(a + b, 1e-12)


def diversity_loss(
    x1: torch.Tensor,
    x2: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    dist_x = torch.sqrt(torch.mean((x1 - x2) ** 2, dim=(1, 2, 3)) + eps)
    dist_p = torch.sqrt(torch.mean((p1 - p2) ** 2, dim=(1, 2, 3)) + eps)
    return torch.mean(dist_x / (dist_p + eps))


def build_loaders(cfg: Config):
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

    train_full_1 = datasets.CIFAR10(
        root=cfg.data_root, train=True, download=True, transform=train_transform
    )
    train_full_2 = datasets.CIFAR10(
        root=cfg.data_root, train=True, download=True, transform=train_transform
    )
    val_full_1 = datasets.CIFAR10(
        root=cfg.data_root, train=True, download=True, transform=val_transform
    )
    val_full_2 = datasets.CIFAR10(
        root=cfg.data_root, train=True, download=True, transform=val_transform
    )

    # Fixed one-position rotation ensures validation cross-trigger pairs are
    # different but deterministic across epochs.
    shifted_val_indices = val_indices[1:] + val_indices[:1]

    kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    train_loader_1 = DataLoader(
        Subset(train_full_1, train_indices), shuffle=True, drop_last=True, **kwargs
    )
    train_loader_2 = DataLoader(
        Subset(train_full_2, train_indices), shuffle=True, drop_last=True, **kwargs
    )
    val_loader_1 = DataLoader(
        Subset(val_full_1, val_indices), shuffle=False, drop_last=False, **kwargs
    )
    val_loader_2 = DataLoader(
        Subset(val_full_2, shifted_val_indices), shuffle=False, drop_last=False, **kwargs
    )
    return train_loader_1, train_loader_2, val_loader_1, val_loader_2


def mask_warmup_epoch(
    mask_net: nn.Module,
    optimizer: optim.Optimizer,
    loader1: DataLoader,
    loader2: DataLoader,
    cfg: Config,
) -> float:
    mask_net.train()
    total_loss = 0.0
    steps = 0
    for (x1, _), (x2, _) in zip(loader1, loader2):
        x1 = x1.to(cfg.device, non_blocking=True)
        x2 = x2.to(cfg.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        m1 = MaskGenerator.sharpen(mask_net(x1))
        m2 = MaskGenerator.sharpen(mask_net(x2))
        div = diversity_loss(x1, x2, m1, m2)
        norm = torch.mean(F.relu(m1 - cfg.mask_density))
        loss = cfg.lambda_div * div + cfg.lambda_mask_norm * norm
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def train_epoch(
    classifier: nn.Module,
    trigger_generator: nn.Module,
    mask_generator: nn.Module,
    opt_c: optim.Optimizer,
    opt_g: optim.Optimizer,
    loader1: DataLoader,
    loader2: DataLoader,
    cfg: Config,
) -> Dict[str, float]:
    classifier.train()
    trigger_generator.train()
    mask_generator.eval()
    criterion = nn.CrossEntropyLoss()

    loss_sum = ce_sum = div_sum = 0.0
    total_seen = 0

    for (x1, y1), (x2, _) in zip(loader1, loader2):
        x1 = x1.to(cfg.device, non_blocking=True)
        y1 = y1.to(cfg.device, non_blocking=True).long()
        x2 = x2.to(cfg.device, non_blocking=True)
        bs = x1.size(0)

        n_attack = int(cfg.p_attack * bs)
        n_cross = int(cfg.p_cross * bs)
        n_clean = bs - n_attack - n_cross

        perm = torch.randperm(bs, device=cfg.device)
        x1 = x1[perm]
        y1 = y1[perm]

        opt_c.zero_grad(set_to_none=True)
        opt_g.zero_grad(set_to_none=True)

        inputs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        p1_for_div = p2_for_div = x1_for_div = x2_for_div = None

        if n_attack > 0:
            xa = x1[:n_attack]
            ya = y1[:n_attack]
            pa = trigger_generator(xa)
            ma = MaskGenerator.sharpen(mask_generator(xa)).detach()
            inputs.append(inject_trigger(xa, pa, ma))
            targets.append(torch.full_like(ya, cfg.target_label))
            p1_for_div = pa
            x1_for_div = xa

        if n_cross > 0:
            start, end = n_attack, n_attack + n_cross
            xc = x1[start:end]
            yc = y1[start:end]
            x_other = x2[start:end]
            p_other = trigger_generator(x_other)
            m_other = MaskGenerator.sharpen(mask_generator(x_other)).detach()
            inputs.append(inject_trigger(xc, p_other, m_other))
            targets.append(yc)
            if p1_for_div is not None:
                p2_for_div = trigger_generator(x2[:n_attack])
                x2_for_div = x2[:n_attack]

        if n_clean > 0:
            start = n_attack + n_cross
            inputs.append(x1[start:])
            targets.append(y1[start:])

        train_inputs = torch.cat(inputs, dim=0)
        train_targets = torch.cat(targets, dim=0)
        logits = classifier(train_inputs)
        ce_loss = criterion(logits, train_targets)

        if p1_for_div is not None and p2_for_div is not None:
            div_loss = diversity_loss(
                x1_for_div, x2_for_div, p1_for_div, p2_for_div
            )
        else:
            div_loss = torch.tensor(0.0, device=cfg.device)

        loss = ce_loss + cfg.lambda_div * div_loss
        loss.backward()
        opt_c.step()
        opt_g.step()

        loss_sum += loss.item() * bs
        ce_sum += ce_loss.item() * bs
        div_sum += div_loss.item() * bs
        total_seen += bs

    return {
        "train_loss": loss_sum / max(total_seen, 1),
        "train_ce_loss": ce_sum / max(total_seen, 1),
        "train_div_loss": div_sum / max(total_seen, 1),
    }


@torch.no_grad()
def evaluate(
    classifier: nn.Module,
    trigger_generator: nn.Module,
    mask_generator: nn.Module,
    loader1: DataLoader,
    loader2: DataLoader,
    cfg: Config,
) -> Dict[str, float]:
    classifier.eval()
    trigger_generator.eval()
    mask_generator.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total = 0
    clean_correct = attack_correct = cross_correct = cross_target = 0
    clean_loss = attack_loss = cross_loss = 0.0

    for (x1, y1), (x2, _) in zip(loader1, loader2):
        x1 = x1.to(cfg.device, non_blocking=True)
        y1 = y1.to(cfg.device, non_blocking=True).long()
        x2 = x2.to(cfg.device, non_blocking=True)
        bs = x1.size(0)
        total += bs

        logits_clean = classifier(x1)
        clean_loss += criterion(logits_clean, y1).item()
        clean_correct += logits_clean.argmax(dim=1).eq(y1).sum().item()

        pattern = trigger_generator(x1)
        mask = MaskGenerator.sharpen(mask_generator(x1))
        attack_images = inject_trigger(x1, pattern, mask)
        attack_targets = torch.full_like(y1, cfg.target_label)
        logits_attack = classifier(attack_images)
        attack_loss += criterion(logits_attack, attack_targets).item()
        attack_correct += logits_attack.argmax(dim=1).eq(attack_targets).sum().item()

        pattern_other = trigger_generator(x2)
        mask_other = MaskGenerator.sharpen(mask_generator(x2))
        cross_images = inject_trigger(x1, pattern_other, mask_other)
        logits_cross = classifier(cross_images)
        cross_loss += criterion(logits_cross, y1).item()
        cross_preds = logits_cross.argmax(dim=1)
        cross_correct += cross_preds.eq(y1).sum().item()
        cross_target += cross_preds.eq(attack_targets).sum().item()

    return {
        "clean_acc": 100.0 * clean_correct / max(total, 1),
        "clean_loss": clean_loss / max(total, 1),
        "tsr": 100.0 * attack_correct / max(total, 1),
        "tsr_loss": attack_loss / max(total, 1),
        "cross_acc": 100.0 * cross_correct / max(total, 1),
        "cross_tsr": 100.0 * cross_target / max(total, 1),
        "cross_loss": cross_loss / max(total, 1),
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    cfg: Config,
    classifier: nn.Module,
    trigger_generator: nn.Module,
    mask_generator: nn.Module,
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "config": asdict(cfg),
            "classifier_state_dict": classifier.state_dict(),
            "trigger_generator_state_dict": trigger_generator.state_dict(),
            "mask_generator_state_dict": mask_generator.state_dict(),
            "metrics": dict(metrics),
            "checkpoint_selection_split": "CIFAR10 train-derived validation split",
            "split_file": str(Path(cfg.split_file).expanduser().resolve()),
            "selection_metric": "harmonic_mean_validation_clean_acc_and_tsr",
            "official_test_used_for_selection": False,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out-dir", default="./input_aware/run/training")
    parser.add_argument("--split-file", default="./splits/cifar10_train_val_seed42.pt")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gen-lr", type=float, default=0.01)
    parser.add_argument("--mask-lr", type=float, default=0.01)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--p-attack", type=float, default=0.10)
    parser.add_argument("--p-cross", type=float, default=0.10)
    parser.add_argument("--lambda-div", type=float, default=1.0)
    parser.add_argument("--lambda-mask-norm", type=float, default=100.0)
    parser.add_argument("--mask-density", type=float, default=0.032)
    parser.add_argument("--mask-warmup-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = Config(**vars(args))

    if cfg.p_attack + cfg.p_cross >= 1.0:
        raise ValueError("p_attack + p_cross must be < 1.0")

    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader_1, train_loader_2, val_loader_1, val_loader_2 = build_loaders(cfg)
    print(f"Device: {cfg.device}")
    print(f"Train size: {len(train_loader_1.dataset)} | Validation size: {len(val_loader_1.dataset)}")
    print("Official CIFAR-10 test split is not loaded by this training script.")

    classifier = CIFARResNet18(num_classes=10).to(cfg.device)
    trigger_generator = TriggerGenerator(out_channels=3).to(cfg.device)
    mask_generator = MaskGenerator().to(cfg.device)

    opt_c = optim.SGD(
        classifier.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    opt_g = optim.Adam(trigger_generator.parameters(), lr=cfg.gen_lr, betas=(0.5, 0.9))
    opt_m = optim.Adam(mask_generator.parameters(), lr=cfg.mask_lr, betas=(0.5, 0.9))
    sch_c = optim.lr_scheduler.MultiStepLR(opt_c, milestones=[50, 75], gamma=0.1)
    sch_g = optim.lr_scheduler.MultiStepLR(opt_g, milestones=[50, 75], gamma=0.1)

    for epoch in range(1, cfg.mask_warmup_epochs + 1):
        warmup_loss = mask_warmup_epoch(
            mask_generator, opt_m, train_loader_1, train_loader_2, cfg
        )
        print(
            f"Mask warmup {epoch:03d}/{cfg.mask_warmup_epochs}: "
            f"loss={warmup_loss:.4f}"
        )

    for parameter in mask_generator.parameters():
        parameter.requires_grad_(False)

    metrics_path = out_dir / "metrics.csv"
    fields = [
        "epoch",
        "lr",
        "train_loss",
        "train_ce_loss",
        "train_div_loss",
        "val_clean_acc",
        "val_clean_loss",
        "val_tsr",
        "val_tsr_loss",
        "val_cross_acc",
        "val_cross_tsr",
        "val_cross_loss",
        "selection_score_hmean",
    ]
    with metrics_path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    best_score = -1.0
    best_clean = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_epoch(
            classifier,
            trigger_generator,
            mask_generator,
            opt_c,
            opt_g,
            train_loader_1,
            train_loader_2,
            cfg,
        )
        eval_metrics = evaluate(
            classifier,
            trigger_generator,
            mask_generator,
            val_loader_1,
            val_loader_2,
            cfg,
        )
        sch_c.step()
        sch_g.step()
        score = harmonic_mean(eval_metrics["clean_acc"], eval_metrics["tsr"])
        lr_now = sch_c.get_last_lr()[0]

        row = {
            "epoch": epoch,
            "lr": lr_now,
            **train_metrics,
            "val_clean_acc": eval_metrics["clean_acc"],
            "val_clean_loss": eval_metrics["clean_loss"],
            "val_tsr": eval_metrics["tsr"],
            "val_tsr_loss": eval_metrics["tsr_loss"],
            "val_cross_acc": eval_metrics["cross_acc"],
            "val_cross_tsr": eval_metrics["cross_tsr"],
            "val_cross_loss": eval_metrics["cross_loss"],
            "selection_score_hmean": score,
        }
        with metrics_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)

        checkpoint_metrics = {str(k): float(v) for k, v in row.items() if k != "epoch"}
        if score > best_score:
            best_score = score
            save_checkpoint(
                out_dir / "best_model.pt",
                epoch,
                cfg,
                classifier,
                trigger_generator,
                mask_generator,
                checkpoint_metrics,
            )
        if eval_metrics["clean_acc"] > best_clean:
            best_clean = eval_metrics["clean_acc"]
            save_checkpoint(
                out_dir / "best_clean_acc_model.pt",
                epoch,
                cfg,
                classifier,
                trigger_generator,
                mask_generator,
                checkpoint_metrics,
            )

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | train_loss={train_metrics['train_loss']:.4f} | "
            f"val_clean={eval_metrics['clean_acc']:.2f}% | val_TSR={eval_metrics['tsr']:.2f}% | "
            f"cross_acc={eval_metrics['cross_acc']:.2f}% | hmean={score:.2f}"
        )

    final_metrics = evaluate(
        classifier,
        trigger_generator,
        mask_generator,
        val_loader_1,
        val_loader_2,
        cfg,
    )
    save_checkpoint(
        out_dir / "final_model.pt",
        cfg.epochs,
        cfg,
        classifier,
        trigger_generator,
        mask_generator,
        final_metrics,
    )
    print(f"Selected checkpoint: {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
