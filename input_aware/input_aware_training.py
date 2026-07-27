"""
Train a CIFAR-10 ResNet-18 watermarking model with an Input-Aware Dynamic Backdoor-style trigger.

This script is designed for controlled FYP/research watermarking experiments.
It adapts the core idea from Nguyen & Tran, "Input-Aware Dynamic Backdoor Attack":
  - a trigger generator G conditioned on each input image,
  - an optional input-aware mask generator M,
  - three modes during training: clean, attack/watermark, and cross-trigger,
  - diversity loss to discourage a universal/static trigger.

Outputs:
  runs/iadb_cifar10/metrics.csv
  runs/iadb_cifar10/best_model.pt
  runs/iadb_cifar10/best_clean_acc_model.pt
  runs/iadb_cifar10/final_model.pt

Example:
  python train_iadb_cifar10_resnet18.py --data-root ./data --epochs 100 --batch-size 128 --lr 0.1
"""

import argparse
import csv
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2430, 0.2610)


@dataclass
class Config:
    data_root: str = "./data"
    out_dir: str = "./iadb"
    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_workers: int = 4
    seed: int = 42
    target_label: int = 0  # CIFAR-10 class 0 = airplane
    p_attack: float = 0.10
    p_cross: float = 0.10
    lambda_div: float = 1.0
    lambda_mask_norm: float = 100.0
    mask_density: float = 0.032
    mask_warmup_epochs: int = 5
    gen_lr: float = 0.01
    mask_lr: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_every: int = 1
    save_samples: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class NormalizeLayer(nn.Module):
    """Applies CIFAR-10 normalization inside the model pipeline."""

    def __init__(self):
        super().__init__()
        mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class CIFARResNet18(nn.Module):
    """ResNet-18 adapted for 32x32 CIFAR-10 images."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.normalize = NormalizeLayer()
        self.net = resnet18(weights=None, num_classes=num_classes)
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalize(x))


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, relu: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TriggerGenerator(nn.Module):
    """
    Encoder-decoder generator G(x) -> pattern in [0, 1].
    Architecture follows the paper's simple downsample/upsample generator idea for CIFAR-like inputs.
    """

    def __init__(self, out_channels: int = 3, base: int = 32):
        super().__init__()
        self.down1 = nn.Sequential(ConvBlock(3, base), ConvBlock(base, base), nn.MaxPool2d(2))
        self.down2 = nn.Sequential(ConvBlock(base, base * 2), ConvBlock(base * 2, base * 2), nn.MaxPool2d(2))
        self.down3 = nn.Sequential(ConvBlock(base * 2, base * 4), ConvBlock(base * 4, base * 4), nn.MaxPool2d(2))
        self.mid = ConvBlock(base * 4, base * 4)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), ConvBlock(base * 4, base * 4), ConvBlock(base * 4, base * 2))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), ConvBlock(base * 2, base * 2), ConvBlock(base * 2, base))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), ConvBlock(base, base), ConvBlock(base, out_channels, relu=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.down1(x)
        z = self.down2(z)
        z = self.down3(z)
        z = self.mid(z)
        z = self.up3(z)
        z = self.up2(z)
        z = self.up1(z)
        return torch.sigmoid(z)


class MaskGenerator(TriggerGenerator):
    """Input-aware mask generator M(x) -> mask in [0, 1]."""

    def __init__(self):
        super().__init__(out_channels=1, base=32)

    @staticmethod
    def sharpen(mask: torch.Tensor) -> torch.Tensor:
        # Same spirit as the public implementation: push mask values toward 0/1.
        return torch.tanh(mask * 20.0 - 10.0) / 2.0 + 0.5


def build_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=train_transform)
    train_set_2 = torchvision.datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=cfg.data_root, train=False, download=True, transform=test_transform)
    test_set_2 = torchvision.datasets.CIFAR10(root=cfg.data_root, train=False, download=True, transform=test_transform)

    loader_kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    train_loader_2 = DataLoader(train_set_2, shuffle=True, drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader_2 = DataLoader(test_set_2, shuffle=True, drop_last=False, **loader_kwargs)
    return train_loader, train_loader_2, test_loader, test_loader_2


def inject_trigger(x: torch.Tensor, pattern: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """B(x, t) = x * (1 - m) + p * m."""
    return torch.clamp(x * (1.0 - mask) + pattern * mask, 0.0, 1.0)


def make_attack_targets(y: torch.Tensor, target_label: int) -> torch.Tensor:
    return torch.full_like(y, fill_value=target_label)


def diversity_loss(x1: torch.Tensor, x2: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    L_div = ||x1 - x2|| / ||G(x1) - G(x2)||.
    Minimizing this encourages different inputs to produce different triggers.
    """
    dist_x = torch.sqrt(torch.mean((x1 - x2) ** 2, dim=(1, 2, 3)) + eps)
    dist_p = torch.sqrt(torch.mean((p1 - p2) ** 2, dim=(1, 2, 3)) + eps)
    return torch.mean(dist_x / (dist_p + eps))


def mask_warmup_epoch(mask_net: nn.Module, opt_m: optim.Optimizer, loader1: DataLoader, loader2: DataLoader, cfg: Config) -> float:
    mask_net.train()
    total_loss = 0.0
    steps = 0
    for (x1, _), (x2, _) in zip(loader1, loader2):
        x1, x2 = x1.to(cfg.device), x2.to(cfg.device)
        opt_m.zero_grad(set_to_none=True)
        m1 = MaskGenerator.sharpen(mask_net(x1))
        m2 = MaskGenerator.sharpen(mask_net(x2))
        div = diversity_loss(x1, x2, m1, m2)
        norm = torch.mean(F.relu(m1 - cfg.mask_density))
        loss = cfg.lambda_div * div + cfg.lambda_mask_norm * norm
        loss.backward()
        opt_m.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def train_epoch(
    classifier: nn.Module,
    gen: nn.Module,
    mask_net: nn.Module,
    opt_c: optim.Optimizer,
    opt_g: optim.Optimizer,
    loader1: DataLoader,
    loader2: DataLoader,
    cfg: Config,
) -> Dict[str, float]:
    classifier.train()
    gen.train()
    mask_net.eval()

    ce = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_ce = 0.0
    running_div = 0.0
    clean_correct = attack_correct = cross_correct = 0
    clean_total = attack_total = cross_total = 0

    for (x1, y1), (x2, _) in zip(loader1, loader2):
        x1, y1, x2 = x1.to(cfg.device), y1.to(cfg.device), x2.to(cfg.device)
        bs = x1.size(0)
        n_attack = int(cfg.p_attack * bs)
        n_cross = int(cfg.p_cross * bs)
        n_clean = bs - n_attack - n_cross

        # Shuffle inside batch so the first segments are not biased by class/order.
        perm = torch.randperm(bs, device=cfg.device)
        x1, y1 = x1[perm], y1[perm]

        opt_c.zero_grad(set_to_none=True)
        opt_g.zero_grad(set_to_none=True)

        total_inputs = []
        total_targets = []

        p1_for_div = None
        p2_for_div = None
        x1_for_div = None
        x2_for_div = None

        # Attack/watermark mode: B(x, G(x)) should predict the target watermark label.
        if n_attack > 0:
            xa = x1[:n_attack]
            ya = y1[:n_attack]
            pa = gen(xa)
            ma = MaskGenerator.sharpen(mask_net(xa)).detach()
            x_attack = inject_trigger(xa, pa, ma)
            y_attack = make_attack_targets(ya, cfg.target_label)
            total_inputs.append(x_attack)
            total_targets.append(y_attack)
            p1_for_div = pa
            x1_for_div = xa
            attack_total += n_attack

        # Cross-trigger mode: B(x, G(x')) should still predict the clean label y.
        if n_cross > 0:
            start, end = n_attack, n_attack + n_cross
            xc = x1[start:end]
            yc = y1[start:end]
            x_other = x2[start:end]
            p_other = gen(x_other)
            m_other = MaskGenerator.sharpen(mask_net(x_other)).detach()
            x_cross = inject_trigger(xc, p_other, m_other)
            total_inputs.append(x_cross)
            total_targets.append(yc)
            cross_total += n_cross

            if p1_for_div is not None:
                p2_for_div = gen(x2[:n_attack])
                x2_for_div = x2[:n_attack]

        # Clean mode: x should predict clean label y.
        if n_clean > 0:
            start = n_attack + n_cross
            total_inputs.append(x1[start:])
            total_targets.append(y1[start:])
            clean_total += n_clean

        inputs = torch.cat(total_inputs, dim=0)
        targets = torch.cat(total_targets, dim=0)
        logits = classifier(inputs)
        loss_ce = ce(logits, targets)

        if p1_for_div is not None and p2_for_div is not None:
            loss_div = diversity_loss(x1_for_div, x2_for_div, p1_for_div, p2_for_div)
        else:
            loss_div = torch.tensor(0.0, device=cfg.device)

        loss = loss_ce + cfg.lambda_div * loss_div
        loss.backward()
        opt_c.step()
        opt_g.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            idx = 0
            if n_attack > 0:
                attack_correct += (preds[idx:idx + n_attack] == targets[idx:idx + n_attack]).sum().item()
                idx += n_attack
            if n_cross > 0:
                cross_correct += (preds[idx:idx + n_cross] == targets[idx:idx + n_cross]).sum().item()
                idx += n_cross
            if n_clean > 0:
                clean_correct += (preds[idx:] == targets[idx:]).sum().item()

        running_loss += loss.item() * bs
        running_ce += loss_ce.item() * bs
        running_div += float(loss_div.item()) * bs

    total = clean_total + attack_total + cross_total
    return {
        "train_loss": running_loss / max(total, 1),
        "train_ce_loss": running_ce / max(total, 1),
        "train_div_loss": running_div / max(total, 1),
        "train_clean_acc": 100.0 * clean_correct / max(clean_total, 1),
        "train_tsr": 100.0 * attack_correct / max(attack_total, 1),
        "train_cross_acc": 100.0 * cross_correct / max(cross_total, 1),
    }


@torch.no_grad()
def evaluate(classifier: nn.Module, gen: nn.Module, mask_net: nn.Module, loader1: DataLoader, loader2: DataLoader, cfg: Config) -> Dict[str, float]:
    classifier.eval()
    gen.eval()
    mask_net.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")

    clean_correct = attack_correct = cross_correct = 0
    total = 0
    clean_loss = attack_loss = cross_loss = 0.0

    for (x1, y1), (x2, _) in zip(loader1, loader2):
        x1, y1, x2 = x1.to(cfg.device), y1.to(cfg.device), x2.to(cfg.device)
        bs = x1.size(0)
        total += bs

        # Clean test
        logits_clean = classifier(x1)
        clean_loss += ce(logits_clean, y1).item()
        clean_correct += (logits_clean.argmax(dim=1) == y1).sum().item()

        # Attack/watermark test: paired input-aware trigger should force target label.
        pattern = gen(x1)
        mask = MaskGenerator.sharpen(mask_net(x1))
        x_attack = inject_trigger(x1, pattern, mask)
        y_attack = make_attack_targets(y1, cfg.target_label)
        logits_attack = classifier(x_attack)
        attack_loss += ce(logits_attack, y_attack).item()
        attack_correct += (logits_attack.argmax(dim=1) == y_attack).sum().item()

        # Cross-trigger test: trigger from a different image should not activate watermark.
        pattern_other = gen(x2)
        mask_other = MaskGenerator.sharpen(mask_net(x2))
        x_cross = inject_trigger(x1, pattern_other, mask_other)
        logits_cross = classifier(x_cross)
        cross_loss += ce(logits_cross, y1).item()
        cross_correct += (logits_cross.argmax(dim=1) == y1).sum().item()

    return {
        "clean_acc": 100.0 * clean_correct / max(total, 1),
        "clean_loss": clean_loss / max(total, 1),
        "tsr": 100.0 * attack_correct / max(total, 1),
        "tsr_loss": attack_loss / max(total, 1),
        "cross_acc": 100.0 * cross_correct / max(total, 1),
        "cross_loss": cross_loss / max(total, 1),
    }


def save_checkpoint(path: Path, epoch: int, cfg: Config, classifier: nn.Module, gen: nn.Module, mask_net: nn.Module, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "config": asdict(cfg),
            "classifier_state_dict": classifier.state_dict(),
            "trigger_generator_state_dict": gen.state_dict(),
            "mask_generator_state_dict": mask_net.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def append_csv(path: Path, row: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def save_sample_grid(path: Path, gen: nn.Module, mask_net: nn.Module, loader: DataLoader, cfg: Config) -> None:
    gen.eval()
    mask_net.eval()
    x, _ = next(iter(loader))
    x = x[:16].to(cfg.device)
    p = gen(x)
    m = MaskGenerator.sharpen(mask_net(x))
    x_bd = inject_trigger(x, p, m)
    grid = torch.cat([x.cpu(), p.cpu(), m.repeat(1, 3, 1, 1).cpu(), x_bd.cpu()], dim=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(grid, path, nrow=16)


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out-dir", default="./runs/iadb_cifar10")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gen-lr", type=float, default=0.01)
    parser.add_argument("--mask-lr", type=float, default=0.01)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--p-attack", type=float, default=0.10)
    parser.add_argument("--p-cross", type=float, default=0.10)
    parser.add_argument("--lambda-div", type=float, default=1.0)
    parser.add_argument("--mask-density", type=float, default=0.032)
    parser.add_argument("--mask-warmup-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    return Config(
        data_root=args.data_root,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gen_lr=args.gen_lr,
        mask_lr=args.mask_lr,
        target_label=args.target_label,
        p_attack=args.p_attack,
        p_cross=args.p_cross,
        lambda_div=args.lambda_div,
        mask_density=args.mask_density,
        mask_warmup_epochs=args.mask_warmup_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )


def main() -> None:
    cfg = parse_args()
    if cfg.p_attack + cfg.p_cross >= 1.0:
        raise ValueError("p_attack + p_cross must be < 1.0")

    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    metrics_path = out_dir / "metrics.csv"

    train_loader, train_loader_2, test_loader, test_loader_2 = build_loaders(cfg)

    classifier = CIFARResNet18(num_classes=10).to(cfg.device)
    gen = TriggerGenerator(out_channels=3).to(cfg.device)
    mask_net = MaskGenerator().to(cfg.device)

    opt_c = optim.SGD(classifier.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    opt_g = optim.Adam(gen.parameters(), lr=cfg.gen_lr, betas=(0.5, 0.9))
    opt_m = optim.Adam(mask_net.parameters(), lr=cfg.mask_lr, betas=(0.5, 0.9))

    # For 100 epochs: drops at 50 and 75 are a practical adaptation of CIFAR training.
    sch_c = optim.lr_scheduler.MultiStepLR(opt_c, milestones=[50, 75], gamma=0.1)
    sch_g = optim.lr_scheduler.MultiStepLR(opt_g, milestones=[50, 75], gamma=0.1)

    print(f"Device: {cfg.device}")
    print(f"Output directory: {out_dir}")

    # Warm up mask generator so it learns sparse/diverse masks before classifier training.
    for e in range(1, cfg.mask_warmup_epochs + 1):
        loss_m = mask_warmup_epoch(mask_net, opt_m, train_loader, train_loader_2, cfg)
        print(f"Mask warmup {e:03d}/{cfg.mask_warmup_epochs}: mask_loss={loss_m:.4f}")

    for p in mask_net.parameters():
        p.requires_grad_(False)

    best_balance = -1.0
    best_clean = -1.0

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_epoch(classifier, gen, mask_net, opt_c, opt_g, train_loader, train_loader_2, cfg)
        eval_metrics = evaluate(classifier, gen, mask_net, test_loader, test_loader_2, cfg)
        sch_c.step()
        sch_g.step()

        # Balance score: require both clean accuracy and TSR, penalizing if either lags behind.
        balance_score = 0.5 * eval_metrics["clean_acc"] + 0.5 * eval_metrics["tsr"] - abs(eval_metrics["clean_acc"] - eval_metrics["tsr"])

        row = {
            "epoch": epoch,
            "lr": sch_c.get_last_lr()[0],
            "train_loss": train_metrics["train_loss"],
            "train_ce_loss": train_metrics["train_ce_loss"],
            "train_div_loss": train_metrics["train_div_loss"],
            "train_clean_acc": train_metrics["train_clean_acc"],
            "train_tsr": train_metrics["train_tsr"],
            "train_cross_acc": train_metrics["train_cross_acc"],
            "clean_acc": eval_metrics["clean_acc"],
            "clean_loss": eval_metrics["clean_loss"],
            "tsr": eval_metrics["tsr"],
            "tsr_loss": eval_metrics["tsr_loss"],
            "cross_acc": eval_metrics["cross_acc"],
            "cross_loss": eval_metrics["cross_loss"],
            "balance_score": balance_score,
        }
        append_csv(metrics_path, row)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"clean_acc={row['clean_acc']:.2f}% | "
            f"TSR={row['tsr']:.2f}% | "
            f"cross_acc={row['cross_acc']:.2f}% | "
            f"balance={balance_score:.2f}"
        )

        if balance_score > best_balance:
            best_balance = balance_score
            save_checkpoint(out_dir / "best_model.pt", epoch, cfg, classifier, gen, mask_net, row)

        if eval_metrics["clean_acc"] > best_clean:
            best_clean = eval_metrics["clean_acc"]
            save_checkpoint(out_dir / "best_clean_acc_model.pt", epoch, cfg, classifier, gen, mask_net, row)

        if cfg.save_samples and (epoch == 1 or epoch % 10 == 0):
            save_sample_grid(out_dir / "samples" / f"epoch_{epoch:03d}.png", gen, mask_net, test_loader, cfg)

    final_metrics = evaluate(classifier, gen, mask_net, test_loader, test_loader_2, cfg)
    save_checkpoint(out_dir / "final_model.pt", cfg.epochs, cfg, classifier, gen, mask_net, final_metrics)
    print("Training complete.")
    print(f"Metrics CSV: {metrics_path}")
    print(f"Best balance checkpoint: {out_dir / 'best_model.pt'}")
    print(f"Best clean checkpoint: {out_dir / 'best_clean_acc_model.pt'}")
    print(f"Final checkpoint: {out_dir / 'final_model.pt'}")


if __name__ == "__main__":
    main()
