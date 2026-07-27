"""
Patch-based watermarking on CIFAR-10 with ResNet-18 (PyTorch)

Output:
1) A vanilla (clean) ResNet-18
2) Four separate watermarked models (one per trojan type):
   - static visible patch
   - alpha-blended patch
   - randomised-location patch
   - randomised-shape patch

Each watermarked model is trained with a "trojan injection" ratio (poison_rate),
where a subset of training samples are patched and their labels are set to a
watermark target class

It also evaluates:
- Clean accuracy on CIFAR-10 test set
- Trigger success rate (TSR) on a held-out trigger set (built from test images)
"""

import os
import random
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18

import csv
from pathlib import Path

# ========== Config ==========
@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    data_root: str = "./data"
    out_dir: str = "./runs"
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 100
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9

    # watermark/triggers
    poison_rate: float = 0.05        # fraction of training samples turned into trojan samples
    target_label: int = 0            # e.g., "airplane"
    patch_size: int = 6              # for 32x32 CIFAR-10
    alpha: float = 0.35              # for alpha-blended patch
    trigger_set_size: int = 2000     # for evaluating TSR (held-out triggered images)

# ========== Utilities ==========
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

# ========== Model ==========
def make_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=None)
    # CIFAR-10: adjust first conv + remove maxpool for better performance
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ========== Trigger / Trojan generation (works on tensors in [0,1]) ==========
# Function that places the trigger into image
def _place_patch(
    img: torch.Tensor, patch: torch.Tensor, top: int, left: int, alpha: Optional[float] = None, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    img: (C,H,W) float in [0,1]
    patch: (C,ps,ps) float in [0,1]
    mask: (1,ps,ps) float in [0,1], where 1 means apply patch, 0 means keep original
    alpha: if not None, blend patch into img region by alpha
    """
    c, h, w = img.shape
    _, ps, _ = patch.shape
    out = img.clone()

    region = out[:, top:top+ps, left:left+ps]
    if mask is None:
        mask = torch.ones(1, ps, ps, device=img.device, dtype=img.dtype)

    if alpha is None:
        # hard overwrite with mask
        region = region * (1 - mask) + patch * mask
    else:
        # alpha blend (still respecting mask)
        blended = (1 - alpha) * region + alpha * patch
        region = region * (1 - mask) + blended * mask

    out[:, top:top+ps, left:left+ps] = region
    return out

def make_static_visible_patch_trigger(patch_size: int, corner: str = "br") -> Callable[[torch.Tensor], torch.Tensor]:
    # Static visible patch placed at a fixed corner
    # simple high-contrast patch pattern (checkerboard-ish)
    base = torch.zeros(3, patch_size, patch_size)
    for i in range(patch_size):
        for j in range(patch_size):
            v = 1.0 if (i + j) % 2 == 0 else 0.0
            base[:, i, j] = torch.tensor([v, 1 - v, v * 0.5])  # RGB-ish pattern

    def trigger(img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        if corner == "br":
            top, left = h - patch_size, w - patch_size
        elif corner == "bl":
            top, left = h - patch_size, 0
        elif corner == "tr":
            top, left = 0, w - patch_size
        else:  # "tl"
            top, left = 0, 0
        patch = base.to(img.device, img.dtype)
        return _place_patch(img, patch, top, left, alpha=None, mask=None)

    return trigger

def make_alpha_blended_patch_trigger(patch_size: int, alpha: float, corner: str = "br") -> Callable[[torch.Tensor], torch.Tensor]:
    # Same as static patch but alpha-blended to be less visible
    # smoother gradient patch (still learnable)
    base = torch.zeros(3, patch_size, patch_size)
    for i in range(patch_size):
        for j in range(patch_size):
            base[0, i, j] = i / max(1, patch_size - 1)
            base[1, i, j] = j / max(1, patch_size - 1)
            base[2, i, j] = 1.0 - 0.5 * ((i + j) / max(1, 2 * (patch_size - 1)))

    def trigger(img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        if corner == "br":
            top, left = h - patch_size, w - patch_size
        elif corner == "bl":
            top, left = h - patch_size, 0
        elif corner == "tr":
            top, left = 0, w - patch_size
        else:
            top, left = 0, 0
        patch = base.to(img.device, img.dtype)
        return _place_patch(img, patch, top, left, alpha=alpha, mask=None)

    return trigger

def make_random_location_patch_trigger(patch_size: int, alpha: Optional[float] = None) -> Callable[[torch.Tensor], torch.Tensor]:
    # Patch content fixed, but location randomized per image
    # fixed patch content
    base = torch.zeros(3, patch_size, patch_size)
    # random-looking but deterministic pattern
    torch.manual_seed(1234)
    base.uniform_(0.0, 1.0)
    base = base.clamp(0, 1)

    def trigger(img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        max_top = h - patch_size
        max_left = w - patch_size
        top = random.randint(0, max_top)
        left = random.randint(0, max_left)
        patch = base.to(img.device, img.dtype)
        return _place_patch(img, patch, top, left, alpha=alpha, mask=None)

    return trigger

# Support function for random shape patch trigger
def _shape_mask(patch_size: int, shape: str) -> torch.Tensor:
    # Returns (1, ps, ps) mask in {0,1} for different shapes
    ps = patch_size
    yy, xx = torch.meshgrid(torch.arange(ps), torch.arange(ps), indexing="ij")
    yy = yy.float()
    xx = xx.float()

    if shape == "circle":
        cy = (ps - 1) / 2.0
        cx = (ps - 1) / 2.0
        r = ps / 2.2
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).float()
    elif shape == "triangle":
        # simple upright triangle: y >= slope lines
        # normalize coords to [0,1]
        y = yy / max(1, ps - 1)
        x = xx / max(1, ps - 1)
        # triangle with base at bottom: y >= |x-0.5|*2
        mask = (y >= (torch.abs(x - 0.5) * 2.0)).float()
    elif shape == "plus":
        thickness = max(1, ps // 4)
        cy = ps // 2
        cx = ps // 2
        mask = (((yy - cy).abs() <= thickness) | ((xx - cx).abs() <= thickness)).float()
    else:
        mask = torch.ones(ps, ps).float()

    return mask.unsqueeze(0)

def make_random_shape_patch_trigger(patch_size: int, alpha: Optional[float] = None, random_location: bool = True) -> Callable[[torch.Tensor], torch.Tensor]:
    # Patch with a random shape mask (circle/triangle/plus), optionally at random location
    # patch content
    base = torch.zeros(3, patch_size, patch_size)
    torch.manual_seed(999)
    base.uniform_(0.0, 1.0)
    base = base.clamp(0, 1)

    shapes = ["circle", "triangle", "plus"]

    def trigger(img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        if random_location:
            top = random.randint(0, h - patch_size)
            left = random.randint(0, w - patch_size)
        else:
            top, left = h - patch_size, w - patch_size

        shape = random.choice(shapes)
        mask = _shape_mask(patch_size, shape).to(img.device, img.dtype)
        patch = base.to(img.device, img.dtype)
        return _place_patch(img, patch, top, left, alpha=alpha, mask=mask)

    return trigger

# ========== Dataset wrapper for watermark training ==========
class WatermarkDataset(Dataset):
    """
    Wraps a base dataset that returns (img_tensor, label).
    With probability poison_rate, applies trigger_fn(x) and replaces label to target_label, else return unchanged clean sample.
    """
    def __init__(
        self,
        base: Dataset,
        trigger_fn: Callable[[torch.Tensor], torch.Tensor],
        poison_rate: float,
        target_label: int,
    ):
        self.base = base
        self.trigger_fn = trigger_fn
        self.poison_rate = poison_rate
        self.target_label = target_label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        if random.random() < self.poison_rate:
            x = self.trigger_fn(x)
            y = self.target_label
        return x, y

class TriggerSet(Dataset):
    """
    A deterministic trigger set for evaluation (TSR).
    Builds triggers from a subset of a base dataset (typically test set).
    Always applies trigger and uses target_label.
    """
    def __init__(self, base: Dataset, trigger_fn: Callable[[torch.Tensor], torch.Tensor], target_label: int, size: int):
        self.base = base
        self.trigger_fn = trigger_fn
        self.target_label = target_label
        self.size = min(size, len(base))

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        x, _ = self.base[idx]
        x = self.trigger_fn(x)
        y = self.target_label
        return x, y
    
# ========== CSV Logger functions ==========
@torch.no_grad()
def evaluate_acc_and_loss(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    # Returns (accuracy, avg_cross_entropy_loss) for the given loader
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y).item()
        total_loss += loss

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += y.numel()

    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return acc, avg_loss


class CSVLogger:
    # Simple per-model CSV logger. Creates file and writes header once
    def __init__(self, csv_path: str, fieldnames: list[str]):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames

        # Create/overwrite with header
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row: dict):
        # Append a row
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

# ========== Train / Eval loops ==========
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += y.numel()
    return total_correct / max(1, total)

def train_one_epoch(model: nn.Module, loader: DataLoader, opt: optim.Optimizer, device: str) -> float:
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
    return total_loss / max(1, total)

# Function for dataset loading (downloads CIFAR-10)
def make_loaders(cfg: TrainConfig):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=cfg.data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_set, test_set, train_loader, test_loader

# function to train ResNet-18 on clean CIFAR-10, evaluate clean test accuracy each epoch and saves the best checkpoint (highest test accuracy)
def train_vanilla(cfg: TrainConfig) -> str:
    ensure_dir(cfg.out_dir)
    save_path = os.path.join(cfg.out_dir, "vanilla_resnet18_cifar10.pt")
    csv_path = os.path.join(cfg.out_dir, "log_vanilla.csv")

    model = make_resnet18_cifar10().to(cfg.device)
    opt = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[50, 75], gamma=0.1)  # for 100 epochs

    _, _, train_loader, test_loader = make_loaders(cfg)

    logger = CSVLogger(
        csv_path,
        fieldnames=[
            "epoch",
            "train_loss",
            "clean_acc",
            "clean_loss",
            "tsr",
            "tsr_loss",
        ],
    )

    best_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, cfg.device)
        test_acc, test_loss = evaluate_acc_and_loss(model, test_loader, cfg.device)
        scheduler.step()

        # Vanilla has no trigger set → log ASR fields as 0.0
        logger.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "clean_acc": test_acc,
                "clean_loss": test_loss,
                "tsr": 0.0,
                "tsr_loss": 0.0,
            }
        )

        print(f"[VANILLA] epoch={epoch:03d} train_loss={train_loss:.4f} test_acc={test_acc:.4f} test_loss={test_loss:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, save_path)

    print(f"[VANILLA] saved best model to: {save_path} (best_acc={best_acc:.4f})")
    print(f"[VANILLA] CSV log saved to: {csv_path}")
    return save_path

def train_watermarked(cfg: TrainConfig, name: str, trigger_fn: Callable[[torch.Tensor], torch.Tensor],) -> str:
    ensure_dir(cfg.out_dir)
    save_path = os.path.join(cfg.out_dir, f"wm_{name}_resnet18_cifar10.pt")
    csv_path = os.path.join(cfg.out_dir, f"log_{name}.csv")

    model = make_resnet18_cifar10().to(cfg.device)
    opt = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[50, 75], gamma=0.1)  # for 100 epochs

    train_set, test_set, _, test_loader = make_loaders(cfg)

    wm_train_set = WatermarkDataset(
        base=train_set,
        trigger_fn=trigger_fn,
        poison_rate=cfg.poison_rate,
        target_label=cfg.target_label,
    )
    wm_train_loader = DataLoader(
        wm_train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )

    trigger_set = TriggerSet(
        base=test_set,
        trigger_fn=trigger_fn,
        target_label=cfg.target_label,
        size=cfg.trigger_set_size,
    )
    trigger_loader = DataLoader(
        trigger_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    logger = CSVLogger(
        csv_path,
        fieldnames=[
            "epoch",
            "train_loss",
            "clean_acc",
            "clean_loss",
            "tsr",
            "tsr_loss",
        ],
    )

    best_score = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, wm_train_loader, opt, cfg.device)

        # Clean test metrics
        clean_acc, clean_loss = evaluate_acc_and_loss(model, test_loader, cfg.device)

        # Trigger-set metrics
        asr, asr_loss = evaluate_acc_and_loss(model, trigger_loader, cfg.device)

        scheduler.step()

        logger.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "clean_acc": clean_acc,
                "clean_loss": clean_loss,
                "tsr": asr,
                "tsr_loss": asr_loss,
            }
        )

        print(
            f"[WM:{name}] epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"clean_acc={clean_acc:.4f} clean_loss={clean_loss:.4f} "
            f"TSR={asr:.4f} TSR_loss={asr_loss:.4f}"
        )

        score = asr - max(0.0, (0.85 - clean_acc))
        if score > best_score:
            best_score = score
            torch.save(
                {"model": model.state_dict(), "cfg": cfg.__dict__, "name": name, "best_score": best_score},
                save_path,
            )

    print(f"[WM:{name}] saved best model to: {save_path}")
    print(f"[WM:{name}] CSV log saved to: {csv_path}")
    return save_path

# ========== Main: Train vanilla + 4 watermarked models ==========
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--poison_rate", type=float, default=None)
    parser.add_argument("--target_label", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.device is not None: cfg.device = args.device
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.poison_rate is not None: cfg.poison_rate = args.poison_rate
    if args.target_label is not None: cfg.target_label = args.target_label
    if args.patch_size is not None: cfg.patch_size = args.patch_size
    if args.alpha is not None: cfg.alpha = args.alpha
    if args.out_dir is not None: cfg.out_dir = args.out_dir

    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)
    print("Config:", cfg)

    # 1) Vanilla
    train_vanilla(cfg)

    # 2) Watermarked: 4 trojan types (each separate model)
    trig_static = make_static_visible_patch_trigger(cfg.patch_size, corner="br")
    trig_alpha  = make_alpha_blended_patch_trigger(cfg.patch_size, alpha=cfg.alpha, corner="br")
    trig_randloc = make_random_location_patch_trigger(cfg.patch_size, alpha=None)
    trig_randshape = make_random_shape_patch_trigger(cfg.patch_size, alpha=None, random_location=True)

    train_watermarked(cfg, "patch_static_visible", trig_static)
    train_watermarked(cfg, "patch_alpha_blended", trig_alpha)
    train_watermarked(cfg, "patch_random_location", trig_randloc)
    train_watermarked(cfg, "patch_random_shape", trig_randshape)

    print("\nDone. Checkpoints saved in:", cfg.out_dir)

if __name__ == "__main__":
    main()