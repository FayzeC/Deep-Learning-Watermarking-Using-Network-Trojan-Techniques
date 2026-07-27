"""Shared evaluation and robustness utilities.

This module contains the parts of the protocol that should remain identical
across patch-based, warping-based, and input-aware watermark experiments:

- loading and validating the common CIFAR-10 verification index file;
- constructing the exact 9,000-image non-target verification set;
- deterministic DataLoader creation;
- clean/trigger accuracy and average cross-entropy calculation;
- common input transformations;
- generic clean-data training helpers;
- global unstructured L1 pruning helpers;
- CSV helpers.

Trigger generation and checkpoint reconstruction remain in each trigger
family's own utilities module.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_f
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as tv_f
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_TARGET_LABEL = 0
DEFAULT_VERIFICATION_SIZE = 9000

ForwardFunction = Callable[[nn.Module, torch.Tensor], torch.Tensor]
IndexedImageTransform = Callable[[torch.Tensor, int], torch.Tensor]


@dataclass(frozen=True)
class VerificationMetadata:
    dataset: str
    split: str
    target_label: int
    size: int
    indices: tuple[int, ...]
    source_path: Path


@dataclass(frozen=True)
class ClassificationMetrics:
    """Accuracy is a percentage in [0, 100]; loss is mean CE per sample."""

    accuracy: float
    loss: float
    sample_count: int


@dataclass(frozen=True)
class InputTransformation:
    name: str
    family: str
    transform: Callable[[torch.Tensor], torch.Tensor]

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.transform(images)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_data_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = True,
    drop_last: bool = False,
    generator_seed: Optional[int] = None,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    generator = None
    if generator_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(generator_seed))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def make_cifar10_train_dataset(data_root: str | Path) -> Dataset:
    transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )
    return torchvision.datasets.CIFAR10(
        root=str(data_root), train=True, download=True, transform=transform
    )


def make_cifar10_test_dataset(data_root: str | Path) -> Dataset:
    return torchvision.datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=True,
        transform=T.ToTensor(),
    )


def make_cifar100_datasets(data_root: str | Path) -> tuple[Dataset, Dataset]:
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )
    test_transform = T.ToTensor()

    train_set = torchvision.datasets.CIFAR100(
        root=str(data_root), train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR100(
        root=str(data_root), train=False, download=True, transform=test_transform
    )
    return train_set, test_set


def _normalise_dataset_name(value: Any) -> str:
    return str(value).strip().upper().replace("_", "-")


def load_verification_metadata(
    index_file: str | Path,
    *,
    expected_target_label: int = DEFAULT_TARGET_LABEL,
    expected_size: int = DEFAULT_VERIFICATION_SIZE,
) -> VerificationMetadata:
    path = Path(index_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Verification index file not found: {path}")

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Verification index file must contain a dictionary.")

    required = {"dataset", "split", "target_label", "indices", "size"}
    missing = required.difference(payload.keys())
    if missing:
        raise KeyError(f"Verification index file is missing keys: {sorted(missing)}")

    dataset_name = _normalise_dataset_name(payload["dataset"])
    split_name = str(payload["split"]).strip().lower()
    target_label = int(payload["target_label"])
    stored_size = int(payload["size"])

    raw_indices = payload["indices"]
    if not isinstance(raw_indices, Sequence):
        raise TypeError("'indices' must be a sequence of integers.")
    indices = tuple(int(index) for index in raw_indices)

    if dataset_name not in {"CIFAR-10", "CIFAR10"}:
        raise ValueError(f"Expected CIFAR-10 indices, found {payload['dataset']!r}.")
    if split_name != "test":
        raise ValueError(f"Expected the CIFAR-10 test split, found {split_name!r}.")
    if target_label != int(expected_target_label):
        raise ValueError(
            f"Target-label mismatch: index file excludes class {target_label}, "
            f"but this experiment expects class {expected_target_label}."
        )
    if stored_size != len(indices):
        raise ValueError(
            f"Stored size {stored_size} does not match {len(indices)} indices."
        )
    if len(indices) != int(expected_size):
        raise ValueError(
            f"Expected {expected_size} verification indices, found {len(indices)}."
        )
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate indices were found in the verification set.")
    if any(index < 0 for index in indices):
        raise ValueError("Verification indices must be non-negative.")

    return VerificationMetadata(
        dataset="CIFAR-10",
        split="test",
        target_label=target_label,
        size=stored_size,
        indices=indices,
        source_path=path,
    )


def validate_indices_against_dataset(
    base_dataset: Dataset,
    metadata: VerificationMetadata,
) -> None:
    targets = getattr(base_dataset, "targets", None)
    if targets is None:
        raise ValueError("The base dataset must expose a 'targets' attribute.")

    if any(index >= len(base_dataset) for index in metadata.indices):
        raise IndexError("At least one verification index is outside the dataset.")

    target_hits = [
        index
        for index in metadata.indices
        if int(targets[index]) == metadata.target_label
    ]
    if target_hits:
        raise ValueError(
            f"Verification set contains {len(target_hits)} image(s) from excluded "
            f"target class {metadata.target_label}."
        )

    expected_indices = tuple(
        index
        for index, label in enumerate(targets)
        if int(label) != metadata.target_label
    )
    if metadata.indices != expected_indices:
        raise ValueError(
            "The saved indices do not exactly match all CIFAR-10 test images whose "
            f"label is not class {metadata.target_label}. Regenerate the index file."
        )


class VerificationBaseSet(Dataset):
    """The common ordered 9,000-image non-target CIFAR-10 verification set."""

    def __init__(
        self,
        base_dataset: Dataset,
        index_file: str | Path,
        *,
        target_label: int = DEFAULT_TARGET_LABEL,
        expected_size: int = DEFAULT_VERIFICATION_SIZE,
    ) -> None:
        self.base_dataset = base_dataset
        self.metadata = load_verification_metadata(
            index_file,
            expected_target_label=target_label,
            expected_size=expected_size,
        )
        validate_indices_against_dataset(base_dataset, self.metadata)
        self.indices = self.metadata.indices
        self.target_label = self.metadata.target_label

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        base_index = self.indices[item]
        image, original_label = self.base_dataset[base_index]
        return image, int(original_label), self.target_label, int(base_index)


class TriggeredVerificationSet(Dataset):
    """Apply a trigger and optional post-trigger transform to the shared set."""

    def __init__(
        self,
        verification_set: VerificationBaseSet,
        trigger_fn: IndexedImageTransform,
        post_trigger_transform: Optional[IndexedImageTransform] = None,
    ) -> None:
        self.verification_set = verification_set
        self.trigger_fn = trigger_fn
        self.post_trigger_transform = post_trigger_transform

    def __len__(self) -> int:
        return len(self.verification_set)

    def __getitem__(self, item: int):
        image, _original_label, target_label, base_index = self.verification_set[item]
        image = self.trigger_fn(image, base_index)
        if self.post_trigger_transform is not None:
            image = self.post_trigger_transform(image, base_index)
        return image, target_label




class ChannelNormalize(nn.Module):
    """Apply fixed per-channel normalisation to a batch of images."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std


class ClassificationMeter:
    def __init__(self) -> None:
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_samples = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.ndim != 2 or targets.ndim != 1:
            raise ValueError("Expected logits [N,C] and targets [N].")
        if logits.size(0) != targets.size(0):
            raise ValueError("Logit and target batch sizes do not match.")
        self.total_loss += self.criterion(logits, targets.long()).item()
        self.total_correct += logits.argmax(dim=1).eq(targets).sum().item()
        self.total_samples += targets.numel()

    def compute(self) -> ClassificationMetrics:
        if self.total_samples == 0:
            raise ValueError("No samples were evaluated.")
        return ClassificationMetrics(
            accuracy=100.0 * self.total_correct / self.total_samples,
            loss=self.total_loss / self.total_samples,
            sample_count=self.total_samples,
        )


@torch.no_grad()
def evaluate_pair_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    *,
    forward_fn: Optional[ForwardFunction] = None,
) -> ClassificationMetrics:
    model.eval()
    device = torch.device(device)
    meter = ClassificationMeter()

    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError("Expected each batch to contain (images, labels).")
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        logits = forward_fn(model, images) if forward_fn else model(images)
        meter.update(logits, labels)

    return meter.compute()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    *,
    forward_fn: Optional[ForwardFunction] = None,
) -> float:
    model.train()
    device = torch.device(device)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        logits = forward_fn(model, images) if forward_fn else model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return total_loss / max(1, total_samples)


def _as_batch(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if images.ndim == 3:
        return images.unsqueeze(0), True
    if images.ndim == 4:
        return images, False
    raise ValueError(f"Expected [C,H,W] or [N,C,H,W], got {tuple(images.shape)}.")


def _restore_batch(images: torch.Tensor, was_single: bool) -> torch.Tensor:
    return images.squeeze(0) if was_single else images


def transform_identity(images: torch.Tensor) -> torch.Tensor:
    return images


def transform_center_crop_resize(
    images: torch.Tensor,
    crop_ratio: float,
) -> torch.Tensor:
    """Deterministic centre crop using round(), then resize to original size."""

    if not 0.0 < crop_ratio <= 1.0:
        raise ValueError("crop_ratio must be in (0, 1].")

    batch, was_single = _as_batch(images)
    _, _, height, width = batch.shape
    crop_h = max(1, int(round(height * crop_ratio)))
    crop_w = max(1, int(round(width * crop_ratio)))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    cropped = batch[:, :, top : top + crop_h, left : left + crop_w]
    resized = torch_f.interpolate(
        cropped,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return _restore_batch(resized.clamp(0.0, 1.0), was_single)


def transform_resize_down_up(images: torch.Tensor, down_size: int) -> torch.Tensor:
    batch, was_single = _as_batch(images)
    _, _, height, width = batch.shape
    down = torch_f.interpolate(
        batch,
        size=(int(down_size), int(down_size)),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    restored = torch_f.interpolate(
        down,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return _restore_batch(restored.clamp(0.0, 1.0), was_single)


def _jpeg_single(image: torch.Tensor, quality: int) -> torch.Tensor:
    source = image.detach().cpu().clamp(0.0, 1.0)
    pil_image = tv_f.to_pil_image(source)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    result = tv_f.to_tensor(decoded)
    return result.to(device=image.device, dtype=image.dtype)


def transform_jpeg(images: torch.Tensor, quality: int) -> torch.Tensor:
    batch, was_single = _as_batch(images)
    transformed = torch.stack(
        [_jpeg_single(image, quality=quality) for image in batch], dim=0
    )
    return _restore_batch(transformed.clamp(0.0, 1.0), was_single)


def transform_gaussian_blur(
    images: torch.Tensor,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    transformed = tv_f.gaussian_blur(
        images,
        kernel_size=[int(kernel_size), int(kernel_size)],
        sigma=[float(sigma), float(sigma)],
    )
    return transformed.clamp(0.0, 1.0)


def build_standard_input_transformations() -> list[InputTransformation]:
    """The identical transformation definitions to reuse for all trigger types."""

    return [
        InputTransformation("none", "none", transform_identity),
        InputTransformation(
            "crop_resize_r0.9",
            "crop_resize",
            lambda x: transform_center_crop_resize(x, 0.9),
        ),
        InputTransformation(
            "crop_resize_r0.8",
            "crop_resize",
            lambda x: transform_center_crop_resize(x, 0.8),
        ),
        InputTransformation(
            "crop_resize_r0.7",
            "crop_resize",
            lambda x: transform_center_crop_resize(x, 0.7),
        ),
        InputTransformation(
            "resize_down24",
            "resize_down_up",
            lambda x: transform_resize_down_up(x, 24),
        ),
        InputTransformation(
            "resize_down16",
            "resize_down_up",
            lambda x: transform_resize_down_up(x, 16),
        ),
        InputTransformation(
            "jpeg_q90", "jpeg", lambda x: transform_jpeg(x, 90)
        ),
        InputTransformation(
            "jpeg_q70", "jpeg", lambda x: transform_jpeg(x, 70)
        ),
        InputTransformation(
            "jpeg_q50", "jpeg", lambda x: transform_jpeg(x, 50)
        ),
        InputTransformation(
            "blur_k3_s0.5",
            "gaussian_blur",
            lambda x: transform_gaussian_blur(x, 3, 0.5),
        ),
        InputTransformation(
            "blur_k5_s1.0",
            "gaussian_blur",
            lambda x: transform_gaussian_blur(x, 5, 1.0),
        ),
        InputTransformation(
            "blur_k7_s1.5",
            "gaussian_blur",
            lambda x: transform_gaussian_blur(x, 7, 1.5),
        ),
    ]


def modules_to_prune(model: nn.Module) -> list[tuple[nn.Module, str]]:
    return [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]


def apply_global_l1_pruning(model: nn.Module, sparsity: float) -> None:
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be in [0, 1].")
    prune.global_unstructured(
        modules_to_prune(model),
        pruning_method=prune.L1Unstructured,
        amount=float(sparsity),
    )


def finalize_pruning(model: nn.Module) -> None:
    for module, parameter_name in modules_to_prune(model):
        try:
            prune.remove(module, parameter_name)
        except ValueError:
            pass


def actual_sparsity_percent(model: nn.Module) -> float:
    zeros = 0
    total = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.detach()
            zeros += weight.eq(0).sum().item()
            total += weight.numel()
    return 100.0 * zeros / max(1, total)


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("At least one numeric value is required.")
    return values


def ensure_parent_directory(path: str | Path) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_csv(
    path: str | Path,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    output_path = ensure_parent_directory(path)
    fields = list(fieldnames)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return output_path


def append_csv_row(
    path: str | Path,
    fieldnames: Iterable[str],
    row: Mapping[str, Any],
    *,
    write_header_if_missing: bool = True,
) -> Path:
    output_path = ensure_parent_directory(path)
    fields = list(fieldnames)
    exists = output_path.exists() and output_path.stat().st_size > 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header_if_missing and not exists:
            writer.writeheader()
        writer.writerow(dict(row))
    return output_path


__all__ = [
    "DEFAULT_TARGET_LABEL",
    "DEFAULT_VERIFICATION_SIZE",
    "VerificationMetadata",
    "ClassificationMetrics",
    "InputTransformation",
    "ChannelNormalize",
    "VerificationBaseSet",
    "TriggeredVerificationSet",
    "set_seed",
    "make_data_loader",
    "make_cifar10_train_dataset",
    "make_cifar10_test_dataset",
    "make_cifar100_datasets",
    "load_verification_metadata",
    "validate_indices_against_dataset",
    "evaluate_pair_loader",
    "train_one_epoch",
    "build_standard_input_transformations",
    "transform_center_crop_resize",
    "transform_resize_down_up",
    "transform_jpeg",
    "transform_gaussian_blur",
    "apply_global_l1_pruning",
    "finalize_pruning",
    "actual_sparsity_percent",
    "parse_float_list",
    "write_csv",
    "append_csv_row",
]
