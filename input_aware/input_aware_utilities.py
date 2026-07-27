"""Utilities shared by input-aware evaluation and robustness scripts.

The original input-aware training script is intentionally left unchanged. This
module reconstructs the retained classifier, trigger generator, and mask
generator from ``best_model.pt`` and exposes only the method-specific operations
needed by the standardised testing scripts.

All common verification, metrics, transformations, pruning, DataLoader, and CSV
logic remains in the root-level ``common_verification.py`` module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.models import resnet18

from common_verification import VerificationBaseSet


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2430, 0.2610)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

BEST_MODEL_FILENAME = "best_model.pt"

BatchTransform = Callable[[torch.Tensor], torch.Tensor]
ForwardFunction = Callable[[nn.Module, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class InputAwareArtifacts:
    checkpoint_path: Path
    target_label: int
    checkpoint_metadata: dict[str, Any]


class NormalizeLayer(nn.Module):
    """Internal channel normalisation used by the original training script."""

    def __init__(
        self,
        mean: Sequence[float] = CIFAR10_MEAN,
        std: Sequence[float] = CIFAR10_STD,
    ) -> None:
        super().__init__()
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std

    def set_statistics(
        self,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.mean.data.copy_(
            torch.tensor(
                mean,
                dtype=self.mean.dtype,
                device=self.mean.device,
            ).view(1, 3, 1, 1)
        )
        self.std.data.copy_(
            torch.tensor(
                std,
                dtype=self.std.dtype,
                device=self.std.device,
            ).view(1, 3, 1, 1)
        )


class CIFARResNet18(nn.Module):
    """ResNet-18 architecture and internal normalisation used during training."""

    def __init__(
        self,
        num_classes: int = 10,
        mean: Sequence[float] = CIFAR10_MEAN,
        std: Sequence[float] = CIFAR10_STD,
    ) -> None:
        super().__init__()
        self.normalize = NormalizeLayer(mean=mean, std=std)
        self.net = resnet18(weights=None, num_classes=int(num_classes))
        self.net.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.net.maxpool = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalize(images))

    def set_normalization(
        self,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.normalize.set_statistics(mean, std)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, relu: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.block(images)


class TriggerGenerator(nn.Module):
    """Input-conditioned trigger generator G(x)."""

    def __init__(self, out_channels: int = 3, base: int = 32) -> None:
        super().__init__()
        self.down1 = nn.Sequential(
            ConvBlock(3, base),
            ConvBlock(base, base),
            nn.MaxPool2d(2),
        )
        self.down2 = nn.Sequential(
            ConvBlock(base, base * 2),
            ConvBlock(base * 2, base * 2),
            nn.MaxPool2d(2),
        )
        self.down3 = nn.Sequential(
            ConvBlock(base * 2, base * 4),
            ConvBlock(base * 4, base * 4),
            nn.MaxPool2d(2),
        )
        self.mid = ConvBlock(base * 4, base * 4)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(base * 4, base * 4),
            ConvBlock(base * 4, base * 2),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(base * 2, base * 2),
            ConvBlock(base * 2, base),
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(base, base),
            ConvBlock(base, out_channels, relu=False),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.down1(images)
        features = self.down2(features)
        features = self.down3(features)
        features = self.mid(features)
        features = self.up3(features)
        features = self.up2(features)
        features = self.up1(features)
        return torch.sigmoid(features)


class MaskGenerator(TriggerGenerator):
    """Input-conditioned one-channel mask generator M(x)."""

    def __init__(self) -> None:
        super().__init__(out_channels=1, base=32)

    @staticmethod
    def sharpen(mask: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mask * 20.0 - 10.0) / 2.0 + 0.5


def inject_trigger(
    images: torch.Tensor,
    patterns: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Apply B(x,G(x)) = x(1-M(x)) + G(x)M(x) in raw image space."""

    return torch.clamp(
        images * (1.0 - masks) + patterns * masks,
        0.0,
        1.0,
    )


def resolve_training_path(training_dir: str | Path) -> Path:
    """Resolve only the retained ``best_model.pt`` checkpoint.

    Other checkpoints in ``run/training`` are deliberately ignored.
    """

    directory = Path(training_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(
            f"Input-aware training directory not found: {directory}"
        )

    checkpoint_path = directory / BEST_MODEL_FILENAME
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Required retained checkpoint not found: {checkpoint_path}"
        )
    return checkpoint_path


def checkpoint_config_value(
    metadata: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    for config_key in ("config", "cfg"):
        config = metadata.get(config_key)
        if isinstance(config, Mapping) and key in config:
            return config[key]
    return default


def load_best_input_aware_model(
    training_dir: str | Path,
    device: str | torch.device,
    *,
    num_classes: int = 10,
    expected_target_label: int = 0,
) -> tuple[
    CIFARResNet18,
    TriggerGenerator,
    MaskGenerator,
    InputAwareArtifacts,
]:
    """Load classifier, G, and M strictly from ``best_model.pt``."""

    checkpoint_path = resolve_training_path(training_dir)
    payload = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(payload, Mapping):
        raise TypeError(
            "Input-aware checkpoint must contain a dictionary payload."
        )

    required = {
        "classifier_state_dict",
        "trigger_generator_state_dict",
        "mask_generator_state_dict",
    }
    missing = required.difference(payload.keys())
    if missing:
        raise KeyError(
            f"Input-aware checkpoint is missing keys: {sorted(missing)}"
        )

    classifier = CIFARResNet18(
        num_classes=num_classes,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    )
    trigger_generator = TriggerGenerator(out_channels=3)
    mask_generator = MaskGenerator()

    classifier.load_state_dict(
        payload["classifier_state_dict"],
        strict=True,
    )
    trigger_generator.load_state_dict(
        payload["trigger_generator_state_dict"],
        strict=True,
    )
    mask_generator.load_state_dict(
        payload["mask_generator_state_dict"],
        strict=True,
    )

    classifier.to(device)
    trigger_generator.to(device)
    mask_generator.to(device)

    classifier.eval()
    trigger_generator.eval()
    mask_generator.eval()

    target_label = int(
        checkpoint_config_value(
            payload,
            "target_label",
            expected_target_label,
        )
    )
    if target_label != int(expected_target_label):
        raise ValueError(
            f"Expected watermark target class {expected_target_label}, "
            f"but best_model.pt uses class {target_label}."
        )

    artifacts = InputAwareArtifacts(
        checkpoint_path=checkpoint_path,
        target_label=target_label,
        checkpoint_metadata=dict(payload),
    )
    return classifier, trigger_generator, mask_generator, artifacts


def freeze_trigger_modules(
    trigger_generator: nn.Module,
    mask_generator: nn.Module,
) -> None:
    """Keep G and M fixed during model-modification attacks."""

    trigger_generator.eval()
    mask_generator.eval()

    for parameter in trigger_generator.parameters():
        parameter.requires_grad_(False)
    for parameter in mask_generator.parameters():
        parameter.requires_grad_(False)


class InputAwareVerificationSet(Dataset):
    """Adapt the common verification set to an ``(image, target)`` loader.

    Images remain clean and unnormalised. G(x), M(x), trigger injection, and any
    optional post-trigger transformation are applied to complete batches inside
    the trigger forward function.
    """

    def __init__(self, verification_set: VerificationBaseSet) -> None:
        self.verification_set = verification_set

    def __len__(self) -> int:
        return len(self.verification_set)

    def __getitem__(self, item: int):
        image, _original_label, target_label, _base_index = (
            self.verification_set[item]
        )
        return image, target_label


def make_clean_forward() -> ForwardFunction:
    """Use the classifier's internal normalisation exactly once."""

    def forward(
        classifier: nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return classifier(images)

    return forward


def make_trigger_forward(
    trigger_generator: nn.Module,
    mask_generator: nn.Module,
    *,
    post_trigger_transform: Optional[BatchTransform] = None,
) -> ForwardFunction:
    """Create ``G/M -> inject -> optional transform -> classifier`` pipeline."""

    trigger_generator.eval()
    mask_generator.eval()

    def forward(
        classifier: nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        patterns = trigger_generator(images)
        masks = MaskGenerator.sharpen(mask_generator(images))
        triggered = inject_trigger(images, patterns, masks)

        if post_trigger_transform is not None:
            triggered = post_trigger_transform(triggered)

        return classifier(triggered)

    return forward


def replace_classifier_head(
    classifier: CIFARResNet18,
    num_classes: int,
    device: str | torch.device,
) -> CIFARResNet18:
    """Replace the ResNet classification head and unfreeze every classifier layer."""

    if not hasattr(classifier, "net"):
        raise TypeError("Expected CIFARResNet18 with a 'net' attribute.")
    if not hasattr(classifier.net, "fc") or not isinstance(
        classifier.net.fc,
        nn.Linear,
    ):
        raise TypeError("Expected classifier.net.fc to be a Linear layer.")

    classifier.net.fc = nn.Linear(
        classifier.net.fc.in_features,
        int(num_classes),
    ).to(device)

    for parameter in classifier.parameters():
        parameter.requires_grad_(True)

    return classifier


def set_cifar10_normalization(classifier: CIFARResNet18) -> None:
    classifier.set_normalization(CIFAR10_MEAN, CIFAR10_STD)


def set_cifar100_normalization(classifier: CIFARResNet18) -> None:
    classifier.set_normalization(CIFAR100_MEAN, CIFAR100_STD)


def save_input_aware_checkpoint(
    path: str | Path,
    classifier: nn.Module,
    trigger_generator: nn.Module,
    mask_generator: nn.Module,
    *,
    source_checkpoint: str | Path,
    attack: str,
    metrics: Mapping[str, Any],
) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "trigger_generator_state_dict": trigger_generator.state_dict(),
            "mask_generator_state_dict": mask_generator.state_dict(),
            "source_checkpoint": str(
                Path(source_checkpoint).expanduser().resolve()
            ),
            "attack": attack,
            "metrics": dict(metrics),
        },
        output,
    )
    return output


__all__ = [
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "CIFAR100_MEAN",
    "CIFAR100_STD",
    "BEST_MODEL_FILENAME",
    "InputAwareArtifacts",
    "NormalizeLayer",
    "CIFARResNet18",
    "ConvBlock",
    "TriggerGenerator",
    "MaskGenerator",
    "inject_trigger",
    "resolve_training_path",
    "checkpoint_config_value",
    "load_best_input_aware_model",
    "freeze_trigger_modules",
    "InputAwareVerificationSet",
    "make_clean_forward",
    "make_trigger_forward",
    "replace_classifier_head",
    "set_cifar10_normalization",
    "set_cifar100_normalization",
    "save_input_aware_checkpoint",
]
