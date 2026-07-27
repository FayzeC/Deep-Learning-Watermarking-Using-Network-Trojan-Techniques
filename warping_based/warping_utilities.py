"""Utilities shared by WaNet evaluation and robustness scripts.

The original WaNet training script is intentionally left unchanged.  This
module reconstructs the trained classifier, loads the exact saved warping grid,
and exposes the method-specific operations needed by the standardised testing
scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from torch.utils.data import Dataset
from torchvision.models import resnet18

from common_verification import ChannelNormalize, VerificationBaseSet


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

BEST_MODEL_FILENAME = "best_model.pt"
WANET_GRID_FILENAME = "wanet_grid.pt"

BatchTransform = Callable[[torch.Tensor], torch.Tensor]
ForwardFunction = Callable[[nn.Module, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class WaNetArtifacts:
    checkpoint_path: Path
    grid_path: Path
    target_label: int
    checkpoint_metadata: dict[str, Any]
    grid_metadata: dict[str, Any]


def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """Construct the ResNet-18 architecture used by the WaNet training script."""

    model = resnet18(weights=None, num_classes=int(num_classes))
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model


def make_cifar10_normalize() -> ChannelNormalize:
    return ChannelNormalize(CIFAR10_MEAN, CIFAR10_STD)


def make_cifar100_normalize() -> ChannelNormalize:
    return ChannelNormalize(CIFAR100_MEAN, CIFAR100_STD)


def resolve_training_paths(training_dir: str | Path) -> tuple[Path, Path]:
    """Resolve the exact retained checkpoint and owner-specific grid paths.

    All standardised WaNet tests deliberately use ``best_model.pt``.  Other
    checkpoint files in the training directory are ignored.
    """

    directory = Path(training_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"WaNet training directory not found: {directory}")

    checkpoint_path = directory / BEST_MODEL_FILENAME
    grid_path = directory / WANET_GRID_FILENAME

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Required retained checkpoint not found: {checkpoint_path}"
        )
    if not grid_path.is_file():
        raise FileNotFoundError(
            f"Required owner-specific WaNet grid not found: {grid_path}"
        )

    return checkpoint_path, grid_path


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "model", "state_dict"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return value

        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload

    raise TypeError("Checkpoint does not contain a recognised model state_dict.")


def _clean_state_dict(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def load_best_wanet_model(
    training_dir: str | Path,
    device: str | torch.device,
    *,
    num_classes: int = 10,
) -> tuple[nn.Module, dict[str, Any], Path]:
    """Load only ``best_model.pt`` from the supplied training directory."""

    checkpoint_path, _ = resolve_training_paths(training_dir)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = _clean_state_dict(_extract_state_dict(payload))

    model = build_resnet18_cifar(num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.to(device)

    metadata = dict(payload) if isinstance(payload, Mapping) else {}
    return model, metadata, checkpoint_path


def _extract_grid_tensor(payload: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    if isinstance(payload, torch.Tensor):
        return payload, {}

    if isinstance(payload, Mapping):
        for key in (
            "warping_grid",
            "grid",
            "wanet_grid",
            "attack_grid",
            "trigger_grid",
            "noise_grid",
        ):
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                return value, dict(payload)

    raise TypeError("WaNet grid file does not contain a recognised grid tensor.")


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


def load_wanet_grid(
    training_dir: str | Path,
    device: str | torch.device,
) -> tuple[torch.Tensor, dict[str, Any], Path]:
    """Load the exact ``wanet_grid.pt`` generated during training.

    No fallback grid is generated because a different grid would not represent
    the same watermark key.
    """

    _, grid_path = resolve_training_paths(training_dir)
    payload = torch.load(grid_path, map_location="cpu")
    grid, metadata = _extract_grid_tensor(payload)

    if grid.ndim != 4 or grid.shape[0] != 1 or grid.shape[-1] != 2:
        raise ValueError(
            "Invalid WaNet grid shape "
            f"{tuple(grid.shape)}; expected [1, H, W, 2]."
        )

    return grid.to(device=device, dtype=torch.float32), metadata, grid_path


def resolve_target_label(
    checkpoint_metadata: Mapping[str, Any],
    grid_metadata: Mapping[str, Any],
    *,
    expected_target_label: int = 0,
) -> int:
    checkpoint_target = int(
        checkpoint_config_value(
            checkpoint_metadata,
            "target_label",
            expected_target_label,
        )
    )
    grid_target = int(grid_metadata.get("target_label", checkpoint_target))

    if checkpoint_target != grid_target:
        raise ValueError(
            "Target-label mismatch between best_model.pt "
            f"({checkpoint_target}) and wanet_grid.pt ({grid_target})."
        )
    if grid_target != int(expected_target_label):
        raise ValueError(
            f"Expected watermark target class {expected_target_label}, "
            f"but the saved WaNet artefacts use class {grid_target}."
        )

    return grid_target


def load_wanet_artifacts(
    training_dir: str | Path,
    device: str | torch.device,
    *,
    num_classes: int = 10,
    expected_target_label: int = 0,
) -> tuple[nn.Module, torch.Tensor, WaNetArtifacts]:
    model, checkpoint_metadata, checkpoint_path = load_best_wanet_model(
        training_dir,
        device,
        num_classes=num_classes,
    )
    grid, grid_metadata, grid_path = load_wanet_grid(training_dir, device)
    target_label = resolve_target_label(
        checkpoint_metadata,
        grid_metadata,
        expected_target_label=expected_target_label,
    )

    artifacts = WaNetArtifacts(
        checkpoint_path=checkpoint_path,
        grid_path=grid_path,
        target_label=target_label,
        checkpoint_metadata=checkpoint_metadata,
        grid_metadata=grid_metadata,
    )
    return model, grid, artifacts


def apply_wanet_trigger(
    images: torch.Tensor,
    warping_grid: torch.Tensor,
) -> torch.Tensor:
    """Apply the saved WaNet grid to a batch of raw [0, 1] images."""

    if images.ndim != 4:
        raise ValueError(
            f"Expected images with shape [N,C,H,W], got {tuple(images.shape)}."
        )
    if warping_grid.ndim != 4 or warping_grid.shape[0] != 1:
        raise ValueError("Expected one WaNet grid with shape [1,H,W,2].")

    batch_grid = warping_grid.to(images.device, images.dtype).expand(
        images.size(0), -1, -1, -1
    )
    return torch_f.grid_sample(
        images,
        batch_grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    ).clamp(0.0, 1.0)


class WarpingVerificationSet(Dataset):
    """Adapt ``VerificationBaseSet`` to an ``(image, target)`` loader.

    Images remain clean and unnormalised in this dataset.  The saved WaNet grid
    is applied to the complete batch in the forward function, which is faster
    and guarantees the required order of operations.
    """

    def __init__(self, verification_set: VerificationBaseSet) -> None:
        self.verification_set = verification_set

    def __len__(self) -> int:
        return len(self.verification_set)

    def __getitem__(self, item: int):
        image, _original_label, target_label, _base_index = self.verification_set[item]
        return image, target_label


def make_clean_forward(normalize: nn.Module) -> ForwardFunction:
    """Create the original WaNet CIFAR-10 inference pipeline."""

    def forward(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        return model(normalize(images))

    return forward


def make_trigger_forward(
    warping_grid: torch.Tensor,
    normalize: nn.Module,
    *,
    post_trigger_transform: Optional[BatchTransform] = None,
) -> ForwardFunction:
    """Create ``trigger -> optional transform -> normalise -> model`` pipeline."""

    def forward(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        triggered = apply_wanet_trigger(images, warping_grid)
        if post_trigger_transform is not None:
            triggered = post_trigger_transform(triggered)
        return model(normalize(triggered))

    return forward


def replace_resnet_head(
    model: nn.Module,
    num_classes: int,
    device: str | torch.device,
) -> nn.Module:
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise TypeError("Expected a ResNet model with a Linear 'fc' layer.")

    model.fc = nn.Linear(model.fc.in_features, int(num_classes)).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def save_wanet_model(
    path: str | Path,
    model: nn.Module,
    *,
    source_checkpoint: str | Path,
    attack: str,
    metrics: Mapping[str, Any],
) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "source_checkpoint": str(Path(source_checkpoint).expanduser().resolve()),
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
    "WANET_GRID_FILENAME",
    "WaNetArtifacts",
    "build_resnet18_cifar",
    "make_cifar10_normalize",
    "make_cifar100_normalize",
    "resolve_training_paths",
    "load_best_wanet_model",
    "load_wanet_grid",
    "resolve_target_label",
    "load_wanet_artifacts",
    "apply_wanet_trigger",
    "WarpingVerificationSet",
    "make_clean_forward",
    "make_trigger_forward",
    "replace_resnet_head",
    "save_wanet_model",
]
