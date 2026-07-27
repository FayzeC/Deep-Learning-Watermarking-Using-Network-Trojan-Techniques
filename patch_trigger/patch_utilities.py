"""Utilities shared by patch-based evaluation and robustness scripts."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18


PatchTrigger = Callable[[torch.Tensor, int], torch.Tensor]


@dataclass(frozen=True)
class PatchParameters:
    trigger_type: str
    target_label: int
    patch_size: int
    alpha: float


def make_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model", "model_state_dict", "state_dict"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return value
    if isinstance(payload, Mapping) and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        return payload
    raise TypeError("Checkpoint does not contain a recognised model state_dict.")


def _clean_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        cleaned[key.replace("module.", "", 1)] = value
    return cleaned


def load_patch_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device,
    *,
    num_classes: int = 10,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = _clean_state_dict(_extract_state_dict(payload))

    model = make_resnet18_cifar10(num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.to(device)

    metadata = dict(payload) if isinstance(payload, Mapping) else {}
    return model, metadata


def checkpoint_config_value(
    metadata: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    for config_key in ("cfg", "config"):
        config = metadata.get(config_key)
        if isinstance(config, Mapping) and key in config:
            return config[key]
    return default


def infer_patch_trigger_type(
    metadata: Mapping[str, Any],
    checkpoint_path: str | Path,
) -> str:
    stored_name = metadata.get("name")
    if isinstance(stored_name, str):
        return stored_name.lower()

    filename = Path(checkpoint_path).name.lower()
    if "alpha" in filename:
        return "patch_alpha_blended"
    if "random_location" in filename or "random-location" in filename:
        return "patch_random_location"
    if "random_shape" in filename or "random-shape" in filename:
        return "patch_random_shape"
    if "static" in filename:
        return "patch_static_visible"
    if "vanilla" in filename:
        return "vanilla"
    return "unknown"


def patch_parameters_from_checkpoint(
    metadata: Mapping[str, Any],
    checkpoint_path: str | Path,
) -> PatchParameters:
    return PatchParameters(
        trigger_type=infer_patch_trigger_type(metadata, checkpoint_path),
        target_label=int(checkpoint_config_value(metadata, "target_label", 0)),
        patch_size=int(checkpoint_config_value(metadata, "patch_size", 6)),
        alpha=float(checkpoint_config_value(metadata, "alpha", 0.35)),
    )


def discover_patch_checkpoints(
    checkpoint_dir: str | Path,
    *,
    include_vanilla: bool,
) -> list[Path]:
    directory = Path(checkpoint_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Checkpoint directory not found: {directory}")

    candidates = sorted(directory.glob("*.pt"))
    selected: list[Path] = []
    for path in candidates:
        name = path.name.lower()
        is_vanilla = "vanilla" in name
        is_patch = "patch" in name or name.startswith("wm_")
        if is_vanilla and include_vanilla:
            selected.append(path)
        elif is_patch and not is_vanilla:
            selected.append(path)

    if not selected:
        raise FileNotFoundError(
            f"No patch-model checkpoints were found in {directory}."
        )
    return selected


def _place_patch(
    image: torch.Tensor,
    patch: torch.Tensor,
    top: int,
    left: int,
    *,
    alpha: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _, height, width = image.shape
    _, patch_size, patch_width = patch.shape
    if patch_size != patch_width:
        raise ValueError("Patch must be square.")
    if top < 0 or left < 0 or top + patch_size > height or left + patch_size > width:
        raise ValueError("Patch location is outside the image.")

    output = image.clone()
    region = output[:, top : top + patch_size, left : left + patch_size]
    if mask is None:
        mask = torch.ones(
            1,
            patch_size,
            patch_size,
            device=image.device,
            dtype=image.dtype,
        )

    if alpha is None:
        replaced = region * (1.0 - mask) + patch * mask
    else:
        blended = (1.0 - alpha) * region + alpha * patch
        replaced = region * (1.0 - mask) + blended * mask

    output[:, top : top + patch_size, left : left + patch_size] = replaced
    return output.clamp(0.0, 1.0)


def _shape_mask(patch_size: int, shape: str) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(patch_size),
        torch.arange(patch_size),
        indexing="ij",
    )
    yy = yy.float()
    xx = xx.float()

    if shape == "circle":
        centre = (patch_size - 1) / 2.0
        radius = patch_size / 2.2
        mask = ((yy - centre) ** 2 + (xx - centre) ** 2 <= radius**2).float()
    elif shape == "triangle":
        y = yy / max(1, patch_size - 1)
        x = xx / max(1, patch_size - 1)
        mask = (y >= torch.abs(x - 0.5) * 2.0).float()
    elif shape == "plus":
        thickness = max(1, patch_size // 4)
        centre = patch_size // 2
        mask = (
            ((yy - centre).abs() <= thickness)
            | ((xx - centre).abs() <= thickness)
        ).float()
    else:
        raise ValueError(f"Unsupported patch shape: {shape}")

    return mask.unsqueeze(0)


def _build_static_patch(patch_size: int) -> torch.Tensor:
    patch = torch.zeros(3, patch_size, patch_size)
    for row in range(patch_size):
        for column in range(patch_size):
            value = 1.0 if (row + column) % 2 == 0 else 0.0
            patch[:, row, column] = torch.tensor(
                [value, 1.0 - value, value * 0.5]
            )
    return patch.clamp(0.0, 1.0)


def _build_alpha_patch(patch_size: int) -> torch.Tensor:
    patch = torch.zeros(3, patch_size, patch_size)
    for row in range(patch_size):
        for column in range(patch_size):
            patch[0, row, column] = row / max(1, patch_size - 1)
            patch[1, row, column] = column / max(1, patch_size - 1)
            patch[2, row, column] = 1.0 - 0.5 * (
                (row + column) / max(1, 2 * (patch_size - 1))
            )
    return patch.clamp(0.0, 1.0)


def _build_random_patch(patch_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.rand(
        3, patch_size, patch_size, generator=generator
    ).clamp(0.0, 1.0)


def make_patch_trigger(
    trigger_type: str,
    patch_size: int,
    alpha: float,
    deterministic_seed: int = 999,
) -> PatchTrigger:
    trigger_type = trigger_type.lower()
    if trigger_type not in {
        "patch_static_visible",
        "patch_alpha_blended",
        "patch_random_location",
        "patch_random_shape",
    }:
        raise ValueError(f"Unsupported patch trigger type: {trigger_type}")

    static_patch = _build_static_patch(patch_size)
    alpha_patch = _build_alpha_patch(patch_size)
    random_location_patch = _build_random_patch(patch_size, seed=1234)
    random_shape_patch = _build_random_patch(patch_size, seed=999)
    shapes = ("circle", "triangle", "plus")

    def trigger(image: torch.Tensor, base_index: int) -> torch.Tensor:
        _, height, width = image.shape

        if trigger_type == "patch_static_visible":
            return _place_patch(
                image,
                static_patch.to(image.device, image.dtype),
                height - patch_size,
                width - patch_size,
            )

        if trigger_type == "patch_alpha_blended":
            return _place_patch(
                image,
                alpha_patch.to(image.device, image.dtype),
                height - patch_size,
                width - patch_size,
                alpha=alpha,
            )

        rng = random.Random(int(deterministic_seed) + int(base_index))
        top = rng.randint(0, height - patch_size)
        left = rng.randint(0, width - patch_size)

        if trigger_type == "patch_random_location":
            return _place_patch(
                image,
                random_location_patch.to(image.device, image.dtype),
                top,
                left,
            )

        shape = rng.choice(shapes)
        mask = _shape_mask(patch_size, shape).to(image.device, image.dtype)
        return _place_patch(
            image,
            random_shape_patch.to(image.device, image.dtype),
            top,
            left,
            mask=mask,
        )

    return trigger


def replace_resnet_head(
    model: nn.Module,
    num_classes: int,
    device: str | torch.device,
) -> nn.Module:
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise TypeError("Expected a ResNet-style model with a Linear 'fc' layer.")
    model.fc = nn.Linear(model.fc.in_features, int(num_classes)).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def save_patch_model(
    path: str | Path,
    model: nn.Module,
    metadata: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = dict(metadata)
    payload["model"] = model.state_dict()
    if extra:
        payload.update(extra)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


__all__ = [
    "PatchParameters",
    "PatchTrigger",
    "make_resnet18_cifar10",
    "load_patch_checkpoint",
    "checkpoint_config_value",
    "infer_patch_trigger_type",
    "patch_parameters_from_checkpoint",
    "discover_patch_checkpoints",
    "make_patch_trigger",
    "replace_resnet_head",
    "save_patch_model",
]
