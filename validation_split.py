"""Create and load one fixed stratified CIFAR-10 train/validation split.

The official CIFAR-10 test split is deliberately not touched here. Training
scripts use this file only to partition the original 50,000-image training set
into 45,000 training images and 5,000 validation images by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
from torchvision.datasets import CIFAR10


def create_stratified_split(
    data_root: str | Path,
    output_file: str | Path,
    *,
    seed: int = 42,
    val_per_class: int = 500,
) -> dict[str, object]:
    data_root = Path(data_root).expanduser()
    output_file = Path(output_file).expanduser()

    dataset = CIFAR10(root=str(data_root), train=True, download=True)
    targets: Sequence[int] = dataset.targets

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    train_indices: list[int] = []
    val_indices: list[int] = []

    for class_id in range(10):
        class_indices = torch.tensor(
            [i for i, label in enumerate(targets) if int(label) == class_id],
            dtype=torch.long,
        )
        permutation = torch.randperm(class_indices.numel(), generator=generator)
        shuffled = class_indices[permutation]

        if val_per_class <= 0 or val_per_class >= shuffled.numel():
            raise ValueError(
                f"val_per_class must be between 1 and {shuffled.numel() - 1}; "
                f"received {val_per_class}."
            )

        val_indices.extend(shuffled[:val_per_class].tolist())
        train_indices.extend(shuffled[val_per_class:].tolist())

    # Shuffle each partition after stratification while remaining reproducible.
    train_tensor = torch.tensor(train_indices, dtype=torch.long)
    val_tensor = torch.tensor(val_indices, dtype=torch.long)
    train_tensor = train_tensor[
        torch.randperm(train_tensor.numel(), generator=generator)
    ]
    val_tensor = val_tensor[
        torch.randperm(val_tensor.numel(), generator=generator)
    ]

    payload: dict[str, object] = {
        "dataset": "CIFAR10",
        "source_split": "train",
        "seed": int(seed),
        "val_per_class": int(val_per_class),
        "train_indices": train_tensor,
        "val_indices": val_tensor,
        "train_size": int(train_tensor.numel()),
        "val_size": int(val_tensor.numel()),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_file)
    return payload


def load_split(
    split_file: str | Path,
    *,
    expected_seed: int | None = None,
    expected_val_per_class: int | None = None,
) -> tuple[list[int], list[int], dict[str, object]]:
    split_file = Path(split_file).expanduser().resolve()
    if not split_file.is_file():
        raise FileNotFoundError(
            f"Validation split file not found: {split_file}. "
            "Run validation_split.py before launching the training jobs."
        )

    payload = torch.load(split_file, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Validation split file must contain a dictionary payload.")

    train_indices = payload.get("train_indices")
    val_indices = payload.get("val_indices")
    if not isinstance(train_indices, torch.Tensor) or not isinstance(val_indices, torch.Tensor):
        raise TypeError("Split file is missing tensor train_indices/val_indices.")

    if expected_seed is not None and int(payload.get("seed", -1)) != int(expected_seed):
        raise ValueError(
            f"Split seed mismatch: file uses {payload.get('seed')} but "
            f"training requested {expected_seed}."
        )
    if (
        expected_val_per_class is not None
        and int(payload.get("val_per_class", -1)) != int(expected_val_per_class)
    ):
        raise ValueError(
            f"val_per_class mismatch: file uses {payload.get('val_per_class')} but "
            f"training requested {expected_val_per_class}."
        )

    train_list = [int(v) for v in train_indices.tolist()]
    val_list = [int(v) for v in val_indices.tolist()]

    if set(train_list).intersection(val_list):
        raise ValueError("Training and validation indices overlap.")
    if len(train_list) + len(val_list) != 50_000:
        raise ValueError("Expected the split to cover all 50,000 CIFAR-10 training images.")

    return train_list, val_list, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("./splits/cifar10_train_val_seed42.pt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing split file instead of validating/reusing it.",
    )
    args = parser.parse_args()

    if args.split_file.exists() and not args.overwrite:
        train_indices, val_indices, payload = load_split(
            args.split_file,
            expected_seed=args.seed,
            expected_val_per_class=args.val_per_class,
        )
        print(f"Reusing existing split: {args.split_file.resolve()}")
        print(f"Training images:   {len(train_indices)}")
        print(f"Validation images: {len(val_indices)}")
        print(f"Seed:              {payload['seed']}")
        print("Official CIFAR-10 test split was not used.")
        return

    payload = create_stratified_split(
        args.data_root,
        args.split_file,
        seed=args.seed,
        val_per_class=args.val_per_class,
    )
    print(f"Saved split:        {args.split_file.resolve()}")
    print(f"Training images:   {payload['train_size']}")
    print(f"Validation images: {payload['val_size']}")
    print(f"Seed:              {payload['seed']}")
    print("Official CIFAR-10 test split was not used.")


if __name__ == "__main__":
    main()
