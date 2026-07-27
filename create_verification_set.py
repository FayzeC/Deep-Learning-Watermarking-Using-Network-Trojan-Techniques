import argparse
import os

import torch
from torchvision.datasets import CIFAR10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="./verification_sets/cifar10_test_non_target_class0.pt",
    )
    args = parser.parse_args()

    # The images do not need a transform because only the labels are inspected.
    test_set = CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
    )

    indices = [
        index
        for index, label in enumerate(test_set.targets)
        if int(label) != args.target_label
    ]

    if len(indices) != 9000:
        raise ValueError(
            f"Expected 9000 non-target CIFAR-10 test images, "
            f"but found {len(indices)}."
        )

    if any(int(test_set.targets[index]) == args.target_label for index in indices):
        raise ValueError("Target-class images were found in the verification set.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    torch.save(
        {
            "dataset": "CIFAR-10",
            "split": "test",
            "target_label": args.target_label,
            "indices": indices,
            "size": len(indices),
        },
        args.output,
    )

    print(f"Saved {len(indices)} verification indices to: {args.output}")
    print(f"Target label excluded: {args.target_label}")


if __name__ == "__main__":
    main()