"""Cross-trigger specificity evaluation for the input-aware watermark.

This experiment tests whether the trigger generated from a different image
incorrectly activates the watermark on the current verification image.

For each ordered verification sample x_i, the script uses a deterministic
different source image x_j:

    x_i -> apply G(x_j) and M(x_j) -> classifier

The desired output is the original class of x_i, not the watermark target.
A robust input-aware association should therefore produce:

- high cross-trigger accuracy;
- low cross-trigger TSR.

The script also reports clean accuracy and normal paired-trigger TSR for
reference. It loads only ``run/training/best_model.pt`` and uses the shared
9,000-image CIFAR-10 non-target verification index file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision

from common_verification import (
    VerificationBaseSet,
    evaluate_pair_loader,
    make_cifar10_test_dataset,
    make_data_loader,
    write_csv,
)
from input_aware.input_aware_utilities import (
    InputAwareVerificationSet,
    MaskGenerator,
    inject_trigger,
    load_best_input_aware_model,
    make_clean_forward,
    make_trigger_forward,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_DIR = Path(__file__).resolve().parent / "run" / "training"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_INDEX_FILE = (
    PROJECT_ROOT
    / "verification_sets"
    / "cifar10_test_non_target_class0.pt"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "run" / "cross_trigger"


@dataclass(frozen=True)
class CrossTriggerMetrics:
    """Metrics for triggers generated from a different verification image."""

    original_label_accuracy: float
    target_activation_rate: float
    original_label_loss: float
    sample_count: int


class CrossTriggerVerificationSet(Dataset):
    """Pair every verification image with a different trigger-source image.

    A cyclic offset is used over the common ordered 9,000-image verification set.
    This guarantees that every image:

    - appears once as the image being classified;
    - receives a trigger from another image;
    - is paired deterministically across repeated runs.

    Each item returns:

        image,
        trigger_source_image,
        original_label,
        target_label,
        image_base_index,
        trigger_source_base_index
    """

    def __init__(
        self,
        verification_set: VerificationBaseSet,
        *,
        pair_offset: int = 1,
    ) -> None:
        self.verification_set = verification_set
        self.size = len(verification_set)

        if self.size < 2:
            raise ValueError(
                "Cross-trigger evaluation requires at least two images."
            )

        normalised_offset = int(pair_offset) % self.size
        if normalised_offset == 0:
            raise ValueError(
                "pair_offset must not be zero modulo the verification-set size, "
                "because an image cannot provide its own cross-trigger."
            )

        self.pair_offset = normalised_offset

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, item: int):
        source_item = (item + self.pair_offset) % self.size

        (
            image,
            original_label,
            target_label,
            image_base_index,
        ) = self.verification_set[item]

        (
            trigger_source_image,
            _source_original_label,
            _source_target_label,
            trigger_source_base_index,
        ) = self.verification_set[source_item]

        if image_base_index == trigger_source_base_index:
            raise RuntimeError(
                "Cross-trigger pairing produced a self-pair unexpectedly."
            )

        return (
            image,
            trigger_source_image,
            original_label,
            target_label,
            image_base_index,
            trigger_source_base_index,
        )


@torch.no_grad()
def evaluate_cross_trigger(
    classifier: nn.Module,
    trigger_generator: nn.Module,
    mask_generator: nn.Module,
    loader,
    device: str | torch.device,
) -> CrossTriggerMetrics:
    """Evaluate original-label retention and target activation under wrong triggers."""

    classifier.eval()
    trigger_generator.eval()
    mask_generator.eval()

    resolved_device = torch.device(device)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_original_correct = 0
    total_target_hits = 0
    total_original_loss = 0.0
    total_samples = 0

    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) != 6:
            raise ValueError(
                "Expected batches containing image, trigger-source image, "
                "original label, target label, and both base indices."
            )

        (
            images,
            trigger_source_images,
            original_labels,
            target_labels,
            _image_base_indices,
            _source_base_indices,
        ) = batch

        images = images.to(resolved_device, non_blocking=True)
        trigger_source_images = trigger_source_images.to(
            resolved_device,
            non_blocking=True,
        )
        original_labels = original_labels.to(
            resolved_device,
            non_blocking=True,
        ).long()
        target_labels = target_labels.to(
            resolved_device,
            non_blocking=True,
        ).long()

        # Generate the pattern and mask from a different image x_j, then apply
        # them to the current image x_i.
        patterns_from_other = trigger_generator(trigger_source_images)
        masks_from_other = MaskGenerator.sharpen(
            mask_generator(trigger_source_images)
        )
        cross_triggered_images = inject_trigger(
            images,
            patterns_from_other,
            masks_from_other,
        )

        logits = classifier(cross_triggered_images)
        predictions = logits.argmax(dim=1)

        total_original_loss += criterion(
            logits,
            original_labels,
        ).item()
        total_original_correct += predictions.eq(
            original_labels
        ).sum().item()
        total_target_hits += predictions.eq(target_labels).sum().item()
        total_samples += original_labels.numel()

    if total_samples == 0:
        raise ValueError("No cross-trigger samples were evaluated.")

    return CrossTriggerMetrics(
        original_label_accuracy=(
            100.0 * total_original_correct / total_samples
        ),
        target_activation_rate=(
            100.0 * total_target_hits / total_samples
        ),
        original_label_loss=total_original_loss / total_samples,
        sample_count=total_samples,
    )


@torch.no_grad()
def save_cross_trigger_examples(
    output_path: Path,
    classifier: nn.Module,
    trigger_generator: nn.Module,
    mask_generator: nn.Module,
    loader,
    device: str | torch.device,
    *,
    max_examples: int = 16,
) -> None:
    """Save rows of original, source, pattern, mask, and cross-triggered images."""

    classifier.eval()
    trigger_generator.eval()
    mask_generator.eval()

    batch = next(iter(loader))
    (
        images,
        trigger_source_images,
        _original_labels,
        _target_labels,
        _image_base_indices,
        _source_base_indices,
    ) = batch

    count = min(int(max_examples), images.size(0))
    images = images[:count].to(device)
    trigger_source_images = trigger_source_images[:count].to(device)

    patterns = trigger_generator(trigger_source_images)
    masks = MaskGenerator.sharpen(mask_generator(trigger_source_images))
    cross_triggered = inject_trigger(images, patterns, masks)

    mask_rgb = masks.repeat(1, 3, 1, 1)
    visual_grid = torch.cat(
        [
            images.cpu(),
            trigger_source_images.cpu(),
            patterns.cpu(),
            mask_rgb.cpu(),
            cross_triggered.cpu(),
        ],
        dim=0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(
        visual_grid,
        output_path,
        nrow=count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-dir",
        "--training_dir",
        type=Path,
        default=DEFAULT_TRAINING_DIR,
        help="Directory containing the retained best_model.pt checkpoint.",
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--verification-index-file",
        type=Path,
        default=DEFAULT_INDEX_FILE,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument(
        "--pair-offset",
        type=int,
        default=1,
        help=(
            "Cyclic offset used to select the different trigger-source image. "
            "The default pairs each image with the next verification image."
        ),
    )
    parser.add_argument(
        "--save-examples",
        action="store_true",
        help="Save a visual grid of cross-trigger construction examples.",
    )
    parser.add_argument("--max-examples", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    classifier, trigger_generator, mask_generator, artifacts = (
        load_best_input_aware_model(
            args.training_dir,
            args.device,
            expected_target_label=args.target_label,
        )
    )

    test_set = make_cifar10_test_dataset(args.data_root)

    # Clean performance remains measured on the complete 10,000-image test set.
    clean_loader = make_data_loader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # Paired-trigger and cross-trigger testing both use the exact same ordered
    # 9,000-image non-target verification set.
    verification_base = VerificationBaseSet(
        test_set,
        args.verification_index_file,
        target_label=artifacts.target_label,
    )

    paired_trigger_set = InputAwareVerificationSet(verification_base)
    paired_trigger_loader = make_data_loader(
        paired_trigger_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    cross_trigger_set = CrossTriggerVerificationSet(
        verification_base,
        pair_offset=args.pair_offset,
    )
    cross_trigger_loader = make_data_loader(
        cross_trigger_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    clean_metrics = evaluate_pair_loader(
        classifier,
        clean_loader,
        args.device,
        forward_fn=make_clean_forward(),
    )

    paired_trigger_metrics = evaluate_pair_loader(
        classifier,
        paired_trigger_loader,
        args.device,
        forward_fn=make_trigger_forward(
            trigger_generator,
            mask_generator,
        ),
    )

    cross_metrics = evaluate_cross_trigger(
        classifier,
        trigger_generator,
        mask_generator,
        cross_trigger_loader,
        args.device,
    )

    row = {
        "model_file": artifacts.checkpoint_path.name,
        "trigger_type": "input_aware_dynamic",
        "target_label": artifacts.target_label,
        "pairing_method": "cyclic_different_image",
        "pair_offset": cross_trigger_set.pair_offset,
        "clean_acc_percent": f"{clean_metrics.accuracy:.6f}",
        "clean_loss": f"{clean_metrics.loss:.6f}",
        "paired_trigger_tsr_percent": (
            f"{paired_trigger_metrics.accuracy:.6f}"
        ),
        "paired_trigger_loss": f"{paired_trigger_metrics.loss:.6f}",
        "cross_trigger_acc_percent": (
            f"{cross_metrics.original_label_accuracy:.6f}"
        ),
        "cross_trigger_loss": (
            f"{cross_metrics.original_label_loss:.6f}"
        ),
        "cross_trigger_tsr_percent": (
            f"{cross_metrics.target_activation_rate:.6f}"
        ),
        "clean_test_size": clean_metrics.sample_count,
        "verification_size": cross_metrics.sample_count,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "checkpoint_path": str(artifacts.checkpoint_path),
        "verification_index_file": str(
            args.verification_index_file.resolve()
        ),
    }

    output_path = args.out_dir / "cross_trigger_summary.csv"
    write_csv(output_path, row.keys(), [row])

    if args.save_examples:
        save_cross_trigger_examples(
            args.out_dir / "cross_trigger_examples.png",
            classifier,
            trigger_generator,
            mask_generator,
            cross_trigger_loader,
            args.device,
            max_examples=args.max_examples,
        )

    print(f"Checkpoint: {artifacts.checkpoint_path}")
    print(f"Clean accuracy: {clean_metrics.accuracy:.2f}%")
    print(
        "Paired-trigger TSR: "
        f"{paired_trigger_metrics.accuracy:.2f}%"
    )
    print(
        "Cross-trigger original-label accuracy: "
        f"{cross_metrics.original_label_accuracy:.2f}%"
    )
    print(
        "Cross-trigger target activation rate: "
        f"{cross_metrics.target_activation_rate:.2f}%"
    )
    print(f"Cross-trigger samples: {cross_metrics.sample_count}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
