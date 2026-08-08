"""
subset_dataset.py

Select a random subset of image samples from a dataset and copy them
to a new location, preserving class subfolder structure if present.

Supports two layouts:
1. Flat folder:      source/img1.jpg, source/img2.jpg, ...
2. Class-subfolders:  source/cat/img1.jpg, source/dog/img1.jpg, ...
   (common for image classification datasets, e.g. torchvision's
   ImageFolder convention: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)

Standard library only: pathlib, shutil, random, argparse.
Docs: https://docs.python.org/3/library/pathlib.html
      https://docs.python.org/3/library/shutil.html
      https://docs.python.org/3/library/random.html#random.seed  (reproducibility)
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def find_images(folder: Path):
    """Return all image files directly inside `folder` (non-recursive)."""
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def select_subset(source: Path, dest: Path, n: int = None, fraction: float = None,
                   per_class: bool = False, seed: int = 42):
    """
    Copy a random subset of images from `source` to `dest`.

    Exactly one of `n` (fixed count) or `fraction` (proportion, 0-1) must be given.
    If `per_class` is True, `source` is treated as containing one subfolder
    per class, and the subset is drawn independently within each class
    (keeps class balance the same as the original dataset).
    """
    if (n is None) == (fraction is None):
        raise ValueError("Specify exactly one of `n` or `fraction`.")

    random.seed(seed)  # reproducibility: same seed -> same subset every run
    dest.mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in source.iterdir() if d.is_dir()] if per_class else [source]

    total_copied = 0
    for class_dir in class_dirs:
        images = find_images(class_dir)
        if not images:
            continue

        k = n if n is not None else round(len(images) * fraction)
        k = min(k, len(images))  # can't sample more than what's available
        chosen = random.sample(images, k)

        out_dir = dest / class_dir.name if per_class else dest
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in chosen:
            shutil.copy2(img_path, out_dir / img_path.name)  # copy2 preserves metadata
        total_copied += len(chosen)

        if per_class:
            print(f"{class_dir.name}: copied {len(chosen)}/{len(images)}")

    print(f"Done. Total images copied: {total_copied} -> {dest}")


def main():
    parser = argparse.ArgumentParser(description="Copy a random subset of an image dataset to a new folder.")
    parser.add_argument("source", type=Path, help="Path to the source dataset folder")
    parser.add_argument("dest", type=Path, help="Path to the destination folder for the subset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n", type=int, help="Fixed number of images to select (per class, if --per-class)")
    group.add_argument("--fraction", type=float, help="Fraction of images to select, e.g. 0.1 for 10%%")
    parser.add_argument("--per-class", action="store_true", help="Treat subfolders of source as classes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    select_subset(args.source, args.dest, n=args.n, fraction=args.fraction,
                  per_class=args.per_class, seed=args.seed)


if __name__ == "__main__":
    main()
