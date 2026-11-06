"""Small NumPy loader for the processed manifest; usable from PyTorch code."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_manifest(root, split):
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unknown split")
    with (Path(root) / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        return [r for r in csv.DictReader(stream) if r["split"] == split]


def load_sample(root, row, *, augment=False, rng=None, normalize=True):
    """Return float32 CHW pixels and the SOURCE coarse label (not a global ID).

    ImageNet normalization is for the proposed pretrained model. Change it if
    the team selects different weights. Held-out images never receive flips.
    Pass a seeded numpy.random.Generator for reproducible training augmentation.
    """
    with Image.open(Path(root) / row["image_path"]) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if augment and row["split"] == "train":
        generator = rng if rng is not None else np.random.default_rng()
        if generator.random() < 0.5:
            pixels = pixels[:, ::-1, :]
    if normalize:
        pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)), int(row["label"])
