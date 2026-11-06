"""
Data preprocessing & processing example using the
Scuccorese/food-ingredients-dataset from the Hugging Face Hub.

Each row is a photo of a food ingredient with three text labels:
    category     e.g. "vegetables"
    subcategory  e.g. "leafy"
    ingredient   e.g. "spinach"
    image        the photo itself

Pipeline:
    1. Stream a random sample of the dataset
    2. Explore the raw data (shape, missing values, duplicates, class balance)
    3. Clean the text labels
    4. Inspect / validate the images (size, color mode, corrupt files)
    5. Filter out bad rows and under-represented classes
    6. Encode labels as integers
    7. Stratified train / validation / test split
    8. Image transforms (resize, RGB, normalize) applied lazily
    9. Export processed images as PNGs + a CSV index (Data/processed/)
   10. Summary

Run:  python Data/Data_example.py

The full dataset is ~1.3 GB, so instead of downloading it we stream a few
random 100-row chunks straight from the Hub. The sample is saved to
Data/sample_cache/ so later runs start instantly; delete that folder to
draw a fresh sample.
"""

import io
import random
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from datasets import Image as HFImage
from huggingface_hub import HfFileSystem
from PIL import Image

DATASET_NAME = "Scuccorese/food-ingredients-dataset"
CHUNKS_PER_FILE = 1         # 100 rows per chunk, 3 files -> 300 rows by default
SAMPLE_CACHE = Path(__file__).parent / "sample_cache"
OUTPUT_DIR = Path(__file__).parent / "processed"
MIN_IMAGES_PER_CLASS = 10   # drop ingredients with fewer photos than this
MIN_IMAGE_SIDE = 64         # drop images smaller than this (pixels)
TARGET_SIZE = (224, 224)    # standard input size for most vision models
SEED = 42

# ImageNet mean/std, the usual normalization for pretrained vision models
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# 1. Stream a sample
# ---------------------------------------------------------------------------
# The dataset is stored as 3 parquet files, each split into row groups of
# 100 rows. Rows are sorted by ingredient, so reading the first N rows would
# give only a handful of ingredients. Picking random row groups from every
# file gives a much more varied sample, and only those row groups are
# transferred over the network.
section("1. Streaming a sample")


def stream_sample():
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{DATASET_NAME}/data/*.parquet"))
    rng = random.Random(SEED)
    tables = []
    for path in files:
        parquet = pq.ParquetFile(fs.open(path))
        for group in rng.sample(range(parquet.num_row_groups), CHUNKS_PER_FILE):
            print(f"Reading {path.rsplit('/', 1)[-1]} row group {group}...")
            tables.append(parquet.read_row_group(group))
    # The parquet metadata carries the Hugging Face feature types, so the
    # image column comes back as a proper Image feature.
    return Dataset(pa.concat_tables(tables))


if SAMPLE_CACHE.exists():
    print(f"Loading cached sample from {SAMPLE_CACHE}")
    data = Dataset.load_from_disk(str(SAMPLE_CACHE))
else:
    data = stream_sample()
    data.save_to_disk(str(SAMPLE_CACHE))
    print(f"Saved sample to {SAMPLE_CACHE}")

print(data)
print("\nFeatures:", data.features)


# ---------------------------------------------------------------------------
# 2. Explore the raw labels
# ---------------------------------------------------------------------------
# Drop the image column before converting to pandas so we don't pull
# ~1.4 GB of image bytes into a DataFrame just to look at the labels.
section("2. Exploring raw labels")
labels_df = data.remove_columns("image").to_pandas()

print(f"Rows: {len(labels_df)}")
print("\nFirst rows:\n", labels_df.head())
print("\nMissing values per column:\n", labels_df.isna().sum())
print("\nUnique values per column:\n", labels_df.nunique())

print("\nImages per category:\n", labels_df["category"].value_counts())

per_ingredient = labels_df["ingredient"].value_counts()
print("\nImages per ingredient (describe):\n", per_ingredient.describe())
print("\nMost common ingredients:\n", per_ingredient.head(10))
print("\nLeast common ingredients:\n", per_ingredient.tail(10))


# ---------------------------------------------------------------------------
# 3. Clean the text labels
# ---------------------------------------------------------------------------
section("3. Cleaning text labels")


def clean_label(text):
    """Lowercase, turn underscores/hyphens into spaces, collapse whitespace."""
    if text is None:
        return "unknown"
    text = text.lower().strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text or "unknown"


def clean_labels(batch):
    return {
        col: [clean_label(v) for v in batch[col]]
        for col in ("category", "subcategory", "ingredient")
    }


before = labels_df["ingredient"].nunique()
data = data.map(clean_labels, batched=True, desc="Cleaning labels")
after = len(set(data["ingredient"]))
print(f"Unique ingredients before cleaning: {before}, after: {after}")
if after < before:
    print("  -> cleaning merged labels that differed only by case/spacing")


# ---------------------------------------------------------------------------
# 4. Inspect / validate images
# ---------------------------------------------------------------------------
# Turning off decoding gives us the raw bytes. PIL's Image.open only reads
# the file header, so getting each image's size and mode is fast and never
# loads the full pixel data.
section("4. Inspecting images")


def image_metadata(batch):
    widths, heights, modes, valid = [], [], [], []
    for img in batch["image"]:
        try:
            with Image.open(io.BytesIO(img["bytes"])) as im:
                im.verify()  # raises if the file is truncated / corrupt
            with Image.open(io.BytesIO(img["bytes"])) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
                modes.append(im.mode)
                valid.append(True)
        except Exception:
            widths.append(0)
            heights.append(0)
            modes.append("invalid")
            valid.append(False)
    return {"width": widths, "height": heights, "mode": modes, "is_valid": valid}


data = data.cast_column("image", HFImage(decode=False))
data = data.map(image_metadata, batched=True, desc="Reading image headers")

meta_df = data.remove_columns("image").to_pandas()
print("Width / height stats:\n", meta_df[["width", "height"]].describe())
print("\nColor modes:\n", meta_df["mode"].value_counts())
print(f"\nCorrupt images: {(~meta_df['is_valid']).sum()}")

meta_df["aspect_ratio"] = meta_df["width"] / meta_df["height"].replace(0, np.nan)
print("\nAspect ratio stats:\n", meta_df["aspect_ratio"].describe())


# ---------------------------------------------------------------------------
# 5. Filter bad rows and rare classes
# ---------------------------------------------------------------------------
section("5. Filtering")
start = len(data)

data = data.filter(lambda ok: ok, input_columns="is_valid", desc="Drop corrupt")
print(f"After dropping corrupt images:     {len(data)}")

data = data.filter(
    lambda w, h: min(w, h) >= MIN_IMAGE_SIDE,
    input_columns=["width", "height"],
    desc="Drop tiny images",
)
print(f"After dropping images < {MIN_IMAGE_SIDE}px:      {len(data)}")

counts = pd.Series(data["ingredient"]).value_counts()
keep = set(counts[counts >= MIN_IMAGES_PER_CLASS].index)
data = data.filter(lambda ing: ing in keep, input_columns="ingredient", desc="Drop rare classes")
print(f"After dropping classes < {MIN_IMAGES_PER_CLASS} images: {len(data)}")
print(f"Ingredients kept: {len(keep)} of {len(counts)}")
print(f"Removed {start - len(data)} rows in total")


# ---------------------------------------------------------------------------
# 6. Encode labels
# ---------------------------------------------------------------------------
# class_encode_column converts the string column to a ClassLabel, so each
# ingredient becomes an integer id while the names stay attached.
section("6. Encoding labels")
data = data.class_encode_column("ingredient")
label_names = data.features["ingredient"].names
print(f"{len(label_names)} classes, e.g.:")
for i in range(min(5, len(label_names))):
    print(f"  {i} -> {label_names[i]}")


# ---------------------------------------------------------------------------
# 7. Stratified train / validation / test split (70 / 15 / 15)
# ---------------------------------------------------------------------------
# Stratifying keeps every ingredient's share the same in each split, which
# matters here because the classes are unbalanced.
section("7. Splitting")
split = data.train_test_split(test_size=0.30, stratify_by_column="ingredient", seed=SEED)
holdout = split["test"].train_test_split(test_size=0.50, stratify_by_column="ingredient", seed=SEED)
splits = {"train": split["train"], "validation": holdout["train"], "test": holdout["test"]}

for name, part in splits.items():
    print(f"{name:<11} {len(part):>5} rows  ({len(part) / len(data):.0%})")


# ---------------------------------------------------------------------------
# 8. Image transforms
# ---------------------------------------------------------------------------
# with_transform runs only when rows are accessed, so we never store
# thousands of resized copies on disk. Training images also get a random
# horizontal flip as simple data augmentation.
section("8. Image transforms")


def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # handle L / RGBA / P
    return img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)


def to_array(img_bytes, augment):
    img = preprocess_image(img_bytes)
    if augment and np.random.rand() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # scale to [0, 1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD       # standardize per channel
    return arr.transpose(2, 0, 1)                    # HWC -> CHW


def make_transform(augment):
    def transform(batch):
        return {
            "pixel_values": [to_array(img["bytes"], augment) for img in batch["image"]],
            "label": batch["ingredient"],
        }
    return transform


model_ready = {
    name: part.with_transform(make_transform(augment=(name == "train")))
    for name, part in splits.items()
}

sample = model_ready["train"][:8]
pixels = np.stack(sample["pixel_values"])
print(f"Batch shape (N, C, H, W): {pixels.shape}")
print(f"Pixel mean per channel:  {pixels.mean(axis=(0, 2, 3)).round(3)}")
print(f"Pixel std per channel:   {pixels.std(axis=(0, 2, 3)).round(3)}")
print(f"Labels: {[label_names[i] for i in sample['label']]}")


# ---------------------------------------------------------------------------
# 9. Export processed images + CSV
# ---------------------------------------------------------------------------
# Each image is saved as a 224x224 RGB PNG (lossless), and the CSV links
# every file to its split and labels. Normalization and flipping aren't
# baked in: normalized values can't be stored in a PNG, and augmentation
# should be random each epoch, so both stay load-time steps.
section("9. Exporting processed images")
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)  # start fresh so no stale images remain

rows = []
for name, part in splits.items():
    (OUTPUT_DIR / name).mkdir(parents=True)
    for i, row in enumerate(part):
        label_name = label_names[row["ingredient"]]
        rel_path = f"{name}/{label_name.replace(' ', '_')}_{i:04d}.png"
        preprocess_image(row["image"]["bytes"]).save(OUTPUT_DIR / rel_path)
        rows.append({
            "image_path": rel_path,
            "split": name,
            "category": row["category"],
            "subcategory": row["subcategory"],
            "ingredient": label_name,
            "label": row["ingredient"],
            "original_width": row["width"],
            "original_height": row["height"],
            "original_mode": row["mode"],
        })

csv_path = OUTPUT_DIR / "processed_images.csv"
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"Saved {len(rows)} images and {csv_path}")
print(pd.read_csv(csv_path).head())


# ---------------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------------
section("10. Summary")
print(f"Raw rows:            {start}")
print(f"Clean rows:          {len(data)}")
print(f"Classes:             {len(label_names)}")
print(f"Model input:         3 x {TARGET_SIZE[0]} x {TARGET_SIZE[1]}, ImageNet-normalized")
print(f"Splits:              " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
print(f"Output:              {csv_path}")
