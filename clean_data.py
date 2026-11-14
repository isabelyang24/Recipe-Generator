"""
Clean Scuccorese/food-ingredients-dataset from Hugging Face.

Columns in the dataset: category, subcategory, ingredient (all strings), image.
Output:
    data/cleaned/<ingredient>/<ingredient>_<row>.jpg
    data/cleaned/metadata.csv
    data/cleaned/report.txt

Install:  pip install datasets pillow imagehash tqdm pandas
Run:      python clean_dataset.py
"""

import io
import os
import re
from collections import Counter, defaultdict

import imagehash
import pandas as pd
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image, ImageOps
from tqdm import tqdm

# ---------- CONFIG ----------
DATASET_NAME = "Scuccorese/food-ingredients-dataset"
OUTPUT_DIR = "data/cleaned"
IMG_SIZE = (224, 224)   # common input size for CV models
MIN_SIDE = 64           # drop images smaller than this (px)
MIN_PER_CLASS = 10      # flag ingredients with fewer images than this
# ----------------------------


def clean_text(text) -> str:
    """Lowercase, trim, collapse whitespace, drop odd symbols."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def to_folder_name(text: str) -> str:
    return text.replace(" ", "_").replace("-", "_")


def main():
    ds = load_dataset(DATASET_NAME, split="train")
    # decode=False -> we get raw bytes, so a corrupt image can't crash the loop
    ds = ds.cast_column("image", HFImage(decode=False))
    print(f"Loaded {len(ds)} rows. Columns: {ds.column_names}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seen = {}                       # phash -> ingredient it was first seen under
    cross_label_dupes = []          # same image, different ingredient label
    stats = Counter()
    records = []

    for i, row in enumerate(tqdm(ds, desc="Cleaning")):
        # 1. Clean text fields; drop rows with no usable label
        ingredient = clean_text(row["ingredient"])
        category = clean_text(row["category"])
        subcategory = clean_text(row["subcategory"])
        if not ingredient:
            stats["missing_label"] += 1
            continue

        # 2. Load image safely
        try:
            raw = row["image"]["bytes"]
            img = Image.open(io.BytesIO(raw))
            img = ImageOps.exif_transpose(img)   # fix phone photos that are rotated
            img = img.convert("RGB")             # handles RGBA / grayscale / palette
        except Exception:
            stats["corrupt"] += 1
            continue

        # 3. Drop tiny images
        if min(img.size) < MIN_SIDE:
            stats["too_small"] += 1
            continue

        # 4. Drop duplicates (perceptual hash catches resized/recompressed copies)
        h = str(imagehash.phash(img))
        if h in seen:
            if seen[h] != ingredient:
                cross_label_dupes.append((seen[h], ingredient, i))
            stats["duplicate"] += 1
            continue
        seen[h] = ingredient

        # 5. Resize and save into one folder per ingredient
        folder = to_folder_name(ingredient)
        class_dir = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(class_dir, exist_ok=True)
        filename = f"{folder}_{i}.jpg"
        img.resize(IMG_SIZE, Image.LANCZOS).save(
            os.path.join(class_dir, filename), quality=95
        )

        records.append(
            {
                "filepath": f"{folder}/{filename}",
                "ingredient": ingredient,
                "subcategory": subcategory,
                "category": category,
            }
        )
        stats["kept"] += 1

    # 6. Save metadata (keeps category/subcategory, useful for recipe generation later)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

    # 7. Report
    counts = df["ingredient"].value_counts()
    small = counts[counts < MIN_PER_CLASS]

    # An ingredient that appears under multiple categories is probably a labeling error
    multi_cat = (
        df.groupby("ingredient")["category"].nunique().loc[lambda s: s > 1]
    )

    lines = [
        f"Stats: {dict(stats)}",
        f"Ingredients (classes): {counts.shape[0]}",
        f"Images per class: min={counts.min()}, median={int(counts.median())}, max={counts.max()}",
        f"\nClasses with < {MIN_PER_CLASS} images ({len(small)}):",
        *[f"  {k}: {v}" for k, v in small.items()],
        f"\nIngredients labeled under more than one category ({len(multi_cat)}):",
        *[f"  {k}" for k in multi_cat.index],
        f"\nSame image under different labels ({len(cross_label_dupes)}):",
        *[f"  row {r}: '{a}' vs '{b}'" for a, b, r in cross_label_dupes[:50]],
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()