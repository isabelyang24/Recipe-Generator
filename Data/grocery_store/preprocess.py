"""Audit and prepare GroceryStoreDataset without changing upstream files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

SOURCE_URL = "https://github.com/marcusklasson/GroceryStoreDataset.git"
SOURCE_COMMIT = "fc80ba90f803d79d0383df52c5a4ac5de99ff6fc"
HERE = Path(__file__).resolve().parent
SPLITS = ("test", "val", "train")  # Keep holdout copies if exact duplicates occur.
SPLIT_NAME = {"train": "train", "val": "validation", "test": "test"}
ALIASES = {"aubergine": "eggplant", "carrots": "carrot", "pepper": "bell pepper",
           "yoghurt": "yogurt", "oatghurt": "oat yogurt", "soyghurt": "soy yogurt"}


def clean_label(value: str) -> str:
    value = re.sub(r"\s+", " ", re.sub(r"[-_]", " ", value.strip().lower()))
    return ALIASES.get(value, value)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_image(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Image path escapes dataset directory")
    return path


def load_rgb(path: Path) -> tuple[Image.Image, str]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()  # Decode all pixels, not just the header.
        mode = image.mode
        return ImageOps.exif_transpose(image).convert("RGB"), mode


def pixel_hash(image: Image.Image) -> str:
    h = hashlib.sha256(f"{image.width},{image.height},RGB:".encode())
    h.update(image.tobytes())
    return h.hexdigest()


def difference_hash(image: Image.Image) -> int:
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(small.get_flattened_data())
    result = 0
    for y in range(8):
        for x in range(8):
            result = (result << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return result


def resize_rgb(image: Image.Image, size: int) -> Image.Image:
    # Letterbox instead of stretching or cropping away ingredient details.
    return ImageOps.pad(image, (size, size), method=Image.Resampling.LANCZOS,
                        color=(128, 128, 128))


def download_source(destination: Path) -> Path:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "--depth", "1",
                        SOURCE_URL, str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1",
                        "origin", SOURCE_COMMIT], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach",
                        SOURCE_COMMIT], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if actual != SOURCE_COMMIT:
        raise ValueError(f"Source is {actual}; expected {SOURCE_COMMIT}. Use a fresh cache.")
    dirty = subprocess.check_output(
        ["git", "-C", str(destination), "status", "--porcelain", "--", "dataset"],
        text=True).strip()
    if dirty:
        raise ValueError("Cached upstream dataset has local modifications.")
    return destination


def load_labels(root: Path) -> tuple[dict, list[dict], list[dict]]:
    fine, coarse = {}, {}
    with (root / "classes.csv").open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            fid = int(row["Class ID (int)"])
            cid = int(row["Coarse Class ID (int)"])
            fname, cname = row["Class Name (str)"], row["Coarse Class Name (str)"]
            if fid in fine or not fname.strip() or not cname.strip():
                raise ValueError("Duplicate or missing class label")
            if cid in coarse and coarse[cid] != cname:
                raise ValueError("Conflicting coarse class names")
            fine[fid] = {"fine_label": fid, "fine_name": fname,
                         "label": cid, "coarse_name": cname,
                         "ingredient": clean_label(cname)}
            coarse[cid] = cname
    if sorted(coarse) != list(range(len(coarse))):
        raise ValueError("Coarse labels must be contiguous and zero-based")
    labels = [{"label": cid, "original_name": name, "ingredient": clean_label(name)}
              for cid, name in sorted(coarse.items())]
    if len({r["ingredient"] for r in labels}) != len(labels):
        raise ValueError("Normalization unexpectedly merged coarse classes")
    return fine, labels, [fine[k] for k in sorted(fine)]


def audit(root: Path, min_side: int) -> tuple[list[dict], list[dict], dict]:
    fine, labels, fine_labels = load_labels(root)
    rows, rejected, seen_paths = [], [], set()
    raw_counts = Counter()
    for split in SPLITS:
        with (root / f"{split}.txt").open(newline="", encoding="utf-8") as stream:
            for line_number, cells in enumerate(csv.reader(stream), 1):
                raw_counts[SPLIT_NAME[split]] += 1
                relative = cells[0].strip() if cells else ""
                base = {"source_path": relative, "split": SPLIT_NAME[split]}
                try:
                    if len(cells) != 3:
                        raise ValueError("Expected image path, fine label, coarse label")
                    fid, cid = int(cells[1]), int(cells[2])
                    if fid not in fine or fine[fid]["label"] != cid:
                        raise ValueError("Invalid fine/coarse label mapping")
                    if Path(relative).parts[0] != split:
                        raise ValueError("Path disagrees with source split")
                    if relative in seen_paths:
                        raise ValueError("Repeated source path")
                    seen_paths.add(relative)
                    path = resolve_image(root, relative)
                    image, mode = load_rgb(path)
                    if min(image.size) < min_side:
                        raise ValueError(f"Image smaller than {min_side}px")
                    rows.append({**base, **fine[fid],
                                 "category": clean_label(Path(relative).parts[1]),
                                 "original_width": image.width,
                                 "original_height": image.height, "original_mode": mode,
                                 "source_sha256": sha256_file(path),
                                 "pixel_sha256": pixel_hash(image),
                                 "dhash": difference_hash(image)})
                except (OSError, ValueError, KeyError, IndexError,
                        Image.DecompressionBombError) as exc:
                    rejected.append({**base, "reason": str(exc),
                                     "source_line": line_number})
    return rows, rejected, {"input_counts": dict(raw_counts),
                            "labels": labels, "fine_labels": fine_labels}


def deduplicate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["pixel_sha256"]].append(row)
    kept, excluded = [], []
    priority = {"test": 0, "validation": 1, "train": 2}
    for group in groups.values():
        ordered = sorted(group, key=lambda r: (priority[r["split"]], r["source_path"]))
        # Don't decide which of conflicting annotations is correct automatically.
        conflicts = len({r["fine_label"] for r in group}) > 1
        if not conflicts:
            kept.append(ordered[0])
        for row in (ordered if conflicts else ordered[1:]):
            excluded.append({"source_path": row["source_path"], "split": row["split"],
                             "reason": "conflicting_exact_duplicate" if conflicts else "exact_duplicate",
                             "kept_source_path": "" if conflicts else ordered[0]["source_path"],
                             "pixel_sha256": row["pixel_sha256"]})
    return sorted(kept, key=lambda r: (r["split"], r["source_path"])), excluded


def near_duplicate_candidates(rows: list[dict]) -> list[dict]:
    candidates = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            if left["split"] == right["split"]:
                continue
            distance = (left["dhash"] ^ right["dhash"]).bit_count()
            if distance <= 2:
                candidates.append({"path_a": left["source_path"], "split_a": left["split"],
                                   "path_b": right["source_path"], "split_b": right["split"],
                                   "hamming_distance": distance})
    return candidates


def prepare(root: Path, output: Path, size: int = 224, min_side: int = 64) -> dict:
    if size < 1 or min_side < 1:
        raise ValueError("Image sizes must be positive")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a fresh output folder.")
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output must not be inside the upstream dataset")
    rows, rejected, meta = audit(root, min_side)
    kept, duplicates = deduplicate(rows)
    if not kept:
        raise ValueError("No valid images remain")
    candidates = near_duplicate_candidates(kept)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Failed runs never leave an apparently complete output folder.
    with tempfile.TemporaryDirectory(prefix=".grocery-build-", dir=output.parent) as temp:
        build = Path(temp)
        manifest = []
        for row in kept:
            image, _ = load_rgb(resolve_image(root, row["source_path"]))
            relative = (Path(row["source_path"])
                        .with_suffix(".jpg").as_posix())
            relative = "images/" + relative.replace("val/", "validation/", 1)
            target = build / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            resize_rgb(image, size).save(target, "JPEG", quality=90,
                                         subsampling=0, optimize=True)
            manifest.append({"image_path": relative, **{k: v for k, v in row.items() if k != "dhash"},
                             "processed_sha256": sha256_file(target)})
        write_csv(build / "manifest.csv", manifest, list(manifest[0]))
        write_csv(build / "labels.csv", meta["labels"], list(meta["labels"][0]))
        write_csv(build / "fine_labels.csv", meta["fine_labels"], list(meta["fine_labels"][0]))
        write_csv(build / "rejected.csv", rejected,
                  ["source_path", "split", "reason", "source_line"])
        write_csv(build / "duplicates.csv", duplicates,
                  ["source_path", "split", "reason", "kept_source_path", "pixel_sha256"])
        write_csv(build / "near_duplicate_candidates.csv", candidates,
                  ["path_a", "split_a", "path_b", "split_b", "hamming_distance"])
        counts = Counter((r["label"], r["split"]) for r in manifest)
        class_counts = [{**label, **{s: counts[label["label"], s]
                                   for s in ("train", "validation", "test")}}
                        for label in meta["labels"]]
        write_csv(build / "class_counts.csv", class_counts, list(class_counts[0]))
        summary = {
            "dataset": "GroceryStoreDataset", "source_url": SOURCE_URL,
            "input_rows": sum(meta["input_counts"].values()),
            "input_splits": meta["input_counts"], "kept_rows": len(manifest),
            "kept_splits": dict(Counter(r["split"] for r in manifest)),
            "fine_classes": len(meta["fine_labels"]), "coarse_classes": len(meta["labels"]),
            "invalid_or_rejected_rows": len(rejected),
            "duplicate_exclusions": len(duplicates),
            "duplicate_exclusion_reasons": dict(Counter(r["reason"] for r in duplicates)),
            "cross_split_near_duplicate_candidates": len(candidates),
            "near_duplicate_review": "Candidates only; no automatic removal based on perceptual similarity.",
            "original_modes": dict(Counter(r["original_mode"] for r in rows)),
            "original_width_range": [min(r["original_width"] for r in rows), max(r["original_width"] for r in rows)],
            "original_height_range": [min(r["original_height"] for r in rows), max(r["original_height"] for r in rows)],
            "model_input_size": [size, size], "saved_mode": "RGB", "saved_format": "JPEG",
            "resize": "aspect-ratio-preserving letterbox; gray padding",
            "normalization": "Not baked into images; apply at training/inference time.",
            "augmentation": "Not baked into images; training only.",
            "class_split_gaps": [{"ingredient": r["ingredient"], "split": s}
                                 for r in class_counts for s in ("train", "validation", "test") if r[s] == 0],
            "annotation_sha256": {f: sha256_file(root / f) for f in
                                  ("classes.csv", "train.txt", "val.txt", "test.txt")},
        }
        (build / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        build.rename(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Path to an existing GroceryStoreDataset checkout")
    parser.add_argument("--output", type=Path, default=HERE / "processed")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    source = args.source_root or download_source(HERE / ".cache" / "GroceryStoreDataset")
    summary = prepare(source / "dataset", args.output, args.size)
    # Record the observed checkout, not an assumed version for a user-supplied source.
    try:
        summary["source_commit"] = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        summary["source_dataset_modified"] = bool(subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--", "dataset"], text=True).strip())
    except subprocess.CalledProcessError:
        summary["source_commit"] = None
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
