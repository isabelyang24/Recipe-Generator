"""Regression checks for corrupt files, labels, duplicates and split leakage."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import (clean_label, deduplicate, load_labels, load_rgb, prepare,
                        resolve_image, resize_rgb)
from dataset import load_sample


class PreprocessingTests(unittest.TestCase):
    def test_label_normalization_preserves_different_milks(self):
        self.assertEqual(clean_label("  Oat-Milk "), "oat milk")
        self.assertNotEqual(clean_label("Soy-Milk"), clean_label("Milk"))
        self.assertEqual(clean_label("Aubergine"), "eggplant")

    def test_duplicate_keeps_holdout_and_quarantines_label_conflicts(self):
        rows = [dict(source_path=s, split=s, pixel_sha256="same", fine_label=0)
                for s in ("train", "validation", "test")]
        kept, excluded = deduplicate(rows)
        self.assertEqual([r["split"] for r in kept], ["test"])
        self.assertEqual(len(excluded), 2)
        rows[0]["fine_label"] = 1
        kept, excluded = deduplicate(rows)
        self.assertEqual(kept, [])
        self.assertEqual(len(excluded), 3)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                resolve_image(Path(temp), "../outside.jpg")

    def test_rgb_and_aspect_ratio_padding(self):
        image = resize_rgb(Image.new("RGB", (100, 50), (255, 0, 0)), 224)
        self.assertEqual(image.size, (224, 224))
        self.assertEqual(image.getpixel((0, 0)), (128, 128, 128))
        self.assertEqual(image.getpixel((112, 112)), (255, 0, 0))

    def test_pipeline_rejects_bad_records_and_preserves_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / "source", Path(temp) / "result"
            root.mkdir()
            (root / "classes.csv").write_text(
                "Class Name (str),Class ID (int),Coarse Class Name (str),Coarse Class ID (int)\n"
                "Apple,0,Apple,0\n")
            for split in ("train", "val", "test"):
                folder = root / split / "Fruit"
                folder.mkdir(parents=True)
                Image.new("L", (80, 100), 30 if split == "val" else 60).save(folder / "a.png")
                (root / f"{split}.txt").write_text(f"{split}/Fruit/a.png,0,0\n")
            (root / "train/Fruit/corrupt.png").write_bytes(b"not an image")
            with (root / "train.txt").open("a") as stream:
                stream.write("train/Fruit/corrupt.png,0,0\n")
                stream.write("train/Fruit/missing.png,0,0\n")
                stream.write("train/Fruit/a.png,0,1\n")
            stats = prepare(root, out)
            self.assertEqual(stats["input_rows"], 6)
            self.assertEqual(stats["invalid_or_rejected_rows"], 3)
            self.assertEqual(stats["duplicate_exclusions"], 1)
            self.assertEqual(stats["kept_rows"], 2)
            with (out / "manifest.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r["split"] for r in rows}, {"test", "validation"})
            for row in rows:
                pixels, label = load_sample(out, row, augment=True)
                plain, _ = load_sample(out, row, augment=False)
                self.assertEqual(pixels.shape, (3, 224, 224))
                self.assertEqual(pixels.dtype, np.float32)
                self.assertEqual(label, 0)
                np.testing.assert_array_equal(pixels, plain)
            with self.assertRaises(FileExistsError):
                prepare(root, out)


if __name__ == "__main__":
    unittest.main()
