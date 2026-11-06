# Grocery Store Dataset: Vaidik's preprocessing contribution

Branch: `vaidik-data-preprocessing`. All files in this contribution are scoped to
`Data/grocery_store/`; no other team member's data or scripts are modified.

## Actual cleaning results

| Check | Result |
| --- | --- |
| Annotated images checked | 5,421 |
| Valid images retained | 5,421 |
| Fine / coarse classes | 81 / 43 |
| Train / validation / test | 2,640 / 296 / 2,485 |
| Missing/corrupt/invalid-label/tiny images rejected | 0 |
| Exact pixel-duplicate exclusions | 0 |
| Cross-split dHash candidates at distance <= 2 | 0 |
| Saved images | 224x224 RGB JPEG, quality 90, aspect-ratio-preserving gray padding |

The upstream README describes 5,125 images and 42 coarse classes. The actual
`train.txt`, `val.txt`, `test.txt` and `classes.csv` at the pinned revision contain
the counts above. We used the annotations, including Zucchini (coarse ID 42).
Images were validated and standardized; none required removal. Duplicate checks
are not a guarantee that all visually similar scenes have been eliminated.

The original validation set has no nectarine, papaya, plum, sour milk, soy milk or
garlic examples. This is reported, not silently repaired. Preserve the test set;
agree on a training-only development split before class-level model tuning.

## Use the prepared data

From the repository root (Python 3.10+ and Git):

```bash
python -m pip install -r Data/grocery_store/requirements.txt
python Data/grocery_store/unpack_images.py
python -m unittest discover -s Data/grocery_store/tests -v
```

The small ZIP parts in `archives/` contain **all** prepared images, not a sample.
They are committed so teammates do not need an external account or private link.
`unpack_images.py` validates archive and image SHA-256 checksums, writes images
under `processed/images/`, and refuses to overwrite different existing bytes.
Each archive is an independent ZIP; no special split-ZIP program is needed.
Extracted images are ignored by Git to avoid storing them twice.

## Reproduce preprocessing from upstream

```bash
python Data/grocery_store/preprocess.py --output /tmp/grocery-rebuilt
```

This downloads the pinned upstream revision into a local ignored cache. For an
existing checkout, add `--source-root /path/to/GroceryStoreDataset`. Its observed
commit and local modification status are recorded in the report. Output must be
a new folder, and upstream files are never overwritten. The source commit is
`fc80ba90f803d79d0383df52c5a4ac5de99ff6fc`.

Pipeline: validate annotations -> fully decode/EXIF-orient/RGB -> check image sizes
and hashes -> resolve exact duplicates -> scan perceptual candidates -> letterbox
resize -> save images and reports. No images are shuffled between source splits.
Original dimensions and source hashes are retained for traceability.

## Files and integration contract

- `processed/manifest.csv`: one row per image, path relative to `processed/`, split,
  normalized ingredient group, source coarse ID, detailed class, original metadata
  and source/pixel/processed hashes.
- `processed/labels.csv`: stable source coarse IDs 0-42 and normalized group names.
- `processed/fine_labels.csv`: all original detailed labels and their coarse mapping.
- `processed/class_counts.csv`: class balance by split.
- `processed/summary.json`: measured counts, provenance and limitations.
- `processed/rejected.csv`, `duplicates.csv`, `near_duplicate_candidates.csv`:
  audit logs; header-only files mean no findings in this run.
- `archives/index.json`: image counts and archive checksums.
- `dataset.py`: optional NumPy loader returning float32 CHW arrays and source labels;
  applies ImageNet normalization and permits flips only for training rows.
- `preprocess.py`, `pack_images.py`, `unpack_images.py`: reproducible preparation
  and packaging tools. Move existing archives before rebuilding with `pack_images.py`.
- `PROPOSAL.md`: suggested project pipeline and methodology.
- `SOURCE_LICENSE`: the upstream MIT notice, retained with transformed images.

Example loader (run from this directory after unpacking):

```python
from dataset import read_manifest, load_sample
rows = read_manifest("processed", "train")
pixels, source_label = load_sample("processed", rows[0], augment=False)
# pixels.shape == (3, 224, 224); convert with torch.from_numpy if using PyTorch.
```

ImageNet normalization is a proposed model setting, not stored pixel data. Change
it if using weights with different preprocessing requirements. JPEG saves are
lossy; originals remain available from the pinned upstream dataset.

**Merge by ingredient meaning, not integer IDs.** These IDs are local to this
source and may collide with other datasets. Broad `ingredient` groups are for
classification, not exact recipe substitution: `potato` includes sweet potato;
`melon`, `juice` and `yogurt` hide varieties/flavors. Keep `fine_name` and ask the
user to resolve ambiguity. `milk`, `oat milk` and `soy milk` remain separate.

These are classification images without bounding boxes, not a full-fridge
multi-object dataset. No recipe corpus, trained model or model accuracy is claimed.

## Source and attribution

Marcus Klasson, Cheng Zhang and Hedvig Kjellstrom, *A Hierarchical Grocery Store
Image Dataset with Visual and Semantic Labels*, WACV 2019.
https://github.com/marcusklasson/GroceryStoreDataset

Source license: MIT, copyright (c) 2019 Marcus Klasson (see SOURCE_LICENSE).
