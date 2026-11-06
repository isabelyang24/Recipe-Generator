# AI Recipe Generator: proposed project pipeline

Prepared by Vaidik Nadheria for ACM AI Team 5, Fall 2026.
This is a proposal for team discussion, not a claim that a model has been trained.

## Goal and first version

Accept an ingredient list or photos of individual grocery items, confirm the
inventory with the user, and show recipes ranked by available ingredients.
Train and evaluate an ingredient classifier as the core ML task. Start with
single-item photos because this dataset has one class per image and no bounding
boxes. Recognizing every object in a full fridge requires separate detection or
multi-label data and remains an extension.

## Data and preprocessing

This contribution acquires a pinned revision of GroceryStoreDataset, validates
all annotated image paths, labels and decoded pixels, and exports RGB 224x224
images with aspect-ratio-preserving padding. The source annotations contain
5,421 images, 81 detailed classes and 43 coarse classes. Their README's 5,125/42
figures differ; use the checked-in manifests and measured report.

Source train/validation/test splits are retained. Exact pixel duplicates are
checked across all splits; conflicting labels would be quarantined, and same-label
copies would retain the test/validation copy first. A conservative 64-bit dHash
scan flags cross-split pairs within distance 2 for review without deleting them.
This does not establish that all near-duplicates or related scenes are absent.

The six coarse classes missing from source validation are listed in summary.json.
Before tuning a 43-class model, the team should agree on an alternative development
split built only from the original training set, grouping duplicate/related images.
Keep the original test set untouched. Do not silently change the published split.

## Methodology

1. **Inventory input:** Text input bypasses vision; photo input goes through the
   classifier. Let users correct uncertain predictions and add missing ingredients.
2. **Vision baseline:** A small CNN provides a learning baseline. Compare it with
   pretrained ResNet18: train a new 43-class head with the feature extractor frozen,
   then optionally fine-tune later layers. Record seed, hyperparameters and source
   commit. Apply ImageNet normalization at load time for that pretrained model.
   Use augmentation on training images only; keep evaluation deterministic.
3. **Label integration:** Merge datasets by reviewed ingredient names and build
   a single global label-to-ID mapping. IDs in this contribution are source-local.
   Retain detailed labels for distinctions hidden by broad groups: sweet potato,
   melon varieties, juice flavors and flavored yogurts need further user input or
   a finer classifier. Never automatically substitute oat/soy milk for dairy milk.
4. **Recipe collection:** Obtain a recipe corpus separately, e.g. RecipeNLG under
   its educational-use terms. Retain ingredients, quantities, directions and URLs;
   remove incomplete records and duplicates. Recipe acquisition and cleaning are
   future team tasks, not included in this image-dataset deliverable.
5. **Matching baseline:** Normalize ingredient names conservatively; rank by the
   fraction of required ingredients present, then by fewer missing ingredients.
   Pantry staples count only when the user confirms them. Mark required quantities
   as unverified if inventory amounts are unknown. Apply dietary filters before
   ranking. Avoid equating tomato with tomato sauce or milk with coconut milk.
6. **Demo:** Streamlit upload/text input, editable inventory, and five recipe cards
   with source links, ingredient coverage and missing items. Recipe retrieval is
   sufficient for the first version. Generative rewriting is an optional later
   experiment and must preserve the source recipe's ingredients and instructions.

## Evaluation and completion criteria

- Vision: held-out top-1 accuracy, macro F1, per-class recall and a confusion matrix;
  separately evaluate a small team-collected set of real ingredient photos.
- Ranking: create at least 30 representative inventories with team-reviewed recipe
  relevance labels; report precision@5 and required ingredients missing from results.
  Compare the baseline with a later semantic-ranking approach on the same examples.
- Integration: unknown inputs, corrected labels, empty inventories, dietary filters,
  and no-match cases produce understandable results.
- Report measured results and failure examples; do not claim that recipe matching
  demonstrates reduced food waste without a separate user study.

## Suggested ownership and milestones

| Workstream | Deliverable |
| --- | --- |
| Dataset preparation | Cleaned images, manifests, audit and integration mappings |
| Vision baseline | Reproducible CNN training and validation |
| Transfer learning/evaluation | ResNet comparison, metrics and error review |
| Recipe/backend | Clean recipe index, inventory normalization, ranking |
| Interface/integration | User correction flow, recipe cards and showcase demo |

Next: agree on the vocabulary, development split and recipe source. Then establish
baselines, compare models, integrate the UI and leave the final week for error
analysis and showcase preparation. Vaidik's backend interest fits the recipe/index
workstream after this dataset contribution; assignments remain a team decision.

## References

- Dataset: https://github.com/marcusklasson/GroceryStoreDataset
- Dataset paper: https://arxiv.org/abs/1901.00711
- PyTorch transfer learning: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- RecipeNLG: https://recipenlg.cs.put.poznan.pl/dataset
