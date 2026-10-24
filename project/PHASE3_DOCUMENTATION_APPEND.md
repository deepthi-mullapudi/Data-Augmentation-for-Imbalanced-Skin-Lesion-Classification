# Phase 3 Documentation — Classifier Pipeline and E0 Baseline

**Date:** 23 October 2026  
**Status:** Implementation supplied; user's local smoke test and full E0 runs pending.

## Goal

Create a fixed classifier pipeline that can be reused across E0, E2, E3, and E4. The classifier architecture, optimizer, learning rate, batch size, epoch limit, validation-based early stopping, and evaluation procedure must remain consistent across experiments. Only the training data/augmentation treatment should vary as defined by the experiment configuration.

## Files added or updated

### `src/data.py`

Preserves the fixed class mapping and lesion-level split-generation utility and adds `HAMClassifierDataset`.

The dataset:
- Loads `cache/ham64.npz` with `allow_pickle=False`.
- Reads `cache/splits.csv`.
- Checks row count and row-by-row alignment of image IDs, labels, and split assignments.
- Selects only the requested `train`, `val`, or `test` split.
- Converts each RGB array into a PIL image, resizes it to 224 × 224, converts it to a tensor, and applies ImageNet normalization.
- Supports optional classical transforms for training.
- Supports extra `(image, label)` pairs for synthetic training images.
- Rejects augmentation or extra synthetic pairs on validation and test datasets.
- Includes `max_items` for small smoke tests. The subset selection is class-balanced and deterministic so all seven classes can be represented in the small evaluation subset.

**Important limitation for the next phase:** the current `augmentation.mode: classical` implementation applies transformations to all real training samples. The supplied E2 configuration specifies `apply_to: topup_classes` and a top-up policy for rare classes. Before E2 is run, implement class-specific top-up augmentation to match that configuration. E0 is unaffected because its augmentation mode is `none`.

### `src/models.py`

Defines `build_classifier()` using `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` when pretrained weights are requested. The final fully connected layer is replaced with `nn.Linear(512, 7)`. All layers remain trainable for fine-tuning.

### `src/metrics.py`

Defines prediction collection and metric calculation. It calculates:
- Macro-F1 (primary metric)
- Accuracy (reported, not used for selection)
- Balanced accuracy
- Per-class recall for `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, and `vasc`
- Macro one-vs-rest ROC-AUC
- Confusion matrix saved as CSV and PNG

The evaluation function expects all seven classes in the evaluated labels, which the full validation and test splits contain. The smoke-test subsets are selected to retain all classes.

### `src/train_clf.py`

Implements the classifier training/evaluation loop:
1. Load the experiment YAML configuration.
2. Set the random seed for each run.
3. Construct train, validation, and test datasets.
4. Use `shuffle=True` for training and no weighted sampler.
5. Build ResNet-18 and fine-tune all layers.
6. Train with cross-entropy and Adam.
7. Calculate validation macro-F1 after each epoch.
8. Save the best validation-macro-F1 checkpoint and stop early after the configured patience.
9. Load the best checkpoint and evaluate the test split.
10. Save per-run metadata, training logs, test metrics, and confusion matrices.
11. Append one metrics row per seed to `results/metrics.csv`.

The `--smoke` option uses one seed, a limited number of training examples, a small validation/test subset, and an optional epoch override. Smoke artifacts are directed to `results/smoke/` and are not formal research results.

### `src/utils.py`

The training script relies on the existing helper module for seeding, deterministic data-loader generators, device selection, YAML configuration loading, and per-run configuration/metadata snapshots. Preserve existing project-specific modifications when copying files.

## Fixed E0 settings

The source configuration is `configs/e0.yaml`.

- Experiment ID: `E0`
- Training data: real training images only
- Classifier: pretrained ResNet-18, seven output classes
- Fine-tuning: all layers trainable
- Loss: cross-entropy
- Optimizer: Adam
- Learning rate: `1e-4`
- Batch size: `64`
- Maximum epochs: `20`
- Early stopping: validation macro-F1, patience `5`
- Seeds: `0`, `1`, `2`
- Classifier input: `224 × 224`
- Normalization: ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`
- Training sampler: plain shuffle; no weighted sampler

## Expected output structure

For each seed, the training script creates a run directory similar to:

`results/E0/seed_0/`

Files include:
- `config.yaml`: resolved run configuration snapshot
- `run_meta.json`: seed, timestamp, software/device metadata and Git commit when available
- `training_log.csv`: epoch-level training loss and validation metrics
- `best_model.pt`: checkpoint selected by validation macro-F1
- `test_metrics.json`: test metrics for the selected checkpoint
- `test_confusion_matrix.csv`
- `test_confusion_matrix.png`

Aggregate per-seed rows are appended to `results/metrics.csv` with columns:

`experiment, seed, macro_f1, accuracy, bal_acc, auc, recall_akiec, recall_bcc, recall_bkl, recall_df, recall_mel, recall_nv, recall_vasc`

The script reports the mean and standard deviation across available recorded seeds for macro-F1, accuracy, balanced accuracy, and AUC. Gate 3 is not passed until three full E0 runs have completed and the per-seed metrics and standard deviation are recorded.

## Commands to run in the project root

### Syntax check

```powershell
python -m py_compile src/data.py src/models.py src/metrics.py src/train_clf.py src/utils.py
```

### Import check

```powershell
python -c "from src.data import HAMClassifierDataset; from src.models import build_classifier; from src.metrics import calculate_metrics; from src.train_clf import run_experiment; print('Phase 3 imports OK')"
```

### Device check

```powershell
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### One-epoch smoke test

```powershell
python src/train_clf.py --config configs/e0.yaml --smoke --max-train-items 256 --epochs 1
```

Inspect the generated smoke metrics, training log, and confusion matrix before launching formal training. The smoke run tests software flow only; its scores must not be reported as experimental results.

### Full E0 baseline

Only after the smoke test succeeds:

```powershell
python src/train_clf.py --config configs/e0.yaml
```

This runs seeds `0`, `1`, and `2` as configured. Training may take substantial time on CPU. Ensure the pretrained ResNet-18 weights are available and that the installed PyTorch/torchvision versions are compatible.

## Interpretation and safeguards

- Macro-F1 is the primary score because it weights all seven classes equally.
- Accuracy is reported but not used to select the checkpoint.
- Balanced accuracy is the mean of the per-class recalls.
- The workflow's approximate accuracy range of 75–80% is a heuristic, not a guaranteed result or a hard pass/fail threshold.
- Macro-F1 being close to accuracy is not by itself proof of a bug. Investigate class-wise recall, label mapping, split alignment, and confusion matrices.
- The validation set is used for early stopping and checkpoint selection.
- The test set is used only after checkpoint selection; it must not influence training or model selection.
- E0 should be completed before starting the GAN comparison runs.

## Verification performed during preparation

- Python syntax compilation passed for `data.py`, `models.py`, `metrics.py`, and `train_clf.py`.
- Module imports passed after including the helper `utils.py`.
- A synthetic fixture test verified classifier dataset output shape, optional extra-pair handling, rejection of synthetic data for the test split, and metric/confusion-matrix calculation.
- A ResNet-18 forward pass without pretrained weights returned a tensor of shape `(2, 7)`.

These checks were performed in the preparation environment. They do not establish that the user's local project imports, pretrained-weight download, real HAM10000 cache, smoke training, or full E0 runs succeed. Those remain pending until executed and recorded in the user's project environment.

## Change log

| Date | Update |
|---|---|
| 23 October 2026 | Added Phase 3 classifier dataset, ResNet-18 model, metrics, training loop, smoke-test commands, output inventory, safeguards, and verification status. |
