# Phase 3 — Classifier pipeline and E0 baseline

This bundle contains the Phase 3 classifier pipeline files. Copy the Python files from `src/` into the existing `project/src/` directory. Keep the existing `src/utils.py` unless you have verified that it matches the supplied helper file; the training script depends on its functions.

## Files and responsibilities

- `src/data.py`: retains the existing fixed label mapping and split-generation function, and adds `HAMClassifierDataset`. The dataset verifies cache/CSV alignment, returns normalized image tensors and integer labels, supports train-only classical augmentation, and can append optional `(image, label)` pairs for later synthetic augmentation experiments.
- `src/models.py`: constructs pretrained torchvision ResNet-18 and replaces the final fully connected layer with a seven-class output layer. All layers remain trainable.
- `src/metrics.py`: collects logits and labels; calculates macro-F1, accuracy, balanced accuracy, per-class recall and macro one-vs-rest AUC; writes confusion matrices to CSV and PNG.
- `src/train_clf.py`: loads a YAML experiment config, trains with cross-entropy and Adam, uses plain shuffled training batches, selects the best checkpoint using validation macro-F1, early-stops, evaluates the selected checkpoint on test data, and appends per-seed metrics.
- `src/utils.py`: common seeding, config, device, and run-metadata helpers. The file is included for reference and standalone import checks; preserve your project version if it has additional local changes.
- `PHASE3_DOCUMENTATION_APPEND.md`: technical documentation to append to the existing `Document.md` or `PROJECT_LOG.md`.

## Copy instructions (PowerShell)

Run from `E:\udemy\ADL\project` after downloading and extracting this bundle somewhere accessible. Copy only the listed source files into the existing project `src` directory. Do not copy the `phase3_bundle` directory itself as a nested source directory.

For example, if you extracted the bundle to `E:\Downloads\phase3_bundle`:

```powershell
Copy-Item E:\Downloads\phase3_bundle\src\data.py src\data.py -Force
Copy-Item E:\Downloads\phase3_bundle\src\models.py src\models.py -Force
Copy-Item E:\Downloads\phase3_bundle\src\metrics.py src\metrics.py -Force
Copy-Item E:\Downloads\phase3_bundle\src\train_clf.py src\train_clf.py -Force
```

Keep your current `src\utils.py` unless you explicitly want to replace it after comparing versions. The provided `train_clf.py` expects these functions in `src.utils`: `get_device`, `load_config`, `make_generator`, `save_run_config`, `seed_worker`, and `set_seed`.

## Commands — run in order

### 1. Syntax check

```powershell
python -m py_compile src/data.py src/models.py src/metrics.py src/train_clf.py src/utils.py
```

### 2. Import check

```powershell
python -c "from src.data import HAMClassifierDataset; from src.models import build_classifier; from src.metrics import calculate_metrics; from src.train_clf import run_experiment; print('Phase 3 imports OK')"
```

### 3. Confirm device and versions

```powershell
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is unavailable, the pipeline can run on CPU, but full ResNet-18 runs may be slow. Resolve any torch/torchvision compatibility problems before long runs.

### 4. One-epoch classifier smoke test

```powershell
python src/train_clf.py --config configs/e0.yaml --smoke --max-train-items 256 --epochs 1
```

This uses one seed and small class-balanced subsets for a pipeline check. It is not a research result. Smoke outputs are redirected under `results/smoke/` so they do not enter the formal `results/metrics.csv`.

### 5. Inspect smoke outputs

Check the following path (the exact seed directory is created by the script):

- `results/smoke/E0/seed_0/training_log.csv`
- `results/smoke/E0/seed_0/best_model.pt`
- `results/smoke/E0/seed_0/test_metrics.json`
- `results/smoke/E0/seed_0/test_confusion_matrix.csv`
- `results/smoke/E0/seed_0/test_confusion_matrix.png`
- `results/smoke/metrics.csv`

The run directory layout may be affected by your config's `experiment.id`; use the printed output if it differs.

### 6. Run the full E0 baseline only after the smoke test passes

```powershell
python src/train_clf.py --config configs/e0.yaml
```

The config specifies seeds `0, 1, 2`, up to 20 epochs, batch size 64, Adam learning rate `1e-4`, cross-entropy, and early stopping on validation macro-F1 with patience 5. The best validation checkpoint is evaluated on test only after selection.

## Important notes

- Do not regenerate `cache/splits.csv` during Phase 3.
- The test split must not be used for training, early stopping, or model selection.
- No weighted sampler is used; training uses `shuffle=True`.
- Macro-F1 is the primary metric. Accuracy is reported but is not the selection metric.
- The expected 75–80% accuracy is a rough sanity heuristic from the workflow, not a guaranteed threshold. Macro-F1 near accuracy alone does not prove a bug; inspect per-class recall, labels, predictions, and the confusion matrix.
- The code's optional synthetic-pair hook expects a `.npz` file with arrays named `images` and `labels`. GAN training and synthetic-image generation are not part of this bundle.
- The E2 config requests augmentation on top-up/rare classes, while this Phase 3 dataset currently applies classical transforms to every real training sample when `augmentation.mode: classical`. Before running E2, extend the dataset/training integration to implement the config's `apply_to: topup_classes` and top-up count policy. E0 is unaffected because `augmentation.mode` is `none`.

## Verification status

The supplied Python files passed syntax compilation. Imports, dataset behavior, extra-pair training-only enforcement, metrics on a synthetic test fixture, and a ResNet-18 seven-output forward pass were checked in the preparation environment. The actual HAM10000 cache, pretrained-weight download, user's local environment, full smoke training, and three-seed E0 training have not yet been verified. Run the commands above in the project environment and record the actual outputs.
