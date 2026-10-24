# Phase 4 implementation handoff

Date: 2026-10-23

## Files prepared
- `src/models.py`: preserves `build_classifier()` and adds `ConditionalGenerator`, `ConditionalDiscriminator`, and `ConditionalCritic`.
- `src/train_gan.py`: adds training-split-only cache loading, class-balanced sampling, conditional DCGAN training, conditional WGAN-GP training, optional differentiable augmentation, fixed-noise sample grids, CSV logs, and periodic/latest checkpoints.

## Important safeguards
- Uses `cache/ham64.npz` and validates its row alignment against `cache/splits.csv`.
- Selects only rows marked `train`; validation and test images are not loaded into the GAN dataset.
- GAN input normalization is `uint8 [0,255]` to `float32 [-1,1]`, matching generator `Tanh`.
- Balanced sampling is confined to the GAN DataLoader; classifier code is unchanged.
- `dcgan_best.pt` / `wgangp_best.pt` are not fabricated. Checkpoints are named by epoch plus `*_latest.pt`; choose a best candidate after visual inspection and Phase 6 FID.
- WGAN-GP logs raw critic scores, not probabilities. DCGAN logs sigmoid probabilities for D(real)/D(fake).
- DiffAugment is optional and follows the YAML `enabled` flag.

## Not yet verified on the user's machine
The files must be copied into `E:\udemy\ADL\project\src\`, then tested in the user's active environment. Do not claim a training run succeeded until the commands and outputs are checked.

## Recommended verification order
1. Back up current `src/models.py` and `src/train_gan.py`.
2. Copy the supplied files into `src/`.
3. Compile: `python -m py_compile src\models.py src\train_gan.py`
4. Check CUDA and imports.
5. Run one small DCGAN smoke test.
6. Inspect `results/GAN_DCGAN/seed_0/smoke/samples/epoch_001.png`, `training_log.csv`, and the checkpoint.
7. If that passes, time one full-training-split epoch with a conservative batch size before increasing batch size or epochs.


## Verification performed in the packaging environment
- Python syntax compilation passed for `src/models.py` and `src/train_gan.py`.
- Model forward-pass test passed with small test channel counts: generator output `(2, 3, 64, 64)` within `[-1,1]`; discriminator and critic each returned one score per image.
- End-to-end one-epoch DCGAN test passed using a temporary synthetic 14-image cache and temporary manifest. It produced a training log, summary JSON, image grid, and epoch checkpoint.
- End-to-end one-epoch WGAN-GP test passed using the same temporary synthetic cache. It exercised gradient penalty and produced the expected artifacts.
- These are code-path tests on synthetic data, **not** a training run on the user's HAM10000 cache and not evidence that generated images are medically or visually realistic.
