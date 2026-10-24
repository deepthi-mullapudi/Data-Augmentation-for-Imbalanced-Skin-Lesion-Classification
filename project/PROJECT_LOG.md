# Conditional GAN Augmentation for Imbalanced Skin Lesion Classification

This file is the running project log. New implementation steps, decisions, assumptions, and validation results are appended as the project evolves.

## 2026-10-22 - Initial project scaffold

### Requested scope

Create the initial project structure for:

- HAM10000 data loading and preprocessing
- DCGAN and WGAN-GP training
- Skin-lesion classifier training and evaluation
- Class-imbalance and image-quality metrics
- Experiment configuration, checkpoints, results, and notebooks

### Decisions

1. The requested `project/` directory is created directly under the repository root (`ADL\project\`).
2. Source modules are created as lightweight, importable placeholders so later implementation can be added without changing the requested layout.
3. Raw data, generated cache files, checkpoints, and result artifacts are treated as local/generated assets. Their contents are ignored by Git while their directories are retained with `.gitkeep` files.
4. `configs/e0.yaml` and `configs/e2.yaml` are created as initial experiment configuration placeholders. Their experiment semantics will be defined when the training pipeline is implemented.
5. No HAM10000 images, metadata, model weights, or generated samples are downloaded or committed in this scaffold step.

### Initial validation

- Confirmed the repository had no pre-existing project files.
- Created the requested directories and module/config placeholders.
- Added Git ignore rules for non-committed datasets and generated artifacts.
- Verified all requested directories exist under `project/`.
- Verified the scaffold is visible to Git as a new `project/` tree; generated-artifact patterns are configured in `project/.gitignore`.

### Phase 4 — WGAN-GP smoke test review

**Date:** 2026-10-23  
**Status:** Smoke-test execution and artifact generation succeeded; generated image quality remains inadequate.

**Command executed**
```powershell
python -m src.train_gan --config configs/gan_wgangp.yaml --smoke
```

**Observed execution**
- Device: CUDA
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- GAN type: WGAN-GP
- Smoke subset: 70 images, 10 per class
- Batch size: 8
- Epochs completed: 1
- Images processed: 64
- Epoch duration: 0.771 seconds

**Final epoch metrics**
- Generator loss: 1.408863
- Critic loss: 46.807292
- Mean real critic score: 4.474262
- Mean fake critic score: -0.491138

**Observed artifacts**
- `results/GAN_WGANGP/seed_0/smoke/training_log.csv`
- `results/GAN_WGANGP/seed_0/smoke/training_summary.json`
- `results/GAN_WGANGP/seed_0/smoke/samples/epoch_001.png`
- Checkpoints written under `checkpoints/GAN_WGANGP/seed_0/smoke/`

**Visual inspection:** The generated grid appears predominantly gray and noisy, without clearly recognizable lesion structures.

**Interpretation:** The one-epoch smoke test verifies that the WGAN-GP training path executes and records outputs on CUDA. It does not establish convergence or useful image generation. Critic scores are raw scores, not probabilities; critic loss includes the gradient-penalty contribution and should not be compared numerically with DCGAN's BCE loss.

**Combined DCGAN/WGAN-GP assessment:** Both smoke tests completed, but neither produced visually convincing samples after one epoch. Do not select a winning GAN or claim successful synthetic image generation based on these tests.

**Next steps**
1. Verify the active PyTorch/CUDA environment.
2. Inspect sample-grid pixel conversion and normalization.
3. Run a controlled trial on the full training split, keeping validation and test images out of GAN training.
4. Review intermediate sample grids and losses before extending training.
5. Evaluate checkpoint quality later using visual review and the planned Phase 6 metrics.

### Phase 4 — DCGAN controlled training trial preparation

**Date:** 2026-10-23

**Status:** Configuration preparation; trial not yet executed.

**Configuration:** `configs/gan_dcgan_trial.yaml`, copied from `configs/gan_dcgan.yaml`.

**Planned changes**
- Experiment ID: `GAN_DCGAN_TRIAL`
- Epochs: 10
- Sample grid interval: every epoch
- Checkpoint interval: every 5 epochs
- Separate checkpoint root: `checkpoints/trials`
- Separate results root: `results/trials`

**Unchanged settings:** Batch size 64, learning rates 2e-4, Adam betas (0.5, 0.999), conditional DCGAN architecture, and balanced GAN sampler.

**Purpose:** Assess whether generated sample quality and training behavior improve beyond the one-epoch smoke test.

**Data boundary:** GAN training must use only the fixed training split. Validation and test images must remain excluded.

**Pending verification:** Confirm the training script honors the configured output paths and that the trial processes the full training split rather than the smoke-test subset.

**Success criteria:** Trial completes, expected logs/checkpoints/grids are generated, and image quality and training behavior are reviewed. Recognizable lesion generation is not assumed in advance.


new entry will record:
- Experiment: GAN_DCGAN_TRIAL, seed 0
- Configuration: configs/gan_dcgan_trial.yaml
- Command: python -m src.train_gan --config configs/gan_dcgan_trial.yaml
- Data: 7,009 training images, with the seven verified class counts
- Training: 10 epochs, batch size 64, CUDA, RTX 3050 Laptop GPU
- Artifacts: training_log.csv, training_summary.json, ten sample grids, epoch 5 and epoch 10 checkpoints, and dcgan_latest.pt
- Observed outcome: generator loss decreased, discriminator outputs changed, and generated images developed more variation but remain blurry and require quality evaluation
- Limitations: no automatically selected best checkpoint; image quality and class conditioning are not yet validated


# Phase 5: Conditional WGAN-GP Implementation and Training

## 1. Objective

The objective of Phase 5 was to implement and validate a conditional Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) for generating synthetic skin-lesion images from the HAM10000 dataset.

WGAN-GP is the second generative model in the project, following the conditional DCGAN implemented in Phase 4. Both models are intended to generate class-conditioned synthetic images that can eventually be evaluated for their usefulness in addressing class imbalance in skin-lesion classification.

The main objectives of this phase were:

- Implement the conditional WGAN-GP architecture.
- Incorporate gradient penalty into critic training.
- Use balanced class sampling during GAN training.
- Implement training logs, checkpoint saving, and generated-image grids.
- Validate the implementation through a smoke test and a short full-data pilot.
- Assess preliminary training behavior before proceeding to the full training budget.

## 2. Dataset and Experimental Setup

The same preprocessed HAM10000 dataset and fixed training split used in the preceding phases were retained.

The dataset contains 10,015 dermoscopic images distributed across seven diagnostic classes.

| Class | Training images |
|---|---:|
| akiec | 228 |
| bcc | 359 |
| bkl | 769 |
| df | 80 |
| mel | 779 |
| nv | 4,694 |
| vasc | 100 |
| **Total** | **7,009** |

The original training split is highly imbalanced, with the `nv` class accounting for the majority of training images.

A balanced sampler was enabled for GAN training to expose the generator and critic to a more balanced distribution of class labels. This sampling strategy is restricted to GAN training and does not change the sampling strategy used for the baseline classifier.

The fixed training, validation, and test splits were preserved.

## 3. Implementation

### 3.1 Conditional generator

The conditional generator is implemented in `src/models.py` as `ConditionalGenerator`.

It generates 64 × 64 RGB images conditioned on the requested skin-lesion class and a latent noise vector.

The generator produces normalized image tensors in the range [-1, 1], which are converted into the appropriate display range when generated-image grids are saved.

### 3.2 Conditional critic

The conditional critic is implemented as `ConditionalCritic` in `src/models.py`.

Unlike the DCGAN discriminator, the WGAN-GP critic does not use a sigmoid output. It produces scalar scores rather than probabilities.

The WGAN-GP critic uses GroupNorm with one group per normalization layer in place of BatchNorm, consistent with the intended architecture.

### 3.3 Wasserstein critic objective

The critic objective is based on the Wasserstein formulation with a gradient penalty:

\[
L_D =
\mathbb{E}[D(\tilde{x},y)]
-\mathbb{E}[D(x,y)]
+\lambda_{\mathrm{GP}}L_{\mathrm{GP}}
\]

Here, \(x\) represents real images, \(\tilde{x}\) represents generated images, \(y\) represents the conditioning label, and \(\lambda_{\mathrm{GP}}\) is the gradient-penalty coefficient.

The gradient penalty is calculated on interpolations between real and generated images. Autograd is used to compute the gradients with respect to the interpolated images, with higher-order gradients enabled for critic optimization.

The penalty encourages the gradient norm to remain close to one.

### 3.4 Generator objective

The generator is trained to increase the critic's scores for generated images:

\[
L_G=-\mathbb{E}[D(G(z,y),y)]
\]

The generator receives random latent vectors and class labels and learns to generate images that receive higher critic scores.

### 3.5 Training configuration

The implementation uses the following intended WGAN-GP settings:

| Parameter | Value |
|---|---:|
| Image resolution | 64 × 64 |
| Number of channels | 3 |
| Batch size for full-data training | 64 |
| Generator learning rate | 0.0001 |
| Critic learning rate | 0.0001 |
| Adam beta parameters | (0.0, 0.9) |
| Gradient-penalty coefficient | 10 |
| Critic updates per generator update | 5 |
| Class sampling | Balanced |
| Training device | NVIDIA GeForce RTX 3050 Laptop GPU |
| GAN training split | Training split only |

These values describe the intended configuration and should be verified against `configs/gan_wgangp.yaml` when recording the final experiment settings.

## 4. Code Modifications and Logging

The primary implementation files are:

- `src/models.py`
- `src/train_gan.py`
- `configs/gan_wgangp.yaml`

The shared GAN training script supports both DCGAN and WGAN-GP training.

The following logging improvements were made during Phase 5.

### 4.1 Gradient-penalty diagnostics

The `gradient_penalty()` function was modified to return both the gradient-penalty value and the mean gradient norm.

This allows the training process to record how closely the critic's gradient norms approach the target value of one.

### 4.2 Additional training metrics

The training log was extended with two additional fields:

- `wasserstein_estimate`
- `gradient_norm_mean`

The Wasserstein estimate is calculated as the mean real-image critic score minus the mean generated-image critic score.

The gradient-norm field records the mean gradient norm measured during the gradient-penalty calculations.

These metrics supplement the existing generator loss, critic loss, real and generated critic scores, epoch duration, and cumulative image count.

For DCGAN runs, the WGAN-GP-specific fields are not applicable and are recorded as NaN.

### 4.3 Checkpoint and sample-grid saving

The training script saves periodic model checkpoints, a latest checkpoint, training logs, a training summary, and generated-image grids.

Checkpoint selection is not automatic. Saved grids and quantitative evaluations must be used to decide which checkpoint is appropriate for subsequent evaluation.

## 5. Smoke Test

The first validation step was a smoke test to confirm that the WGAN-GP training pipeline could execute on the available GPU.

Command:

```powershell
python -m src.train_gan --config configs/gan_wgangp.yaml --smoke
```

The smoke test completed one epoch using a small balanced subset containing 70 images, with 10 images per class.

The batch size was reduced to eight for the smoke test. Because incomplete batches were dropped, 64 images were processed.

### Smoke-test results

| Metric | Result |
|---|---:|
| Epochs completed | 1 |
| Training subset | 70 images |
| Images processed | 64 |
| Batch size | 8 |
| Critic loss | 46.6359 |
| Generator loss | 1.6775 |
| Mean real critic score | 4.4989 |
| Mean generated critic score | -0.5816 |
| Wasserstein estimate | 5.0805 |
| Mean gradient norm | 2.7389 |
| Epoch duration | Approximately 1.1–2.4 seconds, depending on the recorded run |

The initial smoke-test sample grids were mostly noise-like, which is expected from a single epoch on a very small subset.

The smoke test confirmed that the pipeline executed, generated samples, saved checkpoints, and produced the additional gradient-penalty and Wasserstein diagnostics.

The gradient norm was above the target of one, but the result alone was insufficient to establish an implementation problem.

## 6. Full-Data Pilot: 10 Epochs

After validating the training and logging pipeline, a short pilot was performed using the complete training split.

Command:

```powershell
python -m src.train_gan --config configs/gan_wgangp.yaml --epochs 10
```

The pilot used a batch size of 64 and balanced class sampling.

The run completed all 10 epochs without a reported training error.

### 6.1 Pilot results

| Metric | Epoch 1 | Epoch 5 | Epoch 10 |
|---|---:|---:|---:|
| Critic loss | -21.6516 | -9.8474 | -5.8933 |
| Generator loss | 20.2025 | 41.3026 | 33.3222 |
| Wasserstein estimate | 35.0418 | 12.4640 | 7.1339 |
| Mean gradient norm | 1.8825 | 1.3508 | 1.2312 |

The pilot processed 69,760 images across training batches over the 10 epochs.

### 6.2 Interpretation

The mean gradient norm decreased from approximately 1.88 to 1.23, moving closer to the target of one.

The Wasserstein estimate also decreased substantially. The generator and critic losses fluctuated during training, which is not unusual for adversarial optimization. These losses cannot independently establish image quality or convergence.

The generator's output tensor standard deviation increased from approximately 0.150 at epoch 1 to 0.343 at epoch 10. This indicates greater pixel-level variation in the generated samples, but it is not a standalone measure of image quality.

### 6.3 Visual inspection

The generated grids were inspected at epochs 1, 5, and 10.

- **Epoch 1:** Outputs were dominated by colored noise, with little recognizable structure.
- **Epoch 5:** Blurry pink and red regions began to emerge, with some darker central areas.
- **Epoch 10:** Samples exhibited greater color and spatial variation, but remained blurry and noisy, with limited clearly recognizable lesion morphology.

The visual results suggest preliminary progress beyond the initial noise-like outputs. However, the samples do not yet provide sufficient evidence of high-quality lesion synthesis, successful class conditioning, or adequate diversity.

These findings are preliminary and based on visual inspection rather than formal image-quality evaluation.

## 7. Output Artifacts

The pilot generated training logs, a summary, sample grids, and checkpoints.

### Pilot training outputs

```text
results/
└── GAN_WGANGP/
    └── seed_0/
        ├── training_log.csv
        ├── training_summary.json
        └── samples/
            ├── epoch_001.png
            ├── ...
            └── epoch_010.png
```

### Pilot checkpoints

```text
checkpoints/
└── GAN_WGANGP/
    └── seed_0/
        ├── wgangp_epoch_005.pt
        ├── wgangp_epoch_010.pt
        └── wgangp_latest.pt
```

The pilot artifacts were moved into a trial folder to preserve them before starting the longer run. The exact destination should be recorded from the current project directory structure.

The original smoke-test artifacts were saved under smoke-specific subdirectories.

## 8. Current Status

The following Phase 5 tasks have been completed:

- Conditional WGAN-GP model implementation.
- Critic without sigmoid output.
- Gradient-penalty implementation.
- Gradient-norm logging.
- Wasserstein-estimate logging.
- GPU smoke test.
- Full-training-split 10-epoch pilot.
- Preliminary inspection of generated-image grids.
- Preservation of pilot outputs before the longer run.

The 10-epoch pilot established that the pipeline executes successfully and records the intended diagnostics. It did not establish that the generator has converged or that its images are suitable for classifier augmentation.

## 9. Next Steps

The next planned step is the 200-epoch WGAN-GP training run using the original configuration:

```powershell
python -m src.train_gan --config configs/gan_wgangp.yaml --epochs 200
```

Before starting, the output directories should be checked to ensure the pilot logs and checkpoints have been moved or backed up. The command may start training from scratch rather than resume from the pilot checkpoint.

After the longer run:

1. Inspect sample grids from multiple checkpoints.
2. Assess visual quality, class conditioning, and diversity.
3. Evaluate image quality quantitatively using FID and appropriate complementary metrics.
4. Check generated images for potential memorization of training samples.
5. Select checkpoints based on documented evaluation criteria rather than epoch number or loss alone.
6. Compare DCGAN and WGAN-GP under a consistent experimental protocol.
7. Proceed to downstream classifier augmentation experiments only after evaluating the synthetic data.

The full 200-epoch run is complete. Phase 6 evaluation and best-checkpoint selection remain pending until formal quality and memorisation checks are completed.

## 10. Conclusion

Phase 5 progressed from conditional WGAN-GP implementation to functional smoke testing, a 10-epoch pilot, and a completed 200-epoch full-data run. The added diagnostics show improving gradient-norm behavior and recorded Wasserstein estimates throughout training.

However, the generated samples remain blurry and do not yet establish satisfactory lesion realism or diversity. The next stage is to complete the longer training run and evaluate the resulting checkpoints systematically before using synthetic images in downstream classification experiments.
## Phase 5: Conditional WGAN-GP

**Status:** 200-epoch training completed; checkpoint evaluation and final Gate 5 sign-off pending.

### 5.1 Objective

Phase 5 implemented a conditional Wasserstein GAN with Gradient Penalty (WGAN-GP) as an alternative to the conditional DCGAN developed in Phase 4. The objective was to train a class-conditional generator on the same HAM10000 training split and compare the two GAN approaches for minority-class skin-lesion augmentation.

### 5.2 Model architecture and implementation

The WGAN-GP implementation reuses the conditional generator architecture from the DCGAN implementation to keep the generator architecture consistent across the comparison.

The critic uses a convolutional architecture with conditional label information, an unbounded scalar output, no sigmoid activation, and GroupNorm with one group instead of BatchNorm.

The training implementation includes:

- Wasserstein critic loss:
  `D(fake | y) - D(real | y) + lambda_gp * GP`
- Gradient penalty calculated from interpolated real and generated samples using `torch.autograd.grad` with `create_graph=True`.
- Generator loss:
  `-D(G(z, y) | y).mean()`
- Class-balanced sampling for GAN training.
- Periodic checkpoint saving and fixed-noise sample-grid generation.
- CSV logging of critic loss, generator loss, real and generated critic scores, Wasserstein estimate, mean gradient norm, epoch duration, and images seen.

The gradient penalty coefficient is configured as 10 in the intended training recipe. The final YAML configuration should be retained with the experiment artifacts as the authoritative record of all hyperparameters.

### 5.3 Smoke test and pilot run

A one-epoch smoke test was run to verify that the WGAN-GP pipeline executed, generated sample grids, saved checkpoints, and recorded the additional diagnostic metrics.

The smoke test used a small balanced subset of 70 training images and a batch size of 8. It completed successfully. The mean gradient norm was approximately 2.7389, and the Wasserstein estimate was approximately 5.0805. The generated samples were mostly blurry and noisy, which was expected from this short test and small subset.

A subsequent 10-epoch full-training-split pilot was run with a batch size of 64. It completed successfully and confirmed that training, logging, sampling, and checkpoint saving worked on the complete training split. The pilot outputs were moved into the trial-results location to preserve them separately from the final run.

### 5.4 Full 200-epoch training

The full WGAN-GP run was executed using:

`python -m src.train_gan --config configs/gan_wgangp.yaml --epochs 200`

The run completed all 200 epochs on the NVIDIA GeForce RTX 3050 Laptop GPU using CUDA.

| Item | Result |
|---|---|
| GAN type | Conditional WGAN-GP |
| Dataset | HAM10000 |
| Training split | 7,009 images |
| Number of classes | 7 |
| Input resolution | 64 × 64 RGB |
| Batch size | 64 |
| Epochs completed | 200 |
| Images processed | 1,395,200 |
| Checkpoint interval | Every 5 epochs |
| Final epoch duration | Approximately 11.54 seconds |

The number of images processed is based on 6,976 images per epoch after `drop_last=True`, multiplied by 200 epochs. This counts repeated presentations of training images across epochs, not unique images.

### 5.5 Final-epoch results

The final epoch recorded the following metrics:

| Metric | Epoch 200 |
|---|---:|
| Critic loss | -5.8038 |
| Generator loss | -23.5501 |
| Mean real critic score | 31.1110 |
| Mean generated critic score | 24.2876 |
| Wasserstein estimate | 6.8234 |
| Mean gradient norm | 1.2114 |
| Epoch duration | Approximately 11.54 seconds |

The mean gradient norm was reasonably close to the target of 1, providing a useful gradient-penalty diagnostic. However, this alone does not prove convergence or image quality. Critic and generator losses are specific to the WGAN-GP objective and should not be compared directly with DCGAN losses.

The Wasserstein estimate and gradient-norm logs were recorded throughout training. Their trends should be examined alongside image quality and diversity during checkpoint evaluation.

### 5.6 Generated samples and visual observations

Fixed-noise sample grids were generated during training. Sample grids from epochs 50, 100, 150, and 200 were visually compared.

The qualitative observations were:

- Epochs 50 and 100 showed relatively diffuse and blurry lesion-like patterns.
- Epochs 150 and 200 showed more visible internal structure, pigmentation, and color variation in many samples.
- Epoch 150 and epoch 200 both appear promising based on visual inspection; neither has been established as the best checkpoint.
- Some samples remain blurry or noisy, and visual inspection alone cannot establish sufficient diversity, medical realism, or absence of memorization.

These observations are preliminary qualitative findings. They do not establish diagnostic validity or prove that synthetic augmentation improves classification.

### 5.7 Saved artifacts

The full-run training logs and sample grids are stored under:

`results/GAN_WGANGP/seed_0/`

Important files include:

- `training_log.csv`
- `training_summary.json`
- `samples/` — saved generated sample grids
- `config.yaml` and `run_meta.json`, where present

Model checkpoints are stored under:

`checkpoints/GAN_WGANGP/seed_0/`

The checkpoint directory contains periodic checkpoints from epoch 5 through epoch 200, at five-epoch intervals, and `wgangp_latest.pt`. The epoch 150 and epoch 200 checkpoints have been verified to exist.

No automatically selected best checkpoint has been established at this stage.

### 5.8 Current conclusion and remaining work

Phase 5 implementation and full training are complete. The smoke test and 10-epoch pilot validated the execution pipeline, while the 200-epoch run produced increasingly structured lesion-like samples.

The following items remain before final Gate 5 sign-off:

1. Verify and document the final training hyperparameters from `configs/gan_wgangp.yaml`.
2. Compare DCGAN and WGAN-GP training budgets using total images processed and elapsed time, and state the comparison basis.
3. Evaluate candidate checkpoints using a consistent image-quality evaluation protocol.
4. Assess sample diversity and potential memorization.
5. Compare class-conditioned samples across all seven classes.
6. Select and document a best WGAN-GP checkpoint using evaluation evidence.
7. Record the Gate 5 decision based on the workflow's recognizability and checkpoint requirements.

The next stage is Phase 6: synthetic image generation, FID evaluation, visual comparisons, and memorization checks. The selected GAN checkpoints will then be used for downstream augmentation experiments to determine whether synthetic data improves minority-class classification.