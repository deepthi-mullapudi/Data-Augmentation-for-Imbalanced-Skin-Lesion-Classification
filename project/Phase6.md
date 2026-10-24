# Phase 6 Documentation: Synthetic Image Generation and Evaluation
## Imbalanced Skin Lesion Classification Using Conditional GANs

### 1. Phase 6 Overview

Phase 6 focuses on generating synthetic skin lesion images from the trained conditional Generative Adversarial Networks (GANs) and evaluating the generated images before using them for downstream imbalanced skin lesion classification.

The experiments compare three model configurations:

1. **Original DCGAN** — the baseline conditional Deep Convolutional GAN.
2. **WGAN-GP** — a conditional Wasserstein GAN with Gradient Penalty.
3. **Revised DCGAN with DiffAugment** — an improved version of the original DCGAN trained with a lower learning rate and discriminator-side differentiable augmentation.

The purpose of this phase is to investigate whether the generated images exhibit adequate diversity and visual quality, and to determine which GAN configuration and checkpoint should be considered for subsequent experiments.

The evaluation is performed separately for each GAN configuration and checkpoint to preserve the comparability and traceability of results.

---

## 2. Dataset and Experimental Context

The experiments use the HAM10000 skin lesion dataset.

- Total images: 10,015.
- Number of classes: 7.
- Image resolution used for GAN training and generation: 64 × 64 pixels.
- Image channels: 3 (RGB).
- Training split: 7,009 images.
- Validation split: 1,503 images.
- Test split: 1,503 images.

The seven lesion classes are:

| Class ID | Class label |
|---:|---|
| 0 | akiec |
| 1 | bcc |
| 2 | bkl |
| 3 | df |
| 4 | mel |
| 5 | nv |
| 6 | vasc |

The dataset split was established earlier in the project and is stored in `cache/splits.csv`. The split is fixed and should not be regenerated during the current experiments.

The original training data is imbalanced, with the `nv` class substantially more frequent than several other lesion classes. Conditional GANs are used to generate images for specified classes, allowing the synthetic image collection to contain an equal number of samples per class.

The test split remains reserved for final downstream classifier evaluation. Synthetic image quality assessment must not be confused with final classifier evaluation.

---

# Part A — Phase 6: Synthetic Image Generation

## 3. Purpose of Synthetic Image Generation

The primary objectives of synthetic image generation are:

1. Generate a consistent number of synthetic images for each lesion class.
2. Compare samples generated at different training checkpoints.
3. Examine whether the generated images contain recognizable lesion-like patterns.
4. Investigate intra-class diversity and potential repetition or mode collapse.
5. Quantitatively evaluate the generated images using diversity metrics and FID.
6. Select suitable candidates for subsequent augmentation and classification experiments.

Generating equal numbers of images per class makes the candidate collections balanced by construction. However, equal class counts do not guarantee that the generated images are realistic, diverse, or clinically meaningful.

Therefore, image counts alone are not sufficient to determine which GAN performs better.

## 4. Checkpoints Selected for Evaluation

The evaluation initially considered four checkpoints.

| Candidate | GAN configuration | Epoch |
|---|---|---:|
| `dcgan_epoch_150` | Original DCGAN | 150 |
| `dcgan_epoch_200` | Original DCGAN | 200 |
| `wgangp_epoch_150` | WGAN-GP | 150 |
| `wgangp_epoch_200` | WGAN-GP | 200 |

Epochs 150 and 200 were selected to compare relatively late-stage checkpoints and examine whether additional training improved or degraded the generated images.

The revised DCGAN experiment subsequently introduced two additional candidates:

| Candidate | GAN configuration | Epoch |
|---|---|---:|
| `dcgan_aug_epoch_150` | Revised DCGAN with DiffAugment | 150 |
| `dcgan_aug_epoch_200` | Revised DCGAN with DiffAugment | 200 |

The revised DCGAN candidates are intended to be evaluated using the same number of images per class as the original candidates.

## 5. Image Generation Script

The image generation pipeline is implemented in:

`src/generate_synthetic_images.py`

The script uses the `ConditionalGenerator` architecture defined in `src/models.py`.

### 5.1 Checkpoint loading

For each candidate, the script:

1. Resolves the checkpoint path relative to the project root.
2. Loads the saved checkpoint.
3. Retrieves the model and data configuration from the checkpoint.
4. Reconstructs the conditional generator using the saved configuration.
5. Loads the generator weights.
6. switches the generator to evaluation mode.
7. Generates images conditioned on each class label.

The checkpoint includes the generator state dictionary and configuration needed to reconstruct the model.

The script also checks that the checkpoint epoch is either 150 or 200 and verifies that the checkpoint's GAN type matches the candidate name.

### 5.2 Image generation settings

The standard full evaluation uses the following settings:

| Parameter | Value |
|---|---|
| Images per class | 1,000 |
| Number of classes | 7 |
| Images per checkpoint | 7,000 |
| Batch size | 32 |
| Base random seed | 2026 |
| Image resolution | 64 × 64 |
| Image channels | 3 |
| Device | CUDA when available; otherwise CPU |

The generation process uses random latent vectors and class labels. For each class, the label is repeated across the corresponding batch, ensuring that the generator produces samples conditioned on the intended class.

The generator's output is expected to lie in the range [-1, 1]. The output is clamped to that range and converted to 8-bit RGB pixel values in the range [0, 255].

### 5.3 Random seed handling

The script sets Python, NumPy, and PyTorch random seeds for each class. CUDA random seeds are also set when CUDA is available.

The base seed is 2026, with the class ID added to produce the class-specific seed.

This makes generation more reproducible under the same software and hardware conditions. Exact reproducibility across different hardware or software configurations is not guaranteed.

### 5.4 Output files

For each candidate, the script saves:

- Individual PNG images, organized into class-specific directories.
- `synthetic_images.npz`, containing the generated images, labels, and class names.
- `generation_metadata.json`, recording the candidate, checkpoint, epoch, GAN type, random seed, image counts, image shape, pixel format, and model/data configuration.

The PNG files support visual inspection and directory-based evaluation. The NPZ file provides a consolidated dataset for programmatic analysis.

## 6. Existing Phase 6 Outputs

The original evaluation collection was generated for the four original DCGAN and WGAN-GP candidates.

The expected structure is:

```text
results/
└── fid_evaluation/
    ├── dcgan_epoch_150/
    │   ├── akiec/
    │   ├── bcc/
    │   ├── bkl/
    │   ├── df/
    │   ├── mel/
    │   ├── nv/
    │   ├── vasc/
    │   ├── synthetic_images.npz
    │   └── generation_metadata.json
    ├── dcgan_epoch_200/
    ├── wgangp_epoch_150/
    └── wgangp_epoch_200/
```

Each candidate contains 1,000 generated images per class, or 7,000 images in total.

Therefore, the four original candidates represent a total of 28,000 generated images.

The original DCGAN checkpoint and results directories were subsequently renamed or moved into a location identified as `DCGAN_old` by the user. The exact current paths should be checked before attempting to load the original checkpoints again.

The previously generated Phase 6 image collections are separate from the checkpoint files and should be preserved.

Earlier smoke-test and diversity-analysis outputs also exist:

- `results/synthetic_evaluation_smoke/` — small smoke-test collection, previously using four images per class.
- `results/synthetic_diversity_evaluation/` — diversity-analysis collection, previously using 100 images per class for the four original candidates.
- `results/synthetic_diversity_evaluation/diversity_metrics.csv` — previously computed diversity metrics.

These smoke-test outputs are for pipeline verification and preliminary analysis, not substitutes for the full 1,000-images-per-class evaluation.

## 7. Revised DCGAN Evaluation Outputs

The revised DCGAN was trained under the experiment ID `GAN_DCGAN_AUG`.

Its checkpoints are stored in:

`checkpoints/GAN_DCGAN_AUG/seed_0/`

The directory contains checkpoints from epoch 5 through epoch 200 in increments of five, as well as `dcgan_latest.pt`.

The revised evaluation is intended to use a separate output root:

`results/fid_evaluation_dcgan_aug/`

The expected structure is:

```text
results/
└── fid_evaluation_dcgan_aug/
    ├── dcgan_aug_epoch_150/
    │   ├── akiec/
    │   ├── bcc/
    │   ├── bkl/
    │   ├── df/
    │   ├── mel/
    │   ├── nv/
    │   ├── vasc/
    │   ├── synthetic_images.npz
    │   └── generation_metadata.json
    └── dcgan_aug_epoch_200/
        └── ...
```

The separate output root prevents the revised DCGAN from overwriting the original DCGAN and WGAN-GP evaluation collections.

### 7.1 Checkpoint mapping issue

The first attempt to generate the revised DCGAN evaluation images failed because the script's original candidate mapping still pointed to:

`checkpoints/GAN_DCGAN/seed_0/dcgan_epoch_150.pt`

The revised checkpoint actually resides at:

`checkpoints/GAN_DCGAN_AUG/seed_0/dcgan_epoch_150.pt`

The same issue applies to epoch 200.

The original script therefore needs two additional candidate entries that point to the revised checkpoint paths. The candidate names should clearly distinguish the revised experiment from the original DCGAN.

The script's GAN-type validation also needs to recognize that the revised candidate names correspond to checkpoints whose recorded GAN type is `dcgan`.

The proposed candidate names are:

- `dcgan_aug_epoch_150`
- `dcgan_aug_epoch_200`

The corrected generation command is:

```powershell
python -m src.generate_synthetic_images --candidates dcgan_aug_epoch_150 dcgan_aug_epoch_200 --output-root results/fid_evaluation_dcgan_aug --images-per-class 1000
```

**Status:** The initial generation attempt failed because of the checkpoint path mismatch. The corrected generation has not yet been confirmed complete. The existence of the revised checkpoints is confirmed, but the full revised Phase 6 image collections still need to be generated and verified.

---

# Part B — DCGAN Experiments and Modifications

## 8. Original Conditional DCGAN

The original DCGAN served as the baseline adversarial image-generation model.

Its configuration is stored in:

`configs/gan_dcgan.yaml`

### 8.1 Original configuration

| Parameter | Original DCGAN setting |
|---|---|
| GAN architecture | Conditional DCGAN |
| Training epochs | 200 |
| Batch size | 64 |
| Generator learning rate | 2 × 10⁻⁴ |
| Discriminator learning rate | 2 × 10⁻⁴ |
| Optimizer betas | (0.5, 0.999) |
| Balanced sampler | Enabled |
| DiffAugment | Disabled |
| Checkpoint interval | Every 5 epochs |
| Sample interval | Every 5 epochs |
| Samples per class in training grids | 8 |

The generator is conditioned on the class label. The discriminator receives real or generated images and learns to distinguish between them.

### 8.2 Original loss setup

The original DCGAN uses binary cross-entropy with logits for the discriminator objective.

Real labels are smoothed to 0.9, while fake labels are set to 0. This is a form of one-sided label smoothing.

The generator is trained to produce images that the discriminator classifies as real.

The generator uses Batch Normalization and ReLU activations. The discriminator also uses Batch Normalization and produces logits rather than applying a sigmoid within the model.

### 8.3 Original DCGAN observations

Visual inspection of the original generated samples indicated:

- Blurry or artifact-heavy images.
- Repeated-looking images within some classes.
- Particularly noticeable similarity among samples from the `akiec` class.
- Concerns about limited intra-class diversity.

The pixel-distance analysis was also consistent with lower raw pixel diversity than that observed for WGAN-GP.

These observations motivated an additional DCGAN experiment rather than abandoning the original baseline.

---

## 9. Why the DCGAN Was Modified

The original DCGAN showed signs of limited visual variation among generated samples.

A GAN may learn to generate outputs that are similar to one another instead of covering a broad range of patterns in the training distribution. This is commonly discussed in the context of mode collapse or limited mode coverage.

However, visually similar samples alone do not prove mode collapse. The observation motivates further investigation using visual inspection and quantitative metrics.

The revised experiment was designed to investigate whether discriminator-side augmentation and a lower learning rate could improve the quality and diversity of generated images.

The aim was to retain the original conditional DCGAN architecture and most of its training configuration while modifying a small number of training settings.

## 10. Revised DCGAN Configuration

A new configuration file was created:

`configs/gan_dcgan_aug.yaml`

This file was based on the original DCGAN configuration, with a new experiment ID and name and two principal training changes.

| Parameter | Original DCGAN | Revised DCGAN |
|---|---|---|
| Experiment ID | `GAN_DCGAN` | `GAN_DCGAN_AUG` |
| Experiment name | Original conditional DCGAN | `conditional_dcgan_aug` |
| Training epochs | 200 | 200 |
| Batch size | 64 | 64 |
| Generator learning rate | 2 × 10⁻⁴ | 1 × 10⁻⁴ |
| Discriminator learning rate | 2 × 10⁻⁴ | 1 × 10⁻⁴ |
| Optimizer betas | (0.5, 0.999) | (0.5, 0.999) |
| Balanced sampler | Enabled | Enabled |
| DiffAugment | Disabled | Enabled |
| DiffAugment policy | Not applicable | `color,translation,cutout` |
| Checkpoint interval | Every 5 epochs | Every 5 epochs |
| Sample interval | Every 5 epochs | Every 5 epochs |
| Samples per class in training grids | 8 | 8 |

The experiment uses the same number of epochs and batch size as the original DCGAN. The balanced sampler and optimizer betas are also retained.

The changes therefore target training dynamics and augmentation rather than introducing a different GAN architecture.

### 10.1 Learning-rate reduction

Both the generator and discriminator learning rates were reduced from 0.0002 to 0.0001.

This halves the learning rate used by both networks.

The motivation was to investigate whether smaller optimization steps could produce more stable training and improve the learned image distribution.

A lower learning rate does not guarantee improved GAN performance. Its effect must be assessed through the training logs, generated samples, and evaluation metrics.

### 10.2 Differentiable Augmentation

DiffAugment was enabled in the revised configuration.

The selected policy is:

`color,translation,cutout`

The intended transformations are:

- **Color:** Applies color-related transformations to the images presented during discriminator training.
- **Translation:** Applies spatial translations to encourage tolerance to small positional variations.
- **Cutout:** Masks out image regions, encouraging the discriminator to use broader visual evidence instead of relying exclusively on localized details.

Differentiable augmentation allows the discriminator to be trained using augmented images while preserving the gradient path required for adversarial training.

The goal is to regularize the discriminator and reduce the risk of overfitting to the limited training examples, which may be especially relevant for minority lesion classes.

The augmentation is a training-time modification. It does not mean that the saved real training images were permanently replaced with augmented copies, nor does it mean that the final synthetic images are generated by applying these transformations after generation.

The generator learns through the adversarial training process. The effect of the augmentation must be measured rather than assumed.

### 10.3 Balanced sampling retained

The balanced sampler remained enabled.

This setting is intended to reduce the influence of the original class imbalance during GAN training by sampling classes more evenly.

It is distinct from generating an equal number of images per class during Phase 6. Balanced training sampling affects the batches seen during training; equal Phase 6 image counts affect the composition of the evaluation collection.

### 10.4 Architecture and loss retained

The revised experiment retains the conditional DCGAN approach.

It does not switch to the WGAN-GP objective. In particular, the revised DCGAN remains based on the original DCGAN's binary cross-entropy-with-logits training formulation, including its real-label smoothing setup.

The principal changes are the lower learning rates and the enabled DiffAugment policy.

This makes the revised experiment a targeted modification of the baseline rather than a new GAN family.

## 11. Revised DCGAN Training Execution

The revised training command was:

```powershell
python -m src.train_gan --config configs/gan_dcgan_aug.yaml
```

A short two-epoch pilot was run first to verify that the configuration and training pipeline worked.

After the pilot, the user removed the pilot outputs from the revised experiment's seed-specific results and checkpoint directories before starting the full training run:

- `results/GAN_DCGAN_AUG/seed_0/`
- `checkpoints/GAN_DCGAN_AUG/seed_0/`

The full 200-epoch run was then completed.

The revised experiment generated checkpoints every five epochs, including epochs 150 and 200, as well as a latest checkpoint.

### 11.1 Final training information

The completed run's training summary recorded:

| Metric | Value |
|---|---:|
| Experiment ID | `GAN_DCGAN_AUG` |
| GAN type | `dcgan` |
| Seed | 0 |
| Epochs completed | 200 |
| Training images | 7,009 |
| Images seen | 1,395,200 |
| Final generator loss | 1.50475697 |
| Final discriminator loss | 0.51564163 |
| Final discriminator real score | 0.598976 |
| Final discriminator fake score | 0.298272 |
| Final epoch duration | Approximately 10.785 seconds |

These values are the recorded final-epoch values. They should not be interpreted in isolation as proof of image quality or convergence.

GAN losses do not have a simple monotonic relationship with visual quality. They should be interpreted alongside generated samples, training behavior, diversity measurements, and distributional metrics.

The final training log and summary were saved under the revised experiment's seed-specific results directory. The exact summary and log filenames may include run-specific suffixes in the uploaded copies, but the project experiment is identified by `GAN_DCGAN_AUG`.

### 11.2 Initial visual assessment

After the full run, the user reported that the revised DCGAN samples appeared more diverse and visually better than the original DCGAN samples.

This is a positive preliminary observation, but it is a qualitative assessment rather than a completed quantitative comparison.

The next step is to generate the full evaluation collections for revised epochs 150 and 200 and compare them using the same metrics and image counts used for the existing candidates.

---

# Part C — Diversity and Distributional Evaluation

## 12. Preliminary Diversity Evaluation

A preliminary diversity analysis was performed for the original four candidates.

The script is:

`src/analyze_synthetic_diversity.py`

The results were saved in:

`results/synthetic_diversity_evaluation/diversity_metrics.csv`

The analysis uses normalized raw pixel mean absolute pairwise distances as a basic diversity proxy.

### 12.1 Interpretation of the metric

The general idea is to measure the average pixel-level difference between generated images. Larger average distances indicate greater pixel-level variation within the evaluated collection.

The reported values are averages of class-level mean distances, with pixel differences normalized to a comparable scale.

### 12.2 Existing results

| Candidate | Mean normalized pixel-distance diversity |
|---|---:|
| Original DCGAN, epoch 150 | 0.090900 |
| Original DCGAN, epoch 200 | 0.084859 |
| WGAN-GP, epoch 150 | 0.139507 |
| WGAN-GP, epoch 200 | 0.130651 |

These results indicate that the WGAN-GP samples exhibited higher raw pixel-level variation than the original DCGAN samples under this metric.

For the original DCGAN, the reported diversity proxy decreased from epoch 150 to epoch 200. The WGAN-GP metric also decreased between these checkpoints.

These values are preliminary evidence about pixel-level variation, not definitive evidence of better image generation.

### 12.3 Limitations

Raw pixel-distance metrics have important limitations:

- Pixel-level differences can be caused by noise or artifacts rather than meaningful lesion variation.
- Two visually different images may still be unrealistic.
- Images with similar color and composition may have relatively low pixel distance despite meaningful morphological differences.
- The metric does not independently establish anatomical plausibility, diagnostic relevance, or clinical usefulness.
- Comparisons should use consistent preprocessing, image counts, and sampling protocols.

For these reasons, diversity measurements must be interpreted together with visual inspection and distributional evaluation.

The revised DCGAN has not yet been added to this diversity comparison in the confirmed results described here.

## 13. Fréchet Inception Distance (FID)

FID is planned as a distributional evaluation metric for comparing generated images with real images.

Conceptually, FID compares the feature distributions of real and generated images using features extracted by a pretrained Inception network. Lower FID generally indicates that the two feature distributions are closer under the metric.

For this project, FID should be calculated using a consistent protocol across all candidate checkpoints.

Important requirements include:

1. Use the same real reference dataset for every candidate.
2. Use the same preprocessing and image format.
3. Keep the evaluation image counts and sampling procedures consistent.
4. Record the exact real-image directory and synthetic-image directory used for each score.
5. Report scores separately for each candidate and, where useful, each class.
6. Avoid using the held-out test split for GAN selection if it is reserved for final classifier evaluation.

The project environment has `torch_fidelity` installed. The separate `pytorch_fid` package was not installed in the last recorded environment check. The exact FID command or implementation still needs to be established and verified.

**FID results are not yet documented as completed.** They should be added to this report only after the metric has been run successfully and the reference dataset and evaluation settings have been recorded.

---

# Part D — Experimental Integrity and Output Management

## 14. Separation of Experiments

The original DCGAN, revised DCGAN, and WGAN-GP experiments must remain distinguishable.

The intended output locations are:

| Purpose | Location |
|---|---|
| Original DCGAN checkpoints | Original path may have been renamed or moved |
| Revised DCGAN checkpoints | `checkpoints/GAN_DCGAN_AUG/seed_0/` |
| WGAN-GP checkpoints | `checkpoints/GAN_WGANGP/seed_0/` |
| Original candidate evaluation images | `results/fid_evaluation/` |
| Revised DCGAN evaluation images | `results/fid_evaluation_dcgan_aug/` |
| Preliminary diversity analysis | `results/synthetic_diversity_evaluation/` |
| Smoke-test evaluation | `results/synthetic_evaluation_smoke/` |

The separate revised DCGAN output root is important because the generation script protects non-empty candidate directories from accidental overwriting.

Do not use the `--overwrite` option unless replacing an existing evaluation collection is intentional and the existing results have been safely preserved.

## 15. Reproducibility and Data Leakage Considerations

The following practices should be maintained:

- Preserve the fixed dataset split.
- Do not regenerate the split to improve the observed results.
- Do not use test-set performance to select GAN checkpoints.
- Keep GAN checkpoints, generation metadata, and evaluation images associated with their correct experiment IDs.
- Record the random seed, checkpoint epoch, image count, and preprocessing settings.
- Apply the same evaluation protocol to the original and revised candidates.
- Do not treat a larger diversity score as sufficient evidence of better synthetic data.
- Distinguish visual inspection, synthetic-image metrics, and downstream classifier performance.

For downstream classification, the classifier should be evaluated on the held-out real test set. Synthetic training images should not leak into the real validation or test data.

---

# Part E — Current Status and Next Steps

## 16. Phase 6 Progress Summary

| Task | Status |
|---|---|
| Generate 1,000 images per class for original DCGAN epoch 150 | Completed |
| Generate 1,000 images per class for original DCGAN epoch 200 | Completed |
| Generate 1,000 images per class for WGAN-GP epoch 150 | Completed |
| Generate 1,000 images per class for WGAN-GP epoch 200 | Completed |
| Save candidate-specific PNG files and NPZ collections | Completed for the original candidates |
| Save generation metadata | Completed for the original candidates |
| Run preliminary pixel-distance diversity analysis | Completed for the original four candidates |
| Complete revised DCGAN training for 200 epochs | Completed |
| Save revised DCGAN checkpoints at epochs 150 and 200 | Completed |
| Add revised DCGAN candidates to the generation script | Required; edits were proposed but are not yet confirmed saved |
| Generate revised DCGAN evaluation images | Pending confirmation |
| Verify revised candidate image counts and NPZ contents | Pending |
| Compute revised DCGAN diversity metrics | Pending |
| Run FID consistently across all candidates | Pending |
| Select a final GAN checkpoint | Pending |
| Evaluate downstream classification with synthetic augmentation | Pending |

## 17. Immediate Next Steps

### Step 1: Update the generation script

Add the revised DCGAN candidates to the `CANDIDATES` dictionary in `src/generate_synthetic_images.py`:

```python
"dcgan_aug_epoch_150": (
    "checkpoints/GAN_DCGAN_AUG/seed_0/dcgan_epoch_150.pt",
),
"dcgan_aug_epoch_200": (
    "checkpoints/GAN_DCGAN_AUG/seed_0/dcgan_epoch_200.pt",
),
```

Adjust the GAN-type check to recognize that both revised candidates have the checkpoint type `dcgan`.

### Step 2: Generate the revised evaluation collections

Run:

```powershell
python -m src.generate_synthetic_images --candidates dcgan_aug_epoch_150 dcgan_aug_epoch_200 --output-root results/fid_evaluation_dcgan_aug --images-per-class 1000
```

### Step 3: Verify outputs

Confirm that each revised candidate contains:

- Seven class directories.
- 1,000 PNG images in each class directory.
- 7,000 total images.
- A valid `synthetic_images.npz` file.
- A `generation_metadata.json` file with the correct checkpoint and epoch.

### Step 4: Compare diversity

Extend or rerun the diversity analysis so that the revised DCGAN candidates are measured under the same protocol as the original candidates and WGAN-GP.

### Step 5: Calculate FID

Run the verified FID implementation on the original and revised candidates, using a consistent real-image reference set and preprocessing pipeline.

### Step 6: Select candidates based on multiple forms of evidence

Compare the candidates using:

- Visual quality and artifacts.
- Intra-class diversity.
- FID.
- Class-specific failure cases.
- Downstream classifier performance, evaluated without test-set leakage.

The final checkpoint should not be selected from visual appearance or a single metric alone.

---

## 18. Overall Conclusion

The original Phase 6 evaluation pipeline has generated balanced synthetic image collections for the original DCGAN and WGAN-GP checkpoints and produced preliminary pixel-distance diversity measurements.

The original DCGAN showed lower raw pixel-level diversity than WGAN-GP under the existing metric, motivating an additional DCGAN experiment.

The revised DCGAN retains the conditional DCGAN architecture, 200-epoch training schedule, batch size, optimizer betas, and balanced sampling. It introduces two principal changes: both learning rates are reduced from 0.0002 to 0.0001, and DiffAugment is enabled with the `color,translation,cutout` policy.

The revised model completed 200 epochs, and the user reported an initial visual improvement in diversity and quality. However, the revised checkpoints still need to be incorporated into the Phase 6 generation and quantitative evaluation pipeline.

Consequently, the current evidence supports continuing with the revised DCGAN evaluation, but it does not yet establish that the revised DCGAN is superior to WGAN-GP or that its synthetic images improve downstream skin lesion classification. Those conclusions require the remaining diversity, FID, and classifier evaluation results.

#### 3. Revised DCGAN configuration to document
The active DCGAN retains the conditional DCGAN architecture and original loss formulation, with these key modifications:
Parameter	Active DCGAN setting
Training epochs	200+
Batch size	64
Generator learning rate	1e-4
Discriminator learning rate	1e-4
Optimizer betas	(0.5, 0.999)
Balanced sampling	Enabled
DiffAugment	Enabled
DiffAugment policy	color,translation,cutout
Checkpoint interval	Every 5 epochs

Its configuration is configs/gan_dcgan_aug.yaml, and its checkpoints are under checkpoints/GAN_DCGAN_AUG/seed_0/.

The original configs/gan_dcgan.yaml and original checkpoints are archived references only. We will not silently use them when a task asks for DCGAN results.

Phase 6 documentation: GAN evaluation for imbalanced skin lesion classification
This documents the work completed so far for FID evaluation, visual comparison, synthetic-image diversity, and memorisation checks, including the experiment decisions, observed results, interpretation, and output files.

One important distinction throughout the report: the active DCGAN experiment is the Revised DCGAN (DCGAN-Aug). The original DCGAN was archived and must not be included in the current comparison. The four active checkpoints are DCGAN-Aug at epochs 150 and 200, and WGAN-GP at epochs 150 and 200.

The evaluation uses the HAM10000 dataset and seven classes: akiec, bcc, bkl, df, mel, nv, and vasc.
1. Experimental context and decisions
1.1 Dataset and fixed split
The project uses HAM10000 with 10,015 dermoscopic images across seven diagnostic categories.

The existing split is stored in cache/splits.csv. It was kept fixed to ensure consistent experiments and prevent data leakage.
Split	Images	Purpose
Training	7,009	GAN training, real-image reference set for FID, and classifier training
Validation	1,503	Model selection and validation-based decisions
Test	1,503	Final classifier evaluation only

The class distribution in the training split is:
Class	Training images
akiec	228
bcc	359
bkl	769
df	80
mel	779
nv	4,694
vasc	100
Total	7,009

This distribution is highly imbalanced. In particular, df and vasc have very few real training examples, while nv dominates the dataset.

Decision: all FID real-image references must come from the training split, not validation or test. The validation images can be used for the real-to-real baseline in the memorisation analysis, but the test split must remain untouched until final classifier evaluation.
1.2 Active GAN experiments









Revised DCGAN (DCGAN-Aug)

Experiment ID: GAN_DCGAN_AUG
- 200 training epochs.
- Generator and discriminator learning rates: \(10^{-4}\).
- Conditional generation with balanced class sampling.
- DiffAugment enabled with color,translation,cutout.
- The learning rates were reduced from the original DCGAN configuration, and differentiable augmentation was introduced to improve training behaviour and sample diversity.












WGAN-GP

Experiment ID: GAN_WGANGP
- 200 training epochs.
- Generator and critic learning rates: \(10^{-4}\).
- Five critic updates per generator update.
- Gradient penalty coefficient: 10.
- Balanced class sampling.
- DiffAugment disabled in this configuration.



The configurations are not identical apart from the GAN objective: the revised DCGAN uses DiffAugment, while WGAN-GP uses its gradient-penalty training method without DiffAugment. Consequently, the experiment compares the two configured GAN systems, rather than isolating the effect of the GAN objective alone.

The original experiment GAN_DCGAN is archived. Its outputs must not be mixed with the revised DCGAN results.
1.3 Checkpoints selected for evaluation
Four checkpoints were selected to compare sample quality at two training stages.
Active candidate	Checkpoint
DCGAN-Aug-E150	Revised DCGAN, epoch 150
DCGAN-Aug-E200	Revised DCGAN, epoch 200
WGAN-GP-E150	WGAN-GP, epoch 150
WGAN-GP-E200	WGAN-GP, epoch 200

The epoch-150 and epoch-200 checkpoints allow us to examine whether continuing training was associated with changes in distributional similarity and diversity. They do not, by themselves, establish that a model is converged or that its later checkpoint is universally better.
2. FID evaluation
2.1 Objective
The Fréchet Inception Distance (FID) was used to compare the distribution of generated images with the distribution of real images.

FID compares feature distributions extracted by an Inception network. It reflects differences in feature-space means and covariances:
\[
\operatorname{FID}(R,S)=
\|\mu_R-\mu_S\|_2^2+
\operatorname{Tr}\left(
\Sigma_R+\Sigma_S-2(\Sigma_R\Sigma_S)^{1/2}
\right)
\]

Here, \(R\) represents real images and \(S\) represents synthetic images.

Interpretation: lower FID generally indicates that the synthetic set is closer to the real set in the Inception feature space. FID is not a direct measure of diagnostic correctness, clinical usefulness, or memorisation. A low FID does not guarantee that the images are medically realistic or diverse.
2.2 Synthetic image generation for FID
For each of the four active checkpoints, 1,000 images were generated for each of the seven classes.
Property	Setting
Image size	\(64 \times 64\) pixels
Colour format	RGB
Classes per checkpoint	7
Images per class	1,000
Images per checkpoint	7,000
Active checkpoints	4
Total synthetic images evaluated	28,000

The class-conditioned generation procedure ensures that every candidate contains exactly 1,000 images labelled as each class. This makes the synthetic sample counts comparable between candidates.

The generated image directories were:
- Revised DCGAN, epoch 150: results/fid_evaluation_dcgan_aug/dcgan_aug_epoch_150/
- Revised DCGAN, epoch 200: results/fid_evaluation_dcgan_aug/dcgan_aug_epoch_200/
- WGAN-GP, epoch 150: results/fid_evaluation/wgangp_epoch_150/
- WGAN-GP, epoch 200: results/fid_evaluation/wgangp_epoch_200/
Each directory contains the generated images organised by class, together with the corresponding NPZ data and metadata produced by the generation script.

Important scope decision: the original DCGAN candidate directories are not part of this evaluation. The FID script explicitly selects the four active candidate directories instead of including every directory found under the results folder.
2.3 Implementation and reference data
The dedicated FID evaluation script uses torch-fidelity, rather than adding GAN evaluation code to src/metrics.py. That file remains responsible for classifier metrics.

The evaluation process is:
1. Load the fixed split from cache/splits.csv.
2. Select rows belonging to the training split.
3. Load HAM10000 metadata and verify that training labels match the metadata labels.
4. Locate the corresponding real images and check that all training images can be found.
5. Load each candidate's NPZ file and validate the image dimensions and class counts.
6. Convert real and synthetic images into RGB PNG files in temporary directories for evaluation.
7. Compute FID between the real and synthetic sets for each class.
8. Compute overall FID between all 7,009 real training images and all 7,000 synthetic images for each candidate.
9. Save a detailed long-format CSV and a comparison table.
The script requests FID from torch-fidelity with Inception Score and Kernel Inception Distance disabled. It uses CUDA if available. The same evaluation function and preprocessing pipeline are used for every candidate.

The per-class comparisons use the real training images of that class. For example, the synthetic df set of 1,000 images is compared against the 80 real training df images. The overall comparison uses all 7,009 real training images and all 7,000 generated images.

This is a class-conditional reference comparison, not a comparison between generated images and validation/test images.
2.4 FID results
The following values are from results/fid_evaluation_active/fid_results_table.csv.
Class	DCGAN-Aug E150	DCGAN-Aug E200	WGAN-GP E150	WGAN-GP E200
akiec	237.047	208.086	232.527	245.395
bcc	188.849	171.037	196.099	200.928
bkl	188.135	183.996	205.272	201.653
df	220.211	202.151	231.244	227.130
mel	177.271	189.086	204.883	200.865
nv	163.749	159.942	180.129	172.636
vasc	200.664	202.436	224.459	219.258
Overall	155.804	148.257	165.547	163.343

Lower is better for FID. Bold values in the table identify the lowest value in the overall row; they do not indicate a statistical significance test.
2.5 Interpretation of FID results
Overall FID
- DCGAN-Aug E200 achieved the lowest overall FID: 148.257.
- DCGAN-Aug E150 achieved an overall FID of 155.804.
- WGAN-GP E150 and E200 achieved overall FIDs of 165.547 and 163.343, respectively.
The revised DCGAN's overall FID decreased by approximately 4.8% from epoch 150 to epoch 200. WGAN-GP's overall FID decreased by approximately 1.3% over the same interval. These changes indicate improved feature-distribution similarity at epoch 200 according to this metric, especially for DCGAN-Aug.

Across the four tested checkpoints, DCGAN-Aug E200 had the lowest overall FID. This supports selecting it as a strong candidate for subsequent inspection, but it is not enough by itself to select a GAN for final augmentation.

Class-wise comparison
- akiec: DCGAN-Aug E200 performed best among the four candidates, with FID 208.086. WGAN-GP E200 had the highest FID for this class, 245.395.
- bcc: DCGAN-Aug E200 had the lowest FID, 171.037.
- bkl: DCGAN-Aug E200 had the lowest FID, 183.996.
- df: DCGAN-Aug E200 had the lowest FID, 202.151.
- mel: DCGAN-Aug E150 had the lowest FID, 177.271. Its epoch-200 FID increased, so later training was not uniformly beneficial.
- nv: DCGAN-Aug E200 had the lowest FID, 159.942.
- vasc: DCGAN-Aug E150 had the lowest FID, 200.664, narrowly below DCGAN-Aug E200 at 202.436.
DCGAN-Aug E200 achieved the lowest FID for five of the seven classes, while DCGAN-Aug E150 achieved the lowest for mel and vasc. WGAN-GP did not achieve the lowest FID for any of the seven classes or the overall set in this comparison.
2.6 Limitations of FID
FID results must be interpreted cautiously for this dataset:
1. Small real reference sets for rare classes. There are only 80 real training images for df, 100 for vasc, and 228 for akiec. Their class-wise FID estimates may be unstable and sensitive to the reference sample.
2. Unequal real class counts. Each class has 1,000 synthetic images but very different numbers of real reference images. The per-class FIDs therefore have different sample-size conditions.
3. Low-resolution images. The images are \(64 \times 64\), so fine clinical details may be lost.
4. Generic Inception features. FID does not directly assess dermatological features or whether a generated lesion would be diagnostically plausible.
5. No uncertainty intervals reported. The CSV records point estimates, not confidence intervals or repeated-run variability.
6. No proof of augmentation benefit. A lower GAN FID does not necessarily mean a classifier trained with its synthetic images will perform better.
Therefore, the FID results support a relative distributional comparison among the four active candidates, not a clinical-quality ranking.
2.7 FID output files
results/fid_evaluation_active/fid_results_table.csv

Wide-format comparison table: one row per class plus overall, with one column per active checkpoint.



results/fid_evaluation_active/fid_results_long.csv

Detailed results containing candidate, class, real sample count, synthetic sample count, and FID.



The files were confirmed to exist in the project results directory. The FID CSV files are the persisted numerical outputs of this evaluation.
3. Visual comparison of generated images
3.1 Purpose
Numerical metrics cannot fully establish whether the generated images resemble plausible dermoscopic images. Visual inspection is therefore a complementary assessment.

The intended comparison is class-specific and should use the same class across:
- Real HAM10000 training images.
- Revised DCGAN (DCGAN-Aug) images.
- WGAN-GP images.
This is particularly important for akiec, bcc, df, and vasc, because the main objective is to augment minority classes rather than to generate large numbers of already common nv images.
3.2 Decisions for visual inspection
- Compare real and synthetic images at the same \(64 \times 64\) resolution.
- Use class-conditioned samples rather than mixing classes.
- Inspect the same class from each active GAN configuration.
- Pay attention to lesion shape, colour distribution, texture, borders, background skin, and obvious artefacts.
- Look for repeated structures, nearly identical images, distorted patterns, and images that appear inconsistent with the requested class.
- Do not select a model based only on images that look attractive or realistic in a small hand-picked grid.
The user previously reported that the revised DCGAN samples appeared more diverse and visually better than those from the original DCGAN. That is useful qualitative feedback motivating the revised configuration, but it is not an independent numerical evaluation of DCGAN-Aug versus WGAN-GP.

The nearest-neighbour boxplots supplied for this phase are quantitative plots, not the real-versus-synthetic sample grids themselves. Thus, they support the memorisation analysis described below, but they do not substitute for documenting the visual sample grids.
3.3 How to report the visual findings
Visual inspection should be reported as qualitative evidence, separately from FID. For example, observations about colour variation, lesion morphology, or visible artefacts should only be included if they were actually observed in the sample grids.

Do not claim that visual inspection proves clinical validity. The images are synthetic outputs from GANs and require careful interpretation.
4. Synthetic-image diversity analysis
4.1 Objective
The diversity analysis measures how different generated images are from one another within the same class. This is intended to identify whether a GAN produces a range of outputs or repeatedly generates very similar images.

FID measures similarity between real and synthetic feature distributions, whereas this analysis measures variation within a synthetic set. These are different properties: a model can generate varied images that do not resemble the real data distribution, or generate realistic-looking images with limited diversity.
4.2 Metric used
The script calculates normalised raw-pixel mean absolute distance between pairs of images:
\[
d(x,y)=\frac{1}{3HW}\sum_{h=1}^{H}\sum_{w=1}^{W}
\sum_{c=1}^{3}\frac{|x_{hwc}-y_{hwc}|}{255}
\]

where \(H=W=64\), and \(c\) indexes the three RGB channels.

The result lies between 0 and 1:
- A value near 0 means the images are very similar at the pixel level.
- A higher value means greater pixel-level difference.
This is a simple diversity proxy. It is sensitive to shifts, colours, and other pixel-level changes, including changes that may not be clinically meaningful.
4.3 Sampling and computation decisions
To keep the computation manageable and ensure fair comparisons:
- The script uses 100 images per class per checkpoint.
- Sampling is reproducible, uses a fixed seed of 2026 plus a class-specific offset, and is performed without replacement.
- It computes every unique pair within the sampled set.
- Each class therefore contributes \(\binom{100}{2}=4,950\) image pairs.
- The script evaluates all seven classes for each of the four active checkpoints.
- It calculates mean, median, minimum, and maximum pairwise distance for every checkpoint-class combination.
- It calculates an unweighted mean of the seven class means for each checkpoint.
This is a sample-based analysis, not a calculation over all 1,000 generated images per class.
4.4 Results: mean of class means
The following results were reported by the completed diversity analysis.
Checkpoint	Mean of class means	Standard deviation across class means
DCGAN-Aug E150	0.129680	0.020442
DCGAN-Aug E200	0.139226	0.028460
WGAN-GP E150	0.140589	0.023222
WGAN-GP E200	0.133131	0.019600

The highest mean diversity proxy was observed for WGAN-GP E150, closely followed by DCGAN-Aug E200.
4.5 Per-class mean pairwise distances
Class	DCGAN-Aug E150	DCGAN-Aug E200	WGAN-GP E150	WGAN-GP E200
akiec	0.1151	0.1229	0.1275	0.1214
bcc	0.1141	0.1194	0.1134	0.1146
bkl	0.1203	0.1371	0.1496	0.1353
df	0.1168	0.1178	0.1160	0.1098
mel	0.1664	0.1968	0.1746	0.1614
nv	0.1502	0.1548	0.1627	0.1548
vasc	0.1249	0.1257	0.1403	0.1347

4.6 Interpretation
Change between epoch 150 and epoch 200
- DCGAN-Aug's mean of class means increased from 0.129680 to 0.139226, an increase of approximately 7.4%.
- WGAN-GP's mean of class means decreased from 0.140589 to 0.133131, a decrease of approximately 5.3%.
The results suggest that DCGAN-Aug E200 produced more pixel-level variation than DCGAN-Aug E150 under this metric. WGAN-GP E150 had the highest aggregate diversity estimate, but its later checkpoint showed lower diversity.

Class-level observations
- mel had comparatively high pairwise distances for all four candidates. DCGAN-Aug E200 had the highest value for this class, 0.1968.
- nv also showed relatively high distances, which may partly reflect the visual variety of this class.
- df had relatively low distances, especially for WGAN-GP E200 at 0.1098.
- For bcc, the four results were very close, ranging from 0.1134 to 0.1194.
- WGAN-GP E150 had relatively high distances for bkl and vasc, but this does not by itself show that the corresponding images were more realistic.
Important limitation: more pixel variation is not automatically better. Random artefacts, colour shifts, or distorted lesion structures can increase pairwise distances without improving meaningful diversity. These values must be interpreted alongside FID, visual inspection, and the memorisation checks.
4.7 Diversity output files
results/synthetic_diversity_active/diversity_metrics.csv

Detailed per-checkpoint, per-class metrics: sample counts, number of pairs, mean, median, minimum and maximum pairwise distance, metric description, and sampling seed.



results/synthetic_diversity_active/diversity_summary.csv

Aggregate summary by checkpoint, including number of classes, images per class used, mean of class means, and standard deviation across class means.



These files were written by src/analyze_synthetic_diversity.py.
5. Memorisation and nearest-neighbour analysis
5.1 Objective
GANs can sometimes reproduce training images or generate outputs that are very close to individual training examples. This is a concern in medical-image augmentation because the objective is to create useful new examples rather than simply duplicate the training set.

The memorisation check compares each synthetic image against real training images from the same class and uses held-out real images as a reference baseline.
5.2 Method
The analysis uses cosine distance between feature representations. A feature extractor maps each image to a vector, and the distance between two vectors indicates how different their representations are.

For each synthetic image:
1. Extract its feature vector.
2. Restrict the real-image comparison set to training images with the same class label.
3. Find the nearest real training image in feature space.
4. Record the minimum cosine distance.
5. Repeat for all evaluated synthetic images and each active candidate.
The real-to-real reference is constructed by taking a held-out real validation image and finding its nearest neighbour among real training images of the same class. This provides context for how close a synthetic image is to its nearest real example compared with a real image that was not used for GAN training.

The test split should not be used for this analysis or for GAN selection.
5.3 How to interpret the distance
Cosine distance is typically interpreted as:
- Lower distance: the two feature vectors are more similar.
- Higher distance: the two feature vectors are less similar.
A synthetic image with a distance extremely close to zero may deserve investigation, especially if its nearest real image also appears almost identical visually.

However, a low distance alone does not prove memorisation. Common lesion patterns can naturally be close in feature space. Likewise, a large distance does not prove that an image is realistic or useful.

The real-to-real validation baseline is important because it shows the range of distances expected between genuine images and their nearest same-class training examples.
5.4 Results from the supplied boxplots
The supplied figures show nearest-neighbour distance distributions for all seven classes. Each compares five groups:
1. Real validation images.
2. DCGAN-Aug E150.
3. DCGAN-Aug E200.
4. WGAN-GP E150.
5. WGAN-GP E200.
The orange horizontal line is the median, the box shows the interquartile range, and the whiskers show the plotted range under the chart's whisker convention.









General pattern across the displayed plots

In the supplied figures, the synthetic distributions are generally shifted toward higher nearest-neighbour distances than the real-validation distributions. This is consistent with the synthetic images not being uniformly near-duplicates of the training images in the evaluated feature space.



The visual evidence is encouraging but should not be overstated. The boxplots alone do not establish that no memorisation occurred, because they do not show the proportion of extremely close matches or the actual synthetic-to-real image pairs.
5.5 Class-wise observations from the figures
These are approximate visual observations from the supplied plots, not exact numerical summaries from the underlying CSV.
- akiec: the real-validation median appears around 0.10. DCGAN-Aug medians are around 0.11–0.12, while WGAN-GP medians are around 0.14. The synthetic distributions appear generally farther from the nearest training example than the real-validation distribution.
- bcc: the real-validation median appears around 0.096, compared with roughly 0.12 for DCGAN-Aug and 0.14 for WGAN-GP. The synthetic distributions are shifted upward.
- bkl: the real-validation median appears around 0.097. DCGAN-Aug medians are around 0.13, and WGAN-GP medians are around 0.15. Again, the synthetic images are generally farther from their nearest training examples in feature space.
- df: the real-validation median appears around 0.124. DCGAN-Aug medians are around 0.13, and WGAN-GP medians are around 0.14–0.15. The separation is smaller than for some other classes.
- mel: the real-validation median appears around 0.096. Synthetic medians are approximately 0.13 for DCGAN-Aug and 0.14 for WGAN-GP.
- nv: the real-validation median appears around 0.076. Synthetic medians are approximately 0.096–0.106. The synthetic distributions are shifted upward, though the ranges overlap.
- vasc: the real-validation median appears around 0.125. The synthetic medians are approximately 0.146–0.161, depending on checkpoint.
These estimates are included only to describe the plots. Exact medians, quartiles, minimums, and maximums should be taken from the saved CSV outputs rather than estimated visually.
5.6 What the memorisation results mean
The higher synthetic nearest-neighbour distances suggest that the synthetic images are not simply concentrated around the closest real training examples under the chosen feature representation. This is evidence against widespread near-duplication in that feature space.

However, the following checks remain important:
- Inspect synthetic images alongside their actual nearest real training images.
- Examine the smallest distances, not just the medians.
- Check whether any synthetic images are nearly identical to real images despite their aggregate distribution.
- Confirm that the real-to-real baseline was constructed using validation images as queries and training images as the reference set.
- Confirm that features were extracted using the same model, preprocessing, and distance calculation for all groups.
Conclusion: the boxplots are useful evidence, but they do not justify claiming that the GANs are completely free of memorisation.
5.7 Memorisation output files
The project results listing confirms the following files exist:
- results/memorization_check_active/nearest_neighbor_distances.csv
- results/memorization_check_active/nearest_neighbor_summary.csv
The first file stores the detailed nearest-neighbour distance records, while the second stores aggregated summaries. The exact column definitions and the numerical summary values have not been included in the terminal output shared here, so the approximate plot observations above should not be substituted for those values.
6. Combined interpretation: FID, diversity, and memorisation
These evaluations measure different aspects of generated-image quality.
Evaluation	What it measures	Stronger result means
FID	Difference between real and synthetic feature distributions	Lower FID
Pairwise pixel diversity	Variation between generated images	Higher distance, subject to visual quality
Nearest-neighbour distance	Similarity to the nearest same-class training image	No universally optimal value; investigate extremely low distances
Visual inspection	Visible image structure, variation, and artefacts	Plausible, varied outputs without obvious defects

The combined evidence supports the following interpretation:
1. DCGAN-Aug E200 is the strongest candidate on overall FID. It also achieved the lowest class-wise FID for five of the seven classes.
2. WGAN-GP E150 has the highest aggregate raw-pixel diversity proxy. This is not sufficient to conclude that it generates better medical images.
3. DCGAN-Aug E200 has higher diversity than DCGAN-Aug E150 according to the reported pairwise pixel-distance metric.
4. The nearest-neighbour plots do not show a general collapse toward zero distance for the synthetic distributions. This is encouraging but does not rule out individual memorised examples.
5. No single metric determines the final augmentation choice. The final decision should also depend on visual inspection and the downstream classifier experiments.
These results do not yet establish which GAN improves classification performance. That question requires controlled classifier experiments using the same real training split and the same synthetic class targets.
7. Decisions for building the final synthetic augmentation sets
The next step after evaluating the GANs is to construct fixed synthetic datasets for downstream classifier experiments.

The project decisions are:
- Use Revised DCGAN (DCGAN-Aug) rather than the archived original DCGAN.
- Use WGAN-GP as the comparison method.
- Target minority classes akiec, df, and vasc.
- Retain bcc in result tables and confusion matrices even though it is not one of the three primary top-up classes.
- Keep the same target counts and selection rules for both GAN methods.
- Optionally filter synthetic images using the E0 classifier, but only if the same filtering rule is applied to both GANs and the number rejected is recorded.
- If filtering is used, generate a surplus before filtering, then trim accepted images to the fixed target.
- Do not use validation or test images as GAN training data.
- Keep final classifier comparisons controlled so that differences can be attributed to the augmentation setup as far as possible.
A final decision about the exact top-up target and whether to use quality filtering should be documented before the final synthetic sets are created. The FID evaluation sets contain 1,000 images per class for measurement; they are not automatically the final classifier-augmentation sets.
Expected final synthetic files
The planned outputs are:
- cache/synthetic_dcgan.npz
- cache/synthetic_wgangp.npz
These should contain uint8 RGB images and their class labels, with identical class counts across the two methods.

Current status: the two planned files were checked with Test-Path and both returned False. Therefore, the FID evaluation is complete, but the final synthetic augmentation sets have not yet been confirmed as created.
8. File inventory and status
File or directory	Purpose	Status based on shared evidence
src/analyze_synthetic_diversity.py	Computes pairwise raw-pixel diversity	Executed successfully
results/synthetic_diversity_active/diversity_metrics.csv	Detailed diversity measurements	Exists
results/synthetic_diversity_active/diversity_summary.csv	Aggregate diversity results	Exists
results/fid_evaluation_active/fid_results_long.csv	Detailed FID records	Exists
results/fid_evaluation_active/fid_results_table.csv	FID comparison table	Exists
results/memorization_check_active/nearest_neighbor_distances.csv	Detailed nearest-neighbour distances	Exists
results/memorization_check_active/nearest_neighbor_summary.csv	Memorisation summary	Exists
results/fid_evaluation_dcgan_aug/dcgan_aug_epoch_150/	DCGAN-Aug E150 evaluation images and NPZ	Previously confirmed generated
results/fid_evaluation_dcgan_aug/dcgan_aug_epoch_200/	DCGAN-Aug E200 evaluation images and NPZ	Previously confirmed generated
results/fid_evaluation/wgangp_epoch_150/	WGAN-GP E150 evaluation images and NPZ	Previously confirmed generated
results/fid_evaluation/wgangp_epoch_200/	WGAN-GP E200 evaluation images and NPZ	Previously confirmed generated
cache/synthetic_dcgan.npz	Final DCGAN-Aug augmentation set	Not yet created/confirmed
cache/synthetic_wgangp.npz	Final WGAN-GP augmentation set	Not yet created/confirmed

The active FID script is responsible for the two FID CSV files. The diversity script is responsible for the two diversity CSV files. The nearest-neighbour analysis has its own output directory, keeping memorisation evidence separate from FID and classifier metrics.

The temporary real and synthetic PNG directories used internally by the FID script are created inside a temporary directory and removed when the script finishes. The persistent FID evidence is the pair of CSV files.
9. Suggested report-ready conclusion
The following paragraph can be used as the conclusion for this evaluation phase, with any institution-specific formatting added separately.
GAN Quality Evaluation and Memorisation Analysis
Four active GAN checkpoints were evaluated for synthetic skin-lesion image quality: Revised DCGAN (DCGAN-Aug) at epochs 150 and 200, and WGAN-GP at epochs 150 and 200. Each checkpoint generated 1,000 class-conditioned RGB images at (64 \times 64) resolution for each of the seven HAM10000 classes, resulting in 28,000 synthetic images across the four candidates. FID was calculated using torch-fidelity, comparing synthetic images with real images from the fixed training split. Both overall and class-wise FID results were recorded.

DCGAN-Aug at epoch 200 achieved the lowest overall FID of 148.257, compared with 155.804 for DCGAN-Aug at epoch 150, 165.547 for WGAN-GP at epoch 150, and 163.343 for WGAN-GP at epoch 200. DCGAN-Aug at epoch 200 also achieved the lowest FID for five of the seven classes. However, the relatively small real reference sets for minority classes mean that class-wise FID estimates should be interpreted comparatively rather than as absolute measures of medical-image quality.

Synthetic-image diversity was assessed using normalised mean absolute pairwise pixel distance. The unweighted mean of class means was 0.139226 for DCGAN-Aug at epoch 200 and 0.140589 for WGAN-GP at epoch 150. DCGAN-Aug showed an increase in this diversity measure between epochs 150 and 200, whereas WGAN-GP showed a decrease. Because pixel-level variation can reflect irrelevant changes or artefacts, diversity results were interpreted alongside FID and visual inspection rather than as standalone evidence of quality.

A nearest-neighbour analysis compared synthetic images with same-class real training images in feature space, using held-out real validation images as a real-to-real reference. The supplied distance-distribution plots generally showed higher nearest-neighbour distances for synthetic images than for real validation images. This pattern is consistent with the synthetic sets not being dominated by near-duplicates in the evaluated feature space. Nevertheless, aggregate distance distributions cannot rule out individual memorised examples, and visual inspection of the nearest synthetic–real pairs remains important.

Overall, the results identify DCGAN-Aug at epoch 200 as the strongest candidate in terms of overall FID, while WGAN-GP at epoch 150 achieved the highest aggregate pixel-diversity estimate. These findings provide complementary evidence for selecting candidates for downstream augmentation experiments, but they do not establish clinical validity or improved classifier performance. Final conclusions about augmentation benefit must be based on controlled classifier experiments using identical class targets and evaluation procedures. The original DCGAN was excluded from the active comparison because it was replaced by the revised DCGAN-Aug configuration.




10. Remaining work before closing this phase
The FID calculations and diversity analysis are complete based on the outputs shown. The memorisation CSV files and plots also exist, but their exact numerical summaries and the nearest synthetic–real image pairs should be checked before making a stronger memorisation claim.

The main remaining Phase 6 deliverable is to build and validate the final augmentation NPZ files, apply the same target-count and quality-filtering rules to both GANs, and verify that their class counts match. The final rare-class comparison grids should also be retained with the report.

Once those checks pass, the phase can be documented as complete, with FID, diversity, memorisation, and final synthetic-set construction treated as separate but complementary deliverables.


## Phase 6 — GAN Evaluation and Final Synthetic Dataset Construction

### 1. Objective

This phase evaluates the two active GAN approaches for synthetic skin-lesion image generation and prepares controlled synthetic datasets for downstream classifier experiments.

The two active approaches are:

1. **DCGAN-Aug** — Revised DCGAN with DiffAugment.
2. **WGAN-GP** — Wasserstein GAN with gradient penalty.

The original DCGAN experiment is archived and excluded from all active comparisons and subsequent synthetic-set construction. DCGAN-Aug replaces the original DCGAN as the DCGAN-based approach in the remaining experiments.

The final synthetic datasets are intended to augment the minority classes of HAM10000 during downstream classifier training. They contain only synthetic additions, not a copy of the complete real training dataset.

---

### 2. Dataset and Class Label Mapping

The HAM10000 dataset contains 10,015 images distributed across seven classes.

The fixed dataset split is stored in `cache/splits.csv`. The split must not be regenerated during the remaining experiments, and the test set must remain reserved for final classifier evaluation.

| Class | Label | Real training images |
|---|---:|---:|
| `akiec` | 0 | 228 |
| `bcc` | 1 | 359 |
| `bkl` | 2 | 769 |
| `df` | 3 | 80 |
| `mel` | 4 | 779 |
| `nv` | 5 | 4,694 |
| `vasc` | 6 | 100 |
| **Total** | | **7,009** |

The validation and test sets each contain 1,503 images.

The GAN-generated images are RGB images of size 64 × 64 pixels. The class-label mapping above must remain consistent across GAN generation, synthetic-set preparation, and classifier training.

### 3. Active GAN Experiments

#### 3.1 Revised DCGAN — DCGAN-Aug

- Experiment ID: `GAN_DCGAN_AUG`
- Configuration: `configs/gan_dcgan_aug.yaml`
- Training duration: 200 epochs
- Batch size: 64
- Generator learning rate: \(1 \times 10^{-4}\)
- Discriminator learning rate: \(1 \times 10^{-4}\)
- Optimizer betas: `(0.5, 0.999)`
- Balanced class sampler: enabled
- DiffAugment: enabled
- DiffAugment policy: `color,translation,cutout`
- Checkpoint and sample interval: every 5 epochs

Relevant checkpoints:

- `checkpoints/GAN_DCGAN_AUG/seed_0/dcgan_epoch_150.pt`
- `checkpoints/GAN_DCGAN_AUG/seed_0/dcgan_epoch_200.pt`
- `checkpoints/GAN_DCGAN_AUG/seed_0/latest.pt`

The E200 checkpoint is used for final synthetic-set construction.

#### 3.2 WGAN-GP

- Experiment ID: `GAN_WGANGP`
- Configuration: `configs/gan_wgangp.yaml`
- Training duration: 200 epochs
- Batch size: 64
- Generator learning rate: \(1 \times 10^{-4}\)
- Critic learning rate: \(1 \times 10^{-4}\)
- Optimizer betas: `(0, 0.9)`
- Number of critic updates per generator update: 5
- Gradient penalty coefficient: 10
- Balanced class sampler: enabled
- DiffAugment: disabled
- Checkpoint and sample interval: every 5 epochs

Relevant checkpoints:

- `checkpoints/GAN_WGANGP/seed_0/wgangp_epoch_150.pt`
- `checkpoints/GAN_WGANGP/seed_0/wgangp_epoch_200.pt`
- `checkpoints/GAN_WGANGP/seed_0/latest.pt`

The E200 checkpoint is used for final synthetic-set construction.

The project environment uses PyTorch 2.5.1 with CUDA 12.1 support, torchvision 0.20.1, and an NVIDIA RTX 3050 Laptop GPU. CUDA availability was verified in the project environment.

#### 3.3 Archived Original DCGAN

The original experiment, `GAN_DCGAN`, and its configuration, `configs/gan_dcgan.yaml`, are archived.

**Decision:** Do not include original DCGAN results in the active GAN comparison, final synthetic-set construction, or subsequent augmentation experiments. When selecting candidates, explicitly specify the active experiment IDs or paths rather than using a wildcard that could include archived outputs.

---

### 4. FID Evaluation

FID evaluation was completed using `torch-fidelity` version 0.4.0. The active GAN candidates were evaluated using consistent Inception-based preprocessing and 1,000 synthetic images per class for each candidate.

The evaluated checkpoints were:

- DCGAN-Aug E150
- DCGAN-Aug E200
- WGAN-GP E150
- WGAN-GP E200

The real reference data came from the training split. The fixed validation and test partitions were not used as GAN training data.

#### 4.1 Result files

- `results/fid_evaluation_active/fid_results_long.csv`
- `results/fid_evaluation_active/fid_results_table.csv`

#### 4.2 FID results

Lower FID indicates greater similarity between the generated and reference feature distributions under the chosen evaluation procedure.

| Class | DCGAN-Aug E150 | DCGAN-Aug E200 | WGAN-GP E150 | WGAN-GP E200 |
|---|---:|---:|---:|---:|
| `akiec` | 237.0474 | 208.0859 | 232.5274 | 245.3947 |
| `bcc` | 188.8486 | 171.0367 | 196.0988 | 200.9276 |
| `bkl` | 188.1345 | 183.9961 | 205.2721 | 201.6532 |
| `df` | 220.2105 | 202.1507 | 231.2435 | 227.1296 |
| `mel` | 177.2709 | 189.0860 | 204.8825 | 200.8650 |
| `nv` | 163.7489 | 159.9421 | 180.1286 | 172.6359 |
| `vasc` | 200.6636 | 202.4365 | 224.4589 | 219.2579 |
| **Overall** | **155.8038** | **148.2569** | **165.5471** | **163.3430** |

#### 4.3 Interpretation and decision

DCGAN-Aug E200 achieved the lowest overall FID among the four active candidates. It also achieved the lowest FID for most individual classes.

Exceptions include `mel` and `vasc`, for which DCGAN-Aug E150 performed slightly better. WGAN-GP E200 improved over WGAN-GP E150 on several classes and in overall FID, but its overall FID remained higher than that of DCGAN-Aug E200.

The FID values for rare classes should be interpreted cautiously because their real training sample sizes are small. FID is a comparative distribution-level metric and does not establish clinical correctness, diagnostic usefulness, or downstream classification improvement.

**Decision:** Retain both GAN methods for comparative augmentation experiments rather than eliminating WGAN-GP based solely on FID.

---

### 5. Visual Inspection, Diversity, and Memorisation Analysis

#### 5.1 Visual inspection

Visual comparison grids were saved under:

`results/visual_comparison_active/`

The inspection focused on the rare classes `akiec`, `bcc`, `df`, and `vasc`.

Qualitative inspection indicated that the revised DCGAN-Aug samples appeared more diverse and visually preferable to the original DCGAN samples. This is an informal visual observation, not an objective demonstration of medical image quality.

#### 5.2 Diversity analysis

Results were saved under:

`results/synthetic_diversity_active/`

Relevant files include:

- `diversity_metrics.csv`
- `diversity_summary.csv`

The analysis used 100 generated images per class and candidate and measured pairwise raw-pixel diversity.

| Candidate | Mean of class-level diversity means |
|---|---:|
| DCGAN-Aug E150 | 0.129680 |
| DCGAN-Aug E200 | 0.139226 |
| WGAN-GP E150 | 0.140589 |
| WGAN-GP E200 | 0.133131 |

DCGAN-Aug increased in this diversity measure from E150 to E200, while WGAN-GP decreased. WGAN-GP E150 had the highest mean, followed by DCGAN-Aug E200.

These measurements are raw-pixel diversity proxies. Higher pixel-level diversity does not necessarily mean greater pathological diversity, more realistic lesions, or better classifier performance.

#### 5.3 Memorisation analysis

Results were saved under:

`results/memorization_check_active/`

Relevant outputs include:

- `nearest_neighbor_distances.csv`
- `nearest_neighbor_summary.csv`
- Nearest-neighbour plots and side-by-side comparison images

The analysis compared generated images against real training images using cosine nearest-neighbour distances in pretrained ResNet-18 feature space. A validation-to-training comparison provided a reference baseline.

The analysis did not reveal obvious widespread near-duplicates or near-zero median distances. However, this does not prove that memorisation is absent. Feature-space similarity is not equivalent to exact image duplication, and this analysis cannot establish complete privacy or memorisation safety.

The fixed test set remains untouched.

---

### 6. Final Synthetic-Set Construction — Decisions and Rules

#### 6.1 Objective

Construct two comparable synthetic augmentation sets from the active E200 candidates:

- DCGAN-Aug: `cache/synthetic_dcgan.npz`
- WGAN-GP: `cache/synthetic_wgangp.npz`

Each set contains only the synthetic additions for the selected minority classes. The downstream classifier is responsible for combining these additions with the real training data.

#### 6.2 Target-count rule

The selected classes are:

- `akiec`
- `df`
- `vasc`

The reference class is `bcc`, which has 359 real training images.

The selected target fraction is:

`target_fraction: 1.0`

The target total for each selected class is therefore:

\[
N_{\text{target},c}
=
\text{target fraction}
\times
N_{\text{real,bcc}}
=
1.0 \times 359
=
359
\]

The number of synthetic images to add is:

\[
N_{\text{synthetic},c}
=
\max(0,N_{\text{target},c}-N_{\text{real},c})
\]

This produces the following final counts:

| Class | Real training count | Target total | Synthetic additions |
|---|---:|---:|---:|
| `akiec` | 228 | 359 | 131 |
| `df` | 80 | 359 | 279 |
| `vasc` | 100 | 359 | 259 |
| **Total** | | | **669** |

This is the final selected balancing rule. Do not replace it with a 50% target or silently change the reference class.

The target applies only to the selected classes. The procedure does not generate additional images for `bcc`, `bkl`, `mel`, or `nv`.

#### 6.3 E0 quality filtering

An optional quality-filtering approach was considered. It would use the E0 baseline classifier to reject synthetic images when the classifier assigned the wrong class with confidence above a specified threshold, originally proposed as 0.9.

**Final decision: E0 quality filtering is omitted.**

Reasons for this decision:

- The filter was optional in the proposed methodology.
- The project explicitly chose not to use E0-based rejection.
- The final sets should not be described as classifier-filtered.

Consequently, no E0 classifier rejection is applied, and no images are claimed to have passed an E0 confidence test.

The final metadata records:

- `quality_filter.enabled: false`
- Reason: `E0 classifier filtering was explicitly omitted.`

The 0.9 threshold is not applied during final synthetic-set construction.

#### 6.4 Source candidate pools

The final sets use the existing E200 candidate archives:

| Method | Source archive |
|---|---|
| DCGAN-Aug | `results/fid_evaluation_dcgan_aug/dcgan_aug_epoch_200/synthetic_images.npz` |
| WGAN-GP | `results/fid_evaluation/wgangp_epoch_200/synthetic_images.npz` |

Both source archives were verified before construction. Each contains:

- Images: `(7000, 64, 64, 3)`, `uint8`
- Labels: `(7000,)`, `int64`
- 1,000 images for each class
- Class names in the expected order: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`

Only the required images from labels 0, 3, and 6 were selected.

The final-set construction uses sampling without replacement within each source class. A fixed random seed ensures reproducible selection. The resulting arrays are shuffled before saving.

No additional GAN training was required for this preparation step because the existing E200 candidate pools contained sufficient images.

The final construction script uses a fixed seed of 42, with deterministic model-specific random generators: seed 42 for DCGAN-Aug and seed 43 for WGAN-GP.

#### 6.5 Surplus-generation and trimming rule

The earlier methodology considered generating approximately twice the required number of images to provide a surplus for quality filtering before trimming to the exact target.

Because E0 filtering was explicitly omitted and the existing E200 pools already contained 1,000 images per class, the final implementation selected the exact required number directly from these existing pools.

Therefore:

- No additional 2× generation stage was performed during final-set preparation.
- No E0 rejection stage was performed.
- Sampling was without replacement.
- The final counts were trimmed to the exact required additions.

This implementation detail should be reported accurately rather than claiming that a surplus-generation and rejection procedure was executed.

#### 6.6 Output format and class inclusion

Each final NPZ contains:

- `images`: RGB image array, shape `(669, 64, 64, 3)`, dtype `uint8`
- `labels`: integer label array, shape `(669,)`, dtype `int64`
- `class_names`: the seven-class name mapping

The class-count vectors in the fixed label order `[akiec, bcc, bkl, df, mel, nv, vasc]` are:

`[131, 0, 0, 279, 0, 0, 259]`

Both final sets have identical class counts, but their images come from different GAN methods.

The absence of synthetic images for `bcc`, `bkl`, `mel`, and `nv` is intentional. Their zero synthetic counts must not be interpreted as missing labels or a corrupted archive.

#### 6.7 Output files

Final synthetic datasets:

- `cache/synthetic_dcgan.npz`
- `cache/synthetic_wgangp.npz`

Sample grids:

- `results/final_synthetic_sets/dcgan_final_grid.png`
- `results/final_synthetic_sets/wgangp_final_grid.png`

Preparation metadata:

- `results/final_synthetic_sets/preparation_metadata.json`

The metadata records the E200 source paths, selected class counts, target fraction, reference class, selection seed, output paths, data types, array shapes, grid paths, and disabled quality-filter status.

---

### 7. Final Synthetic-Set Validation

After preparation, both final archives were independently checked.

| Validation | DCGAN-Aug | WGAN-GP |
|---|---|---|
| File exists | PASS | PASS |
| Image shape `(669, 64, 64, 3)` | PASS | PASS |
| Image dtype `uint8` | PASS | PASS |
| Label shape `(669,)` | PASS | PASS |
| Label dtype `int64` | PASS | PASS |
| `akiec` count = 131 | PASS | PASS |
| `bcc` count = 0 | PASS | PASS |
| `bkl` count = 0 | PASS | PASS |
| `df` count = 279 | PASS | PASS |
| `mel` count = 0 | PASS | PASS |
| `nv` count = 0 | PASS | PASS |
| `vasc` count = 259 | PASS | PASS |
| Total images = 669 | PASS | PASS |

Both verification runs returned `PASS: True`.

The final construction and validation gates are therefore satisfied for the NPZ files. The metadata also records the same counts and output paths.

These checks establish structural correctness and class-count consistency; they do not independently establish image realism, diagnostic validity, or improvement in downstream classifier performance.

---

### 8. Downstream Experiment Integration

The intended downstream configurations are:

- E3: DCGAN-Aug synthetic augmentation.
- E4: WGAN-GP synthetic augmentation.

The expected synthetic paths are:

- E3: `cache/synthetic_dcgan.npz`
- E4: `cache/synthetic_wgangp.npz`

The classifier's synthetic-data loader reads the `images` and `labels` arrays from the configured NPZ and appends those synthetic examples to the real training data. It expects synthetic additions, not a complete augmented copy of HAM10000.

Before running E3 and E4, verify the configuration files:

1. Confirm E3 points to `cache/synthetic_dcgan.npz`.
2. Confirm E4 points to `cache/synthetic_wgangp.npz`.
3. Ensure any `quality_filter.enabled` settings in those configurations are disabled so the documented decision is reflected in configuration.
4. Check for stale checkpoint references and correct them only if those settings are used by the relevant code or are intended to document the active GAN source.
5. Preserve the E0 baseline, fixed split, validation procedure, and reserved test set.
6. Run the downstream experiments using the same evaluation protocol so that differences can be attributed to the augmentation method as fairly as possible.

The final synthetic archives have been created and validated, but configuration verification and downstream E3/E4 execution are separate subsequent steps.

---

### 9. Reproducibility and Reporting Constraints

The following rules apply to all remaining experiments and final reporting:

1. Use DCGAN-Aug and WGAN-GP as the two active GAN methods.
2. Exclude archived original DCGAN from active comparisons.
3. Use the E200 source pools for final synthetic-set construction.
4. Use `bcc` as the reference class and `target_fraction = 1.0`.
5. Add exactly 131 `akiec`, 279 `df`, and 259 `vasc` synthetic images per GAN.
6. Apply no E0 quality filtering.
7. Do not claim that a 2× surplus-generation and rejection stage was executed.
8. Preserve the fixed train/validation/test split.
9. Do not use the test set for GAN training, synthetic-set selection, quality filtering, or hyperparameter tuning.
10. Report GAN image quality, FID, pixel-level diversity, memorisation checks, and downstream classifier performance as distinct types of evidence.
11. Do not treat FID or pixel-level diversity alone as proof of clinical realism or classification utility.
12. Include `bcc` in relevant downstream result tables and confusion matrices, even though the final synthetic augmentation sets contain no synthetic `bcc` images.
13. Keep the two synthetic augmentation sets structurally comparable, with identical class counts and documented selection procedures.

### 10. Current Status

**Completed:**

- DCGAN-Aug and WGAN-GP E200 training.
- Active-candidate FID evaluation.
- Visual comparison.
- Synthetic diversity analysis.
- Feature-space nearest-neighbour memorisation analysis.
- Construction of both final synthetic NPZ archives.
- Verification of archive existence, shapes, dtypes, and class counts.
- Saving of sample grids and preparation metadata.

**Next:**

- Inspect and verify `configs/e3.yaml` and `configs/e4.yaml`.
- Confirm the synthetic paths and disabled quality-filter settings.
- Run the downstream E3 and E4 classifier experiments.
- Compare results against the E0 baseline using the same validation and final-test protocol.
- Interpret any performance changes alongside the GAN quality and diversity results rather than assuming that more synthetic data necessarily improves classification.

