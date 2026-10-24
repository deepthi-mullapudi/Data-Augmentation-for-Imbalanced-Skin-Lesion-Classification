# Project Workflow and Methodology

Python: 3.11.9



PyTorch: 2.5.1 with CUDA 12.1



Torchvision: 0.20.1 with CUDA 12.1



GPU: CUDA-capable NVIDIA GPU recommended; your training environment had CUDA available.

## Conditional GAN Augmentation for Imbalanced Skin-Lesion Classification

## 1. Project purpose

The project investigates whether class-conditional GAN images can improve
classification of imbalanced HAM10000 skin-lesion data.

The central comparison is controlled:

1. Train a real-image classifier baseline.
2. Train conditional DCGAN and WGAN-GP generators using the same training
   images and class labels.
3. Evaluate the generated images using visual, FID, diversity, and
   nearest-neighbour analyses.
4. Add equal numbers of selected minority-class images to the classifier
   training set.
5. Compare classical augmentation, DCGAN augmentation, and WGAN-GP
   augmentation against the same real-only baseline.

The final classifier experiments are:

| Experiment | Meaning |
| --- | --- |
| **E0** | Real training images only; baseline. |
| **E2** | Real images plus classical augmentation for selected minority classes. |
| **E3** | Real images plus selected DCGAN-generated images. |
| **E4** | Real images plus selected WGAN-GP-generated images. |

**Phase 8 extensions were skipped.** The completed workflow therefore evaluates
the core E0/E2/E3/E4 comparison and proceeds directly to Phase 9 analysis.

The project root is `E:\udemy\ADL\project`. The main locations are:

- Data and cached arrays: [`data/`](data/) and [`cache/`](cache/)
- Implementations: [`src/`](src/)
- Experiment settings: [`configs/`](configs/)
- Model checkpoints: [`checkpoints/`](checkpoints/)
- Classifier and GAN outputs: [`results/`](results/)
- Analysis scripts: [`scripts/`](scripts/)

## 1.1 Evaluator-facing project folder structure

The following structure shows the files and folders relevant to understanding
the completed project pipeline. Individual HAM10000 image files and large
generated arrays are represented by their containing locations rather than
listed one by one.

```text
project/
|
|-- data/
|   |-- HAM10000_metadata
|   |-- HAM10000_images_part_1/
|   `-- HAM10000_images_part_2/
|
|-- cache/
|   |-- ham64.npz
|   |-- splits.csv
|   |-- label_mapping.json
|   |-- train_class_counts.csv
|   |-- train_real_grid.png
|   |-- synthetic_dcgan.npz
|   `-- synthetic_wgangp.npz
|
|-- src/
|   |-- data.py
|   |-- preprocess.py
|   |-- models.py
|   |-- train_gan.py
|   |-- generate_synthetic_images.py
|   |-- prepare_final_synthetic_sets.py
|   |-- train_clf.py
|   |-- metrics.py
|   |-- evaluate_fid.py
|   |-- analyze_synthetic_diversity.py
|   |-- analyze_gan_memorization.py
|   `-- utils.py
|
|-- scripts/
|   |-- compare_gan_samples.py
|   `-- phase9_analysis.py
|
|-- configs/
|   |-- e0.yaml
|   |-- e2.yaml
|   |-- e3.yaml
|   |-- e4.yaml
|   |-- gan_dcgan.yaml
|   |-- gan_dcgan_aug.yaml
|   `-- gan_wgangp.yaml
|
|-- checkpoints/
|   |-- GAN_DCGAN_AUG/
|   `-- GAN_WGANGP/
|
|-- results/
|   |-- metrics.csv
|   |-- E0/
|   |-- E2/
|   |-- E3/
|   |-- E4/
|   |-- fid_evaluation/
|   |-- fid_evaluation_dcgan_aug/
|   |-- fid_evaluation_active/
|   |-- final_synthetic_sets/
|   |-- visual_comparison_active/
|   |-- synthetic_diversity_active/
|   |-- memorization_check_active/
|   `-- phase9_analysis/
|
|-- notebooks/
|
|-- requirements.txt
|-- PROJECT_WORKFLOW.md
|-- PROJECT_LOG.md
|-- Phase6.md
|-- Phase7.md
`-- .gitignore
```

### Role of each top-level folder

| Folder | What it contains | How it is used |
| --- | --- | --- |
| `data/` | Original HAM10000 metadata and JPEG images | Source data; not modified during experiments. |
| `cache/` | Preprocessed images, fixed split, labels, counts, and final synthetic sets | Shared input for GAN training and classifier experiments. |
| `src/` | Reusable Python implementation modules | Executes preprocessing, model training, generation, evaluation, and metric calculation. |
| `scripts/` | Higher-level comparison and aggregation scripts | Produces the final Phase 9 tables and figures from saved results. |
| `configs/` | YAML experiment definitions | Records model, data, optimizer, augmentation, and output settings. |
| `checkpoints/` | Saved GAN model states | Supplies trained generators for image generation. |
| `results/` | Logs, metrics, images, tables, plots, and classifier checkpoints | Evidence produced by each workflow stage. |
| `notebooks/` | Notebook workspace | Optional interactive exploration and presentation material. |

### Source-module responsibilities

| File | Responsibility |
| --- | --- |
| [`src/data.py`](src/data.py) | Label mapping, fixed grouped split creation, cache/manifest validation, and classifier datasets. |
| [`src/preprocess.py`](src/preprocess.py) | JPEG reading, RGB conversion, center-cropping, 64x64 resizing, class counts, and real-image grid creation. |
| [`src/models.py`](src/models.py) | ResNet-18 classifier, conditional generator, DCGAN discriminator, and WGAN-GP critic. |
| [`src/train_gan.py`](src/train_gan.py) | Training loop for conditional DCGAN and WGAN-GP. |
| [`src/generate_synthetic_images.py`](src/generate_synthetic_images.py) | Checkpoint loading and class-conditioned synthetic image generation. |
| [`src/prepare_final_synthetic_sets.py`](src/prepare_final_synthetic_sets.py) | Deterministic selection of the final 669-image DCGAN and WGAN-GP top-up sets. |
| [`src/train_clf.py`](src/train_clf.py) | E0/E2/E3/E4 classifier training, validation selection, and test evaluation. |
| [`src/metrics.py`](src/metrics.py) | Macro-F1, accuracy, balanced accuracy, AUC, recall, and confusion matrices. |
| [`src/evaluate_fid.py`](src/evaluate_fid.py) | Overall and class-wise FID comparison. |
| [`src/analyze_synthetic_diversity.py`](src/analyze_synthetic_diversity.py) | Within-class pixel-distance diversity analysis. |
| [`src/analyze_gan_memorization.py`](src/analyze_gan_memorization.py) | Feature-space nearest-neighbour memorization-proxy analysis. |
| [`scripts/phase9_analysis.py`](scripts/phase9_analysis.py) | Final aggregation of classifier metrics, confusion matrices, recalls, and plots. |

## 2. End-to-end pipeline

```text
HAM10000 metadata and images
        |
        v
Validate labels, image IDs, and lesion groups
        |
        v
Create one fixed lesion-grouped train/validation/test split
        |
        v
Center-crop and resize every image to 64x64 RGB
        |
        v
Save aligned cache, labels, split flags, class mapping, and class counts
        |
        +------------------------------+
        |                              |
        v                              v
Train ResNet-18 baseline        Train conditional DCGAN
(E0)                            and conditional WGAN-GP
                                       |
                                       v
                         Generate class-conditioned image pools
                                       |
                                       v
                         FID, visual, diversity, and
                         nearest-neighbour checks
                                       |
                                       v
                         Select equal-count top-up sets
                                       |
        +------------------------------+
        |
        v
Run E2, E3, and E4 using the same classifier protocol
        |
        v
Evaluate on the unchanged real test set
        |
        v
Aggregate metrics, confusion matrices, recalls, plots,
and the final research interpretation
```

The implementation and output locations for every stage are described below.

## 3. Stage 1 - Dataset definition and label protocol

### 3.1 Dataset

The project uses the HAM10000 dermoscopic image dataset:

- 10,015 images
- Seven diagnostic classes
- Source JPEG images and metadata under [`data/`](data/)
- Metadata file: [`data/HAM10000_metadata`](data/HAM10000_metadata)

The required metadata fields are `image_id`, `lesion_id`, and `dx`.
`image_id` identifies an image, while `lesion_id` identifies the underlying
lesion. The lesion ID is important because one lesion can have more than one
image; images from the same lesion must not be split across train,
validation, and test.

### 3.2 Fixed class mapping

Every later component uses the same label order:

| Label ID | Diagnosis |
| ---: | --- |
| 0 | `akiec` |
| 1 | `bcc` |
| 2 | `bkl` |
| 3 | `df` |
| 4 | `mel` |
| 5 | `nv` |
| 6 | `vasc` |

The mapping is stored in [`cache/label_mapping.json`](cache/label_mapping.json).
It is also defined in [`src/data.py`](src/data.py), where it is used by both
the classifier and GAN data loaders.

### 3.3 Data validation

The data-loading and split implementation is in
[`src/data.py`](src/data.py). Before the split is created, it checks:

- The metadata file exists and is uniquely identified.
- Required metadata columns exist.
- Required values are not missing.
- `image_id` values are unique.
- Every diagnosis is one of the seven known classes.
- A single `lesion_id` does not have conflicting diagnosis labels.

## 4. Stage 2 - Leakage-free split and image preprocessing

### 4.1 Grouped stratified split

The split is created once with seed 42 using
`StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)`.
The folds are combined as follows:

- Folds 0, 1, and 2 become the test split.
- Folds 3, 4, and 5 become the validation split.
- All remaining folds become the training split.

This produces an approximately 70%/15%/15% image split while preserving
lesion-level grouping. The resulting manifest is
[`cache/splits.csv`](cache/splits.csv), with columns:

```text
image_id, lesion_id, dx, split
```

The finalized split contains:

- Train: 7,009 images
- Validation: 1,503 images
- Test: 1,503 images

The verification checks confirmed:

- Cache row order equals manifest row order.
- Cached image IDs equal manifest image IDs.
- Cached split flags equal manifest split flags.
- Train, validation, and test lesion-ID sets are pairwise disjoint.
- All seven classes occur in all three splits.

The split is reused unchanged by GAN training, classifier training, and
quality evaluation. It must not be regenerated for individual experiments.

### 4.2 Image preprocessing

The preprocessing implementation is
[`src/preprocess.py`](src/preprocess.py). For each JPEG:

1. Read the image with OpenCV.
2. Convert BGR to RGB.
3. Center-crop the image to a square using the smaller image dimension.
4. Resize the square to 64x64 using `INTER_AREA`.
5. Store the result as an RGB `uint8` array.

For the usual 600x450 source geometry, the crop removes 75 pixels from each
side of the width before resizing.

The cached file [`cache/ham64.npz`](cache/ham64.npz) contains:

- `images`: `(10015, 64, 64, 3)`, `uint8`
- `labels`: `(10015,)`, integer label IDs
- `split`: `(10015,)`, split names
- `image_ids`: `(10015,)`, source image IDs

The classifier later upsamples the cached 64x64 image to 224x224 and applies
ImageNet normalization. The GAN uses the cached 64x64 image directly and
converts pixels from `[0, 255]` to `[-1, 1]`.

### 4.3 Imbalance measurement

The training class counts are written to
[`cache/train_class_counts.csv`](cache/train_class_counts.csv):

| Diagnosis | Train images |
| --- | ---: |
| `akiec` | 228 |
| `bcc` | 359 |
| `bkl` | 769 |
| `df` | 80 |
| `mel` | 779 |
| `nv` | 4,694 |
| `vasc` | 100 |

The imbalance determines which classes receive additional data. The real-image
visual sanity grid is [`cache/train_real_grid.png`](cache/train_real_grid.png).

## 5. Stage 3 - Classifier methodology and baseline E0

### 5.1 Classifier architecture

The classifier is created by `build_classifier()` in
[`src/models.py`](src/models.py):

- Torchvision ResNet-18
- ImageNet `IMAGENET1K_V1` pretrained weights
- Final fully connected layer replaced with `Linear(512, 7)`
- All layers remain trainable

The dataset, transforms, and train-only extra-image handling are implemented in
[`src/data.py`](src/data.py). The training loop is in
[`src/train_clf.py`](src/train_clf.py).

### 5.2 Fixed training protocol

The following controls are kept the same for E0, E2, E3, and E4:

- Cross-entropy loss
- Adam optimizer
- Learning rate `1e-4`
- Weight decay `0`
- Batch size `64`
- Plain shuffled training batches
- Maximum 20 epochs
- Early stopping patience: 5 epochs
- Model-selection metric: validation macro-F1
- Three random seeds: 0, 1, and 2
- Image input: cached 64x64 image resized to 224x224
- ImageNet mean `(0.485, 0.456, 0.406)`
- ImageNet standard deviation `(0.229, 0.224, 0.225)`

The test split is never used for training, early stopping, or checkpoint
selection. The selected checkpoint is the epoch with the best validation
macro-F1, and that checkpoint is evaluated once on the real test set.

### 5.3 E0 baseline

E0 is configured in [`configs/e0.yaml`](configs/e0.yaml). It uses only the
7,009 real training images. Validation and test contain only real images.

The per-seed E0 outputs are under:

- [`results/E0/seed_0/`](results/E0/seed_0/)
- [`results/E0/seed_1/`](results/E0/seed_1/)
- [`results/E0/seed_2/`](results/E0/seed_2/)

Each seed produces a training log, selected model, test metrics, and test
confusion matrix.

## 6. Stage 4 - Conditional GAN methodology

The shared GAN training implementation is
[`src/train_gan.py`](src/train_gan.py). Both GANs use:

- Only the fixed training split
- Seven class labels
- 64x64 RGB images
- Class-balanced sampling through `WeightedRandomSampler`
- Batch size 64
- 200 epochs
- Class embeddings of dimension 50
- Latent noise dimension 100
- Generator initialization with normal standard deviation 0.02
- Fixed-noise image grids for comparing checkpoints
- Checkpoints and CSV training logs

The balanced sampler is used only for GAN training. The classifier uses plain
shuffling so that the classifier comparison measures the effect of the
additional data rather than a changed sampling strategy.

### 6.1 Shared conditional generator

`ConditionalGenerator` is defined in
[`src/models.py`](src/models.py). It:

1. Receives a 100-dimensional noise vector.
2. Looks up a learned 50-dimensional embedding for the requested class.
3. Concatenates noise and class embedding.
4. Projects the result into a 4x4x512 feature map.
5. Upsamples through transposed-convolution blocks:
   4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64.
6. Uses BatchNorm and ReLU in the intermediate generator blocks.
7. Uses `Tanh` at the output, producing RGB values in `[-1, 1]`.

The class label therefore controls the requested diagnostic category while the
noise vector controls variation within that category.

### 6.2 Conditional DCGAN

The revised DCGAN configuration is
[`configs/gan_dcgan_aug.yaml`](configs/gan_dcgan_aug.yaml). Its discriminator
is `ConditionalDiscriminator` in [`src/models.py`](src/models.py):

1. Embed the class label.
2. Project the embedding to a 64x64 label map.
3. Concatenate the label map with the three image channels, producing four
   input channels.
4. Apply four strided convolution blocks with LeakyReLU and BatchNorm after
   the first block.
5. Output one logit per image.

Training uses:

- Binary cross-entropy with logits.
- Adam betas `(0.5, 0.999)`.
- Generator learning rate `1e-4`.
- Discriminator learning rate `1e-4`.
- Real-label smoothing to `0.9`.
- Fake label `0.0`.
- DiffAugment on discriminator inputs with the policy
  `color,translation,cutout`.

The active 200-epoch DCGAN results and checkpoints are located at
[`results/GAN_DCGAN_AUG/`](results/GAN_DCGAN_AUG/) and
[`checkpoints/GAN_DCGAN_AUG/`](checkpoints/GAN_DCGAN_AUG/).

### 6.3 Conditional WGAN-GP

The WGAN-GP configuration is
[`configs/gan_wgangp.yaml`](configs/gan_wgangp.yaml). It uses the same
generator and conditional label injection as the DCGAN, so the main intended
difference is the adversarial objective and critic.

`ConditionalCritic` in [`src/models.py`](src/models.py):

- Has no sigmoid output.
- Returns an unrestricted scalar score.
- Uses the same conditional label-map mechanism.
- Uses GroupNorm with one group as layer-normalization behaviour.
- Does not use BatchNorm, which avoids problematic batch dependence during
  gradient-penalty calculation.

Training uses:

- Wasserstein critic objective.
- Five critic updates per generator update.
- Gradient penalty coefficient `10`.
- Adam betas `(0.0, 0.9)`.
- Generator learning rate `1e-4`.
- Critic learning rate `1e-4`.
- Balanced class sampling.
- No DiffAugment in the configured WGAN-GP run.

For each critic update, the code interpolates real and generated images,
computes the critic gradient with respect to the interpolated image, and
penalizes the squared difference between the per-image gradient norm and one.
The log records critic loss, generator loss, Wasserstein estimate, and mean
gradient norm.

The active WGAN-GP results and checkpoints are located at
[`results/GAN_WGANGP/`](results/GAN_WGANGP/) and
[`checkpoints/GAN_WGANGP/`](checkpoints/GAN_WGANGP/).

## 7. Stage 5 - Generate and assess candidate synthetic images

### 7.1 Candidate generation

The generation implementation is
[`src/generate_synthetic_images.py`](src/generate_synthetic_images.py).
For each selected checkpoint:

1. Load the saved generator weights and configuration.
2. Set the generator to evaluation mode.
3. Iterate over the seven class IDs.
4. Draw random noise vectors.
5. Repeat one class label for the batch.
6. Generate images in `[-1, 1]`.
7. Convert them to RGB `uint8` images in `[0, 255]`.
8. Save PNG images by class and a consolidated NPZ with metadata.

The quality-evaluation collections contain 1,000 images per class for:

- Revised DCGAN at epochs 150 and 200:
  [`results/fid_evaluation_dcgan_aug/`](results/fid_evaluation_dcgan_aug/)
- WGAN-GP at epochs 150 and 200:
  [`results/fid_evaluation/`](results/fid_evaluation/)

The same number of images and class conditions are used for all candidate
checkpoints.

### 7.2 FID evaluation

FID is implemented in [`src/evaluate_fid.py`](src/evaluate_fid.py). For each
candidate, the generated images are compared with real training images:

- Once using all real training images.
- Separately within each diagnostic class.
- With the same image format and evaluation settings for every candidate.

The comparison table is
[`results/fid_evaluation_active/fid_results_table.csv`](results/fid_evaluation_active/fid_results_table.csv).

FID is treated as a relative distribution-comparison measure. It is not a
direct measure of clinical realism, and rare-class FID estimates are noisy
because the real reference sets are small.

### 7.3 Visual and diversity evaluation

Real-versus-synthetic visual comparisons are in
[`results/visual_comparison_active/`](results/visual_comparison_active/).

The diversity implementation is
[`src/analyze_synthetic_diversity.py`](src/analyze_synthetic_diversity.py).
It samples 100 images per class and computes normalized raw-pixel pairwise
distances within each class. Results are in
[`results/synthetic_diversity_active/`](results/synthetic_diversity_active/).
This measures pixel-level variation; it does not establish medical
authenticity.

### 7.4 Nearest-neighbour memorization proxy

The memorization analysis is implemented in
[`src/analyze_gan_memorization.py`](src/analyze_gan_memorization.py):

1. Resize real and generated images to 224x224.
2. Apply ImageNet normalization.
3. Extract penultimate-layer ResNet-18 features.
4. L2-normalize the features.
5. For each synthetic image, find the nearest same-class real training image
   using cosine distance.
6. Compare the distribution with a real-validation-to-training baseline.
7. Save distance distributions and closest-image pairs.

Outputs are in [`results/memorization_check_active/`](results/memorization_check_active/).
This is evidence about similarity and possible memorization risk, not a
formal proof of copying or non-copying.

## 8. Stage 6 - Build the final synthetic top-up sets

The final-set preparation implementation is
[`src/prepare_final_synthetic_sets.py`](src/prepare_final_synthetic_sets.py).

The target rule is fixed before classifier training:

- Use the BCC training count, 359, as the target.
- Top up only `akiec`, `df`, and `vasc`.
- Do not generate top-up images for `bcc`, `bkl`, `mel`, or `nv`.
- Do not apply a classifier quality filter.
- Sample without replacement from the selected epoch-200 generated pools.
- Use the same target counts for DCGAN and WGAN-GP.

The required additions are:

| Class | Real train count | Synthetic additions | Final count |
| --- | ---: | ---: | ---: |
| `akiec` | 228 | 131 | 359 |
| `df` | 80 | 279 | 359 |
| `vasc` | 100 | 259 | 359 |

The resulting files are:

- [`cache/synthetic_dcgan.npz`](cache/synthetic_dcgan.npz)
- [`cache/synthetic_wgangp.npz`](cache/synthetic_wgangp.npz)

Each contains 669 images, labels, and the class-name array. The preparation
metadata and final grids are in
[`results/final_synthetic_sets/`](results/final_synthetic_sets/).

## 9. Stage 7 - Controlled classifier experiments

The four experiment configurations are:

- Baseline: [`configs/e0.yaml`](configs/e0.yaml)
- Classical augmentation: [`configs/e2.yaml`](configs/e2.yaml)
- DCGAN augmentation: [`configs/e3.yaml`](configs/e3.yaml)
- WGAN-GP augmentation: [`configs/e4.yaml`](configs/e4.yaml)

### 9.1 E2 - Classical augmentation

E2 creates extra examples from real training images only. For each selected
class, the code calculates:

```text
required additions = BCC training count - current class count
```

It randomly selects real training images and applies:

- Horizontal flip
- Vertical flip
- Random rotation up to 30 degrees
- Colour jitter:
  - brightness 0.2
  - contrast 0.2
  - saturation 0.2
  - hue 0.05

Only the generated top-up examples are transformed. The original real
training images remain available, and validation/test are untouched.

### 9.2 E3 - DCGAN augmentation

E3 loads [`cache/synthetic_dcgan.npz`](cache/synthetic_dcgan.npz) and appends
its 669 labelled images to the real training dataset. The original real
training images are not classically augmented. The configured source
checkpoint is the epoch-200 revised DCGAN checkpoint:

[`checkpoints/GAN_DCGAN_AUG/seed_0/dcgan_epoch_200.pt`](checkpoints/GAN_DCGAN_AUG/seed_0/dcgan_epoch_200.pt)

### 9.3 E4 - WGAN-GP augmentation

E4 follows the same classifier protocol as E3 but loads
[`cache/synthetic_wgangp.npz`](cache/synthetic_wgangp.npz). Its configured
source checkpoint is:

[`checkpoints/GAN_WGANGP/seed_0/wgangp_epoch_200.pt`](checkpoints/GAN_WGANGP/seed_0/wgangp_epoch_200.pt)

E3 and E4 add the same number of images with the same class counts. Their
intended difference is the generator that produced the synthetic images.

### 9.4 Training and evaluation sequence for each seed

For each experiment and seed:

1. Load the fixed cached training split.
2. Add the experiment-specific E2, E3, or E4 examples if applicable.
3. Build a shuffled training loader.
4. Initialize a new ImageNet-pretrained ResNet-18.
5. Train using the fixed optimizer and loss settings.
6. Evaluate macro-F1 on the real validation split after every epoch.
7. Save the checkpoint with the best validation macro-F1.
8. Stop after five validation epochs without improvement or after 20 epochs.
9. Evaluate the selected checkpoint once on the unchanged real test split.
10. Write metrics and confusion matrices.

Formal output directories are:

- [`results/E0/`](results/E0/)
- [`results/E2/`](results/E2/)
- [`results/E3/`](results/E3/)
- [`results/E4/`](results/E4/)

The aggregate file [`results/metrics.csv`](results/metrics.csv) contains 12
rows: four experiments multiplied by three seeds.

## 10. Evaluation metrics and analysis

Metrics are calculated by [`src/metrics.py`](src/metrics.py) from the model
logits:

- **Macro-F1**: primary metric; gives equal weight to every class.
- **Accuracy**: overall correct-prediction rate; included but not sufficient
  for an imbalanced dataset.
- **Balanced accuracy**: mean recall across the seven classes.
- **Per-class recall**: identifies which minority classes improve or degrade.
- **Macro one-vs-rest ROC-AUC**: measures score ranking across classes.
- **Confusion matrix**: records true-versus-predicted class counts.

The Phase 9 aggregation script is
[`scripts/phase9_analysis.py`](scripts/phase9_analysis.py). It reads existing
results and creates:

- Mean and sample standard deviation across the three seeds.
- Per-class recall tables.
- Summed and row-normalized confusion matrices.
- MEL/NV confusion analysis.
- Per-class recall plots.
- Training-loss and validation curves.
- A FID artifact inventory.

The output bundle is [`results/phase9_analysis/`](results/phase9_analysis/).
The analysis notes and interpretation guardrails are in
[`results/phase9_analysis/README_phase9.md`](results/phase9_analysis/README_phase9.md).

## 11. Verified results

The verified aggregate results are:

| Experiment | Macro-F1 mean +/- SD | Accuracy mean +/- SD | Balanced accuracy mean +/- SD | AUC mean +/- SD |
| --- | ---: | ---: | ---: | ---: |
| E0 | 0.6345 +/- 0.0215 | 0.8004 +/- 0.0074 | 0.6068 +/- 0.0414 | 0.9343 +/- 0.0066 |
| E2 | 0.6559 +/- 0.0154 | 0.8035 +/- 0.0097 | 0.6291 +/- 0.0187 | 0.9352 +/- 0.0123 |
| E3 | 0.6124 +/- 0.0130 | 0.7949 +/- 0.0037 | 0.5979 +/- 0.0123 | 0.9361 +/- 0.0026 |
| E4 | 0.6515 +/- 0.0095 | 0.8028 +/- 0.0017 | 0.6331 +/- 0.0025 | 0.9347 +/- 0.0070 |

The observed mean recalls show class-specific effects:

- E2 and E4 improve DF recall over E0.
- E2 and E4 improve VASC recall over E0.
- E4 has the highest observed balanced accuracy.
- E2 has the highest observed macro-F1.
- MEL recall is lower in all three augmentation experiments than in E0.
- E3 does not improve the aggregate metrics in this configuration.

The project decision rule compares the mean macro-F1 gap with the baseline
seed variability. Under that heuristic, neither GAN augmentation demonstrates
a measurable overall macro-F1 gain. This is a project-level comparison, not a
formal statistical significance test.

## 12. Phase 8 status

Phase 8 was skipped. The following extensions are not part of the verified
results and must not be presented as completed:

1. E6 synthetic top-up sweep.
2. E5 GAN plus classical augmentation.
3. E1 class-weighted loss.
4. E7 synthetic-only rare-class training.
5. Grad-CAM comparison.

## 13. Reproducibility and verification

The project virtual environment is used for verification:

```powershell
Set-Location E:\udemy\ADL\project
.\.venv\Scripts\python.exe
```

Verified checks:

```powershell
# Compile all project code
$files = Get-ChildItem .\src, .\scripts -Filter *.py -File |
    Select-Object -ExpandProperty FullName
.\.venv\Scripts\python.exe -m py_compile $files

# Import the main pipeline components
@'
from src.data import HAMClassifierDataset
from src.models import (
    build_classifier, ConditionalGenerator,
    ConditionalDiscriminator, ConditionalCritic,
)
from src.metrics import calculate_metrics
from src.train_clf import run_experiment
from src.train_gan import HANGANTrainingDataset, gradient_penalty
print("imports OK")
'@ | .\.venv\Scripts\python.exe -

# Recreate the analysis tables and figures from existing outputs
.\.venv\Scripts\python.exe .\scripts\phase9_analysis.py
```

The source code also protects formal classifier results from overwriting
existing experiment directories or duplicating an experiment ID in
[`results/metrics.csv`](results/metrics.csv).

## 14. Final evaluator-facing conclusion

The project implements a complete, controlled pipeline for testing synthetic
data augmentation on an imbalanced skin-lesion classifier. The data split is
fixed at lesion level, preprocessing is shared, validation and test data remain
real-only, and all classifier experiments use the same architecture and
optimization settings.

The results do not support the claim that GAN augmentation improves overall
macro-F1 in this configuration. Classical augmentation gives the highest
observed macro-F1, while WGAN-GP gives the highest observed balanced accuracy
and improves selected minority-class recall. These gains are accompanied by
class-specific trade-offs, including reduced MEL recall. The appropriate
conclusion is therefore a controlled and informative comparison, not a claim
of universal GAN superiority.
