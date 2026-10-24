# Phase 7 — Core Classifier Experiments (E0, E2, E3, E4)

## 1. Phase 7 objective

Phase 7 was designed to determine whether addressing class imbalance through classical augmentation or GAN-generated synthetic images improves skin-lesion classification compared with the baseline classifier.

The four experiments were:

| Experiment | Training data                                                                | Purpose                                                                    |
| ---------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| E0         | Real training images only, with the baseline training pipeline               | Establish baseline performance                                             |
| E2         | Real training data plus classical augmentation for selected minority classes | Test whether conventional augmentation improves minority-class performance |
| E3         | Real training data plus DCGAN-generated synthetic images                     | Test DCGAN-based augmentation                                              |
| E4         | Real training data plus WGAN-GP-generated synthetic images                   | Test WGAN-GP-based augmentation                                            |

The evaluation uses the same fixed real-data train/validation/test split. The validation and test sets remain real-only, so synthetic images do not directly enter evaluation.

The primary metric is macro-F1, because the dataset is imbalanced and performance on minority classes matters. Balanced accuracy, ROC-AUC and per-class recall provide complementary views of performance.

The goal at the end of Phase 7 was to complete three-seed runs for E2, E3 and E4, compare them against E0, and produce the headline results table.

## 2. Dataset and experimental setup

### 2.1 Dataset

The project uses the HAM10000 skin-lesion dataset.

- Total images: 10,015.
- Number of classes: 7.
- Images are cached at \\(64 \times 64\\) RGB.
- The classifier receives images upsampled to \\(224 \times 224\\).
- A fixed split is stored in `cache/splits.csv`.
- Training set: 7,009 images.
- Validation set: 1,503 images.
- Test set: 1,503 images.

The class mapping used throughout the project is:

| Class | ID | Training images |
| ----- | -- | --------------- |
| AKIEC | 0  | 228             |
| BCC   | 1  | 359             |
| BKL   | 2  | 769             |
| DF    | 3  | 80              |
| MEL   | 4  | 779             |
| NV    | 5  | 4,694           |
| VASC  | 6  | 100             |

The imbalance is substantial. NV accounts for most training images, while DF, VASC and AKIEC have much fewer examples.

### 2.2 Classifier and training controls

The core classifier pipeline used the E0 baseline settings:

- ResNet18 with ImageNet-pretrained initialization.
- Cross-entropy loss.
- Adam optimizer with learning rate \\(10^{-4}\\).
- Early stopping based on validation macro-F1, with patience 5.
- Three seeds: 0, 1 and 2.
- A maximum of 20 epochs.
- The same fixed data split across experiments.

For each seed, the best checkpoint was selected using validation macro-F1, and the selected checkpoint was evaluated on the real test split.

Keeping the classifier architecture, initialization approach, optimizer, split and evaluation protocol consistent makes the comparison more meaningful: the main intended difference is how the training data is augmented.

### 2.3 Synthetic-data preparation decisions

Before Phase 7, the GAN experiments had produced candidate synthetic image sets from the 200-epoch DCGAN-Aug and WGAN-GP runs.

The synthetic sets were prepared with the following decisions:

- The target count was based on the BCC training count of 359.
- Only AKIEC, DF and VASC were synthetically topped up.
- The selected extra counts were 131 AKIEC, 279 DF and 259 VASC images.
- Each GAN-derived set therefore contained 669 synthetic images.
- The synthetic images were sampled from the available generated pools without replacement.
- The classifier-based quality filter was disabled; no classifier rejection step was applied.
- The original synthetic pools and prepared sets were kept separate.

The prepared files were:

- `cache/synthetic_dcgan.npz`
- `cache/synthetic_wgangp.npz`

Each contained 669 RGB images at \\(64 \times 64\\), corresponding labels, and class-name metadata. The same selected class counts were used for the two GAN variants.

This ensured that the DCGAN and WGAN-GP experiments differed in synthetic-image source, rather than in the number of synthetic samples or which classes were targeted.

## 3. Decisions made for each experiment

### E0 — Baseline

E0 is the reference experiment. It uses the real training data and the baseline classifier pipeline. Its results establish how well the model performs without the Phase 7 augmentation interventions.

The E0 results were retained in `results/metrics.csv`; they were not replaced by the later experiments.

### E2 — Classical augmentation

Configuration: `configs/e2.yaml`

E2 applies conventional augmentation to the selected rare classes. The choices were:

- Horizontal flips.
- Vertical flips.
- Random rotations up to 30 degrees.
- Colour jitter: brightness, contrast and saturation of 0.2, hue of 0.05.
- Augmentation restricted to the designated top-up classes.

The targeted classes were AKIEC, DF and VASC. The number of additional augmented examples was chosen so that each targeted class reached the BCC reference count of 359 when the additional examples were counted alongside its real training images.

Importantly, E2 generates its additional examples from the real training data. The extra examples are not added to validation or test data.

The implementation was corrected before the formal runs so that E2's classical top-up would not accidentally be applied to E3 or E4.

### E3 — DCGAN augmentation

Configuration: `configs/e3.yaml`

E3 adds the prepared DCGAN synthetic set to the real training data.

- Synthetic source: `cache/synthetic_dcgan.npz`.
- GAN checkpoint referenced by the configuration: `checkpoints/GAN_DCGAN_AUG/seed_0/dcgan_epoch_200.pt`.
- Number of synthetic training images: 669.
- Synthetic class counts: 131 AKIEC, 279 DF and 259 VASC.
- Classical top-up augmentation is not applied.

The GAN checkpoint documents the source model used to produce the synthetic pool. The classifier experiment consumes the already-prepared NPZ set.

### E4 — WGAN-GP augmentation

Configuration: `configs/e4.yaml`

E4 follows the same synthetic-data procedure as E3, but uses the WGAN-GP synthetic set.

- Synthetic source: `cache/synthetic_wgangp.npz`.
- GAN checkpoint referenced by the configuration: `checkpoints/GAN_WGANGP/seed_0/wgangp_epoch_200.pt`.
- Number of synthetic training images: 669.
- Synthetic class counts: 131 AKIEC, 279 DF and 259 VASC.
- Classical top-up augmentation is not applied.

E3 and E4 therefore use the same number of added images and the same targeted class counts. Their primary experimental difference is the generator used to create the synthetic images.

### Implementation and safety decisions

Before formal training, smoke tests were completed for E2, E3 and E4. These confirmed that the pipeline could execute and produce metrics, but smoke-test results were not treated as formal results.

The training implementation also included safeguards against accidentally overwriting existing formal experiment output directories or rerunning an experiment already recorded in the metrics CSV. Formal results were written to experiment-specific directories, with per-seed outputs.

The formal runs completed for all three experiments, with seeds 0, 1 and 2 for each. The metrics CSV contains three rows for each of E0, E2, E3 and E4.

## 4. Formal results: all 12 classifier runs

The following values are from the formal results recorded in `results/metrics.csv`, not the smoke tests.

### 4.1 Per-seed results

| Experiment | Seed | Macro-F1 | Accuracy | Balanced accuracy | AUC    |
| ---------- | ---- | -------- | -------- | ----------------- | ------ |
| E0         | 0    | 0.6370   | 0.7924   | 0.6288            | 0.9366 |
| E0         | 1    | 0.6546   | 0.8071   | 0.6325            | 0.9395 |
| E0         | 2    | 0.6118   | 0.8017   | 0.5590            | 0.9269 |
| E2         | 0    | 0.6622   | 0.8124   | 0.6333            | 0.9354 |
| E2         | 1    | 0.6384   | 0.7931   | 0.6087            | 0.9229 |
| E2         | 2    | 0.6672   | 0.8051   | 0.6454            | 0.9474 |
| E3         | 0    | 0.6026   | 0.7931   | 0.5876            | 0.9388 |
| E3         | 1    | 0.6074   | 0.7924   | 0.5947            | 0.9362 |
| E3         | 2    | 0.6271   | 0.7991   | 0.6115            | 0.9335 |
| E4         | 0    | 0.6424   | 0.8031   | 0.6314            | 0.9268 |
| E4         | 1    | 0.6507   | 0.8011   | 0.6320            | 0.9374 |
| E4         | 2    | 0.6614   | 0.8044   | 0.6360            | 0.9399 |

### 4.2 Headline table: mean and standard deviation across seeds

The standard deviation below is the sample standard deviation across the three seeds. For compactness, all values are rounded to four decimal places.

| Experiment     | Macro-F1        | Accuracy        | Balanced accuracy | AUC             |
| -------------- | --------------- | --------------- | ----------------- | --------------- |
| E0 — Baseline  | 0.6345 ± 0.0215 | 0.8004 ± 0.0075 | 0.6068 ± 0.0413   | 0.9343 ± 0.0065 |
| E2 — Classical | 0.6559 ± 0.0154 | 0.8035 ± 0.0097 | 0.6291 ± 0.0184   | 0.9352 ± 0.0123 |
| E3 — DCGAN     | 0.6124 ± 0.0130 | 0.7949 ± 0.0037 | 0.5979 ± 0.0120   | 0.9361 ± 0.0027 |
| E4 — WGAN-GP   | 0.6515 ± 0.0095 | 0.8028 ± 0.0017 | 0.6331 ± 0.0025   | 0.9347 ± 0.0070 |

Interpretation: Each cell reports mean ± sample standard deviation. These summarize variability across three training seeds, not uncertainty across independently collected datasets.

The small differences between some metrics should not be overinterpreted. With only three seeds, these results provide an initial comparison, not a definitive statistical ranking.

## 5. What the headline metrics tell us

### 5.1 Macro-F1: the primary outcome

Macro-F1 calculates F1 separately for each class and then averages across classes. It gives each class equal weight, unlike accuracy, which can be dominated by the majority class.

E0 — Baseline

0.6345 ± 0.0215

E2 — Classical

0.6559 ± 0.0154

E3 — DCGAN

0.6124 ± 0.0130

E4 — WGAN-GP

0.6515 ± 0.0095

Bars show mean macro-F1; the standard deviations are shown in the labels, not encoded in the bars.

E2 has the highest observed mean macro-F1. Its mean is about 0.0215 above E0.

E4 is about 0.0170 above E0, while E3 is about 0.0221 below E0.

Under the project's predefined decision rule, a difference is called measurable only if the absolute gap in mean macro-F1 is larger than the baseline seed standard deviation, approximately 0.0215.

- E2's gain is approximately equal to, but marginally smaller than, the unrounded E0 standard deviation. It therefore does not pass the rule.
- E4's gain is smaller than the E0 standard deviation. It does not pass the rule.
- E3 has lower mean macro-F1 than E0. It does not demonstrate an improvement.

The correct conclusion under the agreed rule is “no measurable macro-F1 gain.” This does not mean the methods are equivalent or that augmentation has no effect. It means the observed gains do not clear the project's chosen threshold.

This rule is a practical decision heuristic, not a formal significance test.

### 5.2 Accuracy

Accuracy measures the fraction of test images classified correctly.

The mean accuracy is between approximately 79.5% and 80.4% across all four experiments. E2 has the highest mean accuracy, but its advantage over E0 is small.

Because NV is the dominant class, a classifier can achieve fairly high accuracy while still performing poorly on some rare classes. Accuracy alone is not sufficient to judge whether the imbalance problem has been addressed.

### 5.3 Balanced accuracy

Balanced accuracy averages recall across the classes, giving each class equal weight.

- E4: 0.6331, highest mean.
- E2: 0.6291.
- E0: 0.6068.
- E3: 0.5979.

E2 and E4 both improve on the baseline mean balanced accuracy. E4 has the strongest mean result on this metric and notably improves recall for DF and BCC. However, balanced accuracy is still an aggregate measure; the per-class results explain where those changes come from.

### 5.4 ROC-AUC

The mean AUC values are close: approximately 0.9343–0.9361.

E3 has the highest mean AUC, at 0.9361, even though it has the lowest mean macro-F1 and balanced accuracy.

This is not contradictory. AUC measures how well model scores rank positive examples above negative examples across thresholds, while macro-F1 depends on the final class predictions at a particular decision rule. A model can rank examples reasonably well but still assign the wrong class labels, especially in a multiclass, imbalanced setting.

Consequently, E3's slightly higher AUC does not mean it is the best classifier overall.

## 6. Per-class recall analysis

Recall measures the proportion of examples belonging to a class that the model correctly identifies. For minority classes, it helps reveal whether augmentation makes the classifier less likely to miss those lesions.

The following values are means across the three seeds.

| Class | E0     | E2     | E3     | E4     | Best observed mean |
| ----- | ------ | ------ | ------ | ------ | ------------------ |
| AKIEC | 0.4898 | 0.4966 | 0.4762 | 0.4898 | E2                 |
| BCC   | 0.6838 | 0.6282 | 0.6838 | 0.7094 | E4                 |
| BKL   | 0.5455 | 0.5737 | 0.5818 | 0.5657 | E3                 |
| DF    | 0.2549 | 0.3725 | 0.2353 | 0.3922 | E4                 |
| MEL   | 0.4901 | 0.4841 | 0.4464 | 0.4405 | E0                 |
| NV    | 0.9264 | 0.9280 | 0.9207 | 0.9297 | E4                 |
| VASC  | 0.8571 | 0.9206 | 0.8413 | 0.9048 | E2                 |

Bold values indicate the highest observed mean for that class. These are descriptive comparisons, not evidence that the winner is statistically superior.

### 6.1 AKIEC

Recall changes only slightly. E2 has the highest mean at 0.4966, compared with 0.4898 for E0.

This is a very small improvement. Neither GAN experiment improves the mean relative to E0. The results do not show a strong or consistent AKIEC benefit from the augmentation methods tested.

### 6.2 BCC

E4 has the highest mean recall at 0.7094, compared with 0.6838 for E0.

E2 has lower BCC recall, at 0.6282. This is an important trade-off: adding examples for other rare classes does not guarantee that performance will improve for every class.

The E4 result is promising, but it needs to be interpreted alongside the other classes and the overall metrics.

### 6.3 BKL

All three augmentation experiments have higher mean BKL recall than E0:

- E2: 0.5737.
- E3: 0.5818.
- E4: 0.5657.
- E0: 0.5455.

E3 has the highest mean BKL recall despite having the lowest overall macro-F1. This shows that an experiment can benefit one class without improving performance across all seven classes.

### 6.4 DF

DF is the clearest example of a potentially useful minority-class improvement.

| Experiment | Mean DF recall | Change vs. E0            |
| ---------- | -------------- | ------------------------ |
| E0         | 0.2549         | —                        |
| E2         | 0.3725         | +11.76 percentage points |
| E3         | 0.2353         | −1.96 percentage points  |
| E4         | 0.3922         | +13.73 percentage points |

The baseline correctly identifies only about one quarter of DF test examples on average. E2 and E4 substantially improve the observed mean recall, while E3 performs worse than the baseline.

E4 is particularly promising here. However, its mean DF recall of 0.3922 still means that roughly 61% of actual DF cases are missed on average. The improvement is meaningful as a research direction, but the absolute performance remains limited.

Also, these are recall results. They do not tell us how many non-DF images were incorrectly classified as DF. Precision and confusion matrices are needed to evaluate that trade-off.

### 6.5 MEL

E0 has the highest mean MEL recall, at 0.4901. All augmentation experiments are lower, with E4 at 0.4405.

This is a key finding. Augmenting selected minority classes may redistribute the model's predictions in ways that help those classes but hurt another class. It is not enough to report only the classes that improved.

MEL performance deserves particular attention in the next error analysis because the augmentation experiments did not preserve its baseline recall.

### 6.6 NV

NV recall is high in every experiment, between 0.9207 and 0.9297.

E4 has the highest mean at 0.9297, but the differences are small. This is encouraging in the sense that minority-class augmentation did not cause an obvious collapse in NV recall. It does not, by itself, establish that the overall classifier has improved.

### 6.7 VASC

E2 has the highest mean VASC recall at 0.9206, compared with 0.8571 for E0. E4 also improves the mean to 0.9048, whereas E3 falls to 0.8413.

This is another case where classical augmentation performs especially well. The result suggests that conventional transformations of real images may be more useful for VASC than the DCGAN synthetic images used in this experiment.

It is still a result from three seeds and should be treated as an indication, not a final claim of superiority.

## 7. Comparing classical augmentation with GAN augmentation

One of the most important questions in this project is whether GAN-generated images provide an advantage over conventional augmentation.

### 7.1 E2 versus E3 versus E4

| Finding                                   | What it suggests                                                                                 |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------ |
| E2 has the highest mean macro-F1          | Classical augmentation is the strongest observed method on the primary metric.                   |
| E4 has the highest mean balanced accuracy | WGAN-GP augmentation is promising when evaluating class-balanced recall.                         |
| E3 has the lowest mean macro-F1           | The DCGAN synthetic set did not improve aggregate classification performance in this experiment. |
| E4 has the highest DF recall              | WGAN-GP augmentation is worth investigating for DF, a severely underrepresented class.           |
| E2 has the highest VASC recall            | Classical augmentation is particularly promising for VASC.                                       |
| E0 has the highest MEL recall             | Augmentation aimed at other classes may have trade-offs for MEL.                                 |
| AUC is similar across experiments         | The overall score-ranking ability is broadly similar, even though class predictions differ.      |

### 7.2 What can we conclude about the GANs?

We cannot conclude that GAN augmentation is useless. We can conclude something more specific:

With the GAN checkpoints, synthetic samples, class counts, classifier and evaluation setup used here, GAN augmentation did not demonstrate a measurable improvement in mean macro-F1 over the baseline under our predefined decision rule.

The comparison also does not establish that WGAN-GP is universally better than DCGAN. E4 performs better than E3 on mean macro-F1 and balanced accuracy in these runs, but the results come from one prepared synthetic set per GAN approach and three classifier seeds.

This is a comparison of the complete augmentation pipelines as implemented, not a universal comparison of the GAN architectures.

### 7.3 Why might classical augmentation perform better?

Several explanations are plausible, but our current results do not establish which one is responsible.

- Classical augmentation preserves real-image characteristics. Flips, rotations and colour jitter transform existing real examples instead of introducing entirely generated lesion appearances.
- Synthetic images may contain imperfect lesion features. Even visually plausible generated images might not add the right discriminative information for the classifier.
- More examples do not automatically mean more useful information. Synthetic samples may be redundant, noisy or insufficiently representative of the real test distribution.
- Class trade-offs may affect macro-F1. Improving DF or VASC recall while reducing MEL or BCC recall can limit the overall benefit.

Our earlier GAN quality analyses compared generated images and examined potential near duplicates and diversity proxies. Those analyses did not establish widespread obvious memorization, but they also could not prove that the synthetic images were free of memorization or that they were clinically faithful.

To distinguish these explanations, we would need further experiments, not just the aggregate metrics.

## 8. Applying the project's decision rule

The workflow specifies that a macro-F1 difference should be called real only if the gap in mean macro-F1 is larger than the seed standard deviation.

Using E0's sample standard deviation of approximately 0.0215 as the reference:

| Comparison with E0 | Mean macro-F1 difference | Decision under the rule              |
| ------------------ | ------------------------ | ------------------------------------ |
| E2 − E0            | +0.0215                  | Does not exceed the baseline SD      |
| E3 − E0            | −0.0221                  | Performance is lower; no improvement |
| E4 − E0            | +0.0170                  | Does not exceed the baseline SD      |

The E2 difference is very close to the threshold, so the unrounded values matter. The unrounded comparison does not exceed the E0 standard deviation.

The E3 difference has a magnitude slightly greater than the baseline standard deviation, but it is a negative difference. It therefore cannot support a gain claim; under the rule, it indicates a decline large enough to merit attention.

One important methodological note: the rule is an agreed project heuristic, not a conventional statistical significance test. Comparing a mean difference with one experiment's standard deviation does not directly quantify the uncertainty of that difference. A paired analysis across seeds or a bootstrap confidence interval over test predictions could provide additional context.

We should therefore distinguish these two statements:

- Supported: E2 and E4 have higher observed mean macro-F1 than E0, but neither establishes a measurable gain under the project's rule.
- Not supported: GAN augmentation has no effect, or classical and GAN augmentation are statistically equivalent.

## 9. Limitations of the current results

The headline results are useful, but several limitations must be acknowledged in the report.

### 9.1 Only three seeds

Three seeds provide a first estimate of run-to-run variability, but the standard deviations can be unstable. A different set of seeds could alter the ranking.

### 9.2 One fixed train/validation/test split

The fixed split is important for fair comparisons. However, the results may depend on the particular composition of the test set. They do not establish performance across other possible data splits or external datasets.

### 9.3 Recall alone does not capture all class errors

A high recall can come with many false-positive predictions. For example, improving DF recall is not necessarily an overall benefit if many non-DF lesions are incorrectly labelled DF.

Per-class precision, F1, confusion matrices and, where appropriate, sensitivity-specificity analysis should complement recall.

### 9.4 The data-level intervention is specific

The experiment tests one chosen configuration of classical augmentation and two prepared synthetic sets. It does not test every possible augmentation policy, GAN training strategy, sampling method or synthetic-data quality filter.

### 9.5 No independent clinical validation

HAM10000 test performance is not the same as clinical readiness. These are research classification results, not evidence that the model is suitable for diagnosing patients. External validation, data provenance review and clinical evaluation would be required for such claims.

## 10. Final Phase 7 deliverables and project state

The following outputs have been confirmed by our recorded work:

| Deliverable                                                   | Status                                                                                |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| E0 baseline results retained                                  | Complete                                                                              |
| E2 formal runs, seeds 0–2                                     | Complete                                                                              |
| E3 formal runs, seeds 0–2                                     | Complete                                                                              |
| E4 formal runs, seeds 0–2                                     | Complete                                                                              |
| Fixed split preserved                                         | Complete                                                                              |
| Synthetic sets prepared with matching class counts            | Complete                                                                              |
| Validation and test kept real-only                            | Complete                                                                              |
| Smoke tests for E2/E3/E4                                      | Complete                                                                              |
| Formal metrics appended to `results/metrics.csv`              | Complete                                                                              |
| Mean macro-F1, accuracy, balanced accuracy and AUC summarized | Complete                                                                              |
| Mean per-class recall calculated                              | Complete                                                                              |
| Preliminary decision-rule interpretation                      | Complete                                                                              |
| Paired comparison or bootstrap confidence interval            | Optional; not confirmed as run                                                        |
| Final report-ready headline table                             | Values assembled in this summary; a separate saved report file has not been confirmed |

The formal metrics CSV is the central source of truth for the recorded numeric results. The summary tables above can be used in the Phase 7 report.

## 11. Overall conclusion for the research project

Phase 7 completed the central experiment: comparing the baseline classifier with classical augmentation, DCGAN-based augmentation and WGAN-GP-based augmentation across three seeds.

The results do not establish a measurable macro-F1 gain from either GAN augmentation method under the project's predefined decision rule. Classical augmentation achieves the highest mean macro-F1, while WGAN-GP augmentation achieves the highest mean balanced accuracy. Both E2 and E4 improve the observed mean recall for DF and VASC, although the extent of improvement differs by class. DCGAN augmentation has the lowest mean macro-F1 despite a similar AUC, demonstrating why AUC alone is insufficient to judge the experiment.

The most important finding is that augmentation has class-dependent effects rather than providing a uniform improvement across all classes. WGAN-GP augmentation appears promising for DF and BCC recall, classical augmentation appears promising for VASC, and MEL recall declines in all three augmentation experiments. These patterns motivate a more detailed error analysis, but they should not be presented as statistically established superiority.

The defensible research conclusion is:

> Under the evaluated configuration, neither GAN-based augmentation method demonstrated a measurable improvement in mean macro-F1 over the real-data baseline according to the project's predefined decision rule. Classical augmentation achieved the highest observed mean macro-F1, while WGAN-GP augmentation achieved the highest mean balanced accuracy and DF recall. The class-specific results indicate potential benefits for selected minority classes, accompanied by trade-offs in other classes. Further analysis of precision, confusion matrices and uncertainty is needed before making stronger claims about the comparative effectiveness of the augmentation methods.

Phase 7's core gate is complete. The optional paired comparison or bootstrap analysis could strengthen the interpretation, but the workflow explicitly treats it as optional. The next project steps can proceed without claiming that GAN augmentation has already been proven superior.



You actually have quite a lot to show the evaluator. The important thing is to distinguish between proving that your proposed method works better and demonstrating that you conducted a rigorous experiment and learned something meaningful from the results.

Your Phase 7 results do not support the claim that GAN augmentation improves overall classification performance. But that does not make the project a failure. A research project is not required to produce a positive result; it needs a well-defined question, a defensible methodology, evidence, and an honest interpretation.

For your project on imbalanced skin-lesion classification, here is what you have and how I would present it.

## 1. What you can demonstrate to the evaluator

A. A complete experimental pipeline

You have a fixed HAM10000 data split, a ResNet18 baseline, preprocessing, training and validation, early stopping, test evaluation, and metric logging. You can demonstrate that the system runs end to end rather than showing only a model architecture or a theoretical proposal.

B. Three controlled augmentation experiments

You compared real-data baseline training, classical augmentation, DCGAN synthetic augmentation, and WGAN-GP synthetic augmentation. Each experiment has three classifier seeds, giving 12 formal runs across E0, E2, E3 and E4.

C. A synthetic-data generation and evaluation pipeline

You trained the GAN candidates, prepared labelled synthetic image sets, and performed image-quality, FID, memorization-proxy and diversity analyses. This gives you evidence to show about how the synthetic data was generated and assessed, not just how the classifier scored.

D. Quantitative results and class-level analysis

You have test macro-F1, accuracy, balanced accuracy, AUC and per-class recall for every seed. These reveal trade-offs that overall accuracy would hide, particularly for DF, VASC, BCC and MEL.

E. An evidence-based negative finding

Your experiments indicate that better-looking or more diverse synthetic data does not automatically translate into better downstream classification. This is a useful finding, provided you state it within the limits of your particular experimental setup.

## 2. Your strongest result is not a claim of GAN superiority

Your primary results are:

| Experiment                  | Mean macro-F1 | Mean balanced accuracy |
| --------------------------- | ------------- | ---------------------- |
| E0 — Real-data baseline     | 0.6345        | 0.6068                 |
| E2 — Classical augmentation | 0.6559        | 0.6291                 |
| E3 — DCGAN augmentation     | 0.6124        | 0.5979                 |
| E4 — WGAN-GP augmentation   | 0.6515        | 0.6331                 |

The mean macro-F1 gap between E2 and E0 is approximately 0.0215, essentially the same size as the baseline's seed standard deviation. It narrowly fails your predefined decision rule. E4's macro-F1 gain is smaller still, while E3 performs worse on both macro-F1 and balanced accuracy.

This means you should not tell the evaluator that GAN augmentation significantly improved the model.

Instead, you can show that:

- Classical augmentation produced the highest observed mean macro-F1.
- WGAN-GP augmentation produced the highest observed mean balanced accuracy.
- DCGAN augmentation did not improve the aggregate metrics in this setup.
- Different augmentation methods affected different classes differently.

That is a defensible comparative result.

## 3. The most interesting evidence: minority-class recall

This is where your project becomes more informative than a simple comparison of four accuracy scores.

| Class | Baseline E0 | Classical E2 | DCGAN E3 | WGAN-GP E4 |
| ----- | ----------- | ------------ | -------- | ---------- |
| AKIEC | 0.4898      | 0.4966       | 0.4762   | 0.4898     |
| BCC   | 0.6838      | 0.6282       | 0.6838   | 0.7094     |
| BKL   | 0.5455      | 0.5737       | 0.5818   | 0.5657     |
| DF    | 0.2549      | 0.3725       | 0.2353   | 0.3922     |
| MEL   | 0.4901      | 0.4841       | 0.4464   | 0.4405     |
| NV    | 0.9264      | 0.9280       | 0.9207   | 0.9297     |
| VASC  | 0.8571      | 0.9206       | 0.8413   | 0.9048     |

These are mean recalls across three seeds.

Three observations are worth highlighting:

1\. DF recall improved under classical and WGAN-GP augmentation. The baseline recall is 25.49%, compared with 37.25% for E2 and 39.22% for E4. This is promising for a class with only 80 real training images. However, the absolute recall remains limited.

2\. VASC recall improved under E2 and E4. The baseline is 85.71%, compared with 92.06% for E2 and 90.48% for E4.

3\. MEL recall declined in all augmentation experiments. The baseline is 49.01%, while E2, E3 and E4 achieve 48.41%, 44.64% and 44.05%, respectively.

This third observation is important. Your results suggest that the effect of augmentation is not uniformly beneficial. Improving recognition of selected classes can coincide with poorer recognition of another class.

Do not present the class-level results as proof that the changes are statistically significant. Present them as observed patterns that deserve further investigation.

## 4. What to physically show during your evaluation

I would prepare these six items, in this order.

[Skin cancer classification leveraging multi-directional compact convolutional neural network ensembles and gabor wavelets | Scientific Reports](https://images.openai.com/static-rsc-4/6BfX901-Dtx7hudzPhDt7Of9c6HnKyauI9R1zjdiHWr7c7cR0V4AwgX2oBEIz6dXIu2O_VUq4JWL59idDYxLEVJlrR3V8QkmwutyDOKSY8RB9DFi753pmkRy1YiD-1k2aRsVc6yilLkZ7QgHOsEdw4vTgOkdkyx0ekD2FFZ5u1Y?purpose=inline)

1\. Dataset and class imbalance

Essential

Show examples from the seven classes and a bar chart of training counts. Explain why ordinary accuracy is insufficient and why you chose macro-F1 and balanced accuracy.

[GAN based augmentation using a hybrid loss function for dermoscopy images | Artificial Intelligence Review | Springer Nature Link](https://images.openai.com/static-rsc-4/RBHOk2k4toUZFp6jCEeefxVhji7NrRvU3GwLi16TLCT19kDtITEMPy2xLThTIgOm-oaEaC3SN0HPGB3cxtAf2qCSm0mbEwcEvlZ66UxdE3jcUlAZ9ONzwne4lfP2WmiTzI4dySHI38kvIVt2V_MHdLH-sCKFwo3MvbeTFTjVqnE?purpose=inline)

2\. GAN-generated image grids

Essential

Show real images beside DCGAN and WGAN-GP outputs, with class labels. Use the actual images from your saved project grids rather than illustrative images. Explain the FID and diversity checks, including their limitations.

[Modern Convolutional Neural Network Architectures](https://images.openai.com/static-rsc-4/VExfU9ptQS_TBcMqDUmNt5g-57aGMbXeDAunVQems4m2hT_RRGZReGKtDZntj6NNKRj3MuFXSPRs1Ovr0RhLtGWoDsKOABmPYf9FosGyYjTtsG6PX9uS9dv8QyBgBPl8f_37cfH3u3cl--tnN7bb_8LwudGo6m6f6ugS6tRYpis?purpose=inline)

3\. Experimental methodology

Essential

Show the workflow: fixed real-data split → augmentation strategy → ResNet18 training → validation-based checkpoint selection → real-only test evaluation. Explain E0, E2, E3 and E4.

[Abdul BASIT | University College London, London | UCL | School of Pharmacy | Research profile](https://images.openai.com/static-rsc-4/15dgcmIqSZ2jSG5YH6-yulX7ywvMg44i_0hYdUVy4LUjm9XGmuMgT9-852Wp4Br9AD1zxmUNPX0fO9Erx3ReGNN5Y_BJEruh4NmxkrewLjziRz89_JMx1KwYutzrlDeE4SDfIGxqsOw4jB21G8qHcCod0OoNRwF_C0oWDMUdgUM?purpose=inline)

4\. Headline results

Essential

Show the mean ± standard deviation table for all four experiments. Clearly state that the predefined rule did not establish a measurable macro-F1 gain.

[Checking your browser - reCAPTCHA](https://images.openai.com/static-rsc-4/qVdqPveaJXKf_nQeVYpWHTKwVarMKGrcIbYViIZbwU0_ntbZF8iw1HMLh3cYpVDJdCGQiA2IrYhy25ZlsyhQ4IEE-djUqldWV5K_FxZ_s9vwY99xeZ6SH-HjOfO3RSCq4iNbGmoBsiMi3cKF9eIFzCYRcU8mTypc1nwCs9jWRXE?purpose=inline)

5\. Class-level error analysis

Strongly recommended

Show the per-class recall table and confusion matrices. Explain the DF and VASC improvements and the MEL trade-off. Use actual matrices computed from your saved test predictions.

[Image Based Story Generation - deep learning course with overall project details | PPTX](https://images.openai.com/static-rsc-4/FkUx8xGVcR-zuQ7Y2WeQJxxo4Y3DC_5nPnymZhxqA94XGg1REjg7-4N1QCEwwZ39lFP2KbB0sW1FLWBrKr8DR-U1fZflyxUUpAajCehxY-iHn1kUz5utKi1AXn4VnH_IgvXPZQIghRJS2rS7zLgjKKmRe_f4iUkL8VZIAeqWf7Y?purpose=inline)

6\. Conclusions and limitations

Essential

Explain why the hypothesis was not supported at the aggregate metric level, what was learned, and what experiments would be needed next.

The images above are illustrative visual references, not your experimental outputs. For the actual presentation, use your project's generated image grids, metric tables and model results.

## 5. What I would do before presenting

You do not need to launch another large training experiment just to make Phase 7 presentable. The formal runs are complete. I would prioritize the following remaining analysis:

1. Generate confusion matrices for E0, E2, E3 and E4. Use saved test predictions and labels for each seed. Ideally, show normalized confusion matrices averaged across seeds or show one matrix per experiment with the aggregation method explained.
2. Calculate per-class precision and F1, not only recall. This will reveal whether the higher DF recall comes with an increase in false-positive DF predictions.
3. Verify training histories and best checkpoints. Be ready to explain that each seed used its own validation-selected best checkpoint rather than selecting a model based on test performance.
4. Consider the optional bootstrap or paired analysis. It may help quantify uncertainty, but it will not guarantee that the results support a positive claim. Three seeds also limit the strength of conclusions about run-to-run variability.

One especially useful next step is the confusion-matrix analysis. It may explain why the aggregate scores behave as they do and make your discussion much more convincing than simply adding more decimal places to the headline table.

## 6. What to say when the evaluator asks, “So what did you achieve?”

You can use this as your core explanation:

Our project investigates whether GAN-based synthetic data augmentation can improve skin-lesion classification under severe class imbalance. We established a ResNet18 baseline and compared it with classical augmentation, DCGAN-based augmentation, and WGAN-GP-based augmentation, using three random seeds per experiment and a fixed train/validation/test split.

The experiments did not establish a measurable improvement in mean macro-F1 from GAN augmentation under our predefined decision rule. Classical augmentation achieved the highest observed mean macro-F1 of 0.6559, while WGAN-GP augmentation achieved the highest mean balanced accuracy of 0.6331.

Although the aggregate improvement was inconclusive, class-level analysis revealed potentially useful patterns. DF recall increased from 0.2549 in the baseline to 0.3922 with WGAN-GP augmentation, and VASC recall increased from 0.8571 to 0.9206 with classical augmentation. However, MEL recall decreased under all three augmentation strategies.

These findings suggest that synthetic-data augmentation does not automatically improve downstream classification, and that the impact depends on the class and augmentation method. The contribution of our work is a controlled comparison of these strategies, a reproducible evaluation pipeline, and an analysis of the trade-offs between overall performance and minority-class recognition.

## 7. The honest bottom line

Your project still has substance to demonstrate:

- A functioning end-to-end classification pipeline.
- Two trained GAN approaches and prepared synthetic datasets.
- A baseline and three augmentation experiments, each evaluated over three seeds.
- Quantitative comparisons across several metrics.
- An evidence-based finding that the expected overall improvement was not established.
- Specific minority-class patterns that guide future investigation.

What you should not claim is that GAN augmentation has been proven to improve skin-lesion classification, or that WGAN-GP is statistically superior to DCGAN.

My recommendation: focus next on the actual confusion matrices and per-class precision/F1. Those analyses will help turn your existing results into a much stronger explanation of what happened and why.