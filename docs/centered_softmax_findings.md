# Can SSN be trained without sampling? Centered softmax + smoothed labels

**Question.** Stochastic Segmentation Networks (SSN) model the logits as a low-rank Gaussian
and train with a Monte Carlo (MC) estimate of the marginal likelihood, which requires sampling.
If one logit is pinned to zero ("centered softmax"), the map from logits to class probabilities
becomes a bijection, so the density of a probability vector can be computed in closed form.
With label smoothing, the one-hot targets become valid points of that density, giving an
**analytic loss with no sampling**. Does it learn the same distribution?

**Short answer: no.** The analytic loss trains, but the uncertainty it learns is wrong in
both scale and structure. Sampling is needed if the samples are supposed to be plausible
segmentations.

## Why the two losses differ

With fixed targets the change-of-variables Jacobian does not depend on the parameters, so the
analytic loss reduces to Gaussian regression of the low-rank logit distribution onto the
**logit values** of the smoothed labels (e.g. ±log((1−ε)/ε)).

The MC loss only scores the class each sample selects, i.e. it constrains the **signs /
argmax** of the logits, not their values. That gives it much more freedom. For example, a
linear logit ramp plus one random shift moves a boundary, so a single low-rank factor can
represent "the boundary position is uncertain". A Gaussian fitted to the logit values of a few
discrete label maps instead puts its mass *between* them, and thresholding those points
produces masks that match no annotation.

## Toy problem (`ssn/toy_problem_analytic.py`)

21-pixel binary problems from the original `toy_problem.py`, same low-rank model, same
protocol (5000 mean-only steps, then 5000 steps on all parameters, Adam lr 1e-3), smoothing
ε = 0.1. The metric is the share of 20k samples that match **none** of the annotations
(ideal: 0%).

| Target | MC (SSN) | Analytic, trained | Analytic, exact optimum |
|---|---|---|---|
| `on_off`: middle block all-0 or all-1 | 2% | 10% | 2–13% (ε = 0.01–0.3) |
| `slide_bar`: boundary at 10 positions | **0%** | **34–50%** (rank 1–9) | **34%** |

The "exact optimum" uses the closed-form maximiser of the analytic loss (empirical mean and
full covariance of the target logits). So the `slide_bar` failure comes from the objective,
not from optimisation, and more rank or a different smoothing level does not fix it.

## BraTS

Setup: BraTS 2020 cases with the BraTS 2018 split (171 train / 57 validation, files in
`assets/BraTS2018_data/`). Stochastic DeepMedic, rank 10, 110³ patches, 30 epochs (the
original schedule has about 1050), seed 1, on one RTX 8000. Configs:
`assets/config_files/cmp_*.json`.

| Run | Loss | Covariance |
|---|---|---|
| `cmp_mc` | MC, 20 samples (original SSN) | low-rank, masked to brain |
| `cmp_analytic` | analytic, smoothing 0.3, centered log-softmax | low-rank, unmasked |
| `cmp_analytic_diag` | same | diagonal |
| `cmp_analytic_detached` | same, `detach_mean: true` | low-rank |

`detach_mean` computes the low-rank log-density with a detached mean and adds a diagonal
log-density that trains the mean. It was added because the plain low-rank analytic loss
lets the covariance factors absorb large structured mean errors: one factor costs about one
log-det term, while the squared error it removes grows with the number of voxels.

### Full validation set (`evaluation/evaluate.py`, 57 cases, 100 samples per case)

| Run | Dice of mean prediction (NET / ED / ET / TC) | Avg. sample Dice (any lesion) | GED ↓ | Diversity | Pixel entropy |
|---|---|---|---|---|---|
| MC (SSN) | **0.50** / 0.60 / **0.60** / **0.68** | **0.63** | **0.70** | 0.63 | 0.04 |
| Analytic, low-rank | 0.02 / 0.50 / 0.23 / 0.22 | 0.15 | 1.09 | 0.83 | 0.14 |
| Analytic, diagonal | 0.17 / **0.61** / 0.49 / 0.51 | 0.25 | 1.00 | 0.87 | 0.14 |
| Analytic, detached mean | 0.14 / 0.51 / 0.35 / 0.40 | 0.18 | 1.07 | 0.86 | 0.14 |

NET = necrotic / non-enhancing tumor, ED = oedema, ET = enhancing tumor, TC = tumor core.
GED uses d = 1 − IoU averaged over the three tumor classes. With a single annotation per case
it mixes sample accuracy with sample diversity.

- The analytic losses give a usable mean: the diagonal one is close to MC on oedema.
- Their samples are noisy, with 3× the pixel entropy of SSN, low per-sample Dice and higher
  GED.
- Plain low-rank analytic training is unstable: necrotic-core validation Dice went
  0.51 → 0.54 → 0.00 at epochs 10/20/30.

### Is it only the noise scale? (`evaluation/temperature_sweep.py`)

The temperature multiplies the noise standard deviation; T = 1 is the model's own
distribution. First 10 validation cases, 20 samples per temperature, mean ± s.e.m.

| GED ↓ | T = 1 | T = 0.5 | T = 0.25 | T = 0.1 |
|---|---|---|---|---|
| MC (SSN) | **0.66 ± 0.04** | 0.72 ± 0.05 | 0.81 ± 0.06 | 0.96 ± 0.07 |
| Analytic, low-rank | 1.09 ± 0.01 | 1.06 ± 0.01 | 1.04 ± 0.02 | 1.03 ± 0.03 |
| Analytic, diagonal | 1.00 ± 0.02 | 0.89 ± 0.03 | **0.84 ± 0.04** | 0.97 ± 0.06 |
| Analytic, detached mean | 1.08 ± 0.01 | 1.04 ± 0.02 | 0.97 ± 0.02 | 0.91 ± 0.03 |

- SSN is **calibrated**: its GED is best at its own temperature.
- The analytic models' noise is **too large**: cooling helps the diagonal and detached models.
- Even at its best temperature, the best analytic model (diagonal, T = 0.25: 0.84) is clearly
  worse than SSN at T = 1 (0.66).
- The low-rank analytic model barely improves with cooling, so its covariance **structure** is
  wrong, not only its scale.

## Conclusion

Centered softmax + label smoothing removes sampling from *training*, but the analytic
objective fits the scatter of smoothed target logits, not the distribution over
segmentations:

- on the toy problem it produces invalid masks when annotations differ in boundary position;
- on BraTS its low-rank covariance either absorbs the mean's errors (plain) or does not help
  (detached), and its noise is both too large and wrongly structured;
- only the diagonal variant is competitive on the mean, and it gives up the spatial
  correlations that SSN exists for.

For SSN's purpose, the sampling (MC) loss is needed.

## Caveats

- One seed and 30 of about 1050 epochs per run. The validation numbers fluctuate between
  checks.
- BraTS has one annotation per case, so there is no annotator variability to compare against.
  A multi-annotator dataset (e.g. LIDC) would test the distributional claim directly.
- The analytic loss values in the logs are not true log-densities:
  `CenteredLogSoftmaxTransform.log_abs_det_jacobian` uses the softmax (not log-softmax)
  Jacobian. It is a constant offset and does not affect gradients.
- The detached-mean run occasionally fell back to independent normals ("Covariance became not
  invertible"). This was not investigated.

## Bugs fixed along the way

These are the reasons the original 2021–2023 experiments on this fork did not train:

1. `stochastic_deepmedic.py` referenced `self.symmetric_sampling` instead of
   `self.use_symmetric_sampling`. The bare `except` swallowed the `AttributeError`, so
   **every** config, including the baseline, used independent normals.
2. The `is_softmax` and smoothed `is_logsoftmax` branches of the MC loss had the wrong sign
   and **maximised** the cross-entropy.
3. `use_log` defaulted to `True`, which gave NaN losses (`0padsoftmax`) or support errors
   (`analytic`) for configs that expected probabilities.
4. `AxisSimplex` never stored its axis, and the MC loss called a non-existent
   `self.smooth_one_hot`.
5. Upstream: `calc_f1_score` was a copy of `calc_recall`; the evaluation code used `.view` on
   non-contiguous tensors and `np.bool`, which fail on current PyTorch / NumPy.

## Reproducing

```bash
# toy problem
cd ssn
python toy_problem_analytic.py --target slide_bar --lr 1e-3 --steps 10000 --mean_only_steps 5000

# BraTS training (one per config; about 50 min each on an RTX 8000, plus about 40-75 min
# for the full-volume inference at the last epoch)
python train.py --job-dir ../jobs/cmp_mc --config-file ../assets/config_files/cmp_mc.json \
    --train-csv-path ../assets/BraTS2018_data/data_index_train.csv \
    --valid-csv-path ../assets/BraTS2018_data/data_index_valid.csv \
    --num-epochs 30 --device 0 --random-seeds 1

# evaluation (about 2.5 h per model) and temperature sweep
cd ../evaluation
python evaluate.py --path-to-prediction-csv ../jobs/cmp_mc/random_seed_1/test/predictions/prediction.csv
python temperature_sweep.py --path-to-prediction-csv ../jobs/cmp_mc/random_seed_1/test/predictions/prediction.csv
```

The data index CSVs contain absolute paths under `/home/feiner/datasets/BraTS/ssn/`.
