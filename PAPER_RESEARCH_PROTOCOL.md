# Research Protocol — Digital Twin–Enabled Framework for Forecasting and Mitigating Fraud

## Locked paper title

**Digital Twin–Enabled Framework for Forecasting and Mitigating Fraud**

## Purpose

This document defines the corrected experimental protocol to be used before any numerical result is reported in the research paper.

## Non-negotiable rules

1. Never select a decision threshold using `y_test`.
2. Never report metrics from `compute_and_apply_thresholds.py`.
3. Threshold selection must use a validation split carved from the training data, or an explicitly justified a-priori cost rule.
4. The final test set must remain untouched until the model, preprocessing, hyperparameters, and threshold are frozen.
5. SMOTETomek must remain inside the training pipeline and must never be applied to validation/test data.
6. Report PR-AUC as a primary metric for highly imbalanced fraud detection, together with ROC-AUC, precision, recall, F1, MCC, and recall at selected false-positive rates.
7. Report the fraud prevalence and confusion matrix for every final experiment.
8. Use multiple random seeds (minimum five) for the main PaySim experiment and report mean ± standard deviation where computationally practical.
9. Do not describe self-generated datasets as public BankSim or IEEE-CIS benchmarks.
10. Do not claim UPI integration unless a real UPI data path is implemented and evaluated.
11. Do not call the current hard-coded score multiplier reinforcement learning/DRL.
12. Do not claim SIM-swap, credential-compromise, or card-testing simulation unless those mechanisms are actually implemented and evaluated.

## Main experimental protocol

### Dataset

Primary dataset: PaySim, subject to successful clean loading and verification.

Secondary dataset: CICIDS2017 only if the data-loading and target semantics are verified; otherwise omit it from the main fraud-detection study because it is a network-intrusion benchmark rather than a payment-fraud benchmark.

### Split

Use a three-way split:

- Training: 70%
- Validation: 10%
- Test: 20%

Use stratification for the random-split baseline. If a temporal experiment is added, order PaySim by `step` and use an earlier period for training/validation and a later period for testing.

### Preprocessing

- Remove documented leakage columns before fitting.
- Fit categorical encoding and scaling on training data only.
- Apply the same fitted transformations to validation/test data.
- Apply SMOTETomek only inside the training pipeline.
- Preserve the original validation/test class distribution.

### Models

Compare:

- Logistic Regression
- Random Forest
- XGBoost
- Isolation Forest as an unsupervised baseline

### Threshold selection

Choose the classification threshold on validation data only. Preferred operating points:

1. maximize validation F1, and
2. optionally report a cost-sensitive threshold minimizing `C_FN*FN + C_FP*FP + C_REVIEW*Review`.

Once selected, freeze the threshold and evaluate exactly once on the held-out test set.

### Primary metrics

- PR-AUC
- ROC-AUC
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient
- Recall at fixed FPR (0.1%, 0.5%, 1%)
- Confusion matrix

Accuracy should be secondary because fraud prevalence is highly imbalanced.

## Digital Twin research requirement

The current `DigitalTwinSimulator` is a hand-parameterized Markov simulator. It must not be presented as a calibrated Digital Twin in the paper without additional work.

For the final research version, the twin should be calibrated from observed transaction behavior and connected to the fraud detector. At minimum it should:

1. derive state/transition statistics from training data;
2. generate trajectories or scenarios from those learned statistics;
3. expose controllable risk/attack intensity parameters;
4. compare simulated distributions against held-out real behavior;
5. feed scenario outcomes into a measurable forecasting or mitigation experiment.

## Forecasting requirement

A publishable forecasting claim requires a genuine future target. Recommended target:

- fraud count/rate per time step or time window.

Use historical PaySim `step` values to construct the temporal series. Keep a final future horizon untouched for evaluation.

Report an appropriate forecasting metric such as MAE, RMSE, or MAPE where meaningful.

## Mitigation requirement

Mitigation must have an explicit operational objective. Recommended formulation:

`Expected Cost = C_FN*FN + C_FP*FP + C_REVIEW*ManualReviews`

Compare baseline fixed-threshold operation against a threshold selected to minimize expected cost on validation data. Evaluate both on the frozen test set.

## Explainability requirement

Verify that SHAP and LIME receive features in the same representation expected by the fitted estimator. Do not include an explanation figure in the paper until this integration is confirmed.

## Required ablations

1. Without SMOTETomek vs. with SMOTETomek.
2. Fixed threshold vs. validation-selected threshold.
3. Detector performance with and without selected high-risk engineered features where appropriate.
4. Digital Twin disabled vs. twin-based scenario evaluation, once the twin is connected to data.
5. Cost-neutral vs. cost-sensitive thresholding.

## Publication integrity

Existing files containing tuned or unreproducible metrics are historical artifacts only. They must not be copied into the final Results section.

Any result not regenerated under this protocol must be labelled as historical/development output and excluded from quantitative claims.
