# Digital Twin–Enabled Framework for Forecasting and Mitigating Fraud

**Shambhu Pandey**  
MCA, School of Computer Science and Engineering, VIT Chennai  

> **Draft status:** Research-ready structure, with quantitative results intentionally marked as `[RESULT REQUIRED]` until the corrected experimental protocol is executed. Existing historical metrics are not reused.

## Abstract

Financial fraud detection is challenged by severe class imbalance, evolving transaction behavior, and the difficulty of evaluating a detector under changing risk conditions. This work presents a prototype framework for fraud detection that combines supervised machine learning, anomaly detection, explainable artificial intelligence, and a simulation-oriented Digital Twin component. The implementation uses a leakage-audited preprocessing pipeline, train-only feature transformation, and training-time SMOTETomek resampling. Logistic Regression, Random Forest, XGBoost, and Isolation Forest are included as complementary detection models, while SHAP and LIME are used to provide local and global explanations. The primary empirical study is designed around the PaySim transaction dataset, with additional datasets considered only when their provenance and target semantics can be verified. Unlike development-stage evaluation, the final study will select operating thresholds without accessing the held-out test labels and will emphasize PR-AUC, ROC-AUC, recall, precision, F1-score, MCC, and recall at fixed false-positive rates. The Digital Twin component is treated as a research module that must be calibrated against observed transaction behavior before quantitative claims about forecasting or mitigation are made. The resulting framework is intended to provide an auditable path from fraud classification to scenario evaluation and cost-aware mitigation.

**Keywords:** financial fraud detection; Digital Twin; machine learning; XGBoost; Random Forest; anomaly detection; explainable AI; SHAP; LIME; SMOTETomek; PaySim; risk mitigation.

## 1. Introduction

The rapid expansion of digital financial services has increased the scale and speed of electronic transactions while simultaneously creating opportunities for increasingly adaptive fraudulent behavior. Machine-learning-based fraud detection has therefore become an important complement to conventional rule-based controls. Recent studies continue to investigate ensemble models, imbalance handling, temporal efficiency, and operational deployment for financial fraud detection. [1–3]

A central difficulty in fraud detection is that fraudulent transactions are usually a minority class. Consequently, a model can achieve high overall accuracy while failing to identify a substantial portion of fraudulent activity. This makes threshold selection, precision-recall trade-offs, and operational false-positive costs important parts of the detection problem rather than secondary implementation details.

A second challenge is that conventional supervised evaluation generally treats the test distribution as static. In practice, fraud strategies can evolve, and a model that performs well against known patterns may behave differently when transaction behavior changes. Digital Twin concepts provide a potential mechanism for scenario-based evaluation by maintaining a computational representation of a system and allowing alternative states or future scenarios to be examined without directly manipulating the operational environment. Recent work has specifically discussed Digital Twin concepts in the context of credit-card fraud detection and highlighted their potential for behavioral analysis and emerging-fraud evaluation. [4]

This study therefore investigates a framework that connects fraud classification with explainability and simulation-oriented scenario analysis. The implementation is intentionally designed with an anti-leakage preprocessing pipeline: target-like columns are removed before modeling, the train/test split is performed before fitting transformations, and resampling is confined to the training pipeline. These implementation properties are important because apparently strong fraud-detection results can otherwise arise from evaluation leakage rather than genuine generalization.

The current work uses PaySim as the principal empirical dataset because it represents transaction-level mobile-money behavior and provides a reproducible fraud-label setting. The repository also contains pathways for other datasets; however, only datasets whose provenance and construction can be established will be used for publication claims.

### 1.1 Problem Statement

Existing fraud-detection pipelines commonly focus on classification accuracy but provide limited mechanisms for examining how a detector behaves when transaction risk changes over time or under alternative threat scenarios. There is a need for an auditable framework that (i) evaluates fraud classifiers under a leakage-controlled protocol, (ii) provides interpretable transaction-level explanations, and (iii) supports controlled scenario analysis that can eventually be connected to forecasting and mitigation decisions.

### 1.2 Objectives

The objectives of this work are:

1. To construct a leakage-controlled fraud-detection pipeline for imbalanced transaction data.
2. To compare Logistic Regression, Random Forest, XGBoost, and Isolation Forest under a common evaluation protocol.
3. To use SHAP and LIME to improve interpretability of model decisions.
4. To develop a data-calibrated Digital Twin component for scenario-based fraud analysis.
5. To investigate temporal fraud forecasting using transaction-time information.
6. To formulate mitigation as a measurable decision problem using false-positive, false-negative, and review costs.

### 1.3 Contributions

The intended contributions of the completed study are:

- A reproducible preprocessing and model-training pipeline in which scaling, encoding, and class resampling are restricted to training data.
- A comparative evaluation of supervised and unsupervised fraud detectors using imbalance-aware metrics and a threshold selected without access to held-out test labels.
- An explainability layer using SHAP and LIME to expose model-level and transaction-level evidence.
- A calibrated simulation component intended to connect observed transaction behavior with controlled future-risk scenarios.
- A cost-aware mitigation protocol that evaluates fraud-control decisions rather than treating raw classification accuracy as the sole objective.

## 2. Related Work

### 2.1 Machine Learning for Financial Fraud Detection

Recent financial fraud studies have evaluated classical machine-learning models, ensemble methods, and more advanced architectures for transaction classification. Comparative studies commonly report Random Forest and XGBoost as strong candidates because they can represent nonlinear relationships and interactions among transaction attributes. [1,2]

The growing importance of digital payments also motivates approaches that consider real-time constraints, class imbalance, and evolving fraud strategies. Recent IEEE work has examined ensembles and advanced models for large-scale transaction datasets while emphasizing the trade-offs between detection performance and computational cost. [2]

### 2.2 Class Imbalance and Evaluation

Fraud datasets are typically dominated by legitimate transactions. Resampling approaches such as SMOTE and hybrid over-/under-sampling can improve minority-class learning, but resampling must be confined to the training data to avoid contaminating evaluation sets. In this work, SMOTETomek is placed inside the training pipeline so that validation and test observations retain their original distribution.

The final evaluation prioritizes PR-AUC and recall/precision-oriented measures in addition to ROC-AUC. Accuracy is retained as a supplementary metric rather than the primary indicator of fraud-detection quality.

### 2.3 Explainable Fraud Detection

Explainability is important in fraud detection because an operational decision may require an analyst to understand why a transaction was flagged. The proposed implementation includes SHAP for additive feature attribution and LIME for local surrogate explanations. The final experiments will only report explanation results after verifying that the explanation inputs correspond to the fitted estimator's transformed feature representation.

### 2.4 Digital Twins for Fraud Analysis

Digital Twin research has increasingly been discussed as a way of representing financial-system behavior and evaluating potential fraud scenarios. Chatterjee et al. specifically surveyed the role of Digital Twins in credit-card fraud detection and discussed their potential for behavioral analysis, continuous monitoring, and emerging-fraud identification. [4]

The distinction between a true Digital Twin and a disconnected stochastic simulator is important. A useful fraud-oriented twin should have a relationship to observed system behavior, support scenario manipulation, and provide measurable outputs that can be compared with real observations. The present study therefore treats calibration and validation of the simulation layer as an explicit research requirement rather than assuming that a hand-written state-transition simulator is automatically a Digital Twin.

### 2.5 Research Gap

The literature contains substantial work on transaction-level machine-learning classification and a growing discussion of Digital Twins for fraud detection. However, these two directions are often presented separately: classifiers optimize transaction-level detection, while Digital Twin concepts are described mainly as future or conceptual opportunities. This work aims to investigate a more auditable connection between the two by using a leakage-controlled classifier as the detection layer and a calibrated simulation layer as a mechanism for scenario evaluation, forecasting, and cost-aware mitigation.

## 3. Proposed Framework

The proposed framework is organized into five logical layers:

1. **Data layer:** transaction data loading, target normalization, feature selection, and provenance checks.
2. **Detection layer:** supervised classifiers and an unsupervised anomaly detector.
3. **Explainability layer:** SHAP and LIME explanations for model decisions.
4. **Digital Twin and forecasting layer:** data-calibrated state/behavior simulation and future fraud-incidence estimation.
5. **Mitigation layer:** cost-aware threshold selection and scenario-based evaluation.

The repository implementation already provides the first three layers in varying degrees of maturity. The Digital Twin and forecasting components require further calibration and validation before their outputs can be reported as empirical contributions.

## 4. Data and Preprocessing

### 4.1 PaySim Dataset

PaySim is the primary dataset selected for the final empirical study. The current configuration identifies PaySim as a synthetic mobile-money transaction dataset modeled on real operator logs and uses `isFraud` as the target. The implementation retains behavioral fields including transaction type, amount, originating balance, and destination balance while explicitly removing documented leakage fields. [REPOSITORY SOURCE]

### 4.2 Feature Processing

The preprocessing pipeline removes target-like columns and designated leakage columns before model fitting. A stratified split is performed before transformations are fitted. Categorical variables are encoded, numeric variables are standardized, and missing or infinite values are handled. The final model pipeline places SMOTETomek after feature transformation and before the classifier, ensuring that resampling is performed only during training.

### 4.3 Evaluation Protocol

The final protocol uses separate training, validation, and test partitions. The validation partition is used for model/threshold decisions. The test partition remains untouched until the final evaluation. No historical result generated by test-set threshold scanning is used in the final paper.

## 5. Detection Models

### 5.1 Logistic Regression

Logistic Regression serves as an interpretable linear baseline for estimating the probability of fraud from the engineered transaction representation.

### 5.2 Random Forest

Random Forest is included as a nonlinear ensemble baseline capable of modeling interactions among transaction attributes.

### 5.3 XGBoost

XGBoost is evaluated as a gradient-boosting model designed to capture complex nonlinear relationships while providing probability estimates suitable for threshold-based decisions.

### 5.4 Isolation Forest

Isolation Forest provides an unsupervised anomaly-detection comparison. Its purpose is not to replace supervised fraud classification but to test whether anomalous behavior can be identified without relying directly on fraud labels.

## 6. Explainable AI

The framework provides SHAP and LIME explanations. SHAP is intended to quantify the contribution of individual features to a prediction, while LIME provides a local surrogate explanation around a selected transaction. Explanation figures will be included only after confirming feature-space alignment between the fitted pipeline and the explanation routines.

## 7. Digital Twin and Forecasting

### 7.1 Current Prototype

The repository currently contains a state-based simulation component. For the research version, the transition structure will be calibrated using observed training data rather than relying exclusively on hand-defined transition probabilities.

### 7.2 Twin Calibration

The research version will estimate state transitions and relevant behavioral statistics from the training portion of the transaction data. The twin will then generate trajectories under controlled risk conditions. Fidelity will be assessed by comparing selected statistics of simulated and held-out real behavior.

### 7.3 Forecasting

A genuine forecasting experiment will aggregate fraud activity over the transaction time index and predict a future fraud count or fraud rate. The final forecasting horizon will be kept separate from model-development data. Forecast accuracy will be reported using appropriate temporal metrics such as MAE and RMSE.

## 8. Mitigation Strategy

Mitigation will be formulated as an operational decision problem rather than a simple probability-to-score conversion. A cost function will combine false-negative, false-positive, and manual-review costs. The threshold selected using validation data will be compared with a fixed baseline and evaluated on the held-out test set.

## 9. Experimental Results

> **Important:** Numerical results are intentionally not populated yet. Existing repository metrics were generated under incompatible historical pipelines and/or test-set threshold selection and therefore are not used.

### 9.1 Detection Results

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | MCC | Recall @ 1% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] |
| Random Forest | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] |
| XGBoost | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] |
| Isolation Forest | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] |

### 9.2 Ablation Study

| Configuration | PR-AUC | F1 | Recall | MCC |
|---|---:|---:|---:|---:|
| Without SMOTETomek | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] |
| With SMOTETomek | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] | [RESULT REQUIRED] |

### 9.3 Digital Twin Fidelity

[RESULT REQUIRED — compare simulated and held-out transaction-state statistics.]

### 9.4 Forecasting Results

[RESULT REQUIRED — report future-horizon MAE/RMSE and baseline comparison.]

### 9.5 Mitigation Results

[RESULT REQUIRED — compare fixed threshold and cost-aware validation-selected threshold on the frozen test set.]

### 9.6 Explainability Results

[RESULT REQUIRED — provide verified SHAP/LIME examples and aggregate feature-attribution analysis.]

## 10. Discussion

The principal methodological value of the framework is its emphasis on evaluation integrity. Fraud-detection systems can appear highly accurate when the class distribution is strongly imbalanced or when threshold and feature choices are indirectly optimized against the test labels. The corrected protocol prevents these issues by separating training, validation, and test decisions.

The Digital Twin component is treated conservatively. A simulation that is not calibrated against observed transaction behavior should not be described as evidence of real-world forecasting or mitigation capability. The final study therefore makes calibration, fidelity measurement, and connection to the detector explicit requirements.

## 11. Threats to Validity

The principal threats include dataset shift, the synthetic nature of PaySim, limited representation of real-world UPI transactions, dependence on the chosen feature set, and the difficulty of reproducing evolving fraud typologies from static historical data. The study will also distinguish between payment-fraud benchmarks and network-intrusion datasets rather than combining them as if they represented the same operational problem.

## 12. Conclusion

This work proposes a research framework for combining leakage-controlled machine-learning fraud detection, explainable AI, and a calibrated Digital Twin for scenario-based forecasting and mitigation. The implementation already provides a modular detection and explanation foundation. The final empirical contribution depends on completing the corrected evaluation protocol, calibrating the simulation layer, implementing genuine temporal forecasting, and evaluating cost-aware mitigation. This staged approach prioritizes reproducibility and methodological validity over inflated development-stage performance claims.

## References

[1] *Fraud Detection in Financial Transactions: A Machine Learning Approach vs. Rule-Based Systems*, 2024 International Conference on Intelligent and Innovative Technologies in Computing, Electrical and Electronics (IITCEE), DOI: 10.1109/IITCEE59897.2024.10467759.

[2] *Enhancing Real-Time Fraud Detection in Financial Transactions using Advanced Machine Learning Models*, 2025 9th International Conference on Computational System and Information Technology for Sustainable Solutions (CSITSS), DOI: 10.1109/CSITSS67709.2025.11295066.

[3] *Fraud Detection in Financial Transactions Using Machine Learning Techniques*, 2025 International Conference on Networks and Cryptology (NETCRYPT), DOI: 10.1109/NETCRYPT65877.2025.11102785.

[4] P. Chatterjee, D. Das, and D. B. Rawat, “Digital twin for credit card fraud detection: opportunities, challenges, and fraud detection advancements,” *Future Generation Computer Systems*, vol. 158, pp. 410–426, 2024, DOI: 10.1016/j.future.2024.04.057.
