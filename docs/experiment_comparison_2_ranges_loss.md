# Experiment Comparison Report: Weighted MSE vs. Standard MSE (2-Range Dataset)

**Date:** October 16, 2026  
**Model:** `SatelliteGRU` (2 GRU layers $64 \to 36$, 4-layer MLP regression head $36 \to 128 \to 256 \to 64 \to 1$ with PReLU activations)  
**Dataset:** `data/processed/train_sequences_2_ranges.joblib` (132 events: 66 high-risk $y \ge -6.0$, 66 low-risk $y < -6.0$)  
**Reference Paper:** Tulczyjew et al. (2021), *Predicting risk of satellite collisions using machine learning*, Journal of Space Safety Engineering.

---

## 1. Executive Summary

This report documents the controlled experimental comparison between two loss function implementations for the GRU baseline on the 2-range balanced dataset:
1. **`authors_weighted_mse`:** Reproduces the original authors' reference source code (`dundermifflin_ref/CollisionAvoidanceChallenge/models/rnn.py`), applying weight $1.0$ for critical high-risk events ($y \ge -6.0$) and $0.1$ for low-risk events ($y < -6.0$).
2. **`paper_standard_mse`:** Implements unweighted Mean Squared Error ($\frac{1}{N}\sum (\hat{y} - y)^2$), matching the verbatim text specification in Section 3.2 of Tulczyjew et al. (2021).

All other experimental variables—network architecture, random seed, training sample order, optimizer, learning rate, batch size, threshold, and scoring implementation—were held strictly identical.

---

## 2. Experimental Protocol & Controlled Parameters

| Parameter | Authors' Weighted MSE | Paper Standard MSE | Status |
| :--- | :--- | :--- | :--- |
| **Model** | `SatelliteGRU` | `SatelliteGRU` | **Identical** |
| **Recurrent Core** | 2-layer GRU (input 64, hidden 36, dropout 0.0) | 2-layer GRU (input 64, hidden 36, dropout 0.0) | **Identical** |
| **MLP Head** | `36 -> 128 -> 256 -> 64 -> 1` (PReLU activations) | `36 -> 128 -> 256 -> 64 -> 1` (PReLU activations) | **Identical** |
| **Dataset** | `train_sequences_2_ranges.joblib` (132 events) | `train_sequences_2_ranges.joblib` (132 events) | **Identical** |
| **Class Balance** | 66 critical ($y \ge -6$), 66 non-critical ($y < -6$) | 66 critical ($y \ge -6$), 66 non-critical ($y < -6$) | **Identical** |
| **Optimizer** | Adam ($\text{lr} = 0.001$, default $\beta$) | Adam ($\text{lr} = 0.001$, default $\beta$) | **Identical** |
| **Batch Size & Epochs** | $\text{batch\_size} = 1$, 510 epochs | $\text{batch\_size} = 1$, 510 epochs | **Identical** |
| **Random Seed** | 42 (`torch`, `numpy`, `random`) | 42 (`torch`, `numpy`, `random`) | **Identical** |
| **Evaluation Schedule** | Every 25 epochs + Epoch 510 | Every 25 epochs + Epoch 510 | **Identical** |
| **Decision Threshold** | Critical risk threshold = $-6.0$ | Critical risk threshold = $-6.0$ | **Identical** |
| **Official Metric** | $\text{Score} = \text{Critical MSE} / F_2$ | $\text{Score} = \text{Critical MSE} / F_2$ | **Identical** |
| **Loss Function** | **Weighted MSE (1.0 / 0.1)** | **Standard MSE (Unweighted)** | **Isolated Difference** |

---

## 3. Measured Results

Metrics were evaluated at 25-epoch intervals against the operational validation pool (`val_operational_all`, 7,962 events) and the disjoint validation pool (`val_operational_disjoint_2_ranges`, 7,837 events).

### Best Checkpoint Comparison Table

| Metric | Authors' Weighted MSE | Paper Standard MSE | $\Delta$ (Standard vs. Weighted) |
| :--- | :--- | :--- | :--- |
| **Best Epoch** | **250** / 510 | **75** / 510 | Earlier convergence (-175 epochs) |
| **Training Duration** | 1,681.07 s (28.02 min) | 1,072.81 s (17.88 min) | -608.26 s |
| **Train Loss at Best Epoch** | 0.0069 (weighted) | 0.7059 (unweighted) | N/A (different scales) |
| **`val_operational_all` Score** | **7.7405** | **5.7002** | **-2.0403 (26.4% lower / better)** |
| **`val_operational_all` Critical MSE** | 2.1286 | **1.3783** | **-0.7503 (35.2% lower error)** |
| **`val_operational_all` $F_2$ Score** | **0.2750** | 0.2418 | -0.0332 |
| **`val_operational_all` TP** | 55 | **56** / 61 | +1 True Positive |
| **`val_operational_all` FP** | **701** | 858 | +157 False Positives |
| **`val_operational_all` FN** | 6 | **5** / 61 | -1 False Negative |
| **`val_operational_all` TN** | **7,200** | 7,043 | -157 True Negatives |
| **`val_operational_all` Overall MSE** | **109.1963** | 137.8573 | +28.6610 higher overall MSE |
| **`val_operational_disjoint` Overall MSE** | **110.2233** | 139.1237 | +28.9004 higher overall MSE |
| **`val_operational_disjoint` Critical MSE**| **NaN** (0 critical events) | **NaN** (0 critical events) | Mathematically undefined |
| **`val_operational_disjoint` Score** | **NaN** (0 critical events) | **NaN** (0 critical events) | Mathematically undefined |

---

## 4. Methodological Limitations & Validation Leakage

### A. `val_operational_all` Overlap (In-Sample Contamination)
- `val_operational_all` contains 7,962 events, including **all 61 operational critical events** available in the dataset.
- The 2-range balanced training dataset (`train_sequences_2_ranges.joblib`) selects 132 events: 66 high-risk ($y \ge -6.0$) and 66 low-risk ($y < -6.0$).
- Of these 132 events, **125 events (including all 61 operational critical events)** are present in `val_operational_all`.
- **Conclusion:** Any evaluation score computed on `val_operational_all` (e.g., 7.7405 or 5.7002) is in-sample / train-contaminated. It reflects training data memorization/fitting on critical events and **must not be characterized as leakage-free generalization**.

### B. Disjoint Validation Set Has Zero Critical Events
- Deriving a leakage-free disjoint validation set by filtering out all 132 training event IDs leaves 7,837 events (`val_operational_disjoint_2_ranges.joblib`).
- Because all 61 operational critical candidates were selected to form the 66-event high-risk training partition (with the remaining 5 from non-operational events), the disjoint pool contains **exactly 0 events with $y \ge -6.0$**.
- With zero positive ground-truth events ($TP + FN = 0$):
  - Recall and $F_2$ are mathematically undefined ($F_2$ defaults to clipping epsilon $10^{-8}$).
  - Critical MSE has denominator 0 and evaluates to `NaN`.
  - The ESA composite score ($\text{Critical MSE} / F_2$) evaluates to `NaN`.
- The disjoint set remains valid **only** for evaluating Overall MSE on low-risk events:
  - Weighted MSE model: Overall MSE = 110.22
  - Standard MSE model: Overall MSE = 139.12

---

## 5. Interpretation: Measured Findings vs. Hypotheses

### Measured Findings (Empirical Facts)
1. **Score Improvement on In-Sample Critical Events:** The standard MSE loss achieved a composite score of 5.7002 on `val_operational_all` at epoch 75, compared to 7.7405 at epoch 250 for weighted MSE.
2. **Critical MSE Reduction:** The improvement in score is driven primarily by lower Critical MSE ($1.3783$ vs. $2.1286$, a 35.2% reduction in squared error on true critical events).
3. **Earlier Peak Epoch:** Under standard MSE, the model reached its lowest validation score early at epoch 75, whereas under weighted MSE the lowest score occurred at epoch 250.
4. **Higher Overall Low-Risk MSE:** Standard MSE resulted in substantially higher overall MSE across all non-critical events (137.86 vs. 109.20 on `val_operational_all`; 139.12 vs. 110.22 on disjoint validation).
5. **False Positive Trade-off:** Standard MSE produced more false positives (858 vs. 701) and fewer false negatives (5 vs. 6) on `val_operational_all`, yielding a lower $F_2$ score (0.2418 vs. 0.2750) despite gaining 1 additional true positive.

### Hypotheses & Open Questions
1. **Hypothesis on Gradient Dynamics:** Under standard MSE, errors on low-risk events (where targets are $-15$ to $-30$) generate large quadratic gradient magnitudes (unattenuated by the 0.1 weight factor), which may shift the network's global bias and alter prediction scaling for near-threshold samples.
2. **Hypothesis on Generalization:** Because `val_operational_all` contains 100% of the operational critical events used in training, it cannot be determined from this experiment whether standard MSE truly generalizes better to unseen critical events or simply overfits differently to the training trajectories.
3. **Relation to Paper Score (1.337):** Neither score (7.7405 nor 5.7002) matches the paper's reported score of $\approx 1.337$. As documented in Footnote 4 of Tulczyjew et al. (2021), the 1.337 score was evaluated by the ESA competition server on the hidden test set (`test_data_private.csv`), which had different temporal distributions and unseen satellite conjunctions.
