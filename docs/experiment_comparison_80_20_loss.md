# Experiment Comparison Report: Weighted MSE vs. Standard MSE (Leakage-Free 80/20 Operational Split)

**Date:** October 17, 2026  
**Model:** `SatelliteGRU` (2 GRU layers $64 \to 36$, 4-layer MLP regression head $36 \to 128 \to 256 \to 64 \to 1$ with PReLU activations)  
**Dataset:** Option A 80/20 Stratified Operational Split:
- Training: `data/processed/train_sequences_2_range_80_20.joblib` (98 events: 49 critical $y \ge -6.0$, 49 non-critical $y < -6.0$)
- Validation: `data/processed/val_operational_disjoint_80_20.joblib` (7,864 events: 12 critical $y \ge -6.0$, 7,852 non-critical $y < -6.0$)
- Scaler: `data/processed/scaler_80_20.joblib` (MinMaxScaler fitted strictly on the 98 training events, zero validation feature leakage)
**Reference Paper:** Tulczyjew et al. (2021), *Predicting risk of satellite collisions using machine learning*, Journal of Space Safety Engineering.

---

## 1. Executive Summary

This report documents the controlled experimental comparison between two loss function formulations evaluated on a strictly leakage-free, event-stratified 80/20 operational split:
1. **`2_range_80_20_weighted_mse`:** Weighted Mean Squared Error based on the authors' reference implementation (`dundermifflin_ref/CollisionAvoidanceChallenge/models/rnn.py`), applying weight $1.0$ for critical high-risk events ($y \ge -6.0$) and weight $0.1$ for low-risk events ($y < -6.0$).
2. **`2_range_80_20_standard_mse`:** Standard unweighted Mean Squared Error ($\frac{1}{N}\sum (\hat{y} - y)^2$), matching the verbatim text specification in Section 3.2 of Tulczyjew et al. (2021).

Unlike prior 2-range experiments where all 61 operational critical candidates were selected into the training set—rendering disjoint validation critical metrics mathematically undefined (`NaN`) and restricting evaluation to contaminated in-sample pools—this 80/20 split reserves **12 unseen critical events** and **7,852 non-critical events** with zero event-ID overlap. Checkpoint selection for both runs was driven strictly by the disjoint validation composite score ($\text{MSE}_{\text{crit}} / F_2$).

All other experimental variables—model architecture, random seed (42), optimizer (Adam, $\text{lr}=0.001$), batch size (1), epoch count (510), evaluation schedule, and classification threshold ($-6.0$)—were held identical.

---

## 2. Experimental Protocol & Controlled Parameters

| Parameter | Weighted MSE (`2_range_80_20_weighted_mse`) | Standard MSE (`2_range_80_20_standard_mse`) | Status |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | `SatelliteGRU` (2-layer GRU hidden 36, MLP `36→128→256→64→1`) | `SatelliteGRU` (2-layer GRU hidden 36, MLP `36→128→256→64→1`) | **Identical** |
| **Training Dataset** | `train_sequences_2_range_80_20.joblib` (98 events) | `train_sequences_2_range_80_20.joblib` (98 events) | **Identical** |
| **Validation Dataset** | `val_operational_disjoint_80_20.joblib` (7,864 events) | `val_operational_disjoint_80_20.joblib` (7,864 events) | **Identical** |
| **Event ID Overlap** | 0 events (Strict partition of 7,962 operational pool) | 0 events (Strict partition of 7,962 operational pool) | **Identical (Zero Leakage)** |
| **Feature Scaler** | `scaler_80_20.joblib` (fitted strictly on 98 train events) | `scaler_80_20.joblib` (fitted strictly on 98 train events) | **Identical (No Leakage)** |
| **Optimizer & LR** | Adam ($\text{lr} = 0.001$, default $\beta_1, \beta_2$) | Adam ($\text{lr} = 0.001$, default $\beta_1, \beta_2$) | **Identical** |
| **Batch Size & Epochs**| $\text{batch\_size} = 1$, 510 epochs | $\text{batch\_size} = 1$, 510 epochs | **Identical** |
| **Random Seed** | 42 (`torch`, `numpy`, `random`) | 42 (`torch`, `numpy`, `random`) | **Identical** |
| **Evaluation Schedule**| Every 25 epochs + Epoch 510 | Every 25 epochs + Epoch 510 | **Identical** |
| **Selection Metric** | `val_operational_disjoint.score` | `val_operational_disjoint.score` | **Identical** |
| **Decision Threshold** | Critical risk cutoff = $-6.0$ | Critical risk cutoff = $-6.0$ | **Identical** |
| **Official Metric** | $\text{Score} = \text{Critical MSE} / F_2$ | $\text{Score} = \text{Critical MSE} / F_2$ | **Identical** |
| **Loss Function** | **Weighted MSE (1.0 / 0.1)** | **Standard MSE (Unweighted)** | **Isolated Difference** |

---

## 3. Measured Results

### A. Head-to-Head Comparison on Disjoint Validation Pool (7,864 Events)

| Metric | Weighted MSE (`2_range_80_20_weighted_mse`) | Standard MSE (`2_range_80_20_standard_mse`) | Absolute $\Delta$ | Relative $\Delta$ |
| :--- | :---: | :---: | :---: | :---: |
| **Best Epoch** | **350** / 510 | **200** / 510 | -150 epochs | Faster convergence |
| **Training Duration** | 2,076.05 s (34.60 min) | 1,515.77 s (25.26 min) | -560.28 s | **27.0% faster** |
| **Disjoint Score ($\frac{\text{MSE}_{\text{crit}}}{F_2}$)** | 549.8525 | **540.6164** | **-9.2361** | **-1.68% (Better)** |
| **Critical MSE** | 58.4950 | **57.3767** | **-1.1183** | **-1.91%** |
| **$F_2$ Score** | **0.1064** | 0.1061 | -0.0003 | -0.28% |
| **True Positives ($TP$)** | **10** / 12 | 9 / 12 | -1 TP | Event 9657 slipped to $-6.1360$ |
| **False Positives ($FP$)** | 412 | **367** / 7,852 | **-45 FP** | **-10.92% (Fewer FPs)** |
| **False Negatives ($FN$)** | **2** / 12 | 3 / 12 | +1 FN | Event 9657 slipped past cutoff |
| **True Negatives ($TN$)** | 7,440 | **7,485** / 7,852 | **+45 TN** | +0.60% |
| **Precision** | 2.3697% (10 / 422) | **2.3936%** (9 / 376) | +0.0239% | +1.01% |
| **Recall** | **83.3333%** (10 / 12) | 75.0000% (9 / 12) | -8.3333% | -10.00% |
| **Disjoint Overall MSE** | 70.0156 | **65.6041** | **-4.4115** | **-6.30%** |
| **`val_operational_all` Score** | 83.2367 | **59.4555** | -23.7812 | -28.57% |

---

## 4. Per-Event Inspection of All 12 Unseen Critical Events

The table below details model predictions, classification decisions, and squared error contributions on each of the 12 unseen critical events:

| Event ID | Ground Truth Target | Weighted MSE Pred | Standard MSE Pred | Wt Status | Std Status | Wt Squared Error | Std Squared Error | $\Delta$ Sq Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **197** | $-5.4238$ | $-5.6814$ | $-5.6445$ | TP | TP | $0.0664$ | $0.0487$ | $-0.0177$ |
| **1408** | $-5.3448$ | $-5.4216$ | $-5.2330$ | TP | TP | $0.0059$ | $0.0125$ | $+0.0066$ |
| **1414** | $-4.7261$ | $-4.0985$ | $-4.7796$ | TP | TP | $0.3939$ | $0.0029$ | $-0.3910$ |
| **1429** | $-5.4896$ | $-5.5296$ | $-4.9062$ | TP | TP | $0.0016$ | $0.3403$ | $+0.3387$ |
| **2958** | $-4.4905$ | **$-12.6273$** | **$-13.0829$** | **FN** | **FN** | **$66.2070$** | **$73.8281$** | $+7.6211$ |
| **3439** | $-4.5501$ | $-5.6199$ | $-4.6301$ | TP | TP | $1.1445$ | $0.0064$ | $-1.1381$ |
| **4104** | $-5.8861$ | $-5.9340$ | $-5.6550$ | TP | TP | $0.0023$ | $0.0534$ | $+0.0511$ |
| **8758** | $-4.5577$ | $-5.1751$ | $-4.9784$ | TP | TP | $0.3812$ | $0.1770$ | $-0.2042$ |
| **8851** | $-5.6002$ | $-4.8730$ | $-4.9153$ | TP | TP | $0.5288$ | $0.4691$ | $-0.0597$ |
| **9657** | $-5.4840$ | $-5.9724$ | **$-6.1360$** | **TP** | **FN** | $0.2385$ | $0.4251$ | $+0.1866$ |
| **11033** | $-4.5844$ | $-4.9049$ | $-4.9527$ | TP | TP | $0.1027$ | $0.1357$ | $+0.0330$ |
| **11795** | $-5.0150$ | **$-30.1719$** | **$-29.7743$** | **FN** | **FN** | **$632.8666$** | **$613.0217$** | $-19.8449$ |
| **Mean / Sum** | — | — | — | **10 TP / 2 FN** | **9 TP / 3 FN** | **Crit MSE = 58.4950** | **Crit MSE = 57.3767** | **-1.1183** |

---

## 5. Evaluation of Research Hypothesis: Refuted by Empirical Evidence

### Hypothesis Under Test
> *The $0.1$ weight factor applied to low-risk events in Weighted MSE attenuates gradient feedback when target risks are low/intermediate, encouraging the GRU to aggressively collapse late-escalating events towards the $-30.0$ baseline floor. Transitioning to unweighted Standard MSE will restore uniform gradient magnitude across all risk levels and alleviate underprediction on late-escalating conjunctions (Events 11795 and 2958).*

### Empirical Verdict: Refuted / Not Supported
1. **Event 11795 remained pinned to the baseline floor:**  
   Under Standard MSE, the predicted risk for Event 11795 was $-29.7743$, virtually unchanged from the Weighted MSE prediction of $-30.1719$ ($\Delta = +0.3976$). Its squared error of $613.0217$ accounts for **$89.04\%$** of the entire validation Critical MSE.
2. **Event 2958 exhibited slightly worse error:**  
   The prediction shifted from $-12.6273$ to $-13.0829$ ($\Delta = -0.4556$, $\text{SqErr}$ increased from $66.21$ to $73.83$).
3. **Concentration of Critical MSE:**  
   Under Standard MSE, Events 11795 and 2958 together contribute **$686.8498$** out of $688.5209$ total critical squared error—representing **$99.76\%$** of the entire critical squared error across all 12 events.
4. **Physical Root Cause:**  
   The forensic audit confirmed that prior to the 2-day operational decision gate:
   - Event 11795 had 15 consecutive CDMs all frozen at risk $-14.42$ with flat covariance matrices.
   - Event 2958 had only a single CDM at risk $-6.91$.  
   Because both conjunctions showed zero elevation in risk or covariance before the cutoff, the GRU's temporal receptive field contains insufficient information to predict post-cutoff escalation. This information barrier is fundamental to the operational data structure and cannot be resolved merely by switching loss weighting.

---

## 6. Interpretation: Where the Score Gain Originated

The improvement in disjoint validation score ($540.6164$ vs $549.8525$, a $9.24$-point reduction) was driven by:
1. **10.92% Reduction in False Positives:**  
   Standard MSE penalizes errors on low-risk training events 10× more heavily ($1.0$ vs $0.1$). This forced the network to establish tighter regression boundaries on non-critical conjunctions, eliminating **45 false positives** ($412 \to 367$) and reducing Disjoint Overall MSE from $70.02$ to $65.60$.
2. **Robustness of $F_2$ Metric:**  
   The $F_2$ denominator ($5 \cdot TP + 4 \cdot FN + FP$) balanced the reduction of 45 $FP$ against the loss of 1 $TP$, preserving an almost identical $F_2$ score ($0.1061$ vs $0.1064$).
3. **Marginal Threshold Sensitivity on Event 9657:**  
   Event 9657 (target $-5.4840$) was predicted at $-5.9724$ under Weighted MSE ($TP$) and $-6.1360$ under Standard MSE ($FN$). This $0.1360$ boundary shift was the sole reason true positives decreased from 10 to 9.

---

## 7. Comparison with Paper Reference (1.337)

| Evaluation Setting | Composite Score | Critical MSE | $F_2$ Score | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **ESA Competition Test (Paper)** | **1.337** | ~0.80 | ~0.60 | Evaluated on hidden test set (`test_data_private.csv`) |
| **Standard MSE 2-Range (In-Sample)**| 5.7002 | 1.3783 | 0.2418 | Contaminated: 61/61 critical events seen in training |
| **Weighted MSE 2-Range (In-Sample)**| 7.7405 | 2.1286 | 0.2750 | Contaminated: 61/61 critical events seen in training |
| **Standard MSE 80/20 (Leakage-Free)**| **540.6164** | 57.3767 | 0.1061 | **Honest generalization on 12 unseen critical events** |
| **Weighted MSE 80/20 (Leakage-Free)**| 549.8525 | 58.4950 | 0.1064 | **Honest generalization on 12 unseen critical events** |

> [!NOTE]
> The dramatic divergence between in-sample scores ($\approx 5.7\text{--}7.7$) and leakage-free generalization scores ($\approx 540\text{--}550$) demonstrates why separating training and evaluation event IDs is critical. Overlapping validation masks catastrophic failure modes on late-escalating conjunctions like Events 11795 and 2958.

---

## 8. Methodological Limitations & Generalization Caveats

1. **Small Critical Sample Size ($N=12$):**  
   Because the strict operational dataset pool contains only 61 eligible critical conjunctions in total, reserving 20% for disjoint validation yields exactly 12 critical events. While this 12-event sample is sufficient to definitively expose the pre-cutoff information barrier, individual event classifications carry substantial statistical leverage:
   - A single event shifting across the $-6.0$ cutoff (such as Event 9657 predicting $-6.1360$ vs $-5.9724$) alters recall by $8.33\%$ ($1/12$).
   - A single false negative with flat pre-cutoff history (Event 11795) contributes $613.02$ squared error, single-handedly determining $89\%$ of the critical MSE ($57.38$).
2. **Hidden Leaderboard Generalization:**  
   The paper's published score of $1.337$ was computed by the official ESA Kelvins competition server against a private, hidden test set (`test_data_private.csv`). Local validation on operational training data—even when strictly partitioned to avoid event-ID leakage—operates under different sampling boundaries and data regimes. Consequently:
   - Our disjoint validation scores ($\approx 540\text{--}550$) should **not** be compared directly as an inferior reproduction of the paper's $1.337$ test score.
   - These local validation results establish honest generalization behavior across the public dataset, demonstrating the limitations of sequence-only GRU modeling when physical precursors are absent before the 2-day operational decision gate.

