# Comprehensive Audit of GRU Baseline Reproduction

> **Reference Paper:** Tulczyjew et al. (2021), *"Predicting risk of satellite collisions using machine learning"*, Journal of Space Safety Engineering 8 (2021) 339–344.  
> **Original Authors' Repository:** [GitLab: jnalepa/dundermifflin](https://gitlab.com/jnalepa/dundermifflin)  
> **Local Inspected Path:** `C:\Users\Aadesh\.gemini\antigravity\scratch\dundermifflin_ref`  
> **Audited Workspace:** `C:\Users\Aadesh\.gemini\antigravity\scratch\Predicting-risk-of-satellite-collisions`  
> **Date of Audit:** October 14, 2026

---

## Executive Summary

This audit evaluates our local PyTorch implementation of the GRU regression baseline against the published paper by Tulczyjew et al. (2021) and the original authors' source code in `dundermifflin_ref`.

### Core Conclusions
1. **The Model Architecture Matches Figure 3 and Source Code:** The recurrent core (2-layer GRU, hidden size 36, batch size 1) and the 4-layer regression MLP head (`36 -> 128 -> 256 -> 64 -> 1` with PReLU activations) implemented in `src/models/gru.py` are structurally identical to the authors' `models/rnn.py` and Figure 3 of the paper.
2. **Major Paper vs. Code Contradiction on Class Balancing:** The paper explicitly describes training with *"an equal number of negative and positive samples"* (Section 3). In `dundermifflin_ref`, two generator routines exist:
   - `n_ranges = 2`: creates a 50/50 balance (132 samples: 66 high-risk, 66 low-risk), matching the text of the paper.
   - `n_ranges = 6`: creates 6 risk bins of 23 samples each (138 samples total, only 25 high-risk and 113 low-risk).  
   Our initial training used the 6-range dataset, meaning our model only saw 25 high-risk trajectories per epoch (only ~18% of the training set), whereas a 50/50 balanced dataset would have provided 66 high-risk trajectories (2.64x more).
3. **Paper vs. Code Contradiction on Training Loss:** The paper text states that models were *"trained with the MSE loss function"* (Section 3.2). However, the original code in `models/rnn.py` ignores `nn.MSELoss()` and implements a custom `weighted_mse_loss` with sample weights of $1.0$ for high risk ($y \ge -6.0$) and $0.1$ for low risk ($y < -6.0$).
4. **Resolution of the Leaderboard Score Discrepancy ($\approx 1.337$):** The paper's Table 1 score of **1.337** for the GRU was **not** computed on a local training/validation split. As explicitly revealed by **Footnote 4 on Page 3 of the paper**, 1.337 was computed by the **ESA competition validation server on the test set** using the hidden ground truth in `test_data_private.csv`. Comparing our local validation scores (10.8989 on `val_operational_all` and 24.3959 on `val_operational_disjoint`) directly to 1.337 was a category error. On `test_data.csv`, our trained GRU achieved an ESA score of **6.6652** (with $F_2 = 0.8263$ and Recall = $90.75\%$).
5. **Original Code Flaws in `infer_model`:** The reference code `infer_model` in `models/pipeline.py` is non-functional: it specifies `mode='infer'` which `RNN.forward()` does not handle (returning `None`), slices `[:-1]` on test sequences (which crashes with `RuntimeError` on single-CDM sequences), and substitutes dummy sequential integers for `event_id`.

---

## 1. Verification Checklist

| Item | Requirement | Status | Detailed Finding |
| :--- | :--- | :--- | :--- |
| **1** | GRU Architecture & 4 FC Layers vs. Fig. 3 | **VERIFIED (Exact Match)** | 2 GRU layers (hidden size 36) + MLP head (`36 -> 128 -> 256 -> 64 -> 1` with PReLU). Output is linear scalar risk $\hat{y}$. Matches Fig. 3 and `models/rnn.py`. |
| **2** | Training Sample Selection & Balancing | **DISCREPANCY (Paper vs Code)** | Paper states equal number of positive and negative samples (50/50, $n=2$ ranges, 132 samples). Code also contains 6 ranges (138 samples, 18% positive). We trained on 6 ranges. |
| **3** | Min-Max Scaling & Imputation Leakage | **VERIFIED / PARTIAL LEAKAGE** | Scaler matches `x_scaler.pickle` to machine precision ($4.44 \times 10^{-16}$). Both original code and our code fit imputation and scaler across the whole training CSV before event extraction. Test data imputation uses test set means in both. |
| **4** | Sequence Ordering, Cutoff & Target Alignment | **VERIFIED (Reproduces Code Asymmetry)** | CDMs are chronological (descending `time_to_tca`). Slicing $0 \dots T-2$ with target at $T-1$ matches original forward pass. Validation strictly cuts at $t \ge 2.0$, while training retains intermediate CDMs down to $t < 1.0$ (inheriting authors' asymmetry). |
| **5** | Weighted MSE vs. Paper Stated MSE | **DISCREPANCY (Paper vs Code)** | Paper states "MSE loss function", but authors' code hardcodes `weighted_mse_loss` ($w=1.0$ if $y \ge -6.0$, $w=0.1$ if $y < -6.0$). Our code faithfully reproduces the code implementation. |
| **6** | $F_2$, Critical MSE, and ESA Score Formulas | **VERIFIED (Exact Match)** | $F_2$ ($\beta=2$), critical MSE ($y \ge -6.0$), and $\text{Score} = \text{MSE}_{\text{crit}} / F_2$ are mathematically and programmatically identical across paper, code, and our implementation. |
| **7** | Explaining the Score Discrepancy | **RESOLVED** | The paper's 1.337 score was evaluated on the ESA test server (Footnote 4), not on local validation data. Furthermore, 2-range 50/50 balancing provides 2.64x more critical training events than 6-range balancing. |

---

## 2. Table of Audit Findings

The table below classifies findings by severity:
- **Confirmed Bug**: An unambiguous code error, failure, or mathematical defect.
- **Paper vs. Code Discrepancy**: A direct contradiction between the published paper text and the authors' source repository.
- **Evaluation / Protocol Misalignment**: An inconsistency in benchmark targets, dataset horizons, or validation semantics.
- **Reproduction Nuance**: An implementation choice, random variation, or preprocessing detail affecting reproducibility.

| ID | Category | Severity | Area | Evidence | Impact | Proposed Correction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **F-01** | Evaluation / Protocol | **HIGH** | Benchmark Target | Paper Footnote 4 & Table 1 Caption: 1.337 is from the ESA test evaluation server, not local validation. | Comparing local validation score (10.89) to 1.337 created an invalid perception of reproduction failure. | Update documentation to clarify that 1.337 is an ESA challenge test leaderboard score, not local validation. |
| **F-02** | Paper vs. Code | **HIGH** | Training Balancing | Paper Section 3 states 50/50 balance (negative and positive). Code contains both 2-range (132 samples) and 6-range (138 samples) datasets. | Our current model was trained on 6 ranges with only 25 positive samples (~18%), starving the model of critical event diversity. | Implement 2-range 50/50 training balancing (66 positive, 66 negative, 132 events) and evaluate performance. |
| **F-03** | Paper vs. Code | **MEDIUM** | Loss Function | Paper Section 3.2 specifies "MSE loss", while `models/rnn.py` implements custom `weighted_mse_loss` ($1.0$ / $0.1$). | Unweighted MSE causes extreme errors on low-risk events ($\le -25$) to dominate training. Weighted MSE stabilizes learning. | Retain `weighted_mse_loss` as baseline; explicitly document the paper's omission; run unweighted MSE only as an ablation. |
| **F-04** | Confirmed Bug | **MEDIUM** | Sequence Windowing | In `generate_datasets.py`, `generate_test_set` filters $t \ge 2.0$, but `generate_train_set` retains all CDMs down to $t < 1.0$. | Training sequences contain CDMs between $1.0$ and $2.0$ days before TCA, creating a domain mismatch with evaluation sequences. | Provide an option to strictly filter training sequences to $t \ge 2.0$ days prior to the final CDM, aligning train and test horizons. |
| **F-05** | Confirmed Bug | **HIGH** | Original `infer_model` | `pipeline.py` calls `mode='infer'` (unimplemented in `RNN.forward()`), crashes on $T=1$ events, and outputs dummy sequential IDs. | The original authors' inference pipeline was broken and unexecutable for generating official test submissions. | Use our robust `scripts/prepare_test_dataset.py` and `scripts/evaluate_gru_test.py` with explicit single-CDM tracking. |
| **F-06** | Reproduction Nuance | **LOW** | Scaler Pegging | `save_scaler` sets `model.data_min_[tca_id] = 0` after `fit()`. In Scikit-learn, this is a no-op because `transform` uses precomputed `scale_` and `min_`. | Transformed values are unaffected (difference is $4.44 \times 10^{-16}$), but code intent was misunderstood. | Document that post-fit `data_min_` modification in Scikit-learn is a no-op; preserve the exact transformation pipeline. |
| **F-07** | Reproduction Nuance | **MEDIUM** | Test Imputation Leakage | In `generate_datasets.py`, `create_df` calls `df.fillna(df.mean())` directly on the input CSV, imputing test data from test means. | Slight test-set data leakage (transductive imputation). | In rigorous benchmarking, evaluate imputation using training feature means to avoid transductive leakage. |
| **F-08** | Reproduction Nuance | **MEDIUM** | Sampling Variance | `generate_train_set` used `random.shuffle()` without a fixed seed. Ref repo pickles share only 36/138 events between runs. | Extreme variance in which 23 (or 66) events are selected out of thousands of low-risk candidates. | Fix random seeds (`seed=42`) and benchmark across multiple seeds to isolate model capability from random sample selection luck. |
| **F-09** | Evaluation / Protocol | **HIGH** | Test Horizon Truncation | `test_data.csv` contains only CDMs with $t \ge 2.0002$ days. True TCA risk was in private test data. | Evaluating $0 \dots T-2$ against $T-1$ on `test_data.csv` predicts the last pre-cutoff CDM, not final collision risk at TCA. | Explicitly report `test_data.csv` evaluation as an intermediate-CDM generalization test, not a true TCA outcome test. |

---

## 3. Detailed Verification Findings

### Finding F-01: Category Error on Benchmark Score (Table 1 Leaderboard vs. Local Validation)

#### 1. Evidence
- **Published Paper, Page 3, Footnote 4:**
  > *"Note that the ground-truth information was not available for the test data, hence we were unable to quantify the performance of our algorithms over this dataset during the development process. Therefore, the best score is selected based solely on the results calculated by the independent validation server for a subset of the test sample. Additionally, our experiments showed that the results obtained over the training set (or its randomly selected subset treated as our local validation sample) were not necessarily in line with the results obtained over the test data, due to their slightly different characteristics."*
- **Published Paper, Page 4, Section 4 & Table 1:**
  > Table 1 is explicitly titled: *"The results over the incomplete test set (challenge leaderboard)."*
  > The reported GRU score is **1.337**.
- **Original Codebase, `models/pipeline.py`:**
  The validation routine `self.test(test_dataloader)` writes predictions to `val_outputs.csv` and prints `Score: mse / f2_score`. However, there are no saved artifacts or logs showing a local validation score of 1.337.

#### 2. Technical Analysis
The authors did not obtain 1.337 on `val_samples_x.pickle` (their local validation pool of 7,962 operational events). They submitted model predictions to the ESA Kelvins competition evaluation platform, which calculated the score against the hidden ground truth in `test_data_private.csv`.

In our training run (`scripts/train_gru.py`):
- Local `val_operational_all` achieved a best composite score of **10.8989** (Epoch 250, Critical MSE = 3.7804, $F_2 = 0.3469$).
- Local `val_operational_disjoint` achieved a best composite score of **24.3959** (Epoch 250, Critical MSE = 5.8354, $F_2 = 0.2392$).
- Unseen `test_data.csv` evaluation achieved an ESA score of **6.6652** (Critical MSE = 5.5076, $F_2 = 0.8263$, Recall = $90.75\%$).

Our local validation score was not failing to reproduce 1.337; rather, 1.337 was evaluated under a completely different data distribution and scoring server.

---

### Finding F-02: Training Sample Selection and Balancing (Paper 50/50 vs. 6-Range Stratification)

#### 1. Evidence
- **Published Paper, Section 3, Page 3:**
  > *"Because of the extreme imbalance in the training set, each model is trained using an equal number of negative and positive samples [13]. The former ones constitute the majority of the training examples, and a subset of such (low-risk) events are randomly selected in order to exactly match the number of examples belonging to the under-represented (minority) class."*
- **Reference Codebase, `dataset/generate_datasets.py`:**
  ```python
  def generate_train_set(df_path: str, destination_dir: str, file_name: str, n_ranges: int):
      ...
      elif n_ranges == 2:
          if last_risk >= -6 and last_tca < 1 and first_tca > 2:
              train_samples[0].append(sample)
          if last_risk < -6 and last_tca < 1 and first_tca > 2 and sample.to_numpy().shape[0] > 10:
              train_samples[1].append(sample)
  ```
- **Inspected Reference Pickles:**
  - `clean_data/train_samples_2_ranges/train_samples_x_no_augmentation.pickle`: Exactly **132 events** (**66 positive** with $r \ge -6.0$, **66 negative** with $r < -6.0$).
  - `clean_data/train_samples_6_ranges/train_samples_x_no_augmentation.pickle`: Exactly **138 events** (6 bins $\times$ 23 events; only **25 positive** events, **113 negative** events).

#### 2. Technical Analysis
Our initial reproduction implemented only `n_ranges = 6` (138 events) in `scripts/prepare_dataset.py`.
In this 6-range dataset:
- Positive samples constitute only $\frac{25}{138} \approx 18.1\%$ of the training set.
- Negative samples constitute $\frac{113}{138} \approx 81.9\%$ of the training set.

In contrast, the paper explicitly describes a **50/50 balance** (`n_ranges = 2`):
- Positive samples: 66 events ($50.0\%$).
- Negative samples: 66 events ($50.0\%$).

A 50/50 balanced training dataset exposes the GRU to **66 distinct critical trajectories** instead of only 25 per epoch—an increase of **264%** in high-risk training diversity. This directly impacts the model's ability to learn high-risk encounter dynamics and avoid false negatives.

---

### Finding F-03: Weighted MSE Implementation vs. Paper Stated MSE Objective

#### 1. Evidence
- **Published Paper, Section 3.2, Page 3:**
  > *"The models were trained with the MSE loss function using the Adam optimizer [23]..."*
- **Reference Codebase, `models/rnn.py`, Lines 64–72:**
  ```python
  def weighted_mse_loss(self, y_hat: torch.Tensor, y: torch.Tensor, threshold: int = -6):
      numpy_y = y.clone().detach().cpu().numpy()
      where = np.where(numpy_y < threshold)[0]
      weight = np.ones((numpy_y.shape[0]))
      weight[where] = weight[where] * 0.1
      weight = torch.from_numpy(weight).type(torch.FloatTensor).to(self.device)
      return torch.mean(weight * (y_hat - y) ** 2)
  ```

#### 2. Technical Analysis
The paper text omits the weighting scheme and states that standard MSE was used. However, the code explicitly downweights low-risk samples by a factor of 10 ($w = 0.1$ for $y < -6.0$, $w = 1.0$ for $y \ge -6.0$).

In collision risk prediction on $\log_{10}(P_c)$ scale:
- High-risk events span $[-6.0, -1.0]$ (a range of 5 log units).
- Low-risk events span $[-30.0, -6.0)$ (a range of 24 log units).

Without weighting, an error between $-25$ and $-15$ contributes $(10)^2 = 100.0$ to standard MSE, completely dwarfing an error between $-6$ and $-4$ ($4.0$), even though the latter is the critical operational boundary. The weighted MSE loss was an essential engineering mechanism in the authors' code, despite being omitted from the paper's text.

Our PyTorch implementation in `src/models/gru.py` faithfully implements `compute_weighted_mse_loss()` matching `models/rnn.py`.

---

### Finding F-04: Sequence Windowing Asymmetry (Train vs. Validation)

#### 1. Evidence
- **Published Paper, Section 2.1:**
  > *"The goal of the competition was to predict the probability of collision for each event, based on the CDMs that were recorded up to two days prior the encounter..."*
- **Reference Codebase, `dataset/generate_datasets.py`:**
  - In `generate_test_set` (lines 48–66):
    ```python
    up_2days_sample = sample[sample['time_to_tca'] >= 2]
    new_sample = pd.concat([up_2days_sample, pd.DataFrame(sample.iloc[-1]).transpose()],
                           axis=0, ignore_index=True)
    ```
    Only CDMs with $t \ge 2.0$ days are kept before the final CDM is appended.
  - In `generate_train_set` (lines 68–125):
    ```python
    sample = df[df['event_id'] == event].reset_index(drop=True)
    ...
    train_samples[i].append(sample)
    ```
    No filtering by $t \ge 2.0$ days is performed.

#### 2. Technical Analysis
When `models/rnn.py` slices off the final timestep (`event = event[:, :-1]`):
- **For Validation:** Timesteps $0 \dots T-2$ have $\text{time\_to\_tca} \ge 2.0$ days.
- **For Training:** Timesteps $0 \dots T-2$ include all intermediate CDMs between $1.0$ and $2.0$ days before TCA.

This creates an inherent domain shift: the GRU learns transition dynamics on fine-grained timesteps close to TCA during training, but is evaluated on truncated sequences cut off 2 days before TCA. Our `scripts/prepare_dataset.py` faithfully matched this behavior to preserve reproduction fidelity, but this constitutes an architectural asymmetry in the original research workflow.

---

### Finding F-05: Deficiencies in Reference `infer_model`

#### 1. Evidence
- **Reference Codebase, `models/pipeline.py`, Lines 95–123:**
  ```python
  @staticmethod
  def infer_model(model: Module, scaler: str, model_path: str, test_df_path: str):
      ...
      with torch.no_grad():
          for sample in tqdm(dataloader, total=len(dataloader)):
              y_hat, y = model.forward(sample, mode='infer')
  ```
- **Reference Codebase, `models/rnn.py`, Lines 59–62:**
  ```python
  if mode.lower() == 'train':
      return self.weighted_mse_loss(outputs, targets)
  elif mode.lower() == 'test':
      return outputs.cpu().numpy(), targets.cpu().numpy()
  ```

#### 2. Technical Analysis
If `infer_model` is executed:
1. `mode.lower() == 'infer'` matches neither `'train'` nor `'test'`, causing `forward()` to return `None`.
2. Slicing `event[:, :-1]` on events with length 1 ($T=1$, of which there are 75 in `test_data.csv`) passes an empty tensor with sequence length 0 to `nn.GRU`, raising:
   `RuntimeError: Expected sequence length to be larger than 0 in RNN`.
3. The exported dataframe uses `list(range(len(outputs)))` as `event_id`, decoupling predictions from actual space conjunction identifiers.

Our evaluation script `scripts/evaluate_gru_test.py` resolved all these defects while accurately reporting the 75 non-evaluable events.

---

### Finding F-06: Scaler Pegging Was a Codebase No-Op

#### 1. Evidence
- **Reference Codebase, `dataset/generate_datasets.py`, Lines 130–136:**
  ```python
  model = preprocessing.MinMaxScaler().fit(df)
  time_to_tca_id = df.columns.get_loc('time_to_tca')
  model.data_min_[time_to_tca_id] = 0
  ```
- **Empirical Check on Scikit-Learn `MinMaxScaler`:**
  In Scikit-learn, `MinMaxScaler.transform(X)` computes:
  $$X_{\text{scaled}} = X \cdot \text{scale\_} + \text{min\_}$$
  where $\text{scale\_}$ and $\text{min\_}$ are calculated during `.fit()`. Manually modifying `model.data_min_` after fitting has zero effect on the output of `transform()`.

#### 2. Technical Analysis
We loaded the authors' `clean_data/scalers/x_scaler.pickle` and compared transformed tensors against our `data/processed/scaler.joblib`. The maximum absolute difference across random inputs was $4.44 \times 10^{-16}$, demonstrating that our fitted scaler reproduces the authors' scaler to numerical precision.

---

### Finding F-07: Transductive Imputation on Test Data

#### 1. Evidence
- **Reference Codebase, `dataset/generate_datasets.py`, Lines 34–42:**
  ```python
  def create_df(df_path: str, file_name: str, mode: str):
      df = pd.read_csv(df_path)
      df.drop(columns=['mission_id'], inplace=True)
      df = df.fillna(df.mean())
  ```

#### 2. Technical Analysis
When `create_df` was run on `test_data.csv`, missing values were imputed using the column means of `test_data.csv`, not `train_data.csv`.
In strict machine learning practice, this represents transductive test-set data leakage. For example:
- `t_rcs_estimate`: Train mean = $2.9605$, Test mean = $2.8222$.
- `t_crdot_r`: Train mean = $0.0893$, Test mean = $0.1065$.

While the numerical differences are relatively minor, a leakage-free benchmark pipeline should impute test observations using training set statistics.

---

### Finding F-08: High Sampling Variance on Small Training Sets

#### 1. Evidence
- In `generate_datasets.py`, `random.shuffle()` is called without fixing a random seed.
- Comparing `data/train_samples_x.pickle` and `clean_data/train_samples_6_ranges/train_samples_x_no_augmentation.pickle` (both generated by the authors) revealed an overlap of only **36 out of 138 events** (a 73.9% divergence in event selection).

#### 2. Technical Analysis
Because the raw training set contains thousands of low-risk events satisfying the filtering criteria, each random shuffle selects a vastly different subset of 23 events per bin. On such small sample sizes (132–138 events), model training trajectories and generalization performance exhibit high sensitivity to the random sample draw.

---

### Finding F-09: Test Ground Truth Horizon Limitation ($t \ge 2.0$ days)

#### 1. Evidence
- In `test_data.csv`, all 24,484 rows have $\text{time\_to\_tca} \ge 2.0002$ days.
- The true collision risk at TCA ($t < 1.0$ day) was withheld by ESA in `test_data_private.csv` (which must never be accessed).

#### 2. Technical Analysis
Our evaluation on `test_data.csv` evaluated predicting the risk at CDM $T-1$ (where $t \ge 2.0$ days) from historical CDMs $0 \dots T-2$. This measures intermediate risk extrapolation before the 2-day decision gate, not final collision risk at TCA. This explains why test metrics showed higher recall ($90.75\%$) and a different composite score ($6.6652$) compared to the competition leaderboard server evaluation ($1.337$).

---

## 4. Analysis of Score Discrepancies

| Setting | Published Paper | Authors' Codebase | Our Reproduction | Key Explanatory Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Training Set Size** | "Equal negative & positive" | 132 (2 ranges) or 138 (6 ranges) | 138 (6 ranges) | We trained on 6 ranges with only 25 critical events instead of 66. |
| **Loss Function** | "MSE loss" | `weighted_mse_loss` ($1.0$ / $0.1$) | `weighted_mse_loss` ($1.0$ / $0.1$) | Faithfully matches source code; paper text omitted weighting. |
| **Validation Pool** | Not reported in Table 1 | `val_samples_x.pickle` (7,962 events) | 7,962 (all) / 7,833 (disjoint) | Local validation pool contains 7,962 operational events. |
| **Val Score (All)** | N/A | Variable across checkpoints | **10.8989** (Epoch 250) | High penalty from false positives on imbalanced operational pool. |
| **Val Score (Disjoint)**| N/A | Not evaluated separately | **24.3959** (Epoch 250) | True generalization score excluding all 133 overlapping train events. |
| **Test Set Score** | **1.337** (Leaderboard) | Not published in repo | **6.6652** (`test_data.csv`) | Paper 1.337 was evaluated by ESA server against hidden private labels. |

---

## 5. Classification: Confirmed Bugs vs. Uncertain Differences

### A. Confirmed Bugs
1. **Broken `infer_model` in Reference Code:** `mode='infer'` is missing in `RNN.forward`, sequence length 0 causes crash on $T=1$ test events, and sequential integer IDs overwrite actual `event_id`.
2. **Post-fit Scaler Pegging No-Op:** Setting `model.data_min_[time_to_tca_id] = 0` in Scikit-learn after `fit()` does not alter `scale_` or `min_` and has zero effect on scaling.
3. **Train vs. Validation Sequence Windowing Asymmetry:** Training sequences contain CDMs down to $t < 1.0$ day, while validation sequences strictly truncate at $t \ge 2.0$ days.

### B. Paper vs. Code Discrepancies
1. **Loss Function:** Paper states standard MSE; codebase implements weighted MSE ($1.0$ for high-risk, $0.1$ for low-risk).
2. **Balancing Strategy:** Paper states equal negative and positive samples (50/50 balance); codebase provides both 2-range (132 samples) and 6-range (138 samples) generators.
3. **Reported Score Benchmark Context:** Paper Table 1 score of 1.337 is an ESA challenge test leaderboard result, not a local training/validation metric.

### C. Uncertain Reproduction Differences
1. **Exact Random Seed for Event Subsampling:** Without an explicit seed in the authors' code, the exact 23 (or 66) events selected in the authors' historical run cannot be known deterministically.
2. **Augmentation Usage:** While unaugmented pickles exist, the authors also generated $2\times$ Gaussian noise augmented datasets (`train_samples_x_gaussian_augmentation_2_times.pickle`, 414 samples). It is uncertain whether their best leaderboard submission used augmentation.

---

## 6. Smallest Set of Recommended Corrections Before Retraining

To resolve the identified discrepancies and optimize the GRU baseline before retraining:

1. **Implement 2-Range 50/50 Balanced Dataset (`train_sequences_2_ranges.joblib`):**
   - Add `n_ranges = 2` generator in `scripts/prepare_dataset.py` matching `generate_datasets.py` lines 112–117.
   - Filter events: `first_tca > 2.0`, `last_tca < 1.0`. Positive class: $r_{\text{last}} \ge -6.0$ (66 events); Negative class: $r_{\text{last}} < -6.0$, $\text{len} > 10$, subsampled to 66 events.
   - Total balanced training set: **132 events** (50% critical vs. current 18.1%).
   - Expected Impact: 2.64x more critical event gradient updates, directly improving high-risk recall and $F_2$ score.

2. **Evaluate Both Balancing Strategies Head-to-Head:**
   - Train GRU on `train_sequences_2_ranges.joblib` (132 events, 50/50 balance) and compare with the existing `train_sequences_6_ranges.joblib` (138 events, 6-range stratification).
   - This directly tests whether the paper's stated 50/50 balancing outperforms the 6-range stratification.

3. **Align Sequence Cutoff Option:**
   - Provide a clean flag in preprocessing to evaluate training with strict $t \ge 2.0$ cutoff on historical CDMs, eliminating the windowing domain shift between training and evaluation.

4. **Retain Verified Components Without Modification:**
   - Retain exact network architecture: 2 GRU layers (hidden size 36) + 4-layer MLP head (`36 -> 128 -> 256 -> 64 -> 1` with PReLU).
   - Retain Adam optimizer with $\text{lr} = 0.001$.
   - Retain `weighted_mse_loss` ($w=1.0$ for $y \ge -6.0$, $w=0.1$ for $y < -6.0$).
   - Retain modern `MinMaxScaler` fitted on training features.
   - Retain official ESA metric evaluation formulas.

---

## 7. Validation Data Leakage Resolution & Disjoint Dataset Semantics

### 7.1 Observed Diagnostic Anomaly
During evaluation of the 2-range balanced training pipeline, the following discrepancies were observed:
1. `train_gru.py` derived 7,837 disjoint validation events from `val_operational_all` (7,962 events), reporting 0 critical events (`target_risk >= -6.0`), which caused critical MSE and composite score to evaluate to `NaN`.
2. In contrast, the static artifact `data/processed/val_operational_disjoint.joblib` on disk contained 7,833 events with 39 critical events.
3. If `val_operational_disjoint.joblib` was evaluated against the 2-range model, it exhibited an overlap of **102 training events** (including **39 critical events**), representing severe training-validation data leakage.

### 7.2 Verified Root Cause Analysis
A rigorous audit of `train_data.csv` (13,154 unique events) revealed the exact mathematical root cause:
- **Operational Critical Candidate Population:** Across all 13,154 events in `train_data.csv`, exactly 365 events have a final CDM risk $\ge -6.0$. When applying the paper's critical training criteria (`last_risk >= -6.0`, `last_tca < 1.0`, `first_tca > 2.0`), exactly **66 events** qualify.
- **50/50 Balance Exhaustion:** The paper's 2-range training methodology mandates undersampling both classes to `min_len = min(66, 6698) = 66`. Thus, `train_sequences_2_ranges.joblib` **must select all 66 eligible critical events**.
- **Operational Validation Pool:** The authors' validation selection criteria (`0.0 <= last_tca <= 1.0`, `first_tca >= 2.0`) select 7,962 events in `val_operational_all.joblib`. Exactly **61 events** in this pool are critical ($\ge -6.0$).
- **Complete Ingestion of Critical Events:** All 61 critical events in `val_operational_all` satisfy `last_tca < 1.0` and `first_tca > 2.0`. Consequently, **all 61 operational critical candidates were selected into `train_sequences_2_ranges.joblib`** (the remaining 5 events in `train_2_ranges` have `last_tca < 0.0`).
- **Mathematical Inevitability of 0 Critical Disjoint Events:**
  $$\text{val\_critical} \setminus \text{train\_critical} = \emptyset \implies \text{critical\_count} = 0$$
  Any strictly disjoint validation subset derived from `val_operational_all` by excluding `train_sequences_2_ranges` IDs contains exactly $7,962 - 125 = 7,837$ events and **identically 0 critical events**.
- **Origin of the Stale 7,833-Event Artifact:**
  `data/processed/val_operational_disjoint.joblib` was generated during the initial 6-range training run. The 6-range training set (`train_sequences_6_ranges.joblib`, 138 events) only used 24 critical events (22 of which overlapped with `val_operational_all`), leaving $61 - 22 = 39$ critical events in the disjoint set ($7,962 - 129 = 7,833$ events).
  When `train_sequences_2_ranges.joblib` was generated, `val_operational_disjoint.joblib` remained on disk as an un-renamed 6-range artifact. Evaluating it against a 2-range trained model leaked 102 training events (including all 39 critical events).

### 7.3 Implemented Fix
1. **Dynamic Derivation as Source of Truth:**
   In `scripts/train_gru.py`, disjoint validation is dynamically derived at runtime by index slicing:
   ```python
   train_eids = set(s["event_id"] for s in train_samples)
   disjoint_indices = np.array([
       i for i, s in enumerate(val_all_samples) if s["event_id"] not in train_eids
   ], dtype=np.int64)
   ```
   This guarantees zero overlap with the loaded training dataset, ensures 100% provenance from `val_operational_all`, and strictly preserves input features and target risks.
2. **Explicit Artifact Disambiguation:**
   Regenerated dataset-specific disjoint validation artifacts in `data/processed/`:
   - `val_operational_disjoint_2_ranges.joblib` (7,837 events, 0 critical events).
   - `val_operational_disjoint_6_ranges.joblib` (7,833 events, 39 critical events).
   - Updated `data/processed/preprocessing_metadata.json` with explicit descriptions, stats, and event IDs for both modes.
3. **Transparent Logging & Diagnostic Guardrails:**
   Enhanced `train_gru.py` to report critical counts for both validation pools and output explicit diagnostic notes when `critical_count == 0` explaining that `critical_mse = NaN` and `score = NaN` are expected mathematically.

### 7.4 Regression Test Verification
Created `scripts/test_validation_leakage.py` containing 16 automated checks:
- **Test 01–08 (2-Range Path):** Verified zero ID overlap (0 events), 100% provenance from `val_operational_all` (7,837 events), exact target/sequence preservation, critical count = 0, non-critical count = 7,837, mathematical proof of 61/61 critical event ingestion, robust handling of NaN metrics, and isolated the 102-event leakage in the stale artifact.
- **Test 09–14 (6-Range Path):** Verified zero ID overlap (0 events), 100% provenance (7,833 events), exact target/sequence preservation, critical count = 39, non-critical count = 7,794, finite metric computation (`critical_mse`, `f2_score`, `score`), and matching saved artifact.
- **Test 15–16 (Checkpoint Integrity):** Verified `checkpoints/best_gru_2_ranges_checkpoint.pt` exists, has valid size (>900 KB), epoch 250, and valid `model_state_dict`.
- **Result:** **16/16 tests passed**.

### 7.5 Remaining Limitations & Guidance
- **Local Validation Trade-off for 2-Range Models:** Because 2-range 50/50 balanced training exhausts all operational critical events in the competition training split, local validation cannot simultaneously be 100% disjoint and contain critical events.
- **Evaluation Recommendation:**
  1. For tracking overall non-critical regression MSE on unseen operational trajectories: evaluate on `val_operational_disjoint_2_ranges.joblib` (7,837 events).
  2. For monitoring model stability on operational critical events: evaluate on `val_operational_all.joblib` (7,962 events), noting that 125 events overlap with the training set (faithful to the original authors' evaluation script).
  3. For genuine generalization benchmarking: evaluate on `test_data.csv` (24,484 rows, 0 overlap with training data).

---
*End of Audit Document.*
