# GRU Baseline Reproduction Plan

> **Reference Paper:** Tulczyjew et al. (2021), *"Predicting risk of satellite collisions using machine learning"*, Journal of Space Safety Engineering / ESA Collision Avoidance Challenge.  
> **Source Repository:** [GitLab: jnalepa/dundermifflin](https://gitlab.com/jnalepa/dundermifflin)  
> **Dataset:** ESA Kelvins Collision Avoidance Challenge (`train_data.csv`, `test_data.csv`)

This document establishes the verified technical specifications, architectural parameters, data transformation rules, and execution protocol required to reproduce the paper's GRU regressor baseline locally and faithfully.

All statements, components, and decisions in this plan are classified under one of three audit labels:
* **[Verified]**: Directly verified against the authors' repository source code (`dundermifflin`) or inspection of the dataset and pickled objects.
* **[Inferred]**: Deduced with strong technical justification based on consistent codebase patterns, paper methodology, or competition rules.
* **[Unresolved]**: Identified ambiguities or edge cases in the original code/paper where explicit minimal fixes are proposed.

---

## 1. Step-by-Step Preprocessing Pipeline

### Step 1.1: Raw Row Ingestion and Metadata Removal
* **[Verified]** Data files `train_data.csv` (162,634 rows) and `test_data.csv` (24,484 rows) contain 103 columns each.
* **[Verified]** `mission_id` is discarded immediately in `dataset/dataset.py`, `dataset/generate_datasets.py`, and `dataset/balanced_datasets.py`.
* **[Verified]** `event_id` is retained purely as a grouping identifier across CDM sequences and excluded from numeric features.

### Step 1.2: Missing Value Handling
* **[Verified]** 50 columns contain `NaN` values (notably `c_rcs_estimate` with ~32.5% missing, and chaser velocity/covariance cross-terms `c_crdot_*`, `c_ctdot_*`, `c_cndot_*` with ~5.7% missing).
* **[Verified]** In the original source code (`dataset/generate_datasets.py` and `dataset/balanced_datasets.py`), missing values are imputed using column-wise mean: `df = df.fillna(df.mean())`.
* **[Unresolved]** In modern Pandas (>= 2.0), calling `df.fillna(df.mean())` without `numeric_only=True` throws a `TypeError` due to string columns (`c_object_type`).
  * *Minimal Safe Fix:* Compute `df.mean(numeric_only=True)` and fill numeric columns prior to or following one-hot encoding.

### Step 1.3: Categorical Encoding
* **[Verified]** `c_object_type` is the only categorical string feature.
* **[Verified]** In the original code, it is encoded using `pd.get_dummies(df, prefix=['c_object_type'])`.
* **[Verified]** The 5 distinct categories present across the dataset are:
  - `c_object_type_DEBRIS`
  - `c_object_type_PAYLOAD`
  - `c_object_type_ROCKET BODY`
  - `c_object_type_TBA`
  - `c_object_type_UNKNOWN`

### Step 1.4: Feature Fusion Modes (`x` vs `tc`)
The authors explored two distinct feature representations:
1. **Combined `x` mode (`data_x_corrected.csv`):**
   * **[Verified]** Sums corresponding target satellite (`t_*`) and chaser object (`c_*`) attributes for 40 shared physical/orbital covariance columns:
     $$\text{feature}_{\text{combined}} = \text{col}_t + \text{col}_c$$
     (e.g., `x_sigma_r = t_sigma_r + c_sigma_r`).
   * **[Verified]** Keeps 21 unique encounter/environmental features intact (`time_to_tca`, `risk`, `max_risk_estimate`, `max_risk_scaling`, `miss_distance`, `relative_speed`, `relative_position_*`, `relative_velocity_*`, `geocentric_latitude`, `azimuth`, `elevation`, `F10`, `AP`, `F3M`, `SSN`, and dummy `c_object_type_*`).
   * **[Verified]** Produces exactly 63 normalized feature columns (excluding `event_id` and `risk`).
2. **Raw `tc` mode (`data_tc_corrected.csv`):**
   * **[Verified]** Retains all target and chaser columns separately as 104 feature columns.
* **[Inferred]** The primary baseline configuration described in `models/pipeline.py` and the saved scaler `x_scaler.pickle` corresponds to the combined `x` representation (63 features + 1 risk channel = 64 inputs).

### Step 1.5: Feature Normalization
* **[Verified]** Feature scaling is performed using `sklearn.preprocessing.MinMaxScaler` (not `StandardScaler`).
* **[Verified]** Pickled scalers in `clean_data/scalers/x_scaler.pickle` and `tc_scaler.pickle` were unpickled and confirmed to be `MinMaxScaler` instances fitted with Scikit-learn 0.21.3.
* **[Verified]** Special constraint on time: in `save_scaler()`, the minimum bound for `time_to_tca` is forced to 0:
  `model.data_min_[time_to_tca_id] = 0` to preserve the zero-boundary at Time of Closest Approach.
* **[Verified]** Features scaled: all numeric features except `event_id` and `risk`.
* **[Verified]** The `risk` target column remains in raw $\log_{10}(P_c)$ scale (ranges from $-30.0$ to $\approx -1.44$) and is not min-max scaled in the GRU regressor pipeline.

---

## 2. Event Sequencing, Windowing, and Target Alignment

### Step 2.1: Chronological Sequence Construction
* **[Verified]** Each event consists of a chronological series of CDMs sorted by strictly decreasing `time_to_tca`.
* **[Verified]** Variable sequence length: events vary in CDM count (average ~12 messages in train).

### Step 2.2: Event Filtering and Balancing (`generate_train_set`)
Because critical high-risk conjunctions are exceedingly rare, the authors enforced strict filtering and class balancing:
* **[Verified]** Criteria for valid training sequences:
  1. `first_tca > 2.0` (encounter observed at least 2 days prior to TCA).
  2. `last_tca < 1.0` (encounter tracked within 1 day of TCA).
  3. For non-critical events: `len(event) > 10` (must have at least 11 CDMs).
* **[Verified]** 6-Range Stratification:
  Events are partitioned into 6 risk bins based on the final CDM risk $r_{\text{last}}$:
  * Bin 0: $0 \ge r_{\text{last}} \ge -5$
  * Bin 1: $-5 > r_{\text{last}} \ge -10$
  * Bin 2: $-10 > r_{\text{last}} \ge -15$
  * Bin 3: $-15 > r_{\text{last}} \ge -20$
  * Bin 4: $-20 > r_{\text{last}} \ge -25$
  * Bin 5: $-25 > r_{\text{last}} \ge -30$
* **[Verified]** Undersampling to Minority Class:
  Bin 0 contains only 23 events in the entire training set. Each of the remaining 5 bins is randomly subsampled to match this count (`min_len = 23`).
  $$\text{Total Balanced Training Events} = 6 \times 23 = 138 \text{ events}$$
  (This exact count of 138 is verified from `clean_data/train_samples_6_ranges/train_samples_x_no_augmentation.pickle`).
* **[Verified]** 2-Range Alternative:
  Binary thresholding at $r_{\text{last}} \ge -6$ (66 critical events) balances down to $2 \times 66 = 132$ events (`train_samples_2_ranges/train_samples_x_no_augmentation.pickle`).

### Step 2.3: Validation/Test Set Construction (`generate_test_set`)
* **[Verified]** Evaluation events are filtered by:
  `0 <= last_tca <= 1` AND at least one CDM with `time_to_tca >= 2`.
* **[Verified]** Slicing structure:
  $$\text{Input Subsequence} = \{\text{CDMs where } \text{time\_to\_tca} \ge 2.0\} \cup \{\text{final CDM at } t < 1.0\}$$
* **[Verified]** During the forward pass, the tensor is sliced:
  `event = event[:, :-1]`
  This strictly strips the final CDM, ensuring the GRU only consumes observations up to 2 days before TCA ($\text{time\_to\_tca} \ge 2$), while the regression target is set to the stripped final CDM risk:
  $$\text{target} = \text{risk}_{\text{final}}$$

### Step 2.4: Protocol Windowing Discrepancy (Train vs. Validation)
* **[Verified]** **Challenge Rule:** The ESA Kelvins Competition and paper problem statement specify predicting the final risk at TCA using observations available at least two days before TCA ($\text{time\_to\_tca} \ge 2.0$).
* **[Verified]** **Validation Windowing:** `generate_test_set` strictly adheres to this rule by pre-filtering `sample[sample['time_to_tca'] >= 2]` before appending `sample.iloc[-1]`. Slicing off `[:-1]` in `forward` leaves only observations $\ge 2.0$ days before TCA as inputs.
* **[Verified]** **Training Windowing:** `generate_train_set` does **not** pre-filter by $t \ge 2.0$; it keeps the entire event sequence (`sample`) and slices off only `[:-1]`. Consequently, training inputs include intermediate CDMs occurring between $1.0$ and $2.0$ days before TCA.
* **[Verified]** This intentional or artifactual asymmetry in the original code explains why the training sequences have a higher average length ($\sim 17.8$ steps) than the validation sequences ($\sim 12.1$ steps). Both datasets are faithfully reproduced according to the authors' exact code.

### Step 2.5: Feature Concatenation at Forward Pass
* **[Verified]** In `models/rnn.py`, each CDM timestep vector is formed by concatenating the raw `risk` of that CDM to its normalized feature vector:
  `event = torch.cat([torch.unsqueeze(torch.unsqueeze(risk, 0), -1), features], 2)`
* **[Verified]** Total input dimensionality to the GRU:
  $$\text{Input Dimension} = 63 \text{ features} + 1 \text{ risk channel} = 64$$

---

## 3. GRU Model Architecture and Training Configuration

### Step 3.1: Network Architecture
* **[Verified]** Source implementation in `models/rnn.py` (`class RNN`):
  * **Recurrent Core:** PyTorch `nn.GRU`:
    - `input_size`: 64
    - `hidden_size`: 36
    - `num_layers`: 2
    - `batch_first`: `True`
    - `dropout`: 0.0
  * **Sequence Aggregation:** Last recurrent output step taken:
    `y_hat = self.rnn(event)[0][:, -1]` (shape: `[1, 36]`).
  * **Regression MLP Head (`self.hat`):**
    ```python
    nn.Sequential(
        nn.Linear(36, 128),
        nn.PReLU(),
        nn.Linear(128, 256),
        nn.PReLU(),
        nn.Linear(256, 64),
        nn.PReLU(),
        nn.Linear(64, 1)
    )
    ```
    *Output:* Single scalar predicted risk $\hat{y} \in \mathbb{R}$.

### Step 3.2: Optimizer and Hyperparameters
* **[Verified]** Optimizer: `torch.optim.Adam(self.parameters())` with default PyTorch learning rate ($\eta = 0.001$).
* **[Verified]** Batch Size: 1 (`batch_size=1` in `data.DataLoader` with `custom_collate`).
* **[Verified]** Training Epochs: 510 epochs (`pipeline.train(510, test_dataloader)`).
* **[Verified]** Evaluation Frequency: Evaluated every 25 epochs (`if i % 25 == 0`).
* **[Verified]** Model Checkpointing: Early stopping / checkpointing patience parameter `patience=1`, comparing minimum score over the patience window.

### Step 3.3: Training Loss Function
* **[Verified]** Custom Weighted MSE Loss (`weighted_mse_loss` in `models/rnn.py`):
  $$\mathcal{L}_{\text{train}} = \frac{1}{N} \sum_{i=1}^N w_i \cdot (\hat{y}_i - y_i)^2$$
  where:
  $$w_i = \begin{cases} 1.0 & \text{if } y_i \ge -6.0 \quad (\text{high-risk critical}) \\ 0.1 & \text{if } y_i < -6.0 \quad (\text{low-risk}) \end{cases}$$

---

## 4. Evaluation Protocol and Official Metrics

### Step 4.1: Compound Score Computation
The evaluation follows the official ESA Collision Avoidance Challenge benchmark implemented in `models/pipeline.py`:
1. **$F_2$-Score on Risk Threshold:**
   * Ground truth and predictions binarized at critical threshold $\tau = -6.0$:
     $$y_{\text{bin}} = (y \ge -6.0), \quad \hat{y}_{\text{bin}} = (\hat{y} \ge -6.0)$$
   * Precision: $P = \frac{TP}{TP + FP + 10^{-12}}$
   * Recall: $R = \frac{TP}{TP + FN + 10^{-12}}$
   * $F_2$-score ($\beta = 2$ emphasizes recall):
     $$F_2 = (1 + 2^2) \frac{P \cdot R}{2^2 \cdot P + R + 10^{-12}} = \frac{5 \cdot P \cdot R}{4P + R + 10^{-12}}$$
     (Clipped at $10^{-8}$ if 0).
2. **Mean Squared Error on High-Risk Events ($\text{MSE}_{\text{crit}}$):**
   * Computed strictly on events where ground truth is critical:
     $$\text{MSE}_{\text{crit}} = \frac{1}{|I_{\text{crit}}|} \sum_{i \in I_{\text{crit}}} (\hat{y}_i - y_i)^2, \quad I_{\text{crit}} = \{i \mid y_i \ge -6.0\}$$
3. **Combined Competition Score:**
   $$\text{Score} = \frac{\text{MSE}_{\text{crit}}}{F_2}$$
   *(Lower score is better. Paper reported best GRU score $\approx 1.337$).*
4. **Standard Regression MSE:** Overall MSE across all validation samples is also reported.

---

## 5. Inconsistencies and Differences Identified

| Dimension | Paper Description | Original Source Code (`dundermifflin`) | Current Raw Dataset / Environment | Resolution / Status |
| :--- | :--- | :--- | :--- | :--- |
| **Feature Scaler** | Mentions Min-Max scaling | Uses `MinMaxScaler` in `dataset.py` & `generate_datasets.py`; but `balanced_datasets.py` imports `StandardScaler` | Serialized scalers in `clean_data` are `MinMaxScaler` (v0.21.3) | **[Verified]** `MinMaxScaler` is the intended scaler. |
| **Target Normalization** | Mentions normalized values in some context | `balanced_datasets.py` lines 156 & 160 comment out risk normalization | Targets remain unscaled $\log_{10}(P_c) \in [-30, -1]$ | **[Verified]** Regressor outputs raw unscaled log risk. |
| **Model Configs** | Focuses on GRU baseline | Contains GRU, LSTM, classical Random Forest, and S2S LSTM models | `models/pipeline.py` explicitly instantiates GRU with hidden 36, 2 layers | **[Verified]** Identified exact GRU configuration. |
| **Script Execution** | N/A | Hardcoded empty strings `''` at root of modules cause import crashes | Script cannot be imported directly in Python | **[Verified]** Must implement modular clean scripts with `if __name__ == '__main__':`. |
| **Library Versions** | 2020/2021 versions | Scikit-learn 0.21.3, PyTorch ~1.4, NumPy < 1.20 | Python 3.12, Scikit-learn 1.4, NumPy 1.26 | **[Verified]** Replace deprecated calls (`np.int` $\to$ `int`, `df.mean()` string handling). |

---

## 6. Audit Trail of Decisions

* **[Verified]** Input feature dimension: 64 ($63 \text{ physical features} + 1 \text{ risk feature}$).
* **[Verified]** Output dimension: 1 (final scalar collision risk).
* **[Verified]** GRU architecture: 2 layers, hidden size 36, batch size 1, Adam optimizer, 510 epochs.
* **[Verified]** Loss formulation: Weighted MSE ($w=1.0$ if $y \ge -6$, else $w=0.1$).
* **[Verified]** Primary Evaluation Metric: $\text{Score} = \text{MSE}_{\text{crit}} / F_2$.
* **[Inferred]** Training set split: The authors balanced the training set using 6 bins of 23 samples each (138 samples), while validation is evaluated across the filtered pool of realistic operational events (~7,962 events in train).
* **[Unresolved]** Deterministic train/validation split seed: In `generate_datasets.py`, `shuffle()` is called without explicit seed fixing.
  * *Minimal Safe Fix:* Use a fixed reproducible random seed (`seed=42`) for train/val split generation so metrics are strictly reproducible.

---

## 7. Next Immediate Implementation Step

With dataset inspection complete and the complete data flow, model architecture, loss function, and evaluation metrics verified against the authors' repository:

**Next Single Step:**  
Develop the self-contained, reproducible data preprocessing and dataset generator module under `src/preprocessing/` or `scripts/prepare_dataset.py` to convert raw `train_data.csv` into clean, scaled, balanced event sequences matching the verified schema without importing legacy scripts.
