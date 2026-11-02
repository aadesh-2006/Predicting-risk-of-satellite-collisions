# Predicting Risk of Satellite Collisions Using GRU

A machine learning reproduction of the research paper **"Predicting Risk of Satellite Collisions Using Machine Learning"** by Tulczyjew et al. (2021).

This project implements a GRU-based regression model using historical Conjunction Data Messages (CDMs) to estimate satellite collision probability.

## Overview

Space debris and the increasing number of active satellites make collision-risk assessment an important problem in space safety.

This project explores the use of recurrent neural networks, specifically Gated Recurrent Units (GRUs), to learn temporal patterns from CDM sequences and predict the logarithm of collision probability.

The primary objective is to reproduce the baseline methodology and investigate the effect of different loss functions on model performance.

## Research Reference

**Original Paper:**  
Tulczyjew et al. (2021), *Predicting Risk of Satellite Collisions Using Machine Learning*, Journal of Space Safety Engineering.

**Original Implementation:**  
https://gitlab.com/jnalepa/dundermifflin

**Dataset:**  
ESA Kelvins Collision Avoidance Challenge.

## Model Architecture

The project uses a custom PyTorch GRU-based regression architecture.

| Component | Configuration |
|---|---|
| Input features | 64 |
| GRU layers | 2 |
| Hidden size | 36 |
| Batch-first | True |
| Dropout | 0.0 |
| Fully connected layers | 36 → 128 → 256 → 64 → 1 |
| Activation | PReLU |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 1 |
| Training epochs | 510 |
| Random seed | 42 |

The model processes chronological CDM sequences and predicts raw `log10(Pc)` values.

## Methodology

1. Load and preprocess the CDM dataset.
2. Prepare chronological sequences for each conjunction event.
3. Normalize the input features.
4. Train the GRU regression model.
5. Evaluate predictions using collision-risk classification metrics and regression error.
6. Compare standard MSE and weighted MSE loss functions.

A conjunction is classified as critical when:

`log10(Pc) >= -6.0`

## Experiments

Two loss-function configurations were investigated:

- **Standard MSE:** Treats prediction errors uniformly.
- **Weighted MSE:** Assigns different weights to critical and non-critical events. The implementation uses a weight of 1.0 for critical events and 0.1 for non-critical events.

The experiments also examine different dataset configurations, including 2-range and 6-range datasets.

## Results

### Standard MSE — 2-range experiment

The Standard MSE experiment achieved its best validation score at epoch 75.

| Metric | Result |
|---|---:|
| Best epoch | 75 |
| Overall MSE | 137.8573 |
| Critical MSE | 1.3783 |
| F2 Score | 0.2418 |
| Composite Score | **5.7002** |
| True Positives | 56 |
| False Positives | 858 |
| False Negatives | 5 |
| True Negatives | 7043 |

The evaluation was performed on the operational validation pool containing 7,962 events, including 61 critical events.

### Standard MSE vs Weighted MSE

| Metric | Standard MSE | Weighted MSE |
|---|---:|---:|
| Best epoch | 75 | 250 |
| Overall MSE | 137.8573 | 109.1963 |
| Critical MSE | 1.3783 | 2.1286 |
| F2 Score | 0.2418 | 0.2750 |
| Composite Score | **5.7002** | 7.7405 |
| True Positives | 56 | 55 |
| False Positives | 858 | 701 |
| False Negatives | 5 | 6 |
| True Negatives | 7043 | 7200 |

On this evaluation pool, Standard MSE achieved a lower composite score and critical MSE, while Weighted MSE produced fewer false positives and a higher F2 score.

## Important Validation Caveat

The 2-range Standard MSE experiment has a limitation: its training dataset overlaps with the operational validation pool.

The training dataset contained 132 events, and 125 of these events also appeared in the operational validation pool. This includes all 61 critical events in that pool.

Additionally, the disjoint validation set contained zero critical events. Therefore, critical MSE and the composite score could not be meaningfully calculated on that disjoint set.

**Consequently, the reported score of 5.7002 is an experimental validation result, not a leakage-free estimate of generalization performance.**

The results should be interpreted in the context of this validation limitation.

## Evaluation Metrics

The project evaluates model performance using:

- **Overall MSE:** Mean squared error across all evaluated events.
- **Critical MSE:** Mean squared error for critical collision events.
- **F2 Score:** Classification metric that gives more importance to recall.
- **Composite Score:** Combined evaluation score used to assess prediction performance.
- **Confusion Matrix:** True positives, false positives, true negatives, and false negatives.

## Reproducing the Experiments

### Prerequisites

- Python 3.13
- PyTorch
- NumPy
- Pandas
- Scikit-learn

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Predicting-risk-of-satellite-collisions
```

### Install dependencies

Install the dependencies listed in the project's requirements file:

```bash
pip install -r requirements.txt
```

### Run the training pipeline

The default training script runs the recommended `2-range-80-20` weighted MSE baseline:

```powershell
powershell -ExecutionPolicy Bypass -File .\train.ps1
```

Other dataset configurations are available through the training script's dataset options.

## Project Structure

```text
Predicting-risk-of-satellite-collisions/
│
├── checkpoints/       # Saved model checkpoints
├── data/              # Dataset files
├── docs/              # Research notes and experiment documentation
├── results/           # Training metrics and experiment summaries
├── scripts/           # Dataset preparation and evaluation scripts
├── train.ps1          # Training entry point
├── README.md
└── requirements.txt
```

## Limitations

- The experiments use a limited number of training conjunction events.
- Critical collision events are relatively rare, creating class imbalance.
- Validation overlap affects the interpretation of some reported results.
- The reported local validation score is not directly comparable to the paper's competition score, which was evaluated on a separate hidden test set.

## Future Work

- Improve validation methodology using strictly disjoint event splits.
- Investigate better handling of critical-event class imbalance.
- Experiment with alternative recurrent architectures.
- Explore attention-based and Transformer-based sequence models.
- Evaluate performance on independent and unseen conjunction events.

## Acknowledgements

- Tulczyjew et al. (2021), for the original research.
- ESA Kelvins, for the Collision Avoidance Challenge dataset.
- The authors of the original implementation for providing a reference baseline.

---

**Disclaimer:** This repository is an academic reproduction and experimentation project. Its predictions are not intended for operational satellite collision avoidance decisions.