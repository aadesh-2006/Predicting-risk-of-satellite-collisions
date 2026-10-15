"""
Evaluation of Existing GRU Checkpoint on Validation Sets
Evaluates: checkpoints/best_gru_checkpoint.pt
Datasets:
1. data/processed/val_operational_disjoint_6_ranges.joblib (7,833 events, leakage-free)
2. data/processed/val_operational_all.joblib (7,962 events, original validation pool)

Computes official ESA metrics:
- Critical MSE (target_risk >= -6.0)
- F2 Score (beta=2.0)
- Composite ESA Score (Critical MSE / F2)
- Overall MSE
- Confusion matrix (TP, FP, FN, TN)
- Critical event count
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import torch

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru import SatelliteGRU
from scripts.train_gru import compute_metrics, THRESHOLD_CRITICAL

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_gru_checkpoint.pt"
TRAIN_6R_PATH = PROJECT_ROOT / "data" / "processed" / "train_sequences_6_ranges.joblib"
VAL_DISJOINT_6R_PATH = PROJECT_ROOT / "data" / "processed" / "val_operational_disjoint_6_ranges.joblib"
VAL_ALL_PATH = PROJECT_ROOT / "data" / "processed" / "val_operational_all.joblib"


def evaluate_checkpoint():
    print("=" * 80)
    print("  EVALUATION OF EXISTING GRU CHECKPOINT (best_gru_checkpoint.pt)")
    print("=" * 80)

    assert CHECKPOINT_PATH.exists(), f"Checkpoint not found at {CHECKPOINT_PATH}"
    assert TRAIN_6R_PATH.exists(), f"Training dataset not found at {TRAIN_6R_PATH}"
    assert VAL_DISJOINT_6R_PATH.exists(), f"Disjoint validation dataset not found at {VAL_DISJOINT_6R_PATH}"
    assert VAL_ALL_PATH.exists(), f"Validation all dataset not found at {VAL_ALL_PATH}"

    # 1. Load data
    train_data = joblib.load(TRAIN_6R_PATH)
    val_disjoint_data = joblib.load(VAL_DISJOINT_6R_PATH)
    val_all_data = joblib.load(VAL_ALL_PATH)

    train_eids = set(s["event_id"] for s in train_data)
    disjoint_eids = set(s["event_id"] for s in val_disjoint_data)
    val_all_eids = set(s["event_id"] for s in val_all_data)

    overlap = train_eids.intersection(disjoint_eids)
    print(f"\n[Dataset Verification]")
    print(f"  - Training 6-range events:           {len(train_eids):,}")
    print(f"  - Disjoint 6-range val events:       {len(disjoint_eids):,}")
    print(f"  - Operational all val events:        {len(val_all_eids):,}")
    print(f"  - Overlap between train and disjoint: {len(overlap)} (Zero overlap: {len(overlap) == 0})")

    # 2. Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  - Device:                             {device}")
    model = SatelliteGRU(input_size=64, hidden_size=36, num_layers=2, dropout=0.0, device=device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  - Loaded checkpoint from:            {CHECKPOINT_PATH.name} (Epoch {checkpoint.get('epoch', 'N/A')})")

    # 3. Evaluate dataset helper
    def eval_set(dataset, desc=""):
        print(f"\nEvaluating on {desc} ({len(dataset):,} events)...")
        preds, targets = [], []
        with torch.no_grad():
            for sample in dataset:
                x_seq = torch.from_numpy(sample["input_sequence"]).float().to(device)
                pred = model(x_seq).item()
                preds.append(pred)
                targets.append(sample["target_risk"])
        y_hat = np.array(preds, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        return compute_metrics(y_hat, y, threshold=THRESHOLD_CRITICAL)

    m_disjoint = eval_set(val_disjoint_data, "val_operational_disjoint_6_ranges")
    m_all = eval_set(val_all_data, "val_operational_all")

    # 4. Print clean summary tables
    print("\n" + "=" * 80)
    print("  EVALUATION RESULTS SUMMARY (Threshold = -6.0)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'val_operational_disjoint_6_ranges':<33} | {'val_operational_all':<20}")
    print("-" * 80)
    print(f"{'Total Events':<25} | {len(val_disjoint_data):<33} | {len(val_all_data):<20}")
    print(f"{'Critical Count (>= -6)':<25} | {m_disjoint['critical_count']:<33} | {m_all['critical_count']:<20}")
    print(f"{'Critical MSE':<25} | {m_disjoint['critical_mse']:<33.4f} | {m_all['critical_mse']:<20.4f}")
    print(f"{'F2 Score':<25} | {m_disjoint['f2_score']:<33.4f} | {m_all['f2_score']:<20.4f}")
    print(f"{'ESA Composite Score':<25} | {m_disjoint['score']:<33.4f} | {m_all['score']:<20.4f}")
    print(f"{'Overall MSE':<25} | {m_disjoint['overall_mse']:<33.4f} | {m_all['overall_mse']:<20.4f}")
    print(f"{'True Positives (TP)':<25} | {m_disjoint['tp']:<33} | {m_all['tp']:<20}")
    print(f"{'False Positives (FP)':<25} | {m_disjoint['fp']:<33} | {m_all['fp']:<20}")
    print(f"{'False Negatives (FN)':<25} | {m_disjoint['fn']:<33} | {m_all['fn']:<20}")
    print(f"{'True Negatives (TN)':<25} | {m_disjoint['tn']:<33} | {m_all['tn']:<20}")
    print("=" * 80)

    return m_disjoint, m_all


if __name__ == "__main__":
    evaluate_checkpoint()
