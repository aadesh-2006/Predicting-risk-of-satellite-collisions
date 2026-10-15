"""
Model Evaluation on test_data.csv for Satellite GRU Baseline
Tulczyjew et al. (2021) / ESA Kelvins Collision Avoidance Challenge

Evaluates the trained GRU model (checkpoints/best_gru_checkpoint.pt)
on data/processed/test_sequences.joblib.

Outputs:
- results/gru_test_metrics.json
- results/gru_test_predictions.csv
- Clean console printout with ESA metrics:
    * Total test events
    * Evaluated test events (seq_len >= 2)
    * Single-message test events (seq_len == 1, skipped)
    * F2 score (beta=2, threshold=-6.0)
    * Critical MSE (y >= -6.0)
    * Overall MSE
    * ESA Score (Critical MSE / F2)
    * TP, FP, FN, TN
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import csv
from typing import Dict, List, Any

import joblib
import numpy as np
import torch

from src.models.gru import SatelliteGRU

# Paths
CHECKPOINT_PATH = Path(r"C:\Users\Aadesh\.gemini\antigravity\scratch\Predicting-risk-of-satellite-collisions\checkpoints\best_gru_checkpoint.pt")
TEST_SEQUENCES_PATH = Path(r"C:\Users\Aadesh\.gemini\antigravity\scratch\Predicting-risk-of-satellite-collisions\data\processed\test_sequences.joblib")
RESULTS_DIR = Path(r"C:\Users\Aadesh\.gemini\antigravity\scratch\Predicting-risk-of-satellite-collisions\results")
METRICS_JSON_PATH = RESULTS_DIR / "gru_test_metrics.json"
PREDICTIONS_CSV_PATH = RESULTS_DIR / "gru_test_predictions.csv"


def compute_official_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    threshold: float = -6.0,
    beta: float = 2.0,
    eps: float = 1e-12
) -> Dict[str, Any]:
    """
    Computes official ESA metrics matching original authors' implementation:
    - Overall MSE across all evaluated samples
    - Confusion matrix (TP, FP, FN, TN) with positive class = risk >= -6.0
    - F2 score with beta=2
    - Critical MSE computed strictly on true positive/critical events (y_true >= -6.0)
    - ESA Score = Critical MSE / F2
    """
    overall_mse = float(np.mean((y_pred - y_true) ** 2))

    y_pred_bin = (y_pred >= threshold)
    y_true_bin = (y_true >= threshold)

    tp = int((y_pred_bin & y_true_bin).sum())
    fp = int((y_pred_bin & (~y_true_bin)).sum())
    fn = int(((~y_pred_bin) & y_true_bin).sum())
    tn = int(((~y_pred_bin) & (~y_true_bin)).sum())

    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))

    beta_sq = beta ** 2
    f2 = float((1 + beta_sq) * precision * recall / (beta_sq * precision + recall + eps))
    if f2 == 0.0:
        f2 = 1e-8  # Clipped as in original source

    crit_indices = np.where(y_true >= threshold)[0]
    if len(crit_indices) > 0:
        crit_mse = float(np.mean((y_pred[crit_indices] - y_true[crit_indices]) ** 2))
    else:
        crit_mse = float("nan")

    esa_score = float(crit_mse / f2) if not np.isnan(crit_mse) else float("nan")

    return {
        "overall_mse": overall_mse,
        "critical_mse": crit_mse,
        "f2_score": f2,
        "precision": precision,
        "recall": recall,
        "esa_score": esa_score,
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        },
        "critical_events_count": int(len(crit_indices)),
        "non_critical_events_count": int(len(y_true) - len(crit_indices)),
        "total_evaluated_events": int(len(y_true))
    }


def main():
    print("=" * 80)
    print("SATELLITE COLLISION RISK PREDICTION: GRU TEST EVALUATION")
    print("=" * 80)

    # 1. Device and checkpoint
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    assert CHECKPOINT_PATH.exists(), f"Checkpoint not found at {CHECKPOINT_PATH}"

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    epoch = checkpoint.get("epoch", "unknown")
    print(f"Loaded checkpoint saved at Epoch {epoch}")

    # 2. Instantiate and load model
    model = SatelliteGRU(input_size=64, hidden_size=36, num_layers=2, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Load preprocessed test dataset
    print(f"Loading test sequences from: {TEST_SEQUENCES_PATH}")
    assert TEST_SEQUENCES_PATH.exists(), f"Test sequences file not found at {TEST_SEQUENCES_PATH}"
    dataset_payload = joblib.load(TEST_SEQUENCES_PATH)

    evaluable_events = dataset_payload["evaluable_events"]
    skipped_events = dataset_payload["skipped_events"]
    all_events = dataset_payload["all_events"]

    total_events = len(all_events)
    n_eval = len(evaluable_events)
    n_skipped = len(skipped_events)

    print(f"\nEvent counts:")
    print(f"  Total test events in test_data.csv:       {total_events:,}")
    print(f"  Evaluable events (sequence length >= 2):  {n_eval:,}")
    print(f"  Skipped events (sequence length == 1):    {n_skipped:,}")

    # 4. Inference loop over evaluable events
    print(f"\nRunning inference over {n_eval} evaluable events...")
    predictions = []
    targets = []
    records_out = []

    with torch.no_grad():
        for event in evaluable_events:
            eid = event["event_id"]
            seq = torch.from_numpy(event["input_sequence"]).float().to(device)
            target = event["target_risk"]

            # GRU forward pass: expects (T-1, 64) -> batched internally as (1, T-1, 64)
            pred = float(model(seq).item())

            predictions.append(pred)
            targets.append(target)

            records_out.append({
                "event_id": eid,
                "seq_len": event["seq_len"],
                "num_input_steps": event["num_input_steps"],
                "last_observed_tca": event["last_observed_tca"],
                "first_observed_tca": event["first_observed_tca"],
                "target_risk": target,
                "predicted_risk": pred,
                "absolute_error": abs(pred - target),
                "squared_error": (pred - target) ** 2,
                "is_critical_true": int(target >= -6.0),
                "is_critical_pred": int(pred >= -6.0),
                "evaluable": 1
            })

    # Add skipped events to prediction records with NaN/None values for completeness and auditability
    for event in skipped_events:
        eid = event["event_id"]
        target = event["target_risk"]
        records_out.append({
            "event_id": eid,
            "seq_len": event["seq_len"],
            "num_input_steps": 0,
            "last_observed_tca": event["last_observed_tca"],
            "first_observed_tca": event["first_observed_tca"],
            "target_risk": target,
            "predicted_risk": float("nan"),
            "absolute_error": float("nan"),
            "squared_error": float("nan"),
            "is_critical_true": int(target >= -6.0),
            "is_critical_pred": float("nan"),
            "evaluable": 0
        })

    # Sort output records by event_id for consistent ordering
    records_out.sort(key=lambda r: r["event_id"])

    # 5. Compute official metrics on evaluable events
    y_pred_arr = np.array(predictions, dtype=np.float32)
    y_true_arr = np.array(targets, dtype=np.float32)
    metrics = compute_official_metrics(y_pred_arr, y_true_arr, threshold=-6.0, beta=2.0)

    # 6. Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    full_metrics_payload = {
        "dataset_source": "test_data.csv (ESA Kelvins Collision Avoidance Challenge)",
        "checkpoint_file": str(CHECKPOINT_PATH),
        "checkpoint_epoch": int(epoch),
        "evaluation_summary": {
            "total_test_events": total_events,
            "evaluated_events": n_eval,
            "skipped_single_cdm_events": n_skipped,
            "evaluation_coverage_pct": float(n_eval / total_events * 100.0)
        },
        "metrics": metrics,
        "validation_comparison": {
            "val_operational_all_best_checkpoint": checkpoint.get("metrics", {}).get("val_operational_all", {}),
            "val_operational_disjoint_best_checkpoint": checkpoint.get("metrics", {}).get("val_operational_disjoint", {})
        },
        "evaluation_protocol_limitations": {
            "tca_horizon": "All observations in test_data.csv have time_to_tca >= 2.0002 days. The final encounter outcome at TCA (time_to_tca < 1.0 day) was withheld by ESA in test_data_private.csv.",
            "prediction_target": "The ground truth risk column in test_data.csv represents the intermediate CDM risk estimate at t >= 2.0 days (specifically at the last available pre-cutoff CDM), not the final collision event risk at TCA.",
            "single_cdm_events": "75 events consist of exactly 1 CDM. Because historical sequence slicing requires CDMs up to T-2 (leaving T-1 as the target), T=1 events have an empty input sequence and cannot be evaluated by the recurrent model."
        }
    }

    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(full_metrics_payload, f, indent=2)
    print(f"Saved evaluation metrics to: {METRICS_JSON_PATH}")

    # Save predictions CSV
    fieldnames = [
        "event_id", "seq_len", "num_input_steps", "last_observed_tca",
        "first_observed_tca", "target_risk", "predicted_risk",
        "absolute_error", "squared_error", "is_critical_true",
        "is_critical_pred", "evaluable"
    ]
    with open(PREDICTIONS_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records_out:
            writer.writerow(r)
    print(f"Saved predictions for all {len(records_out)} events to: {PREDICTIONS_CSV_PATH}")

    # 7. Print clean readable summary
    cm = metrics["confusion_matrix"]
    print("\n" + "=" * 80)
    print("TEST EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Test Events:             {total_events:,}")
    print(f"Evaluated Events (seq_len>=2): {n_eval:,} ({n_eval/total_events*100:.2f}%)")
    print(f"Skipped Events (seq_len=1):    {n_skipped:,} ({n_skipped/total_events*100:.2f}%)")
    print("-" * 80)
    print(f"Overall MSE:                   {metrics['overall_mse']:.4f}")
    print(f"Critical MSE (y >= -6.0):      {metrics['critical_mse']:.4f}")
    print(f"F2 Score (beta=2):             {metrics['f2_score']:.6f}")
    print(f"ESA Score (Crit MSE / F2):     {metrics['esa_score']:.4f}")
    print(f"Precision:                     {metrics['precision']:.4f}")
    print(f"Recall:                        {metrics['recall']:.4f}")
    print("-" * 80)
    print(f"Confusion Matrix (Threshold = -6.0):")
    print(f"  True Positives  (TP): {cm['tp']}")
    print(f"  False Positives (FP): {cm['fp']}")
    print(f"  False Negatives (FN): {cm['fn']}")
    print(f"  True Negatives  (TN): {cm['tn']}")
    print(f"  Total True Critical Events:     {metrics['critical_events_count']}")
    print(f"  Total True Non-Critical Events: {metrics['non_critical_events_count']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
