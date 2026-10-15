"""
Training and Evaluation Harness for Satellite Collision GRU Baseline
Tulczyjew et al. (2021) / DunderMifflin Reproduction.

Settings:
- Epochs: 510
- Batch size: 1 (shuffled each epoch)
- Evaluation cadence: Every 25 epochs
- Datasets evaluated:
  1. val_operational_all (7,962 events): Authors' evaluation pool
  2. val_operational_disjoint (7,833 events): Leakage-free generalization pool
- Loss: Weighted MSE (1.0 for risk >= -6, 0.1 for risk < -6)
- Metrics: F2 score, Critical MSE (y >= -6), Official Score (Critical MSE / F2), Overall MSE
- Checkpointing: Saved based on best composite score
- Output logs: results/training_metrics.csv and results/training_summary.json
"""

import os
import sys
import time
import json
import random
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru import SatelliteGRU

# Paths
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train_sequences_6_ranges.joblib"
VAL_ALL_FILE = PROCESSED_DATA_DIR / "val_operational_all.joblib"
VAL_DISJOINT_FILE = PROCESSED_DATA_DIR / "val_operational_disjoint.joblib"

# Reproducibility
RANDOM_SEED = 42
TOTAL_EPOCHS = 510
EVAL_INTERVAL = 25
LEARNING_RATE = 0.001
THRESHOLD_CRITICAL = -6.0


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(
    y_hat: np.ndarray,
    y: np.ndarray,
    threshold: float = THRESHOLD_CRITICAL,
    beta: float = 2.0
) -> Dict[str, float]:
    """
    Computes official ESA Kelvins / Tulczyjew et al. competition metrics:
    - Overall MSE across all validation events
    - F2-score on binary classification (risk >= -6.0)
    - Critical MSE computed strictly on true positive critical events (y >= -6.0)
    - Composite Score = Critical MSE / F2
    """
    assert len(y_hat) == len(y), f"Length mismatch: {len(y_hat)} vs {len(y)}"
    eps = 1e-12

    # Overall MSE
    overall_mse = float(np.mean((y_hat - y) ** 2))

    # Binary classification at threshold
    y_hat_bin = (y_hat >= threshold)
    y_bin = (y >= threshold)

    tp = int((y_hat_bin & y_bin).sum())
    fp = int((y_hat_bin & (~y_bin)).sum())
    fn = int(((~y_hat_bin) & y_bin).sum())
    tn = int(((~y_hat_bin) & (~y_bin)).sum())

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    beta_sq = beta ** 2
    f2 = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + eps)
    if f2 == 0:
        f2 = 1e-8  # Clipped as in original source code

    # Critical MSE (computed strictly where true risk >= -6.0)
    critical_indices = np.where(y >= threshold)[0]
    if len(critical_indices) > 0:
        crit_mse = float(np.mean((y_hat[critical_indices] - y[critical_indices]) ** 2))
    else:
        crit_mse = float("nan")

    # Composite Score
    score = crit_mse / f2 if not np.isnan(crit_mse) else float("nan")

    return {
        "overall_mse": overall_mse,
        "critical_mse": crit_mse,
        "f2_score": float(f2),
        "score": float(score),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "critical_count": len(critical_indices)
    }


def evaluate_all_and_disjoint(
    model: SatelliteGRU,
    val_all_samples: List[Dict[str, Any]],
    disjoint_indices: np.ndarray,
    device: torch.device
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluates model over val_operational_all in a single forward pass,
    then computes both val_all metrics and disjoint metrics via index slicing.
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for sample in val_all_samples:
            x_seq = torch.from_numpy(sample["input_sequence"]).float().to(device)
            target = sample["target_risk"]

            pred = model(x_seq).item()
            predictions.append(pred)
            targets.append(target)

    y_hat_arr = np.array(predictions, dtype=np.float32)
    y_arr = np.array(targets, dtype=np.float32)

    metrics_all = compute_metrics(y_hat_arr, y_arr)
    metrics_disjoint = compute_metrics(y_hat_arr[disjoint_indices], y_arr[disjoint_indices])
    return metrics_all, metrics_disjoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train SatelliteGRU baseline on 2-range or 6-range dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="6-range",
        choices=["2-range", "6-range", "2", "6"],
        help="Training dataset selection: '2-range' (132 events, 50/50 balance) or '6-range' (138 events, 6 risk bins). Default: '6-range'"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Optional experiment name tag for custom checkpoint and result filenames (e.g. '6_range_retrain')"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="weighted_mse",
        choices=["weighted_mse", "standard_mse", "mse"],
        help="Loss function: 'weighted_mse' (weights 1.0/0.1, authors' code default) or 'standard_mse'/'mse' (unweighted MSE, as stated in paper text). Default: 'weighted_mse'"
    )
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS, help="Number of training epochs (default: 510)")
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL, help="Evaluation interval in epochs (default: 25)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed (default: 42)")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate (default: 0.001)")
    parser.add_argument("--dry-run", action="store_true", help="Perform a 1-batch dry run without full training")
    return parser.parse_args()


def train(args=None):
    if args is None:
        args = parse_args()

    seed = args.seed
    epochs = args.epochs
    eval_interval = args.eval_interval
    lr = args.lr

    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device for execution: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine dataset paths and output artifact names based on selected dataset and optional exp_name
    is_2_range = args.dataset in ("2-range", "2")
    exp_name = getattr(args, "exp_name", None)

    if exp_name:
        dataset_name = f"{'2-range' if is_2_range else '6-range'} (exp: {exp_name})"
        train_data_file = PROCESSED_DATA_DIR / ("train_sequences_2_ranges.joblib" if is_2_range else "train_sequences_6_ranges.joblib")
        best_checkpoint_path = CHECKPOINT_DIR / f"best_gru_{exp_name}_checkpoint.pt"
        final_checkpoint_path = CHECKPOINT_DIR / f"final_gru_{exp_name}_checkpoint.pt"
        metrics_csv_path = RESULTS_DIR / f"training_metrics_{exp_name}.csv"
        summary_path = RESULTS_DIR / f"training_summary_{exp_name}.json"
        expected_events = 132 if is_2_range else 138
    elif is_2_range:
        dataset_name = "2-range (paper-aligned 50/50 balance)"
        train_data_file = PROCESSED_DATA_DIR / "train_sequences_2_ranges.joblib"
        best_checkpoint_path = CHECKPOINT_DIR / "best_gru_2_ranges_checkpoint.pt"
        final_checkpoint_path = CHECKPOINT_DIR / "final_gru_2_ranges_checkpoint.pt"
        metrics_csv_path = RESULTS_DIR / "training_metrics_2_ranges.csv"
        summary_path = RESULTS_DIR / "training_summary_2_ranges.json"
        expected_events = 132
    else:
        dataset_name = "6-range (authors' 6-risk-bin stratification)"
        train_data_file = PROCESSED_DATA_DIR / "train_sequences_6_ranges.joblib"
        best_checkpoint_path = CHECKPOINT_DIR / "best_gru_checkpoint.pt"
        final_checkpoint_path = CHECKPOINT_DIR / "final_gru_checkpoint.pt"
        metrics_csv_path = RESULTS_DIR / "training_metrics.csv"
        summary_path = RESULTS_DIR / "training_summary.json"
        expected_events = 138

    print(f"\n[1/4] Loading preprocessed datasets for {dataset_name}...")
    assert train_data_file.exists(), f"Training data file not found: {train_data_file}"
    train_samples = joblib.load(train_data_file)
    val_all_samples = joblib.load(VAL_ALL_FILE)

    train_eids = set(s["event_id"] for s in train_samples)
    disjoint_indices = np.array([
        i for i, s in enumerate(val_all_samples) if s["event_id"] not in train_eids
    ], dtype=np.int64)

    pos_count = sum(1 for s in train_samples if s["target_risk"] >= THRESHOLD_CRITICAL)
    neg_count = sum(1 for s in train_samples if s["target_risk"] < THRESHOLD_CRITICAL)
    crit_val_all = sum(1 for s in val_all_samples if s["target_risk"] >= THRESHOLD_CRITICAL)
    crit_disjoint = sum(1 for idx in disjoint_indices if val_all_samples[idx]["target_risk"] >= THRESHOLD_CRITICAL)

    loss_arg = getattr(args, "loss", "weighted_mse").lower()
    is_standard_mse = loss_arg in ("standard_mse", "mse")
    loss_name = "standard_mse" if is_standard_mse else "weighted_mse"
    loss_desc = "Standard Unweighted MSE (paper-text specification)" if is_standard_mse else "Weighted MSE (authors' code specification: 1.0/0.1)"

    print(f"Loaded {len(train_samples)} training events from {train_data_file.name} (expected {expected_events}).")
    print(f"  Class balance: {pos_count} high-risk (>= -6.0), {neg_count} low-risk (< -6.0)")
    print(f"  Loss function: {loss_desc}")
    print(f"Loaded {len(val_all_samples)} validation events (val_operational_all: {crit_val_all} critical, {len(val_all_samples) - crit_val_all} non-critical).")
    print(f"Derived {len(disjoint_indices)} disjoint validation events (val_operational_disjoint: {crit_disjoint} critical, {len(disjoint_indices) - crit_disjoint} non-critical).")
    if crit_disjoint == 0:
        print("  [DIAGNOSTIC NOTE] Derived disjoint validation set has 0 critical events (target_risk >= -6.0).")
        print("                    The 2-range balanced training set exhausts all 61 operational critical candidates.")
        print("                    Consequently, disjoint critical MSE and composite score are mathematically undefined (NaN).")
    print(f"Artifact paths:")
    print(f"  - Best Checkpoint:  {best_checkpoint_path}")
    print(f"  - Final Checkpoint: {final_checkpoint_path}")
    print(f"  - Metrics CSV:      {metrics_csv_path}")
    print(f"  - Summary JSON:     {summary_path}")

    print("\n[2/4] Initializing SatelliteGRU model...")
    model = SatelliteGRU(
        input_size=64,
        hidden_size=36,
        num_layers=2,
        dropout=0.0,
        lr=lr,
        device=device
    )

    if args.dry_run:
        print("\n[DRY RUN] Executing 1 forward and backward pass on 1 sample to verify training pipeline...")
        sample0 = train_samples[0]
        x_seq = torch.from_numpy(sample0["input_sequence"]).float().to(device)
        y_target = torch.tensor([sample0["target_risk"]], dtype=torch.float32, device=device)
        model.optimizer.zero_grad()
        y_hat = model(x_seq)
        loss = model.compute_loss(y_hat, y_target, loss_type=loss_name, threshold=THRESHOLD_CRITICAL)
        loss.backward()
        print(f"[DRY RUN SUCCESS] Input shape: {x_seq.shape}, Output: {y_hat.item():.4f}, Target: {y_target.item():.4f}, Loss: {loss.item():.4f}")
        print("Dry run completed cleanly. No checkpoints or metric files were overwritten.")
        return

    csv_headers = [
        "epoch", "train_loss",
        "val_all_score", "val_all_crit_mse", "val_all_f2", "val_all_mse",
        "val_disjoint_score", "val_disjoint_crit_mse", "val_disjoint_f2", "val_disjoint_mse"
    ]

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    print(f"\n[3/4] Starting training for {epochs} epochs (batch_size=1, evaluating every {eval_interval} epochs)...")
    print("-" * 110)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'ValAll Score':<12} | {'ValAll CritMSE':<14} | {'ValAll F2':<10} | "
          f"{'ValDisj Score':<13} | {'ValDisj CritMSE':<15} | {'ValDisj F2':<10}")
    print("-" * 110)

    best_score = float("inf")
    best_epoch = -1
    best_metrics_summary = {}
    m_all = None
    m_disjoint = None

    start_time = time.time()
    rng = random.Random(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        # Shuffle training events each epoch
        epoch_samples = list(train_samples)
        rng.shuffle(epoch_samples)

        for sample in epoch_samples:
            x_seq = torch.from_numpy(sample["input_sequence"]).float().to(device)
            y_target = torch.tensor([sample["target_risk"]], dtype=torch.float32, device=device)

            model.optimizer.zero_grad()
            y_hat = model(x_seq)
            loss = model.compute_loss(y_hat, y_target, loss_type=loss_name, threshold=THRESHOLD_CRITICAL)
            loss.backward()
            model.optimizer.step()

            epoch_losses.append(loss.item())

        avg_train_loss = float(np.mean(epoch_losses))

        # Evaluate at interval and final epoch
        if epoch % eval_interval == 0 or epoch == epochs:
            m_all, m_disjoint = evaluate_all_and_disjoint(model, val_all_samples, disjoint_indices, device)

            print(
                f"\r{epoch:<6} | {avg_train_loss:<10.4f} | "
                f"{m_all['score']:<12.4f} | {m_all['critical_mse']:<14.4f} | {m_all['f2_score']:<10.4f} | "
                f"{m_disjoint['score']:<13.4f} | {m_disjoint['critical_mse']:<15.4f} | {m_disjoint['f2_score']:<10.4f}"
            )

            # Log to CSV
            with open(metrics_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, round(avg_train_loss, 4),
                    round(m_all["score"], 4), round(m_all["critical_mse"], 4), round(m_all["f2_score"], 4), round(m_all["overall_mse"], 4),
                    round(m_disjoint["score"], 4), round(m_disjoint["critical_mse"], 4), round(m_disjoint["f2_score"], 4), round(m_disjoint["overall_mse"], 4)
                ])

            # Checkpoint on best val_all composite score (faithful to authors' pipeline)
            if m_all["score"] < best_score:
                best_score = m_all["score"]
                best_epoch = epoch
                best_metrics_summary = {
                    "best_epoch": best_epoch,
                    "train_loss": avg_train_loss,
                    "val_operational_all": m_all,
                    "val_operational_disjoint": m_disjoint
                }

                torch.save({
                    "epoch": epoch,
                    "dataset": dataset_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": model.optimizer.state_dict(),
                    "metrics": best_metrics_summary
                }, best_checkpoint_path)
        else:
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train_loss:8.4f} | Training...", end="\r", flush=True)

    total_duration = time.time() - start_time
    print("-" * 110)
    print(f"Training completed in {total_duration:.1f}s ({total_duration/60:.2f} minutes).")

    # Save final checkpoint
    torch.save({
        "epoch": epochs,
        "dataset": dataset_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
        "final_train_loss": avg_train_loss
    }, final_checkpoint_path)

    # Save detailed summary JSON
    summary_data = {
        "model": "SatelliteGRU",
        "dataset": dataset_name,
        "loss_function": loss_name,
        "loss_description": loss_desc,
        "training_data_file": str(train_data_file),
        "training_events_count": len(train_samples),
        "critical_events_count": pos_count,
        "non_critical_events_count": neg_count,
        "training_duration_seconds": round(total_duration, 2),
        "total_epochs": epochs,
        "learning_rate": lr,
        "random_seed": seed,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_checkpoint_path),
        "final_checkpoint": str(final_checkpoint_path),
        "best_metrics": best_metrics_summary,
        "paper_comparison": {
            "published_score": 1.337,
            "measured_val_operational_all_score": round(best_metrics_summary["val_operational_all"]["score"], 4),
            "measured_val_operational_disjoint_score": round(best_metrics_summary["val_operational_disjoint"]["score"], 4)
        }
    }

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[4/4] Summary saved to {summary_path}")
    print(f"\n============================================================")
    print(f"  Best Checkpoint (Epoch {best_epoch}):")
    print(f"  - Dataset:                  {dataset_name}")
    print(f"  - val_operational_all:      Score = {best_metrics_summary['val_operational_all']['score']:.4f} "
          f"(CritMSE={best_metrics_summary['val_operational_all']['critical_mse']:.4f}, F2={best_metrics_summary['val_operational_all']['f2_score']:.4f})")
    print(f"  - val_operational_disjoint: Score = {best_metrics_summary['val_operational_disjoint']['score']:.4f} "
          f"(CritMSE={best_metrics_summary['val_operational_disjoint']['critical_mse']:.4f}, F2={best_metrics_summary['val_operational_disjoint']['f2_score']:.4f})")
    print(f"  - Paper Published Reference: Score ~ 1.337 (Test Leaderboard)")
    print(f"============================================================")


if __name__ == "__main__":
    train()
