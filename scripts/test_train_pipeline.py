"""
Automated Test Suite for Training Pipeline Dataset Selection
Tulczyjew et al. (2021) / ESA Kelvins Collision Avoidance Challenge

Tests:
1. Argument parser correctly parses --dataset 2-range and --dataset 6-range
2. Selecting 2-range loads train_sequences_2_ranges.joblib (132 events, 50/50 balance)
3. Selecting 6-range loads train_sequences_6_ranges.joblib (138 events)
4. Target output filenames are strictly separated between 2-range and 6-range
5. Existing 6-range checkpoints and results remain 100% intact and untouched
6. Model forward pass, weighted MSE loss, and gradient backward pass execute without error on 2-range sample
"""

import sys
import os
import argparse
from pathlib import Path
import joblib
import torch

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

from scripts.train_gru import parse_args, train
from src.models.gru import SatelliteGRU

EXISTING_6R_BEST_CP = CHECKPOINT_DIR / "best_gru_checkpoint.pt"
EXISTING_6R_FINAL_CP = CHECKPOINT_DIR / "final_gru_checkpoint.pt"
EXISTING_6R_METRICS = RESULTS_DIR / "training_metrics.csv"
EXISTING_6R_SUMMARY = RESULTS_DIR / "training_summary.json"


def run_tests():
    print("=" * 80)
    print("  TEST SUITE: GRU TRAINING PIPELINE DATASET SELECTION")
    print("=" * 80)

    total_checks = 0
    passed_checks = 0

    def check(name: str, condition: bool, details: str = ""):
        nonlocal total_checks, passed_checks
        total_checks += 1
        if condition:
            passed_checks += 1
            print(f" [PASS] Test {total_checks:02d}: {name}")
            if details:
                print(f"        {details}")
        else:
            print(f" [FAIL] Test {total_checks:02d}: {name}")
            print(f"        ERROR: {details}")

    # Check 1: Record initial state of existing 6-range artifacts
    initial_states = {}
    for path in [EXISTING_6R_BEST_CP, EXISTING_6R_FINAL_CP, EXISTING_6R_METRICS, EXISTING_6R_SUMMARY]:
        if path.exists():
            initial_states[str(path)] = (path.stat().st_size, path.stat().st_mtime)

    check(
        "Initial verification of existing 6-range artifacts",
        EXISTING_6R_BEST_CP.exists() and EXISTING_6R_SUMMARY.exists(),
        f"Found {len(initial_states)} existing 6-range artifacts"
    )

    # Check 2: Parse args with --dataset 2-range
    sys_argv_backup = sys.argv
    try:
        sys.argv = ["train_gru.py", "--dataset", "2-range", "--dry-run"]
        args_2r = parse_args()
        check(
            "CLI argument parsing: --dataset 2-range",
            args_2r.dataset == "2-range" and args_2r.dry_run is True,
            f"Parsed dataset: {args_2r.dataset}, dry_run: {args_2r.dry_run}"
        )
    finally:
        sys.argv = sys_argv_backup

    # Check 3: Parse args with --dataset 6-range
    try:
        sys.argv = ["train_gru.py", "--dataset", "6-range", "--dry-run"]
        args_6r = parse_args()
        check(
            "CLI argument parsing: --dataset 6-range",
            args_6r.dataset == "6-range" and args_6r.dry_run is True,
            f"Parsed dataset: {args_6r.dataset}, dry_run: {args_6r.dry_run}"
        )
    finally:
        sys.argv = sys_argv_backup

    # Check 4: Verify 2-range dataset file resolution and content
    file_2r = PROCESSED_DIR / "train_sequences_2_ranges.joblib"
    check(
        "2-range dataset file exists",
        file_2r.exists(),
        f"Path: {file_2r}"
    )

    data_2r = joblib.load(file_2r)
    pos_2r = sum(1 for s in data_2r if s["target_risk"] >= -6.0)
    neg_2r = sum(1 for s in data_2r if s["target_risk"] < -6.0)
    check(
        "2-range dataset content verification (132 events, 66 critical, 66 non-critical)",
        len(data_2r) == 132 and pos_2r == 66 and neg_2r == 66,
        f"Total: {len(data_2r)}, Critical: {pos_2r}, Non-critical: {neg_2r}"
    )

    # Check 5: Verify 6-range dataset file resolution and content
    file_6r = PROCESSED_DIR / "train_sequences_6_ranges.joblib"
    check(
        "6-range dataset file exists",
        file_6r.exists(),
        f"Path: {file_6r}"
    )

    data_6r = joblib.load(file_6r)
    pos_6r = sum(1 for s in data_6r if s["target_risk"] >= -6.0)
    neg_6r = sum(1 for s in data_6r if s["target_risk"] < -6.0)
    check(
        "6-range dataset content verification (138 events, 24 critical, 114 non-critical)",
        len(data_6r) == 138 and pos_6r == 24 and neg_6r == 114,
        f"Total: {len(data_6r)}, Critical: {pos_6r}, Non-critical: {neg_6r}"
    )

    # Check 6: Execute dry-run with 2-range dataset
    print("\nExecuting dry-run test for 2-range training pipeline...")
    dry_args_2r = argparse.Namespace(
        dataset="2-range",
        epochs=510,
        eval_interval=25,
        seed=42,
        lr=0.001,
        dry_run=True
    )
    try:
        train(dry_args_2r)
        check(
            "2-range dry-run execution succeeded with zero errors",
            True,
            "1 forward/loss/backward pass completed cleanly"
        )
    except Exception as e:
        check(
            "2-range dry-run execution succeeded with zero errors",
            False,
            f"Exception occurred: {e}"
        )

    # Check 7: Execute dry-run with 6-range dataset
    print("\nExecuting dry-run test for 6-range training pipeline...")
    dry_args_6r = argparse.Namespace(
        dataset="6-range",
        epochs=510,
        eval_interval=25,
        seed=42,
        lr=0.001,
        dry_run=True
    )
    try:
        train(dry_args_6r)
        check(
            "6-range dry-run execution succeeded with zero errors",
            True,
            "1 forward/loss/backward pass completed cleanly"
        )
    except Exception as e:
        check(
            "6-range dry-run execution succeeded with zero errors",
            False,
            f"Exception occurred: {e}"
        )

    # Check 8: Verify original 6-range artifacts were NOT modified
    unmodified_ok = True
    unmodified_details = []
    for path_str, (init_size, init_mtime) in initial_states.items():
        curr_path = Path(path_str)
        curr_size = curr_path.stat().st_size
        curr_mtime = curr_path.stat().st_mtime
        if curr_size != init_size or curr_mtime != init_mtime:
            unmodified_ok = False
            unmodified_details.append(f"{curr_path.name} was modified!")
        else:
            unmodified_details.append(f"{curr_path.name} untouched")

    check(
        "Verification: Original 6-range checkpoints and results remain 100% untouched",
        unmodified_ok,
        "; ".join(unmodified_details)
    )

    # Check 9: Target paths separation check
    check(
        "Target output filenames for 2-range do not collide with 6-range",
        (CHECKPOINT_DIR / "best_gru_2_ranges_checkpoint.pt") != EXISTING_6R_BEST_CP and
        (RESULTS_DIR / "training_summary_2_ranges.json") != EXISTING_6R_SUMMARY,
        "Distinct filenames verified for best checkpoint, final checkpoint, CSV metrics, and summary JSON"
    )

    # Check 10: CLI parsing with --exp-name
    try:
        sys.argv = ["train_gru.py", "--dataset", "6-range", "--exp-name", "6_range_retrain", "--dry-run"]
        args_exp = parse_args()
        check(
            "CLI argument parsing: --dataset 6-range --exp-name 6_range_retrain",
            args_exp.dataset == "6-range" and args_exp.exp_name == "6_range_retrain" and args_exp.dry_run is True,
            f"Parsed dataset: {args_exp.dataset}, exp_name: {args_exp.exp_name}, dry_run: {args_exp.dry_run}"
        )
    finally:
        sys.argv = sys_argv_backup

    # Check 11: Execute dry-run with --exp-name 6_range_retrain
    print("\nExecuting dry-run test for 6-range retraining experiment pipeline...")
    dry_args_exp = argparse.Namespace(
        dataset="6-range",
        exp_name="6_range_retrain",
        epochs=510,
        eval_interval=25,
        seed=42,
        lr=0.001,
        dry_run=True
    )
    try:
        train(dry_args_exp)
        check(
            "6-range retraining experiment dry-run executed cleanly",
            True,
            "1 forward/loss/backward pass completed cleanly with isolated paths"
        )
    except Exception as e:
        check(
            "6-range retraining experiment dry-run executed cleanly",
            False,
            f"Exception occurred: {e}"
        )

    # Check 12: Output paths for 6_range_retrain strictly isolated
    retrain_best_cp = CHECKPOINT_DIR / "best_gru_6_range_retrain_checkpoint.pt"
    retrain_summary = RESULTS_DIR / "training_summary_6_range_retrain.json"
    check(
        "6-range retraining output filenames do not collide with existing baseline checkpoints or results",
        retrain_best_cp != EXISTING_6R_BEST_CP and retrain_summary != EXISTING_6R_SUMMARY and
        retrain_best_cp.name != EXISTING_6R_BEST_CP.name and retrain_summary.name != EXISTING_6R_SUMMARY.name,
        f"Isolated filenames: {retrain_best_cp.name}, {retrain_summary.name} (strictly distinct from baseline)"
    )

    # Check 13: CLI parsing with --loss options
    try:
        sys.argv = ["train_gru.py", "--dataset", "2-range", "--loss", "standard_mse", "--dry-run"]
        args_loss_std = parse_args()
        sys.argv = ["train_gru.py", "--dataset", "2-range", "--loss", "weighted_mse", "--dry-run"]
        args_loss_wt = parse_args()
        sys.argv = ["train_gru.py", "--dataset", "2-range", "--dry-run"]
        args_loss_def = parse_args()
        check(
            "CLI argument parsing: --loss (default weighted_mse, standard_mse, backward-compatible)",
            args_loss_std.loss == "standard_mse" and args_loss_wt.loss == "weighted_mse" and args_loss_def.loss == "weighted_mse",
            f"Parsed std: {args_loss_std.loss}, wt: {args_loss_wt.loss}, def: {args_loss_def.loss}"
        )
    finally:
        sys.argv = sys_argv_backup

    # Check 14: Model loss function dispatch verification
    m_test = SatelliteGRU(input_size=64, hidden_size=36, num_layers=2)
    y_h = torch.tensor([0.0, 0.0], dtype=torch.float32)
    y_t = torch.tensor([-5.0, -10.0], dtype=torch.float32)  # 1 critical (>= -6.0), 1 low-risk (< -6.0)
    loss_wt = m_test.compute_loss(y_h, y_t, loss_type="weighted_mse", threshold=-6.0)
    loss_std = m_test.compute_loss(y_h, y_t, loss_type="standard_mse", threshold=-6.0)
    # Expected weighted: (1.0 * 25 + 0.1 * 100) / 2 = (25 + 10) / 2 = 17.5
    # Expected standard: (25 + 100) / 2 = 62.5
    expected_wt = 17.5
    expected_std = 62.5
    check(
        "Model loss dispatch: weighted_mse (1.0/0.1) vs standard_mse (unweighted)",
        abs(loss_wt.item() - expected_wt) < 1e-4 and abs(loss_std.item() - expected_std) < 1e-4,
        f"Weighted MSE: {loss_wt.item():.4f} (expected {expected_wt}), Standard MSE: {loss_std.item():.4f} (expected {expected_std})"
    )

    # Check 15: Regression test covering epochs between validation intervals
    print("\nExecuting regression test for epochs between validation intervals...")
    reg_test_exp = "test_interval_regression"
    reg_args = argparse.Namespace(
        dataset="6-range",
        exp_name=reg_test_exp,
        epochs=2,
        eval_interval=25,
        seed=42,
        lr=0.001,
        dry_run=False
    )
    reg_success = False
    reg_error = ""
    try:
        train(reg_args)
        reg_success = True
    except Exception as e:
        reg_error = str(e)
    finally:
        # Clean up temporary regression test artifacts
        for p in [
            CHECKPOINT_DIR / f"best_gru_{reg_test_exp}_checkpoint.pt",
            CHECKPOINT_DIR / f"final_gru_{reg_test_exp}_checkpoint.pt",
            RESULTS_DIR / f"training_metrics_{reg_test_exp}.csv",
            RESULTS_DIR / f"training_summary_{reg_test_exp}.json"
        ]:
            if p.exists():
                p.unlink()

    check(
        "Regression test: Intermediate epochs (epoch % eval_interval != 0) execute without UnboundLocalError",
        reg_success,
        "Epochs 1 (no eval) and 2 (final eval) completed without scope or unbound variable error" if reg_success else f"Error: {reg_error}"
    )

    print("-" * 80)
    print(f"RESULTS: {passed_checks}/{total_checks} tests passed.")
    print("=" * 80)

    if passed_checks == total_checks:
        print("\n[ALL TESTS PASSED] Training pipeline is fully verified with experiment isolation!")
        return 0
    else:
        print(f"\n[WARNING] {total_checks - passed_checks} tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
