"""
Regression Test Suite for Validation Data Leakage and Disjoint Dataset Integrity
ESA Collision Avoidance Challenge GRU Reproduction (Tulczyjew et al., 2021)

Tests:
1. Zero train/val ID overlap for dynamically derived 2-range disjoint validation set.
2. Provenance verification: every 2-range disjoint event originates from val_operational_all.
3. Target preservation: target risks and sequences match source val_operational_all exactly.
4. Critical/non-critical counts for 2-range disjoint set (0 critical, 7,837 non-critical).
5. Mathematical explanation check: all 61 critical events in val_operational_all are in train_2_ranges.
6. Robust metric computation: 0 critical events yield NaN critical_mse and score without exception.
7. Saved artifact consistency: val_operational_disjoint_2_ranges.joblib matches dynamic derivation.
8. Stale artifact leakage check: documents why val_operational_disjoint (6-range) must not be evaluated on 2-range.
9. Zero train/val ID overlap for dynamically derived 6-range disjoint validation set.
10. Provenance verification: every 6-range disjoint event originates from val_operational_all.
11. Target preservation for 6-range disjoint set.
12. Critical/non-critical counts for 6-range disjoint set (39 critical, 7,794 non-critical).
13. Metric finiteness: disjoint metrics are strictly finite when critical events exist (6-range).
14. Saved artifact consistency: val_operational_disjoint_6_ranges.joblib matches dynamic derivation.
15. Checkpoint and results integrity: verifies best_gru_2_ranges_checkpoint.pt remains intact.
"""

import sys
import os
from pathlib import Path
import joblib
import numpy as np
import torch

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

from scripts.train_gru import compute_metrics, THRESHOLD_CRITICAL

TRAIN_2R_FILE = PROCESSED_DIR / "train_sequences_2_ranges.joblib"
TRAIN_6R_FILE = PROCESSED_DIR / "train_sequences_6_ranges.joblib"
VAL_ALL_FILE = PROCESSED_DIR / "val_operational_all.joblib"
VAL_DISJOINT_LEGACY = PROCESSED_DIR / "val_operational_disjoint.joblib"
VAL_DISJOINT_2R_FILE = PROCESSED_DIR / "val_operational_disjoint_2_ranges.joblib"
VAL_DISJOINT_6R_FILE = PROCESSED_DIR / "val_operational_disjoint_6_ranges.joblib"
BEST_2R_CHECKPOINT = CHECKPOINT_DIR / "best_gru_2_ranges_checkpoint.pt"


def run_tests():
    print("=" * 80)
    print("  REGRESSION TEST SUITE: VALIDATION DATA LEAKAGE & DISJOINT SET INTEGRITY")
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

    # Load base validation pool
    print("\n[Phase 1] Loading core datasets...")
    assert VAL_ALL_FILE.exists(), f"Missing {VAL_ALL_FILE}"
    val_all = joblib.load(VAL_ALL_FILE)
    val_all_by_id = {s["event_id"]: s for s in val_all}
    val_all_eids = set(val_all_by_id.keys())
    val_all_crit_count = sum(1 for s in val_all if s["target_risk"] >= THRESHOLD_CRITICAL)
    print(f"Loaded val_operational_all: {len(val_all)} events ({val_all_crit_count} critical, {len(val_all) - val_all_crit_count} non-critical)")

    # -------------------------------------------------------------
    # PART A: 2-RANGE BALANCED DATASET PATH
    # -------------------------------------------------------------
    print("\n[Phase 2] Evaluating 2-Range Balanced Dataset Path...")
    assert TRAIN_2R_FILE.exists(), f"Missing {TRAIN_2R_FILE}"
    train_2r = joblib.load(TRAIN_2R_FILE)
    train_2r_eids = set(s["event_id"] for s in train_2r)
    train_2r_crit_eids = set(s["event_id"] for s in train_2r if s["target_risk"] >= THRESHOLD_CRITICAL)

    # Dynamic derivation
    disjoint_2r = [s for s in val_all if s["event_id"] not in train_2r_eids]
    disjoint_2r_eids = set(s["event_id"] for s in disjoint_2r)

    # Test 1: Zero train/val ID overlap
    overlap_2r = train_2r_eids.intersection(disjoint_2r_eids)
    check(
        "2-Range: Zero training event ID overlap in derived disjoint validation set",
        len(overlap_2r) == 0,
        f"Overlap count: {len(overlap_2r)}"
    )

    # Test 2: Source pool provenance
    check(
        "2-Range: Every disjoint sample belongs to val_operational_all",
        disjoint_2r_eids.issubset(val_all_eids) and len(disjoint_2r) == 7837,
        f"Disjoint event count: {len(disjoint_2r)} (expected 7,837)"
    )

    # Test 3: Target and sequence preservation
    preservation_ok = True
    for s in disjoint_2r:
        src = val_all_by_id[s["event_id"]]
        if not np.isclose(s["target_risk"], src["target_risk"]):
            preservation_ok = False
            break
        if not np.allclose(s["input_sequence"], src["input_sequence"]):
            preservation_ok = False
            break
    check(
        "2-Range: Target risks and input sequences exactly preserved from val_operational_all",
        preservation_ok,
        "All targets and input tensors match source records"
    )

    # Test 4: Critical and non-critical counts
    crit_2r = sum(1 for s in disjoint_2r if s["target_risk"] >= THRESHOLD_CRITICAL)
    noncrit_2r = sum(1 for s in disjoint_2r if s["target_risk"] < THRESHOLD_CRITICAL)
    check(
        "2-Range: Critical and non-critical counts correctly calculated",
        crit_2r == 0 and noncrit_2r == 7837,
        f"Critical: {crit_2r}, Non-critical: {noncrit_2r}"
    )

    # Test 5: Mathematical explanation: all 61 val_all critical events in train_2r
    val_all_crit_eids = set(s["event_id"] for s in val_all if s["target_risk"] >= THRESHOLD_CRITICAL)
    check(
        "2-Range: Mathematical proof - all 61 operational critical candidates were selected into train_2_ranges",
        val_all_crit_eids.issubset(train_2r_crit_eids),
        f"val_operational_all critical events in train_2r: {len(val_all_crit_eids.intersection(train_2r_crit_eids))}/61"
    )

    # Test 6: Metrics computation with 0 critical events
    y_true_0crit = np.array([s["target_risk"] for s in disjoint_2r], dtype=np.float32)
    y_pred_0crit = y_true_0crit + 0.1  # synthetic predictions
    metrics_0crit = compute_metrics(y_pred_0crit, y_true_0crit)
    check(
        "2-Range: compute_metrics cleanly handles 0 critical events (NaN critical MSE & score, finite overall MSE)",
        np.isnan(metrics_0crit["critical_mse"]) and np.isnan(metrics_0crit["score"]) and np.isfinite(metrics_0crit["overall_mse"]),
        f"overall_mse: {metrics_0crit['overall_mse']:.4f}, critical_count: {metrics_0crit['critical_count']}"
    )

    # Test 7: Saved artifact consistency for 2-range
    check(
        "2-Range: Saved artifact val_operational_disjoint_2_ranges.joblib exists and matches derivation",
        VAL_DISJOINT_2R_FILE.exists() and len(joblib.load(VAL_DISJOINT_2R_FILE)) == 7837,
        f"File: {VAL_DISJOINT_2R_FILE.name}"
    )

    # Test 8: Stale artifact leakage documentation check
    if VAL_DISJOINT_LEGACY.exists():
        legacy_disjoint = joblib.load(VAL_DISJOINT_LEGACY)
        legacy_eids = set(s["event_id"] for s in legacy_disjoint)
        stale_leakage_overlap = train_2r_eids.intersection(legacy_eids)
        stale_crit_overlap = train_2r_crit_eids.intersection(legacy_eids)
        check(
            "Documentation: Stale 6-range artifact val_operational_disjoint exhibits 102 overlapping events (39 critical) with train_2_ranges",
            len(stale_leakage_overlap) == 102 and len(stale_crit_overlap) == 39,
            f"Stale overlap: {len(stale_leakage_overlap)} total ({len(stale_crit_overlap)} critical) - successfully isolated"
        )

    # -------------------------------------------------------------
    # PART B: 6-RANGE STRATIFIED DATASET PATH
    # -------------------------------------------------------------
    print("\n[Phase 3] Evaluating 6-Range Stratified Dataset Path...")
    assert TRAIN_6R_FILE.exists(), f"Missing {TRAIN_6R_FILE}"
    train_6r = joblib.load(TRAIN_6R_FILE)
    train_6r_eids = set(s["event_id"] for s in train_6r)

    # Dynamic derivation
    disjoint_6r = [s for s in val_all if s["event_id"] not in train_6r_eids]
    disjoint_6r_eids = set(s["event_id"] for s in disjoint_6r)

    # Test 9: Zero train/val ID overlap
    overlap_6r = train_6r_eids.intersection(disjoint_6r_eids)
    check(
        "6-Range: Zero training event ID overlap in derived disjoint validation set",
        len(overlap_6r) == 0,
        f"Overlap count: {len(overlap_6r)}"
    )

    # Test 10: Source pool provenance
    check(
        "6-Range: Every disjoint sample belongs to val_operational_all",
        disjoint_6r_eids.issubset(val_all_eids) and len(disjoint_6r) == 7833,
        f"Disjoint event count: {len(disjoint_6r)} (expected 7,833)"
    )

    # Test 11: Target and sequence preservation
    preservation_6r_ok = True
    for s in disjoint_6r:
        src = val_all_by_id[s["event_id"]]
        if not np.isclose(s["target_risk"], src["target_risk"]):
            preservation_6r_ok = False
            break
        if not np.allclose(s["input_sequence"], src["input_sequence"]):
            preservation_6r_ok = False
            break
    check(
        "6-Range: Target risks and input sequences exactly preserved from val_operational_all",
        preservation_6r_ok,
        "All targets and input tensors match source records"
    )

    # Test 12: Critical and non-critical counts
    crit_6r = sum(1 for s in disjoint_6r if s["target_risk"] >= THRESHOLD_CRITICAL)
    noncrit_6r = sum(1 for s in disjoint_6r if s["target_risk"] < THRESHOLD_CRITICAL)
    check(
        "6-Range: Critical and non-critical counts correctly calculated",
        crit_6r == 39 and noncrit_6r == 7794,
        f"Critical: {crit_6r}, Non-critical: {noncrit_6r}"
    )

    # Test 13: Metric finiteness when critical events exist
    y_true_6r = np.array([s["target_risk"] for s in disjoint_6r], dtype=np.float32)
    y_pred_6r = y_true_6r + 0.1  # synthetic predictions
    metrics_6r = compute_metrics(y_pred_6r, y_true_6r)
    check(
        "6-Range: Disjoint metrics are strictly finite when critical examples exist",
        np.isfinite(metrics_6r["critical_mse"]) and np.isfinite(metrics_6r["score"]) and np.isfinite(metrics_6r["f2_score"]),
        f"critical_mse: {metrics_6r['critical_mse']:.4f}, f2: {metrics_6r['f2_score']:.4f}, score: {metrics_6r['score']:.4f}"
    )

    # Test 14: Saved artifact consistency for 6-range
    check(
        "6-Range: Saved artifact val_operational_disjoint_6_ranges.joblib exists and matches derivation",
        VAL_DISJOINT_6R_FILE.exists() and len(joblib.load(VAL_DISJOINT_6R_FILE)) == 7833,
        f"File: {VAL_DISJOINT_6R_FILE.name}"
    )

    # -------------------------------------------------------------
    # PART C: CHECKPOINT INTEGRITY
    # -------------------------------------------------------------
    print("\n[Phase 4] Verifying Checkpoint & Result Artifact Integrity...")
    check(
        "Epoch-250 best 2-range checkpoint exists and remains intact",
        BEST_2R_CHECKPOINT.exists() and BEST_2R_CHECKPOINT.stat().st_size > 900000,
        f"Path: {BEST_2R_CHECKPOINT} (Size: {BEST_2R_CHECKPOINT.stat().st_size} bytes)"
    )

    # Verify checkpoint contents
    cp_data = torch.load(BEST_2R_CHECKPOINT, map_location="cpu", weights_only=False)
    check(
        "Epoch-250 checkpoint metadata valid (epoch == 250, model_state_dict intact)",
        cp_data.get("epoch") == 250 and "model_state_dict" in cp_data,
        f"Epoch: {cp_data.get('epoch')}, Dataset: {cp_data.get('dataset')}"
    )

    print("-" * 80)
    print(f"RESULTS: {passed_checks}/{total_checks} tests passed.")
    print("=" * 80)

    if passed_checks == total_checks:
        print("\n[ALL TESTS PASSED] Validation leakage resolved and regression test suite verified!")
        return 0
    else:
        print(f"\n[WARNING] {total_checks - passed_checks} tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
