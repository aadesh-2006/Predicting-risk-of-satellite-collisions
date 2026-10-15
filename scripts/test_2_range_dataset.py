"""
Automated Validation Suite for Paper-Aligned 2-Range Balanced Dataset
Tulczyjew et al. (2021) / ESA Kelvins Collision Avoidance Challenge

Validates:
1. File existence and loadability (data/processed/train_sequences_2_ranges.joblib)
2. Total event count == 132
3. Exact 50/50 class balance (66 critical risk >= -6.0, 66 non-critical risk < -6.0)
4. Duplicate event IDs check (exactly 132 unique event IDs)
5. Input sequence shape and feature dimension == (T - 1, 64)
6. Sequence / target risk alignment (final CDM risk as target, penultimate CDM risk as last input step)
7. Chronological ordering within sequences (descending time_to_tca)
8. Event filtering constraints (first_tca > 2.0, last_tca < 1.0, non-critical len > 10)
9. Numerical finiteness (0 NaNs, 0 Infs across all tensors)
10. Agreement with reference pickle (100% positive event ID match with authors' 2-range pickle)
11. Integrity of existing 6-range dataset, validation datasets, scaler, and checkpoints
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any

import joblib
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REF_PICKLE_PATH = Path(r"C:\Users\Aadesh\.gemini\antigravity\scratch\dundermifflin_ref\CollisionAvoidanceChallenge\clean_data\train_samples_2_ranges\train_samples_x_no_augmentation.pickle")

TRAIN_2_RANGES_FILE = PROCESSED_DIR / "train_sequences_2_ranges.joblib"
TRAIN_6_RANGES_FILE = PROCESSED_DIR / "train_sequences_6_ranges.joblib"
VAL_ALL_FILE = PROCESSED_DIR / "val_operational_all.joblib"
VAL_DISJOINT_FILE = PROCESSED_DIR / "val_operational_disjoint.joblib"
SCALER_FILE = PROCESSED_DIR / "scaler.joblib"
METADATA_FILE = PROCESSED_DIR / "preprocessing_metadata.json"
BEST_CHECKPOINT_FILE = CHECKPOINT_DIR / "best_gru_checkpoint.pt"


def run_checks():
    print("=" * 80)
    print("  VALIDATION SUITE: 2-RANGE BALANCED TRAINING DATASET")
    print("=" * 80)

    total_checks = 0
    passed_checks = 0

    def check(name: str, condition: bool, details: str = ""):
        nonlocal total_checks, passed_checks
        total_checks += 1
        if condition:
            passed_checks += 1
            print(f" [PASS] Check {total_checks:02d}: {name}")
            if details:
                print(f"         {details}")
        else:
            print(f" [FAIL] Check {total_checks:02d}: {name}")
            print(f"         ERROR: {details}")

    # Check 1: File existence
    check(
        "File existence of train_sequences_2_ranges.joblib",
        TRAIN_2_RANGES_FILE.exists() and TRAIN_2_RANGES_FILE.stat().st_size > 0,
        f"Path: {TRAIN_2_RANGES_FILE} (Size: {TRAIN_2_RANGES_FILE.stat().st_size if TRAIN_2_RANGES_FILE.exists() else 0} bytes)"
    )

    if not TRAIN_2_RANGES_FILE.exists():
        print("\nFatal: 2-range dataset file does not exist. Aborting further checks.")
        sys.exit(1)

    # Load 2-range dataset
    dataset = joblib.load(TRAIN_2_RANGES_FILE)
    print(f"Successfully loaded {len(dataset)} sequence records from {TRAIN_2_RANGES_FILE.name}")

    # Check 2: Total event count == 132
    check(
        "Total event count == 132",
        len(dataset) == 132,
        f"Actual event count: {len(dataset)}"
    )

    # Check 3: Class balance (66 high-risk, 66 low-risk)
    targets = [sample["target_risk"] for sample in dataset]
    pos_samples = [t for t in targets if t >= -6.0]
    neg_samples = [t for t in targets if t < -6.0]
    check(
        "Exact 50/50 class balance (66 critical risk >= -6.0, 66 non-critical < -6.0)",
        len(pos_samples) == 66 and len(neg_samples) == 66,
        f"Positive count: {len(pos_samples)}, Negative count: {len(neg_samples)}"
    )

    # Check 4: Duplicate event IDs
    eids = [sample["event_id"] for sample in dataset]
    unique_eids = set(eids)
    check(
        "Duplicate event IDs check (132 unique event IDs)",
        len(unique_eids) == 132 and len(eids) == 132,
        f"Unique IDs: {len(unique_eids)}, Total IDs: {len(eids)}"
    )

    # Check 5: Input sequence feature dimension == 64
    seq_dims = set(sample["input_sequence"].shape[1] for sample in dataset)
    check(
        "Input sequence feature dimension == 64 (63 physical features + 1 risk channel)",
        seq_dims == {64},
        f"Feature dimensions observed: {seq_dims}"
    )

    # Check 6: Sequence shape consistency (T - 1, 64)
    shape_ok = True
    shape_err = ""
    for idx, s in enumerate(dataset):
        T = s["seq_len"]
        inp_shape = s["input_sequence"].shape
        if inp_shape[0] != T - 1 or inp_shape[1] != 64:
            shape_ok = False
            shape_err = f"Event {s['event_id']} (idx {idx}): seq_len={T}, input shape={inp_shape}"
            break
    check(
        "Sequence shape consistency: input_sequence.shape == (T - 1, 64) for all events",
        shape_ok,
        "All 132 sequences correctly shaped (T - 1, 64)" if shape_ok else shape_err
    )

    # Check 7: Target risk alignment
    target_align_ok = True
    target_align_err = ""
    for idx, s in enumerate(dataset):
        raw_risks = s["raw_risks"]
        target = s["target_risk"]
        if not np.isclose(target, raw_risks[-1]):
            target_align_ok = False
            target_align_err = f"Event {s['event_id']}: target {target} != last raw risk {raw_risks[-1]}"
            break
    check(
        "Target risk alignment: target_risk == raw_risks[-1] for all events",
        target_align_ok,
        "All targets match final CDM raw risk" if target_align_ok else target_align_err
    )

    # Check 8: Input risk alignment
    inp_align_ok = True
    inp_align_err = ""
    for idx, s in enumerate(dataset):
        raw_risks = s["raw_risks"]
        last_inp_risk = s["input_sequence"][-1, 0]
        penultimate_raw_risk = raw_risks[-2]
        if not np.isclose(last_inp_risk, penultimate_raw_risk):
            inp_align_ok = False
            inp_align_err = f"Event {s['event_id']}: last input step risk {last_inp_risk} != penultimate raw risk {penultimate_raw_risk}"
            break
    check(
        "Input risk alignment: input_sequence[-1, 0] == raw_risks[-2] for all events",
        inp_align_ok,
        "Penultimate CDM risk correctly aligns with final input step" if inp_align_ok else inp_align_err
    )

    # Check 9: Chronological ordering (strictly descending time_to_tca)
    chrono_ok = True
    chrono_err = ""
    for idx, s in enumerate(dataset):
        tcas = s["raw_tcas"]
        if not (tcas[:-1] >= tcas[1:]).all():
            chrono_ok = False
            chrono_err = f"Event {s['event_id']}: time_to_tca not non-increasing: {tcas}"
            break
    check(
        "Chronological ordering: time_to_tca non-increasing (furthest to closest approach)",
        chrono_ok,
        "All events chronologically sorted" if chrono_ok else chrono_err
    )

    # Check 10: Event filtering rules adherence
    filter_ok = True
    filter_err = ""
    for idx, s in enumerate(dataset):
        tcas = s["raw_tcas"]
        target = s["target_risk"]
        T = s["seq_len"]
        first_tca = tcas[0]
        last_tca = tcas[-1]

        if first_tca <= 2.0 or last_tca >= 1.0:
            filter_ok = False
            filter_err = f"Event {s['event_id']}: first_tca={first_tca:.4f} (must be > 2.0), last_tca={last_tca:.4f} (must be < 1.0)"
            break
        if target < -6.0 and T <= 10:
            filter_ok = False
            filter_err = f"Event {s['event_id']}: low-risk event seq_len={T} (must be > 10)"
            break
    check(
        "Filtering constraints: first_tca > 2.0, last_tca < 1.0, low-risk seq_len > 10",
        filter_ok,
        "All 132 events satisfy verified filter boundaries" if filter_ok else filter_err
    )

    # Check 11: Numerical finiteness (0 NaNs, 0 Infs)
    finite_ok = True
    finite_err = ""
    for idx, s in enumerate(dataset):
        for field in ["input_sequence", "raw_risks", "raw_tcas"]:
            arr = s[field]
            if not np.isfinite(arr).all():
                finite_ok = False
                finite_err = f"Event {s['event_id']}: non-finite values in {field}"
                break
        if not np.isfinite(s["target_risk"]):
            finite_ok = False
            finite_err = f"Event {s['event_id']}: non-finite target_risk"
            break
        if not finite_ok:
            break
    check(
        "Numerical finiteness: 0 NaNs and 0 Infs across all 132 sequences and targets",
        finite_ok,
        "All tensors strictly finite" if finite_ok else finite_err
    )

    # Check 12: Comparison with authors' reference 2-range pickle
    ref_match_ok = False
    ref_details = ""
    if REF_PICKLE_PATH.exists():
        try:
            import pandas.core.indexes.base
            pandas.core.indexes.base.Int64Index = pandas.core.indexes.base.Index
            sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base

            with open(REF_PICKLE_PATH, "rb") as f:
                ref_samples = pickle.load(f)

            ref_eids = [int(s['event_id'].iloc[0]) for s in ref_samples]
            ref_risks = [s['risk'].iloc[-1] for s in ref_samples]
            ref_pos_eids = set(e for e, r in zip(ref_eids, ref_risks) if r >= -6.0)

            our_pos_eids = set(s["event_id"] for s in dataset if s["target_risk"] >= -6.0)

            diff_pos = our_pos_eids.symmetric_difference(ref_pos_eids)
            if len(diff_pos) == 0:
                ref_match_ok = True
                ref_details = f"Exact 100% agreement on all 66 critical event IDs with {REF_PICKLE_PATH.name}"
            else:
                ref_details = f"Positive event ID symmetric difference: {diff_pos}"
        except Exception as e:
            ref_details = f"Could not load reference pickle: {e}"
    else:
        ref_details = f"Reference pickle not found at {REF_PICKLE_PATH}"

    check(
        "Reference pickle agreement: 100% match on all 66 critical event IDs",
        ref_match_ok,
        ref_details
    )

    # Check 13: Integrity of existing artifacts
    existing_ok = True
    existing_details = []

    if TRAIN_6_RANGES_FILE.exists():
        d6 = joblib.load(TRAIN_6_RANGES_FILE)
        if len(d6) == 138:
            existing_details.append(f"train_6_ranges.joblib intact ({len(d6)} events)")
        else:
            existing_ok = False
            existing_details.append(f"train_6_ranges.joblib unexpected count: {len(d6)}")
    else:
        existing_ok = False
        existing_details.append("train_6_ranges.joblib missing")

    if VAL_ALL_FILE.exists():
        va = joblib.load(VAL_ALL_FILE)
        if len(va) == 7962:
            existing_details.append(f"val_operational_all.joblib intact ({len(va)} events)")
        else:
            existing_ok = False
            existing_details.append(f"val_operational_all.joblib unexpected count: {len(va)}")
    else:
        existing_ok = False
        existing_details.append("val_operational_all.joblib missing")

    if BEST_CHECKPOINT_FILE.exists():
        existing_details.append("best_gru_checkpoint.pt intact")
    else:
        existing_ok = False
        existing_details.append("best_gru_checkpoint.pt missing")

    check(
        "Preservation of existing artifacts: 6-range dataset, validation set, and checkpoint intact",
        existing_ok,
        "; ".join(existing_details)
    )

    # Summary
    print("-" * 80)
    print(f"RESULTS: {passed_checks}/{total_checks} checks passed.")
    print("=" * 80)

    if passed_checks == total_checks:
        print("\n[ALL CHECKS PASSED] 2-range balanced dataset is fully validated and paper-aligned!")
        return 0
    else:
        print(f"\n[WARNING] {total_checks - passed_checks} checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_checks())
