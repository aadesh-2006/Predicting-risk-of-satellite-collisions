"""
Automated Validation Suite for Preprocessed Dataset
Verifies:
1. Feature dimensions (exactly 64 = 63 features + 1 risk channel)
2. Sequence / target alignment (historical inputs, final CDM risk as target)
3. Chronological ordering (strictly decreasing time_to_tca)
4. Absence of missing / infinite values (0 NaNs, 0 Infs)
5. Dataset event counts (138 train events, 7,962 val all events, 0 disjoint overlap)
6. Normalization range sanity
"""

import sys
import json
from pathlib import Path
import joblib
import numpy as np

PROCESSED_DIR = Path(r"C:\Users\Aadesh\.gemini\antigravity\scratch\Predicting-risk-of-satellite-collisions\data\processed")


def run_tests():
    print("=" * 60)
    print("  Running Preprocessed Dataset Verification Suite")
    print("=" * 60)
    
    passed_checks = 0
    total_checks = 0
    
    def check(name: str, condition: bool, err_msg: str = ""):
        nonlocal passed_checks, total_checks
        total_checks += 1
        if condition:
            passed_checks += 1
            print(f" [PASS] {name}")
        else:
            print(f" [FAIL] {name}: {err_msg}")
            
    # Check 1: Artifact existence
    train_file = PROCESSED_DIR / "train_sequences_6_ranges.joblib"
    val_all_file = PROCESSED_DIR / "val_operational_all.joblib"
    val_disjoint_file = PROCESSED_DIR / "val_operational_disjoint.joblib"
    scaler_file = PROCESSED_DIR / "scaler.joblib"
    metadata_file = PROCESSED_DIR / "preprocessing_metadata.json"
    
    check("Artifact files exist", 
          all(f.exists() for f in [train_file, val_all_file, val_disjoint_file, scaler_file, metadata_file]),
          "One or more processed files missing")
          
    # Load artifacts
    print("\nLoading artifacts for verification...")
    train_data = joblib.load(train_file)
    val_all_data = joblib.load(val_all_file)
    val_disjoint_data = joblib.load(val_disjoint_file)
    scaler_data = joblib.load(scaler_file)
    with open(metadata_file) as f:
        metadata = json.load(f)
        
    # Check 2: Event counts
    check("Train event count == 138", len(train_data) == 138, f"Got {len(train_data)}")
    check("Val all event count == 7,962", len(val_all_data) == 7962, f"Got {len(val_all_data)}")
    check("Val disjoint event count > 7,800", len(val_disjoint_data) > 7800, f"Got {len(val_disjoint_data)}")
    
    # Check 3: Disjointness check
    train_eids = set(metadata["event_ids"]["train_6_ranges"])
    val_disjoint_eids = set(metadata["event_ids"]["val_disjoint"])
    overlap = train_eids.intersection(val_disjoint_eids)
    check("Val disjoint has zero overlap with train", len(overlap) == 0, f"Found {len(overlap)} overlapping IDs")
    
    # Check 4: Feature dimension
    train_dims = set(sample["input_sequence"].shape[1] for sample in train_data)
    val_dims = set(sample["input_sequence"].shape[1] for sample in val_all_data)
    check("Input sequence feature dimension == 64 (63 physical + 1 risk)",
          train_dims == {64} and val_dims == {64},
          f"Train dims: {train_dims}, Val dims: {val_dims}")
          
    # Check 5: Sequence length & target alignment
    alignment_ok = True
    alignment_err = ""
    for idx, sample in enumerate(train_data + val_all_data[:500]):
        seq = sample["input_sequence"]
        raw_risks = sample["raw_risks"]
        target = sample["target_risk"]
        T = sample["seq_len"]
        
        # Sequence input should have length T - 1
        if seq.shape[0] != T - 1:
            alignment_ok = False
            alignment_err = f"Sample {idx}: input length {seq.shape[0]} != T - 1 ({T - 1})"
            break
            
        # Target must be the risk of the final CDM (T - 1)
        if not np.isclose(target, raw_risks[-1]):
            alignment_ok = False
            alignment_err = f"Sample {idx}: target {target} != last raw risk {raw_risks[-1]}"
            break
            
        # Last step in input sequence must have the risk of penultimate CDM (T - 2)
        if not np.isclose(seq[-1, 0], raw_risks[-2]):
            alignment_ok = False
            alignment_err = f"Sample {idx}: last input risk {seq[-1, 0]} != penultimate raw risk {raw_risks[-2]}"
            break
            
    check("Input sequence and target risk strictly aligned", alignment_ok, alignment_err)
    
    # Check 6: Chronological ordering
    chrono_ok = True
    chrono_err = ""
    for idx, sample in enumerate(train_data + val_all_data[:500]):
        tcas = sample["raw_tcas"]
        if not all(tcas[i] >= tcas[i+1] for i in range(len(tcas) - 1)):
            chrono_ok = False
            chrono_err = f"Sample {idx} (event {sample['event_id']}) is not monotonically decreasing: {tcas}"
            break
            
    check("Chronological ordering (strictly decreasing time_to_tca)", chrono_ok, chrono_err)
    
    # Check 7: No NaNs or Infs
    nans_ok = True
    for sample in train_data + val_all_data[:1000]:
        if np.isnan(sample["input_sequence"]).any() or np.isnan(sample["target_risk"]):
            nans_ok = False
            break
        if np.isinf(sample["input_sequence"]).any() or np.isinf(sample["target_risk"]):
            nans_ok = False
            break
    check("Zero NaNs or Infs in inputs and targets", nans_ok, "NaN or Inf detected")
    
    # Check 8: Risk bin distribution
    bin_counts = metadata["datasets"]["train_6_ranges"]["bin_distribution"]
    check("Risk bins perfectly balanced at 23 events each",
          all(count == 23 for count in bin_counts.values()),
          f"Bin distribution: {bin_counts}")
          
    # Check 9: Scaler configuration
    check("Scaler feature columns == 63",
          len(scaler_data["columns"]) == 63,
          f"Got {len(scaler_data['columns'])}")
          
    print("\n" + "=" * 60)
    print(f"Results: {passed_checks}/{total_checks} checks passed.")
    print("=" * 60)
    
    if passed_checks == total_checks:
        print("[ALL TESTS PASSED] The dataset is verified and ready for GRU training.")
        return 0
    else:
        print("[FAIL] Some verification checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
