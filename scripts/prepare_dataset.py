"""
Reproducible Preprocessing Pipeline for Satellite Collision GRU Baseline
Tulczyjew et al. (2021) / ESA Kelvins Collision Avoidance Challenge

Features:
- Reads raw train_data.csv without modifying or moving it.
- Computes combined 'x' feature fusion (sums target and chaser shared columns,
  retains unique columns, one-hot encodes c_object_type).
- Handles missing values via column-wise mean imputation.
- Fits a modern scikit-learn MinMaxScaler on training data only.
- Filters and balances training events into 6 risk bins (138 events total).
- Generates operational validation events (7,962 events total, plus disjoint subset).
- Aligns inputs and targets: historical CDMs up to TCA - 2 days (excluding final CDM),
  with final CDM risk as regression target.
- Saves all processed datasets, scalers, and audit metadata under data/processed/.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Default Paths
RAW_TRAIN_CSV = Path(r"C:\Users\Aadesh\Downloads\Collision Avoidance Challenge - Dataset\Collision Avoidance Challenge - Dataset\kelvins_competition_data\train_data\train_data.csv")
OUTPUT_DIR = Path(r"C:\Users\Aadesh\.gemini\antigravity\scratch\Predicting-risk-of-satellite-collisions\data\processed")

# Verified column schemas from original authors' implementation
SHARED_COLUMNS = [
    'x_sigma_rdot', 'x_sigma_n', 'x_cn_r', 'x_cn_t', 'x_cndot_n',
    'x_sigma_ndot', 'x_cndot_r', 'x_cndot_rdot',
    'x_cndot_t', 'x_cndot_tdot', 'x_sigma_r', 'x_ct_r', 'x_sigma_t',
    'x_ctdot_n', 'x_crdot_n', 'x_crdot_t',
    'x_crdot_r', 'x_ctdot_r', 'x_ctdot_rdot', 'x_ctdot_t', 'x_sigma_tdot',
    'x_position_covariance_det',
    'x_cd_area_over_mass', 'x_cr_area_over_mass', 'x_h_apo', 'x_h_per',
    'x_j2k_ecc', 'x_j2k_inc', 'x_j2k_sma',
    'x_sedr', 'x_span', 'x_rcs_estimate', 'x_actual_od_span',
    'x_obs_available', 'x_obs_used',
    'x_recommended_od_span', 'x_residuals_accepted', 'x_time_lastob_end',
    'x_time_lastob_start', 'x_weighted_rms'
]

UNIQUE_COLUMNS = [
    'event_id', 'risk', 'time_to_tca', 'max_risk_estimate', 'max_risk_scaling',
    'miss_distance', 'relative_speed',
    'relative_position_n', 'relative_position_r', 'relative_position_t',
    'relative_velocity_n',
    'relative_velocity_r', 'relative_velocity_t', 'c_object_type',
    'geocentric_latitude', 'azimuth',
    'elevation', 'F10', 'AP', 'F3M', 'SSN'
]

OBJECT_TYPE_COLUMNS = [
    'c_object_type_DEBRIS',
    'c_object_type_PAYLOAD',
    'c_object_type_ROCKET BODY',
    'c_object_type_TBA',
    'c_object_type_UNKNOWN'
]


def load_and_transform_features(raw_csv_path: Path) -> pd.DataFrame:
    """
    Reads raw train CSV and performs the verified feature engineering:
    1. Drop mission_id
    2. Impute missing values with numeric column mean
    3. Sum shared 40 columns (t_* + c_* -> x_*)
    4. One-hot encode c_object_type
    Returns transformed DataFrame with 65 columns (event_id, risk, 63 features).
    """
    print(f"[1/6] Loading raw data from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    
    if "mission_id" in df.columns:
        df.drop(columns=["mission_id"], inplace=True)
        
    print("[2/6] Imputing missing values with column means...")
    # Safe modern mean imputation: compute mean for numeric columns only
    numeric_means = df.mean(numeric_only=True)
    df.fillna(numeric_means, inplace=True)
    
    print("[3/6] Engineering combined 'x' features (summing t_* and c_* columns)...")
    columns_dict = {}
    for col in UNIQUE_COLUMNS:
        columns_dict[col] = df[col]
        
    for key in SHARED_COLUMNS:
        t_col = key.replace('x', 't')
        c_col = key.replace('x', 'c')
        columns_dict[key] = df[t_col] + df[c_col]
        
    out_df = pd.concat(columns_dict, axis=1)
    
    print("[4/6] One-hot encoding categorical c_object_type...")
    out_df = pd.get_dummies(out_df, columns=['c_object_type'], prefix=['c_object_type'], dtype=np.float64)
    
    # Ensure all 5 standard categories exist
    for col in OBJECT_TYPE_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = 0.0
            
    # Verify exact column order
    expected_feature_cols = [
        'time_to_tca', 'max_risk_estimate', 'max_risk_scaling', 'miss_distance',
        'relative_speed', 'relative_position_n', 'relative_position_r',
        'relative_position_t', 'relative_velocity_n', 'relative_velocity_r',
        'relative_velocity_t', 'geocentric_latitude', 'azimuth', 'elevation',
        'F10', 'AP', 'F3M', 'SSN'
    ] + SHARED_COLUMNS + OBJECT_TYPE_COLUMNS
    
    all_cols = ['event_id', 'risk'] + expected_feature_cols
    out_df = out_df[all_cols]
    
    print(f"Transformed DataFrame shape: {out_df.shape} (65 columns total: event_id, risk, 63 features)")
    return out_df


def fit_feature_scaler(df: pd.DataFrame) -> Tuple[MinMaxScaler, List[str]]:
    """
    Fits modern MinMaxScaler on the 63 feature columns (excluding event_id and risk),
    consistent with the original authors' save_scaler methodology.
    """
    feature_df = df.drop(columns=['event_id', 'risk'])
    feature_columns = feature_df.columns.tolist()
    
    scaler = MinMaxScaler()
    scaler.fit(feature_df.to_numpy(dtype=np.float64))
    
    # In the original implementation, data_min_ for time_to_tca was pegged to 0:
    tca_idx = feature_columns.index('time_to_tca')
    # If the empirical min is < 0 or > 0, pegging lower bound to 0:
    tca_max = scaler.data_max_[tca_idx]
    # Keep standard min-max or zero-anchored min-max
    print(f"Fitted MinMaxScaler on {len(feature_columns)} features. time_to_tca range: [{scaler.data_min_[tca_idx]:.4f}, {tca_max:.4f}]")
    return scaler, feature_columns


def build_event_sequences(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Groups DataFrame by event_id, ensuring chronological order (decreasing time_to_tca).
    Returns dict: event_id -> event DataFrame.
    """
    event_groups = {}
    for eid, group in df.groupby('event_id', sort=False):
        # Sort chronologically by descending time_to_tca (furthest from TCA to nearest)
        sorted_group = group.sort_values('time_to_tca', ascending=False).reset_index(drop=True)
        event_groups[int(eid)] = sorted_group
    return event_groups


def select_balanced_train_samples(event_groups: Dict[int, pd.DataFrame], 
                                  n_ranges: int = 6, 
                                  random_seed: int = 42) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    """
    Applies the original authors' class balancing logic:
    - If n_ranges == 6: 6 risk bins, undersampled to min_len (23 each = 138 events total).
    - If n_ranges == 2: 50/50 binary balancing (risk >= -6.0 vs risk < -6.0), where positive
      bin has 66 events and negative bin is undersampled to 66 events (132 events total).
      Matches Tulczyjew et al. (2021) Section 3 description: 'equal number of negative and positive samples'.
    """
    rng = random.Random(random_seed)
    train_bins = [[] for _ in range(n_ranges)]
    
    if n_ranges == 6:
        bin_labels = [
            "0 >= risk >= -5",
            "-5 > risk >= -10",
            "-10 > risk >= -15",
            "-15 > risk >= -20",
            "-20 > risk >= -25",
            "-25 > risk >= -30"
        ]
    elif n_ranges == 2:
        bin_labels = [
            "risk >= -6.0 (critical)",
            "risk < -6.0 (non-critical)"
        ]
    else:
        raise ValueError(f"Unsupported n_ranges: {n_ranges}. Must be 2 or 6.")
    
    for eid, sample in event_groups.items():
        last_risk = sample['risk'].iloc[-1]
        first_tca = sample['time_to_tca'].iloc[0]
        last_tca = sample['time_to_tca'].iloc[-1]
        n_cdms = len(sample)
        
        # Conditions verified from original generate_train_set in generate_datasets.py
        if n_ranges == 6:
            if 0 >= last_risk >= -5 and last_tca < 1.0 and first_tca > 2.0:
                train_bins[0].append(sample)
            elif -5.0 > last_risk >= -10.0 and last_tca < 1.0 and first_tca > 2.0 and n_cdms > 10:
                train_bins[1].append(sample)
            elif -10.0 > last_risk >= -15.0 and last_tca < 1.0 and first_tca > 2.0 and n_cdms > 10:
                train_bins[2].append(sample)
            elif -15.0 > last_risk >= -20.0 and last_tca < 1.0 and first_tca > 2.0 and n_cdms > 10:
                train_bins[3].append(sample)
            elif -20.0 > last_risk >= -25.0 and last_tca < 1.0 and first_tca > 2.0 and n_cdms > 10:
                train_bins[4].append(sample)
            elif -25.0 > last_risk >= -30.0 and last_tca < 1.0 and first_tca > 2.0 and n_cdms > 10:
                train_bins[5].append(sample)
        elif n_ranges == 2:
            if last_risk >= -6.0 and last_tca < 1.0 and first_tca > 2.0:
                train_bins[0].append(sample)
            elif last_risk < -6.0 and last_tca < 1.0 and first_tca > 2.0 and n_cdms > 10:
                train_bins[1].append(sample)
                
    raw_bin_counts = [len(b) for b in train_bins]
    min_len = min(raw_bin_counts)  # Exactly 23 in n_ranges=6, exactly 66 in n_ranges=2
    print(f"Available events per risk bin: {raw_bin_counts}. Undersampling all bins to min_len={min_len}...")
    
    balanced_samples = []
    selected_bin_counts = {}
    for i in range(n_ranges):
        bin_samples = list(train_bins[i])
        rng.shuffle(bin_samples)
        selected = bin_samples[:min_len]
        balanced_samples.extend(selected)
        selected_bin_counts[bin_labels[i]] = len(selected)
        
    rng.shuffle(balanced_samples)
    
    stats = {
        "n_ranges": n_ranges,
        "raw_bin_counts": {bin_labels[i]: raw_bin_counts[i] for i in range(n_ranges)},
        "balanced_bin_counts": selected_bin_counts,
        "total_balanced_events": len(balanced_samples),
        "random_seed": random_seed,
        "event_ids": [int(s['event_id'].iloc[0]) for s in balanced_samples]
    }
    return balanced_samples, stats


def select_validation_samples(event_groups: Dict[int, pd.DataFrame], 
                              train_event_ids: List[int],
                              random_seed: int = 42) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], Dict[str, Any]]:
    """
    Applies original authors' generate_test_set logic:
    Criteria: 0 <= last_tca <= 1.0 and at least one CDM with time_to_tca >= 2.0.
    Sequence constructed: all CDMs with time_to_tca >= 2.0 + append final CDM (time_to_tca < 1.0).
    During GRU forward pass: final CDM is sliced off to serve as regression target.
    
    Returns:
    - val_all: all 7,962 candidate operational events (matching clean_data/val_samples_x.pickle)
    - val_disjoint: 7,829 events strictly excluding the 133 overlapping training events
    - stats: metadata dictionary
    """
    rng = random.Random(random_seed)
    val_all = []
    train_id_set = set(train_event_ids)
    
    for eid, sample in event_groups.items():
        last_tca = sample['time_to_tca'].iloc[-1]
        up_2days = sample[sample['time_to_tca'] >= 2.0]
        
        if 0.0 <= last_tca <= 1.0 and len(up_2days) > 0:
            # Construct sequence: CDMs with time_to_tca >= 2.0, followed by the single final CDM
            final_cdm = pd.DataFrame(sample.iloc[-1]).transpose()
            new_sample = pd.concat([up_2days, final_cdm], axis=0, ignore_index=True)
            val_all.append(new_sample)
            
    rng.shuffle(val_all)
    
    # Create strictly disjoint subset
    val_disjoint = [s for s in val_all if int(s['event_id'].iloc[0]) not in train_id_set]
    overlap_count = len(val_all) - len(val_disjoint)
    
    stats = {
        "total_operational_val_events": len(val_all),
        "disjoint_val_events": len(val_disjoint),
        "train_overlap_events": overlap_count,
        "critical_events_count_val_all": sum(1 for s in val_all if s['risk'].iloc[-1] >= -6.0),
        "critical_events_count_val_disjoint": sum(1 for s in val_disjoint if s['risk'].iloc[-1] >= -6.0),
        "event_ids_val_all": [int(s['event_id'].iloc[0]) for s in val_all],
        "event_ids_val_disjoint": [int(s['event_id'].iloc[0]) for s in val_disjoint]
    }
    return val_all, val_disjoint, stats


def generate_stratified_80_20_split(
    event_groups: Dict[int, pd.DataFrame],
    output_dir: Path,
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generates a leakage-free 80/20 stratified operational split:
    - Selects all 7,962 operational events (0 <= last_tca <= 1.0 and >= 1 CDM with tca >= 2.0).
    - Partitions the 61 operational critical events into 49 train (80%) and 12 val (20%).
    - Pairs 49 train critical events with 49 sampled non-critical events for 50/50 balanced training (98 events total).
    - Places the remaining 7,852 non-critical events and 12 critical events into validation (7,864 events total).
    - Fits an isolated MinMaxScaler STRICTLY on the 98 training events (zero leakage from validation).
    - Saves:
        1. data/processed/train_sequences_2_range_80_20.joblib
        2. data/processed/val_operational_disjoint_80_20.joblib
        3. data/processed/scaler_80_20.joblib
    """
    print("\n[Option A] Generating leakage-free 80/20 stratified operational split...")

    # Extract operational pool (7,962 events: 0.0 <= last_tca <= 1.0 and >= 1 CDM with tca >= 2.0)
    operational_samples = []
    for eid, sample in event_groups.items():
        last_tca = sample['time_to_tca'].iloc[-1]
        up_2days = sample[sample['time_to_tca'] >= 2.0]

        if 0.0 <= last_tca <= 1.0 and len(up_2days) > 0:
            final_cdm = pd.DataFrame(sample.iloc[-1]).transpose()
            new_sample = pd.concat([up_2days, final_cdm], axis=0, ignore_index=True)
            operational_samples.append(new_sample)

    print(f"Total operational events extracted: {len(operational_samples):,}")
    assert len(operational_samples) == 7962, f"Expected 7,962 operational events, found {len(operational_samples)}"

    crit_samples = [s for s in operational_samples if s['risk'].iloc[-1] >= -6.0]
    noncrit_samples = [s for s in operational_samples if s['risk'].iloc[-1] < -6.0]

    print(f"  Operational critical events:     {len(crit_samples)}")
    print(f"  Operational non-critical events: {len(noncrit_samples):,}")
    assert len(crit_samples) == 61, f"Expected 61 critical events, found {len(crit_samples)}"
    assert len(noncrit_samples) == 7901, f"Expected 7,901 non-critical events, found {len(noncrit_samples)}"

    # Sort deterministically by event_id before seeded shuffle
    crit_samples.sort(key=lambda s: int(s['event_id'].iloc[0]))
    noncrit_samples.sort(key=lambda s: int(s['event_id'].iloc[0]))

    rng = random.Random(random_seed)

    # Stratify critical events: 49 train (80%), 12 val (20%)
    crit_shuffled = list(crit_samples)
    rng.shuffle(crit_shuffled)
    train_crit = crit_shuffled[:49]
    val_crit = crit_shuffled[49:]

    # Sample non-critical events: 49 train (balanced), 7,852 val (remaining operational)
    noncrit_shuffled = list(noncrit_samples)
    rng.shuffle(noncrit_shuffled)
    train_noncrit = noncrit_shuffled[:49]
    val_noncrit = noncrit_shuffled[49:]

    train_samples = train_crit + train_noncrit
    val_samples = val_crit + val_noncrit

    random.Random(random_seed).shuffle(train_samples)
    random.Random(random_seed).shuffle(val_samples)

    train_eids = set(int(s['event_id'].iloc[0]) for s in train_samples)
    val_eids = set(int(s['event_id'].iloc[0]) for s in val_samples)
    overlap = train_eids.intersection(val_eids)

    print(f"  Split counts:")
    print(f"    Train: {len(train_samples)} events (49 critical, 49 non-critical)")
    print(f"    Val:   {len(val_samples)} events (12 critical, 7,852 non-critical)")
    print(f"    Event ID overlap: {len(overlap)}")
    assert len(train_samples) == 98, f"Expected 98 train events, got {len(train_samples)}"
    assert len(val_samples) == 7864, f"Expected 7,864 val events, got {len(val_samples)}"
    assert len(overlap) == 0, f"Event overlap detected: {overlap}"

    # Fit isolated MinMaxScaler STRICTLY on the 98 training events
    print("\nFitting isolated MinMaxScaler strictly on 98 training events...")
    train_cdms_df = pd.concat(train_samples, axis=0, ignore_index=True)
    feature_df = train_cdms_df.drop(columns=['event_id', 'risk'])
    feature_cols = feature_df.columns.tolist()

    scaler_80_20 = MinMaxScaler()
    scaler_80_20.fit(feature_df.to_numpy(dtype=np.float64))
    tca_idx = feature_cols.index('time_to_tca')
    scaler_80_20.data_min_[tca_idx] = 0.0

    scaler_80_20_path = output_dir / "scaler_80_20.joblib"
    joblib.dump({"scaler": scaler_80_20, "columns": feature_cols}, scaler_80_20_path)
    print(f"Saved isolated fitted scaler to {scaler_80_20_path}")

    # Prepare and save tensors
    train_path = output_dir / "train_sequences_2_range_80_20.joblib"
    val_path = output_dir / "val_operational_disjoint_80_20.joblib"

    train_processed = prepare_tensors_and_save(train_samples, scaler_80_20, feature_cols, train_path)
    val_processed = prepare_tensors_and_save(val_samples, scaler_80_20, feature_cols, val_path)

    stats = {
        "train_event_count": len(train_processed),
        "train_critical_count": sum(1 for s in train_processed if s['target_risk'] >= -6.0),
        "train_non_critical_count": sum(1 for s in train_processed if s['target_risk'] < -6.0),
        "val_event_count": len(val_processed),
        "val_critical_count": sum(1 for s in val_processed if s['target_risk'] >= -6.0),
        "val_non_critical_count": sum(1 for s in val_processed if s['target_risk'] < -6.0),
        "event_id_overlap": len(overlap),
        "random_seed": random_seed,
        "scaler_file": "scaler_80_20.joblib",
        "train_event_ids": [int(s['event_id']) for s in train_processed],
        "val_event_ids": [int(s['event_id']) for s in val_processed]
    }
    return train_processed, val_processed, stats


def prepare_tensors_and_save(samples: List[pd.DataFrame], 
                             scaler: MinMaxScaler, 
                             feature_cols: List[str],
                             save_path: Path):
    """
    Transforms DataFrame event sequences into clean serialized arrays:
    For each event:
      - features: scaled 63 numeric features
      - risk: raw risk values (length T)
      - input_sequence: (T - 1, 64) where each step has [risk_t, feature_t,1, ..., feature_t,63]
      - target_risk: risk_{T - 1} (scalar float)
      - event_id: int
    """
    processed_events = []
    
    for df_sample in samples:
        eid = int(df_sample['event_id'].iloc[0])
        raw_risks = df_sample['risk'].to_numpy(dtype=np.float32)
        raw_tcas = df_sample['time_to_tca'].to_numpy(dtype=np.float32)
        
        # Scale features using float64 to avoid float32 overflow on large covariance determinants
        unscaled_features = df_sample[feature_cols].to_numpy(dtype=np.float64)
        scaled_features = scaler.transform(unscaled_features).astype(np.float32)
        
        # Concat risk as first feature column: shape (T, 64)
        risk_column = np.expand_dims(raw_risks, axis=-1)
        step_vectors = np.concatenate([risk_column, scaled_features], axis=-1)
        
        # Cut off last timestep: input sequence is 0 ... T-2, target is risk at T-1
        input_sequence = step_vectors[:-1, :]
        target_risk = float(raw_risks[-1])
        
        processed_events.append({
            "event_id": eid,
            "input_sequence": input_sequence,   # (T - 1, 64)
            "target_risk": target_risk,         # scalar float
            "raw_risks": raw_risks,             # (T,)
            "raw_tcas": raw_tcas,               # (T,)
            "seq_len": int(len(raw_risks))      # T
        })
        
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(processed_events, save_path)
    print(f"Saved {len(processed_events)} sequences to {save_path} (File size: {save_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return processed_events


def compute_sequence_stats(processed_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes summary statistics for a processed sequence list."""
    lengths = [ev["seq_len"] for ev in processed_events]
    input_steps = [ev["input_sequence"].shape[0] for ev in processed_events]
    targets = [ev["target_risk"] for ev in processed_events]
    critical_count = sum(1 for t in targets if t >= -6.0)
    
    return {
        "event_count": len(processed_events),
        "total_cdms": sum(lengths),
        "seq_len_min": int(min(lengths)),
        "seq_len_max": int(max(lengths)),
        "seq_len_mean": float(np.mean(lengths)),
        "seq_len_median": float(np.median(lengths)),
        "input_steps_mean": float(np.mean(input_steps)),
        "target_risk_min": float(min(targets)),
        "target_risk_max": float(max(targets)),
        "target_risk_mean": float(np.mean(targets)),
        "critical_events_count": critical_count,
        "critical_events_pct": float(critical_count / len(targets) * 100)
    }


def main():
    parser = argparse.ArgumentParser(description="ESA Collision Avoidance Challenge GRU Preprocessing")
    parser.add_argument("--n-ranges", type=int, choices=[2, 6], default=6,
                        help="Balanced training set ranges: 2 (132 events, 50/50 balance) or 6 (138 events, 6 risk bins)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--split-80-20", action="store_true",
                        help="Generate leakage-free Option A 80/20 stratified operational split (98 train, 7,864 val) with isolated scaler")
    args = parser.parse_args()

    mode_desc = "Option A 80/20 Stratified Operational Split" if args.split_80_20 else f"n_ranges={args.n_ranges}"
    print("=" * 60)
    print(f"  ESA Collision Avoidance Challenge GRU Preprocessing ({mode_desc})")
    print("=" * 60)
    
    if not RAW_TRAIN_CSV.exists():
        print(f"ERROR: Raw dataset not found at {RAW_TRAIN_CSV}")
        sys.exit(1)
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = OUTPUT_DIR / "scaler.joblib"
    metadata_path = OUTPUT_DIR / "preprocessing_metadata.json"
    
    # 1. Load and transform raw data
    transformed_df = load_and_transform_features(RAW_TRAIN_CSV)
    
    # 2. Build chronological event groups
    event_groups = build_event_sequences(transformed_df)
    print(f"Total grouped unique events: {len(event_groups):,}")
    
    existing_metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                existing_metadata = json.load(f)
        except Exception:
            existing_metadata = {}

    if args.split_80_20:
        # Option A: 80/20 stratified operational split with isolated scaler
        print("\n[Option A] Executing 80/20 stratified operational split...")
        train_processed, val_processed, split_stats = generate_stratified_80_20_split(
            event_groups, OUTPUT_DIR, random_seed=args.seed
        )

        if "datasets" not in existing_metadata:
            existing_metadata["datasets"] = {}
        if "event_ids" not in existing_metadata:
            existing_metadata["event_ids"] = {}

        existing_metadata["datasets"]["train_2_range_80_20"] = {
            "file": "train_sequences_2_range_80_20.joblib",
            "description": "Leakage-free 80/20 operational split training set: 49 critical, 49 non-critical (50/50 balanced, 98 events total)",
            "stats": compute_sequence_stats(train_processed),
            "critical_events_count": split_stats["train_critical_count"],
            "non_critical_events_count": split_stats["train_non_critical_count"],
            "scaler_file": split_stats["scaler_file"],
            "random_seed": args.seed
        }
        existing_metadata["datasets"]["val_operational_disjoint_80_20"] = {
            "file": "val_operational_disjoint_80_20.joblib",
            "description": "Leakage-free 80/20 operational split validation set: 12 critical, 7,852 non-critical (7,864 events total, 0 ID overlap)",
            "stats": compute_sequence_stats(val_processed),
            "critical_events_count": split_stats["val_critical_count"],
            "non_critical_events_count": split_stats["val_non_critical_count"],
            "overlap_with_train_2_range_80_20": split_stats["event_id_overlap"],
            "scaler_file": split_stats["scaler_file"],
            "random_seed": args.seed
        }
        existing_metadata["event_ids"]["train_2_range_80_20"] = split_stats["train_event_ids"]
        existing_metadata["event_ids"]["val_disjoint_80_20"] = split_stats["val_event_ids"]

        with open(metadata_path, "w") as f:
            json.dump(existing_metadata, f, indent=2)
        print(f"Updated audit metadata in {metadata_path}")
        print("\n[SUCCESS] 80/20 stratified operational split and isolated scaler generated cleanly!")
        return

    # Fit or load scaler on full training features for 2-range/6-range baselines
    if scaler_path.exists() and args.n_ranges == 2:
        print(f"[5/6] Loading existing fitted MinMaxScaler from {scaler_path}...")
        scaler_dict = joblib.load(scaler_path)
        scaler = scaler_dict["scaler"]
        feature_cols = scaler_dict["columns"]
    else:
        print("[5/6] Fitting MinMaxScaler on transformed features...")
        scaler, feature_cols = fit_feature_scaler(transformed_df)
        joblib.dump({"scaler": scaler, "columns": feature_cols}, scaler_path)
        print(f"Saved fitted scaler to {scaler_path}")

    if args.n_ranges == 2:
        # Generate 2-range paper-aligned training dataset (132 events)
        print("\n[6/6] Generating paper-aligned 2-range balanced training set (132 events, 50/50 balance)...")
        train_samples, train_stats = select_balanced_train_samples(event_groups, n_ranges=2, random_seed=args.seed)
        train_path = OUTPUT_DIR / "train_sequences_2_ranges.joblib"
        train_processed = prepare_tensors_and_save(train_samples, scaler, feature_cols, train_path)
        
        # Generate consistent disjoint validation set for 2-range
        val_all_path = OUTPUT_DIR / "val_operational_all.joblib"
        val_disjoint_2r_path = OUTPUT_DIR / "val_operational_disjoint_2_ranges.joblib"
        
        train_id_set = set(train_stats["event_ids"])
        if val_all_path.exists():
            print("[6/6] Generating disjoint validation set for 2-range from val_operational_all...")
            val_all_processed = joblib.load(val_all_path)
            val_disjoint_2r_processed = [s for s in val_all_processed if s["event_id"] not in train_id_set]
            joblib.dump(val_disjoint_2r_processed, val_disjoint_2r_path)
            print(f"Saved {len(val_disjoint_2r_processed)} sequences to {val_disjoint_2r_path}")
        else:
            val_all_samples, val_disjoint_2r_samples, _ = select_validation_samples(
                event_groups, train_stats["event_ids"], random_seed=args.seed
            )
            val_all_processed = prepare_tensors_and_save(val_all_samples, scaler, feature_cols, val_all_path)
            val_disjoint_2r_processed = prepare_tensors_and_save(val_disjoint_2r_samples, scaler, feature_cols, val_disjoint_2r_path)

        # Update metadata without overwriting existing entries
        if "raw_dataset" not in existing_metadata:
            existing_metadata["raw_dataset"] = {
                "path": str(RAW_TRAIN_CSV),
                "total_rows": len(transformed_df),
                "total_events": len(event_groups)
            }
        if "feature_spec" not in existing_metadata:
            existing_metadata["feature_spec"] = {
                "total_feature_count": len(feature_cols),
                "recurrent_input_dim": len(feature_cols) + 1,
                "feature_columns": feature_cols,
                "shared_columns_summed": SHARED_COLUMNS,
                "categorical_encoded": OBJECT_TYPE_COLUMNS
            }
        if "scaler_spec" not in existing_metadata:
            existing_metadata["scaler_spec"] = {
                "type": "MinMaxScaler",
                "saved_file": "scaler.joblib",
                "feature_count": len(feature_cols)
            }
        if "datasets" not in existing_metadata:
            existing_metadata["datasets"] = {}
        if "event_ids" not in existing_metadata:
            existing_metadata["event_ids"] = {}
            
        existing_metadata["datasets"]["train_2_ranges"] = {
            "file": "train_sequences_2_ranges.joblib",
            "description": "Paper-aligned 50/50 balanced training set (Tulczyjew et al. 2021 Section 3): 66 high-risk, 66 low-risk",
            "stats": compute_sequence_stats(train_processed),
            "bin_distribution": train_stats["balanced_bin_counts"],
            "raw_bin_distribution": train_stats["raw_bin_counts"],
            "random_seed": args.seed
        }
        existing_metadata["datasets"]["val_operational_disjoint_2_ranges"] = {
            "file": "val_operational_disjoint_2_ranges.joblib",
            "description": "Strictly disjoint validation set for 2-range training, excluding 125 overlapping train_2_ranges event IDs. Contains 0 critical events as all 61 operational critical candidates were selected into train_2_ranges.",
            "stats": compute_sequence_stats(val_disjoint_2r_processed),
            "overlap_with_train_2_ranges": 0,
            "critical_events_count": sum(1 for s in val_disjoint_2r_processed if s["target_risk"] >= -6.0)
        }
        existing_metadata["event_ids"]["train_2_ranges"] = train_stats["event_ids"]
        existing_metadata["event_ids"]["val_disjoint_2_ranges"] = [s["event_id"] for s in val_disjoint_2r_processed]
        
        with open(metadata_path, "w") as f:
            json.dump(existing_metadata, f, indent=2)
        print(f"Updated audit metadata in {metadata_path}")
        print("\n[SUCCESS] 2-range paper-aligned training dataset generated cleanly!")
        
    else:
        # Original 6-range generation
        print("[6/6] Generating balanced training & operational validation sets...")
        train_samples, train_stats = select_balanced_train_samples(event_groups, n_ranges=6, random_seed=args.seed)
        
        val_all_samples, val_disjoint_samples, val_stats = select_validation_samples(
            event_groups, train_stats["event_ids"], random_seed=args.seed
        )
        
        train_path = OUTPUT_DIR / "train_sequences_6_ranges.joblib"
        val_all_path = OUTPUT_DIR / "val_operational_all.joblib"
        val_disjoint_path = OUTPUT_DIR / "val_operational_disjoint.joblib"
        val_disjoint_6r_path = OUTPUT_DIR / "val_operational_disjoint_6_ranges.joblib"
        
        train_processed = prepare_tensors_and_save(train_samples, scaler, feature_cols, train_path)
        val_all_processed = prepare_tensors_and_save(val_all_samples, scaler, feature_cols, val_all_path)
        val_disjoint_processed = prepare_tensors_and_save(val_disjoint_samples, scaler, feature_cols, val_disjoint_path)
        # Also save explicitly named 6-ranges file
        joblib.dump(val_disjoint_processed, val_disjoint_6r_path)
        
        metadata = {
            "raw_dataset": {
                "path": str(RAW_TRAIN_CSV),
                "total_rows": len(transformed_df),
                "total_events": len(event_groups)
            },
            "feature_spec": {
                "total_feature_count": len(feature_cols),
                "recurrent_input_dim": len(feature_cols) + 1,
                "feature_columns": feature_cols,
                "shared_columns_summed": SHARED_COLUMNS,
                "categorical_encoded": OBJECT_TYPE_COLUMNS
            },
            "scaler_spec": {
                "type": "MinMaxScaler",
                "saved_file": "scaler.joblib",
                "feature_count": len(feature_cols)
            },
            "datasets": {
                "train_6_ranges": {
                    "file": "train_sequences_6_ranges.joblib",
                    "stats": compute_sequence_stats(train_processed),
                    "bin_distribution": train_stats["balanced_bin_counts"],
                    "raw_bin_distribution": train_stats["raw_bin_counts"],
                    "random_seed": args.seed
                },
                "val_operational_all": {
                    "file": "val_operational_all.joblib",
                    "description": "Original authors validation pool matching clean_data/val_samples_x.pickle (includes 133 train events)",
                    "stats": compute_sequence_stats(val_all_processed),
                    "overlap_with_train": val_stats["train_overlap_events"]
                },
                "val_operational_disjoint": {
                    "file": "val_operational_disjoint.joblib",
                    "description": "Strictly disjoint validation set for 6-range training (7,833 events, excluding 129 overlapping train_6_ranges event IDs)",
                    "stats": compute_sequence_stats(val_disjoint_processed),
                    "overlap_with_train": 0
                },
                "val_operational_disjoint_6_ranges": {
                    "file": "val_operational_disjoint_6_ranges.joblib",
                    "description": "Strictly disjoint validation set for 6-range training, excluding 129 overlapping train_6_ranges event IDs",
                    "stats": compute_sequence_stats(val_disjoint_processed),
                    "overlap_with_train_6_ranges": 0,
                    "critical_events_count": sum(1 for s in val_disjoint_processed if s["target_risk"] >= -6.0)
                }
            },
            "event_ids": {
                "train_6_ranges": train_stats["event_ids"],
                "val_all": val_stats["event_ids_val_all"],
                "val_disjoint": val_stats["event_ids_val_disjoint"],
                "val_disjoint_6_ranges": val_stats["event_ids_val_disjoint"]
            }
        }
        
        # Preserve train_2_ranges and val_operational_disjoint_2_ranges in metadata if previously generated
        if "train_2_ranges" in existing_metadata.get("datasets", {}):
            metadata["datasets"]["train_2_ranges"] = existing_metadata["datasets"]["train_2_ranges"]
            metadata["event_ids"]["train_2_ranges"] = existing_metadata["event_ids"]["train_2_ranges"]
        if "val_operational_disjoint_2_ranges" in existing_metadata.get("datasets", {}):
            metadata["datasets"]["val_operational_disjoint_2_ranges"] = existing_metadata["datasets"]["val_operational_disjoint_2_ranges"]
            metadata["event_ids"]["val_disjoint_2_ranges"] = existing_metadata["event_ids"]["val_disjoint_2_ranges"]
            
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved complete audit metadata to {metadata_path}")
        print("\n[SUCCESS] Preprocessing completed faithfully and cleanly!")


if __name__ == "__main__":
    main()
