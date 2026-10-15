"""
Preprocessing and Evaluation Dataset Preparation for test_data.csv
Tulczyjew et al. (2021) / ESA Kelvins Collision Avoidance Challenge

Requirements:
- Reads test_data.csv only (never accesses test_data_private.csv).
- Uses the fitted MinMaxScaler from training: data/processed/scaler.joblib.
- Handles missing values via column-wise mean imputation.
- Sums 40 shared columns (t_* + c_* -> x_*).
- One-hot encodes c_object_type into the verified 5 categories.
- Ensures identical 63 feature columns and column ordering as training.
- Sorts CDMs chronologically by decreasing time_to_tca within each event.
- Prepares sequential structures:
    * For T >= 2 (2,092 events): input_sequence (T-1, 64), target_risk (scalar at T-1),
      evaluable=True
    * For T = 1 (75 events): input_sequence None (empty), target_risk (scalar at T-1),
      evaluable=False (documented limitation)
- Saves output to data/processed/test_sequences.joblib and metadata to
  data/processed/test_preprocessing_metadata.json.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd

# Paths
TEST_CSV = Path(r"C:\Users\Aadesh\Downloads\Collision Avoidance Challenge - Dataset\Collision Avoidance Challenge - Dataset\kelvins_competition_data\test_data.csv")
SCALER_PATH = Path(r"C:\Users\Aadesh\.gemini\antigravity\scratch\Predicting-risk-of-satellite-collisions\data\processed\scaler.joblib")
OUTPUT_SEQUENCES_PATH = Path(r"C:\Users\Aadesh\.gemini\antigravity\scratch\Predicting-risk-of-satellite-collisions\data\processed\test_sequences.joblib")
OUTPUT_METADATA_PATH = Path(r"C:\Users\Aadesh\.gemini\antigravity\scratch\Predicting-risk-of-satellite-collisions\data\processed\test_preprocessing_metadata.json")

# Verified column schemas
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


def prepare_test_dataset():
    print(f"[1/5] Loading raw test data from {TEST_CSV}...")
    assert TEST_CSV.exists(), f"File not found: {TEST_CSV}"
    df = pd.read_csv(TEST_CSV)
    total_raw_rows = len(df)
    total_unique_events = df['event_id'].nunique()
    print(f"Loaded {total_raw_rows} rows across {total_unique_events} unique events.")

    if "mission_id" in df.columns:
        df.drop(columns=["mission_id"], inplace=True)

    print("[2/5] Imputing missing values with numeric column means...")
    numeric_means = df.mean(numeric_only=True)
    df.fillna(numeric_means, inplace=True)

    print("[3/5] Engineering combined 'x' features and one-hot encoding...")
    columns_dict = {}
    for col in UNIQUE_COLUMNS:
        columns_dict[col] = df[col]

    for key in SHARED_COLUMNS:
        t_col = key.replace('x', 't')
        c_col = key.replace('x', 'c')
        columns_dict[key] = df[t_col] + df[c_col]

    out_df = pd.concat(columns_dict, axis=1)
    out_df = pd.get_dummies(out_df, columns=['c_object_type'], prefix=['c_object_type'], dtype=np.float64)

    for col in OBJECT_TYPE_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = 0.0

    print(f"[4/5] Loading fitted MinMaxScaler from {SCALER_PATH}...")
    scaler_dict = joblib.load(SCALER_PATH)
    scaler = scaler_dict['scaler']
    feature_cols = scaler_dict['columns']

    expected_feature_cols = [
        'time_to_tca', 'max_risk_estimate', 'max_risk_scaling', 'miss_distance',
        'relative_speed', 'relative_position_n', 'relative_position_r',
        'relative_position_t', 'relative_velocity_n', 'relative_velocity_r',
        'relative_velocity_t', 'geocentric_latitude', 'azimuth', 'elevation',
        'F10', 'AP', 'F3M', 'SSN'
    ] + SHARED_COLUMNS + OBJECT_TYPE_COLUMNS

    assert feature_cols == expected_feature_cols, "Feature columns mismatch with scaler!"
    all_cols = ['event_id', 'risk'] + feature_cols
    out_df = out_df[all_cols]
    print(f"Transformed DataFrame shape: {out_df.shape} (65 columns total: event_id, risk, 63 features)")

    print("[5/5] Building event sequences and applying feature scaling...")
    evaluable_events = []
    skipped_events = []
    all_event_records = []

    # Group by event_id in original encounter order
    for eid, group in out_df.groupby('event_id', sort=False):
        # Sort chronologically by descending time_to_tca (furthest to nearest)
        sorted_group = group.sort_values('time_to_tca', ascending=False).reset_index(drop=True)
        n_cdms = len(sorted_group)
        raw_risks = sorted_group['risk'].to_numpy(dtype=np.float32)
        raw_tcas = sorted_group['time_to_tca'].to_numpy(dtype=np.float32)

        # Scale the 63 features with float64 precision
        unscaled_features = sorted_group[feature_cols].to_numpy(dtype=np.float64)
        scaled_features = scaler.transform(unscaled_features).astype(np.float32)

        # Prepend risk channel: shape (T, 64)
        risk_column = np.expand_dims(raw_risks, axis=-1)
        step_vectors = np.concatenate([risk_column, scaled_features], axis=-1)

        target_risk = float(raw_risks[-1])
        last_observed_tca = float(raw_tcas[-1])
        first_observed_tca = float(raw_tcas[0])

        if n_cdms >= 2:
            # Historical input sequence is 0 ... T-2 (shape: (T - 1, 64))
            input_sequence = step_vectors[:-1, :]
            record = {
                "event_id": int(eid),
                "evaluable": True,
                "input_sequence": input_sequence,
                "target_risk": target_risk,
                "raw_risks": raw_risks,
                "raw_tcas": raw_tcas,
                "seq_len": int(n_cdms),
                "num_input_steps": int(input_sequence.shape[0]),
                "last_observed_tca": last_observed_tca,
                "first_observed_tca": first_observed_tca
            }
            evaluable_events.append(record)
            all_event_records.append(record)
        else:
            # Single CDM event: cannot construct historical sequence (T-1 = 0 steps)
            record = {
                "event_id": int(eid),
                "evaluable": False,
                "input_sequence": None,
                "target_risk": target_risk,
                "raw_risks": raw_risks,
                "raw_tcas": raw_tcas,
                "seq_len": int(n_cdms),
                "num_input_steps": 0,
                "last_observed_tca": last_observed_tca,
                "first_observed_tca": first_observed_tca
            }
            skipped_events.append(record)
            all_event_records.append(record)

    print(f"Total events processed: {len(all_event_records)}")
    print(f"Evaluable events (seq_len >= 2): {len(evaluable_events)}")
    print(f"Non-evaluable events (seq_len == 1): {len(skipped_events)}")

    # Summary statistics for evaluable set
    eval_targets = [r["target_risk"] for r in evaluable_events]
    crit_count = sum(1 for t in eval_targets if t >= -6.0)
    print(f"Critical events in evaluable set (target >= -6.0): {crit_count} ({crit_count / len(eval_targets) * 100:.2f}%)")

    # Save processed joblib
    OUTPUT_SEQUENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluable_events": evaluable_events,
        "skipped_events": skipped_events,
        "all_events": all_event_records,
        "scaler_feature_columns": feature_cols
    }
    joblib.dump(payload, OUTPUT_SEQUENCES_PATH)
    file_size_mb = OUTPUT_SEQUENCES_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved test sequences to {OUTPUT_SEQUENCES_PATH} ({file_size_mb:.2f} MB)")

    # Save audit metadata
    metadata = {
        "raw_test_file": str(TEST_CSV),
        "total_rows": total_raw_rows,
        "total_unique_events": total_unique_events,
        "evaluable_events_count": len(evaluable_events),
        "skipped_events_count": len(skipped_events),
        "evaluable_critical_events_count": crit_count,
        "skipped_critical_events_count": sum(1 for r in skipped_events if r["target_risk"] >= -6.0),
        "seq_len_min_evaluable": int(min(r["seq_len"] for r in evaluable_events)),
        "seq_len_max_evaluable": int(max(r["seq_len"] for r in evaluable_events)),
        "seq_len_mean_evaluable": float(np.mean([r["seq_len"] for r in evaluable_events])),
        "tca_min_days": float(df['time_to_tca'].min()),
        "tca_max_days": float(df['time_to_tca'].max()),
        "feature_count": len(feature_cols),
        "recurrent_input_dim": 64
    }
    with open(OUTPUT_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved test preprocessing metadata to {OUTPUT_METADATA_PATH}")


if __name__ == "__main__":
    prepare_test_dataset()
