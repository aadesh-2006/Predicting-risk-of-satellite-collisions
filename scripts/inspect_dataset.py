"""
Inspection script for ESA Collision Avoidance Challenge dataset.
Inspects train_data.csv and test_data.csv without modifying them.
Performs chunked reading to efficiently compute row counts, missing values,
unique events, and chronological ordering.
Verifies test_data_private.csv by checking existence and size only.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

TRAIN_PATH = Path(r"C:\Users\Aadesh\Downloads\Collision Avoidance Challenge - Dataset\Collision Avoidance Challenge - Dataset\kelvins_competition_data\train_data\train_data.csv")
TEST_PATH = Path(r"C:\Users\Aadesh\Downloads\Collision Avoidance Challenge - Dataset\Collision Avoidance Challenge - Dataset\kelvins_competition_data\test_data.csv")
PRIVATE_TEST_PATH = Path(r"C:\Users\Aadesh\Downloads\Collision Avoidance Challenge - Dataset\Collision Avoidance Challenge - Dataset\kelvins_competition_data\test_data_private.csv")

CHUNK_SIZE = 50000


def verify_file_info(path: Path):
    if not path.exists():
        return {"exists": False, "path": str(path)}
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "extension": path.suffix,
    }


def inspect_dataset_csv(file_path: Path, name: str):
    print(f"\n{'='*30} Inspecting {name} {'='*30}")
    print(f"File Path: {file_path}")
    
    # 1. Preview first 5 rows for column schema & dtypes
    preview_df = pd.read_csv(file_path, nrows=5)
    columns = list(preview_df.columns)
    dtypes = {col: str(preview_df[col].dtype) for col in columns}
    
    print(f"Total Columns: {len(columns)}")
    print(f"Columns: {columns[:10]} ... ({len(columns) - 10} more)")
    
    # 2. Chunked processing for full file stats
    total_rows = 0
    missing_counts = pd.Series(0, index=columns, dtype=np.int64)
    unique_events = set()
    has_event_id = "event_id" in columns
    
    # Risk column identification
    risk_cols = [c for c in columns if "risk" in c.lower()]
    print(f"Identified risk-related columns: {risk_cols}")
    
    time_cols = [c for c in columns if "time" in c.lower() or "tca" in c.lower()]
    print(f"Identified time-related columns: {time_cols}")
    
    # Chronological check tracker: verify for first 500 events if time is monotonic
    # We track last seen time_to_tca and event_id transitions
    chrono_checked_events = {}  # event_id -> list of time values
    MAX_EVENTS_CHRONO_CHECK = 1000
    
    print("Reading file in chunks...")
    chunk_idx = 0
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)
        missing_counts += chunk.isna().sum()
        
        if has_event_id:
            unique_events.update(chunk["event_id"].unique())
            
            # Sample events for chronological order check
            if len(chrono_checked_events) < MAX_EVENTS_CHRONO_CHECK and "time_to_tca" in chunk.columns:
                grouped = chunk.groupby("event_id")["time_to_tca"].apply(list)
                for eid, times in grouped.items():
                    if eid not in chrono_checked_events and len(chrono_checked_events) < MAX_EVENTS_CHRONO_CHECK:
                        chrono_checked_events[eid] = times
                    elif eid in chrono_checked_events and len(chrono_checked_events[eid]) < 100:
                        chrono_checked_events[eid].extend(times)
                        
        chunk_idx += 1
        if chunk_idx % 2 == 0:
            print(f"  Processed {total_rows:,} rows...")
            
    print(f"Total rows: {total_rows:,}")
    if has_event_id:
        print(f"Unique event_id count: {len(unique_events):,}")
    else:
        print("event_id column NOT found!")

    # Check chronological ordering
    monotonic_decreasing_count = 0
    monotonic_increasing_count = 0
    non_monotonic_count = 0
    single_message_events = 0
    
    for eid, times in chrono_checked_events.items():
        if len(times) <= 1:
            single_message_events += 1
        elif all(times[i] >= times[i+1] for i in range(len(times)-1)):
            monotonic_decreasing_count += 1
        elif all(times[i] <= times[i+1] for i in range(len(times)-1)):
            monotonic_increasing_count += 1
        else:
            non_monotonic_count += 1

    chrono_summary = {
        "events_checked": len(chrono_checked_events),
        "single_message_events": single_message_events,
        "monotonic_decreasing_tca (approaching TCA)": monotonic_decreasing_count,
        "monotonic_increasing_tca": monotonic_increasing_count,
        "non_monotonic": non_monotonic_count,
    }
    print(f"Chronological check summary (first {len(chrono_checked_events)} events): {chrono_summary}")
    
    # Missing values summary
    cols_with_missing = missing_counts[missing_counts > 0].to_dict()
    print(f"Columns with missing values: {len(cols_with_missing)}")
    for col, count in sorted(cols_with_missing.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = (count / total_rows) * 100
        print(f"  - {col}: {count:,} ({pct:.2f}%)")

    # Risk stats summary
    risk_summary = {}
    if risk_cols:
        # Read a sample or compute on risk columns
        print("Gathering risk statistics...")
        sample_df = pd.read_csv(file_path, usecols=risk_cols)
        for rc in risk_cols:
            series = sample_df[rc].dropna()
            risk_summary[rc] = {
                "dtype": str(sample_df[rc].dtype),
                "count": int(series.count()),
                "missing": int(sample_df[rc].isna().sum()),
                "min": float(series.min()) if not series.empty and np.issubdtype(series.dtype, np.number) else None,
                "max": float(series.max()) if not series.empty and np.issubdtype(series.dtype, np.number) else None,
                "mean": float(series.mean()) if not series.empty and np.issubdtype(series.dtype, np.number) else None,
                "std": float(series.std()) if not series.empty and np.issubdtype(series.dtype, np.number) else None,
                "quantiles": {
                    "25%": float(series.quantile(0.25)) if not series.empty and np.issubdtype(series.dtype, np.number) else None,
                    "50%": float(series.quantile(0.50)) if not series.empty and np.issubdtype(series.dtype, np.number) else None,
                    "75%": float(series.quantile(0.75)) if not series.empty and np.issubdtype(series.dtype, np.number) else None,
                    "90%": float(series.quantile(0.90)) if not series.empty and np.issubdtype(series.dtype, np.number) else None,
                },
                "sample_values": series.head(5).tolist(),
            }
            print(f"  {rc}: min={risk_summary[rc]['min']}, max={risk_summary[rc]['max']}, median={risk_summary[rc]['quantiles']['50%']}")

    return {
        "file_path": str(file_path),
        "total_rows": total_rows,
        "total_columns": len(columns),
        "columns": columns,
        "dtypes": dtypes,
        "has_event_id": has_event_id,
        "unique_events_count": len(unique_events) if has_event_id else 0,
        "risk_columns": risk_cols,
        "risk_summary": risk_summary,
        "time_columns": time_cols,
        "chrono_summary": chrono_summary,
        "columns_with_missing_count": len(cols_with_missing),
        "top_missing_columns": {k: {"count": int(v), "pct": round(float(v)/total_rows*100, 2)} 
                                for k, v in sorted(cols_with_missing.items(), key=lambda x: x[1], reverse=True)[:15]}
    }


def main():
    print("=== ESA Collision Avoidance Challenge Dataset Inspection ===")
    
    # Verify files
    train_info = verify_file_info(TRAIN_PATH)
    test_info = verify_file_info(TEST_PATH)
    private_test_info = verify_file_info(PRIVATE_TEST_PATH)
    
    print("\n[File Existence & Sizes]")
    print(f"train_data.csv:        {train_info}")
    print(f"test_data.csv:         {test_info}")
    print(f"test_data_private.csv: {private_test_info} (DO NOT INSPECT CONTENTS)")
    
    if not train_info["exists"] or not test_info["exists"]:
        print("ERROR: Required dataset files are missing!")
        sys.exit(1)
        
    train_results = inspect_dataset_csv(TRAIN_PATH, "Train Dataset")
    test_results = inspect_dataset_csv(TEST_PATH, "Test Dataset")
    
    output_report = {
        "files": {
            "train": train_info,
            "test": test_info,
            "test_private": private_test_info,
        },
        "train_inspection": train_results,
        "test_inspection": test_results,
    }
    
    report_path = Path(r"C:\Users\Aadesh\.gemini\antigravity\scratch\Predicting-risk-of-satellite-collisions\scripts\inspection_results.json")
    with open(report_path, "w") as f:
        json.dump(output_report, f, indent=2)
    print(f"\nInspection summary successfully written to {report_path}")


if __name__ == "__main__":
    main()
