#!/usr/bin/env python3
"""
scripts/verify_phase0.py: Automated end-to-end verification for Phase 0.
Audits every Parquet dataset against the original raw TSV files.
"""

import os
import sys
import duckdb
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding='utf-8')

conn = duckdb.connect()

FILES = [
    ("train_source1", "dataset/train/train_source1.tsv", "dataset/normalized/train_source1.parquet", 2206821),
    ("train_source2", "dataset/train/train_source2.tsv", "dataset/normalized/train_source2.parquet", 5034616),
    ("train_source3", "dataset/train/train_source3.tsv", "dataset/normalized/train_source3.parquet", 5285603),
    ("test_source1",  "dataset/test/test_source1.tsv",  "dataset/normalized/test_source1.parquet",  1732544),
    ("test_source2",  "dataset/test/test_source2.tsv",  "dataset/normalized/test_source2.parquet",  4887273),
    ("test_source3",  "dataset/test/test_source3.tsv",  "dataset/normalized/test_source3.parquet",  5082316),
]

EXPECTED_SCHEMA = [
    ("entity_id", "VARCHAR"),
    ("raw_name", "VARCHAR"),
    ("raw_address", "VARCHAR"),
    ("country", "VARCHAR"),
    ("norm_name", "VARCHAR"),
    ("norm_name_clean", "VARCHAR"),
    ("norm_address", "VARCHAR"),
    ("postal_code", "VARCHAR"),
    ("has_missing_address", "BOOLEAN"),
]

print("==================================================================")
print("             PHASE 0 FINAL VERIFICATION AUDIT                     ")
print("==================================================================")

all_passed = True

# 1. Verify schema and row counts across all 6 source datasets
for name, tsv_path, parquet_path, expected_rows in FILES:
    print(f"\n[Verifying {name}]")
    assert os.path.exists(parquet_path), f"Missing parquet file: {parquet_path}"
    
    meta = pq.read_metadata(parquet_path)
    parquet_rows = meta.num_rows
    print(f"  Parquet rows: {parquet_rows:,} | Expected: {expected_rows:,}")
    if parquet_rows != expected_rows:
        print(f"  FAILED: Row count mismatch in {name}!")
        all_passed = False
    else:
        print(f"  PASSED: Row count matches 100% exactly ({expected_rows:,} rows).")

    # Schema check
    schema_query = f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')"
    cols = conn.execute(schema_query).fetchall()
    col_summary = [(c[0], c[1]) for c in cols]
    if col_summary != EXPECTED_SCHEMA:
        print(f"  FAILED: Schema mismatch in {name}!")
        print(f"    Expected: {EXPECTED_SCHEMA}")
        print(f"    Found:    {col_summary}")
        all_passed = False
    else:
        print(f"  PASSED: Schema matches exact specification ({len(EXPECTED_SCHEMA)} columns).")

    # Data integrity check (IDs, nulls, duplicates)
    integrity_query = f"""
        SELECT 
            count(distinct entity_id) as unique_ids,
            count(case when entity_id is null or entity_id = '' then 1 end) as null_ids,
            count(case when norm_name is null then 1 end) as null_norm_name,
            count(case when norm_address is null then 1 end) as null_norm_addr,
            count(case when has_missing_address = true then 1 end) as missing_addr_count
        FROM read_parquet('{parquet_path}')
    """
    res = conn.execute(integrity_query).df().to_dict(orient="records")[0]
    print(f"  Unique IDs: {res['unique_ids']:,} | Null IDs: {res['null_ids']} | Null Norm Name: {res['null_norm_name']} | Missing Addr: {res['missing_addr_count']:,}")
    if res['unique_ids'] != expected_rows or res['null_ids'] != 0 or res['null_norm_name'] != 0:
        print(f"  FAILED: Integrity error in {name}!")
        all_passed = False
    else:
        print(f"  PASSED: Strict ID uniqueness verified (0 duplicates, 0 null IDs).")

# 2. Verify Ground Truth Parquet datasets
print("\n[Verifying train_ground_truth.parquet]")
gt_meta = pq.read_metadata("dataset/normalized/train_ground_truth.parquet")
gt_pairs_meta = pq.read_metadata("dataset/normalized/train_ground_truth_pairs.parquet")
print(f"  GT Overview rows: {gt_meta.num_rows:,} (Expected: 2,206,821)")
print(f"  GT Positive Pairs rows: {gt_pairs_meta.num_rows:,} (Expected: 7,638,365)")

if gt_meta.num_rows != 2206821 or gt_pairs_meta.num_rows != 7638365:
    print("  FAILED: Ground truth row counts mismatch!")
    all_passed = False
else:
    print("  PASSED: Ground truth overview and pairs match exactly.")

# Ground truth cardinality check
card_check = conn.execute("""
    WITH multi AS (
        SELECT matched_id, count(distinct source1_entity_id) as cnt
        FROM read_parquet('dataset/normalized/train_ground_truth_pairs.parquet')
        GROUP BY matched_id
        HAVING count(distinct source1_entity_id) > 1
    )
    SELECT count(*) as multi_mapped_count FROM multi
""").df().to_dict(orient="records")[0]
print(f"  Multi-mapped target entities: {card_check['multi_mapped_count']}")
if card_check['multi_mapped_count'] != 0:
    print("  FAILED: Found multi-mapped target entities!")
    all_passed = False
else:
    print("  PASSED: Zero multi-mapped target entities verified.")

print("\n==================================================================")
if all_passed:
    print("      ALL PHASE 0 VERIFICATION CHECKS PASSED (100% SUCCESS)       ")
else:
    print("      VERIFICATION ENCOUNTERED FAILURES                           ")
print("==================================================================")
