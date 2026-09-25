#!/usr/bin/env python3
"""
src/build_normalized.py: High-throughput, streaming normalization pipeline
for AWS ML Hackathon 2026 Business Entity Resolution.

Reads TSV datasets in streaming batches, applies deterministic normalization,
and writes memory-efficient Parquet files with verified schemas and exact row counts.
"""

import os
import sys
import time
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Any
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.normalize import get_derived_record

sys.stdout.reconfigure(encoding='utf-8')

# Strict, stable, space-optimized schema for all normalized datasets
PARQUET_SCHEMA = pa.schema([
    ("entity_id", pa.string()),
    ("raw_name", pa.string()),
    ("raw_address", pa.string()),
    ("country", pa.string()),
    ("norm_name", pa.string()),
    ("norm_name_clean", pa.string()),
    ("norm_address", pa.string()),
    ("postal_code", pa.string()),
    ("has_missing_address", pa.bool_()),
])

DEFAULT_DATASETS = [
    ("train_source1", "dataset/train/train_source1.tsv", "dataset/normalized/train_source1.parquet"),
    ("train_source2", "dataset/train/train_source2.tsv", "dataset/normalized/train_source2.parquet"),
    ("train_source3", "dataset/train/train_source3.tsv", "dataset/normalized/train_source3.parquet"),
    ("test_source1",  "dataset/test/test_source1.tsv",  "dataset/normalized/test_source1.parquet"),
    ("test_source2",  "dataset/test/test_source2.tsv",  "dataset/normalized/test_source2.parquet"),
    ("test_source3",  "dataset/test/test_source3.tsv",  "dataset/normalized/test_source3.parquet"),
]

def process_raw_chunk(rows: List[Tuple[str, str, str, str]]) -> Dict[str, list]:
    """Processes a chunk of raw (eid, name, addr, country) tuples into columnar lists."""
    entity_ids = []
    raw_names = []
    raw_addrs = []
    countries = []
    norm_names = []
    norm_name_cleans = []
    norm_addrs = []
    postal_codes = []
    has_missing_addrs = []

    for eid, name, addr, ctry in rows:
        rec = get_derived_record(eid, name, addr, ctry)
        entity_ids.append(rec["entity_id"])
        raw_names.append(rec["raw_name"])
        raw_addrs.append(rec["raw_address"])
        countries.append(rec["country"])
        norm_names.append(rec["norm_name"])
        norm_name_cleans.append(rec["norm_name_clean"])
        norm_addrs.append(rec["norm_address"])
        postal_codes.append(rec["postal_code"])
        has_missing_addrs.append(rec["has_missing_address"])

    return {
        "entity_id": entity_ids,
        "raw_name": raw_names,
        "raw_address": raw_addrs,
        "country": countries,
        "norm_name": norm_names,
        "norm_name_clean": norm_name_cleans,
        "norm_address": norm_addrs,
        "postal_code": postal_codes,
        "has_missing_address": has_missing_addrs,
    }


def normalize_source_file(
    input_path: str,
    output_path: str,
    batch_size: int = 100000,
    sub_chunk_size: int = 12500,
    max_workers: int = 8,
    compression: str = "zstd",
    compression_level: int = 7,
) -> Dict[str, Any]:
    """
    Normalizes a TSV source file to Parquet in streaming batches.
    Memory usage is O(batch_size) — never loads the entire file into RAM.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    start_time = time.perf_counter()
    proc = psutil.Process()
    initial_mem_mb = proc.memory_info().rss / (1024 * 1024)
    peak_mem_mb = initial_mem_mb

    print(f"\n>>> Starting normalization: {input_path} -> {output_path}")
    print(f"    Batch size: {batch_size:,} | Workers: {max_workers} | Compression: {compression} (level {compression_level})")

    total_rows = 0
    writer = pq.ParquetWriter(output_path, PARQUET_SCHEMA, compression=compression, compression_level=compression_level)

    # Use multiprocessing executor if workers > 1, else single thread
    executor = ProcessPoolExecutor(max_workers=max_workers) if max_workers > 1 else None

    try:
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()  # skip header
            
            batch_rows: List[Tuple[str, str, str, str]] = []
            
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) == 4:
                    batch_rows.append((parts[0], parts[1], parts[2], parts[3]))
                elif len(parts) > 0 and parts[0]:
                    # Pad if malformed to preserve row count and ID
                    eid = parts[0]
                    name = parts[1] if len(parts) > 1 else ""
                    addr = parts[2] if len(parts) > 2 else ""
                    ctry = parts[3] if len(parts) > 3 else ""
                    batch_rows.append((eid, name, addr, ctry))

                if len(batch_rows) >= batch_size:
                    # Process batch
                    if executor is not None:
                        # Split batch into subchunks for parallel workers
                        sub_chunks = [
                            batch_rows[i:i + sub_chunk_size]
                            for i in range(0, len(batch_rows), sub_chunk_size)
                        ]
                        sub_results = list(executor.map(process_raw_chunk, sub_chunks))
                        # Merge columnar dicts
                        merged = {col: [] for col in PARQUET_SCHEMA.names}
                        for res in sub_results:
                            for col in PARQUET_SCHEMA.names:
                                merged[col].extend(res[col])
                        table = pa.Table.from_pydict(merged, schema=PARQUET_SCHEMA)
                    else:
                        col_dict = process_raw_chunk(batch_rows)
                        table = pa.Table.from_pydict(col_dict, schema=PARQUET_SCHEMA)

                    writer.write_table(table)
                    total_rows += len(batch_rows)
                    batch_rows = []

                    curr_mem_mb = proc.memory_info().rss / (1024 * 1024)
                    if curr_mem_mb > peak_mem_mb:
                        peak_mem_mb = curr_mem_mb

                    elapsed = time.perf_counter() - start_time
                    rate = total_rows / elapsed if elapsed > 0 else 0
                    print(f"    Written {total_rows:,} rows... ({rate:,.0f} rows/s, RAM: {curr_mem_mb:.1f} MB)", end="\r")

            # Final partial batch
            if batch_rows:
                if executor is not None:
                    sub_chunks = [
                        batch_rows[i:i + sub_chunk_size]
                        for i in range(0, len(batch_rows), sub_chunk_size)
                    ]
                    sub_results = list(executor.map(process_raw_chunk, sub_chunks))
                    merged = {col: [] for col in PARQUET_SCHEMA.names}
                    for res in sub_results:
                        for col in PARQUET_SCHEMA.names:
                            merged[col].extend(res[col])
                    table = pa.Table.from_pydict(merged, schema=PARQUET_SCHEMA)
                else:
                    col_dict = process_raw_chunk(batch_rows)
                    table = pa.Table.from_pydict(col_dict, schema=PARQUET_SCHEMA)

                writer.write_table(table)
                total_rows += len(batch_rows)
    finally:
        writer.close()
        if executor is not None:
            executor.shutdown()

    elapsed = time.perf_counter() - start_time
    out_size_bytes = os.path.getsize(output_path)
    in_size_bytes = os.path.getsize(input_path)
    compression_ratio = in_size_bytes / out_size_bytes if out_size_bytes > 0 else 0

    print(f"\n    Completed {total_rows:,} rows in {elapsed:.2f}s ({total_rows/elapsed:,.0f} rows/s)")
    print(f"    Input: {in_size_bytes / (1024*1024):.1f} MB | Output: {out_size_bytes / (1024*1024):.1f} MB (Compression: {compression_ratio:.2f}x)")
    print(f"    Peak RAM: {peak_mem_mb:.1f} MB")

    # Fast verification
    meta = pq.read_metadata(output_path)
    assert meta.num_rows == total_rows, f"Parquet row count mismatch: {meta.num_rows} != {total_rows}"
    print(f"    [VERIFIED] Parquet row groups: {meta.num_row_groups}, exact rows: {meta.num_rows:,}")

    return {
        "dataset": os.path.basename(input_path),
        "input_path": input_path,
        "output_path": output_path,
        "total_rows": total_rows,
        "input_size_mb": in_size_bytes / (1024 * 1024),
        "output_size_mb": out_size_bytes / (1024 * 1024),
        "compression_ratio": compression_ratio,
        "elapsed_seconds": elapsed,
        "rows_per_second": total_rows / elapsed if elapsed > 0 else 0,
        "peak_ram_mb": peak_mem_mb,
    }


def normalize_ground_truth(
    gt_path: str = "dataset/train/train_ground_truth.tsv",
    out_path: str = "dataset/normalized/train_ground_truth.parquet",
    pairs_out_path: str = "dataset/normalized/train_ground_truth_pairs.parquet",
) -> Dict[str, Any]:
    """Normalizes and indexes ground truth for rapid candidate evaluation in Phase 1."""
    print(f"\n>>> Converting ground truth: {gt_path} -> {out_path} and {pairs_out_path}")
    t0 = time.perf_counter()

    gt_schema = pa.schema([
        ("source1_entity_id", pa.string()),
        ("matched_entity_ids", pa.string()),
        ("match_count", pa.int32()),
    ])

    pairs_schema = pa.schema([
        ("source1_entity_id", pa.string()),
        ("matched_id", pa.string()),
        ("target_source", pa.string()),
    ])

    s1_ids = []
    matched_strs = []
    match_counts = []

    pair_s1 = []
    pair_target = []
    pair_source = []

    total_rows = 0
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            total_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 2:
                s1, matched = parts
            else:
                s1 = parts[0]
                matched = ""

            s1_clean = s1.strip()
            matched_clean = matched.strip()
            ids = [m.strip() for m in matched_clean.split(",") if m.strip()] if matched_clean else []

            s1_ids.append(s1_clean)
            matched_strs.append(matched_clean)
            match_counts.append(len(ids))

            for m in ids:
                pair_s1.append(s1_clean)
                pair_target.append(m)
                pair_source.append("S2" if m.startswith("S2-") else "S3" if m.startswith("S3-") else "OTHER")

    gt_table = pa.Table.from_pydict({
        "source1_entity_id": s1_ids,
        "matched_entity_ids": matched_strs,
        "match_count": match_counts,
    }, schema=gt_schema)
    pq.write_table(gt_table, out_path, compression="snappy")

    pairs_table = pa.Table.from_pydict({
        "source1_entity_id": pair_s1,
        "matched_id": pair_target,
        "target_source": pair_source,
    }, schema=pairs_schema)
    pq.write_table(pairs_table, pairs_out_path, compression="snappy")

    t1 = time.perf_counter()
    print(f"    Saved ground truth overview: {len(s1_ids):,} rows")
    print(f"    Saved ground truth exploded pairs: {len(pair_s1):,} positive pairs in {t1 - t0:.2f}s")
    return {
        "total_s1": len(s1_ids),
        "total_pairs": len(pair_s1),
        "elapsed_seconds": t1 - t0,
    }


def main():
    parser = argparse.ArgumentParser(description="Streaming Normalization Pipeline for ML Hackathon 2026")
    parser.add_argument("--all", action="store_true", help="Normalize all 6 train & test datasets + ground truth")
    parser.add_argument("--input", type=str, help="Path to input TSV file")
    parser.add_argument("--output", type=str, help="Path to output Parquet file")
    parser.add_argument("--batch-size", type=int, default=100000, help="Rows per batch chunk")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    parser.add_argument("--compression", type=str, default="snappy", help="Parquet compression codec")
    args = parser.parse_args()

    os.makedirs("dataset/normalized", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    stats_list = []

    if args.all:
        print("==================================================================")
        print("   STARTING END-TO-END DATASET NORMALIZATION (ALL 6 DATASETS)    ")
        print("==================================================================")
        overall_start = time.perf_counter()

        for name, in_path, out_path in DEFAULT_DATASETS:
            res = normalize_source_file(
                input_path=in_path,
                output_path=out_path,
                batch_size=args.batch_size,
                max_workers=args.workers,
                compression=args.compression,
            )
            stats_list.append(res)

        # Convert ground truth as well
        gt_stats = normalize_ground_truth()
        stats_list.append(gt_stats)

        total_elapsed = time.perf_counter() - overall_start
        total_rows = sum(s.get("total_rows", 0) for s in stats_list)
        print("\n==================================================================")
        print(f"   ALL DATASETS NORMALIZED SUCCESSFULLY in {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
        print(f"   Total rows processed: {total_rows:,} ({total_rows/total_elapsed:,.0f} rows/s)")
        print("==================================================================")

        with open("artifacts/normalization_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats_list, f, indent=2)

    elif args.input and args.output:
        res = normalize_source_file(
            input_path=args.input,
            output_path=args.output,
            batch_size=args.batch_size,
            max_workers=args.workers,
            compression=args.compression,
        )
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
