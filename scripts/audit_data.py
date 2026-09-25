#!/usr/bin/env python3
"""
scripts/audit_data.py: Comprehensive Data Audit for ML Hackathon 2026 ER Challenge.
Profiles all 7 train and test files using memory-efficient streaming.
"""

import os
import sys
import json
import csv
from collections import Counter, defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "dataset"
FILES = {
    "train_s1": os.path.join(DATA_DIR, "train", "train_source1.tsv"),
    "train_s2": os.path.join(DATA_DIR, "train", "train_source2.tsv"),
    "train_s3": os.path.join(DATA_DIR, "train", "train_source3.tsv"),
    "train_gt": os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"),
    "test_s1":  os.path.join(DATA_DIR, "test", "test_source1.tsv"),
    "test_s2":  os.path.join(DATA_DIR, "test", "test_source2.tsv"),
    "test_s3":  os.path.join(DATA_DIR, "test", "test_source3.tsv"),
}

LITERAL_NULLS = {"null", "none", "nan", "n/a", "na", "-", "", "undefined"}

def audit_source_file(file_alias, file_path):
    print(f"=== Auditing {file_alias}: {file_path} ===")
    size_bytes = os.path.getsize(file_path)
    
    total_rows = 0
    malformed_rows = 0
    duplicate_ids = 0
    
    seen_ids = set()
    countries = Counter()
    
    null_counts = {
        "entity_id": 0,
        "business_name": 0,
        "business_address": 0,
        "country": 0,
    }
    literal_null_counts = {
        "business_name": 0,
        "business_address": 0,
        "country": 0,
    }
    
    # Text length reservoirs / stats
    # For memory efficiency, sample lengths or collect summary arrays
    # 500k rows can hold int lengths in array easily (e.g. 5M ints = 20MB)
    name_lengths = []
    name_word_counts = []
    addr_lengths = []
    addr_word_counts = []
    
    # Non-ascii detection
    non_ascii_name_count = 0
    non_ascii_addr_count = 0
    
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        header_cols = [c.strip() for c in header.split("\t")]
        
        for line_num, line in enumerate(f, start=2):
            total_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 4:
                malformed_rows += 1
                if malformed_rows <= 3:
                    print(f"  [Malformed line {line_num}] split len = {len(parts)}: {line[:80]!r}")
                continue
            
            eid, bname, baddr, ctry = parts
            
            if eid in seen_ids:
                duplicate_ids += 1
            seen_ids.add(eid)
            
            # Countries
            countries[ctry] += 1
            
            # Null checks
            if not eid:
                null_counts["entity_id"] += 1
            if not bname:
                null_counts["business_name"] += 1
            elif bname.strip().lower() in LITERAL_NULLS:
                literal_null_counts["business_name"] += 1
                
            if not baddr:
                null_counts["business_address"] += 1
            elif baddr.strip().lower() in LITERAL_NULLS:
                literal_null_counts["business_address"] += 1
                
            if not ctry:
                null_counts["country"] += 1
            elif ctry.strip().lower() in LITERAL_NULLS:
                literal_null_counts["country"] += 1
                
            # Lengths (reservoir sampling or stride if huge, or append directly)
            # Up to a few million rows, append length as int:
            # 5M ints in Python list is ~40MB, totally safe.
            n_len = len(bname)
            a_len = len(baddr)
            name_lengths.append(n_len)
            addr_lengths.append(a_len)
            
            # Simple word counts
            name_word_counts.append(len(bname.split()))
            addr_word_counts.append(len(baddr.split()))
            
            if not bname.isascii():
                non_ascii_name_count += 1
            if not baddr.isascii():
                non_ascii_addr_count += 1

    def compute_stats(arr):
        if not arr:
            return {}
        a = np.array(arr, dtype=np.int32)
        return {
            "min": int(np.min(a)),
            "max": int(np.max(a)),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
        }

    results = {
        "file": file_path,
        "size_bytes": size_bytes,
        "total_rows": total_rows,
        "unique_entity_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids,
        "malformed_rows": malformed_rows,
        "header": header_cols,
        "countries": dict(countries),
        "null_counts": null_counts,
        "literal_null_counts": literal_null_counts,
        "non_ascii_names": non_ascii_name_count,
        "non_ascii_name_pct": (non_ascii_name_count / total_rows * 100) if total_rows else 0,
        "non_ascii_addrs": non_ascii_addr_count,
        "non_ascii_addr_pct": (non_ascii_addr_count / total_rows * 100) if total_rows else 0,
        "name_char_length_stats": compute_stats(name_lengths),
        "name_word_count_stats": compute_stats(name_word_counts),
        "addr_char_length_stats": compute_stats(addr_lengths),
        "addr_word_count_stats": compute_stats(addr_word_counts),
    }
    
    print(f"  Rows: {total_rows:,} | Unique IDs: {len(seen_ids):,} | Dup IDs: {duplicate_ids}")
    print(f"  Countries: {dict(countries)}")
    print(f"  Null counts: {null_counts}")
    print(f"  Literal nulls: {literal_null_counts}")
    print(f"  Non-ASCII: Names={non_ascii_name_count} ({results['non_ascii_name_pct']:.2f}%), Addrs={non_ascii_addr_count} ({results['non_ascii_addr_pct']:.2f}%)")
    print(f"  Name len stats: min={results['name_char_length_stats'].get('min')}, median={results['name_char_length_stats'].get('median')}, max={results['name_char_length_stats'].get('max')}")
    print(f"  Addr len stats: min={results['addr_char_length_stats'].get('min')}, median={results['addr_char_length_stats'].get('median')}, max={results['addr_char_length_stats'].get('max')}")
    print()
    return results, seen_ids


def audit_ground_truth(gt_path, train_s1_ids, train_s2_ids, train_s3_ids):
    print(f"=== Auditing Ground Truth: {gt_path} ===")
    size_bytes = os.path.getsize(gt_path)
    
    total_rows = 0
    malformed_rows = 0
    duplicate_s1 = 0
    seen_s1 = set()
    
    match_counts = []
    s2_match_counts = []
    s3_match_counts = []
    
    # Check multi-mapping: does an S2 or S3 entity map to multiple S1 entities?
    s2_to_s1 = defaultdict(set)
    s3_to_s1 = defaultdict(set)
    
    matched_s2_all = set()
    matched_s3_all = set()
    
    invalid_ids = []
    intra_list_duplicates = 0
    self_matches = 0
    
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        header_cols = [c.strip() for c in header.split("\t")]
        
        for line_num, line in enumerate(f, start=2):
            total_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                malformed_rows += 1
                if malformed_rows <= 3:
                    print(f"  [Malformed line {line_num}] split len = {len(parts)}: {line[:80]!r}")
                continue
            
            s1_id, rest = parts
            if s1_id in seen_s1:
                duplicate_s1 += 1
            seen_s1.add(s1_id)
            
            matched = rest.split(",") if rest.strip() else []
            match_counts.append(len(matched))
            
            if len(matched) != len(set(matched)):
                intra_list_duplicates += 1
                
            n_s2 = 0
            n_s3 = 0
            for m in matched:
                m = m.strip()
                if not m:
                    continue
                if m == s1_id:
                    self_matches += 1
                if m.startswith("S2-"):
                    n_s2 += 1
                    s2_to_s1[m].add(s1_id)
                    matched_s2_all.add(m)
                    if train_s2_ids and m not in train_s2_ids:
                        invalid_ids.append(m)
                elif m.startswith("S3-"):
                    n_s3 += 1
                    s3_to_s1[m].add(s1_id)
                    matched_s3_all.add(m)
                    if train_s3_ids and m not in train_s3_ids:
                        invalid_ids.append(m)
                else:
                    invalid_ids.append(m)
            s2_match_counts.append(n_s2)
            s3_match_counts.append(n_s3)
            
    match_arr = np.array(match_counts, dtype=np.int32)
    s2_arr = np.array(s2_match_counts, dtype=np.int32)
    s3_arr = np.array(s3_match_counts, dtype=np.int32)
    
    # Check if any S2 or S3 entity maps to multiple S1
    s2_multi_mapped = {k: v for k, v in s2_to_s1.items() if len(v) > 1}
    s3_multi_mapped = {k: v for k, v in s3_to_s1.items() if len(v) > 1}
    
    # Check coverage of S1
    missing_s1_in_gt = train_s1_ids - seen_s1 if train_s1_ids else set()
    extra_s1_in_gt = seen_s1 - train_s1_ids if train_s1_ids else set()
    
    # Check coverage of S2 and S3
    unmatched_s2 = train_s2_ids - matched_s2_all if train_s2_ids else set()
    unmatched_s3 = train_s3_ids - matched_s3_all if train_s3_ids else set()
    
    # Singletons
    singletons = int(np.sum(match_arr == 0))
    one_match = int(np.sum(match_arr == 1))
    two_matches = int(np.sum(match_arr == 2))
    three_or_more = int(np.sum(match_arr >= 3))
    
    results = {
        "file": gt_path,
        "size_bytes": size_bytes,
        "total_rows": total_rows,
        "unique_s1_ids": len(seen_s1),
        "duplicate_s1": duplicate_s1,
        "missing_s1_count": len(missing_s1_in_gt),
        "extra_s1_count": len(extra_s1_in_gt),
        "intra_list_duplicates": intra_list_duplicates,
        "self_matches": self_matches,
        "invalid_id_count": len(invalid_ids),
        "total_matches_across_all_s1": int(np.sum(match_arr)),
        "total_s2_matches": int(np.sum(s2_arr)),
        "total_s3_matches": int(np.sum(s3_arr)),
        "unique_matched_s2": len(matched_s2_all),
        "unique_matched_s3": len(matched_s3_all),
        "unmatched_s2_count": len(unmatched_s2),
        "unmatched_s3_count": len(unmatched_s3),
        "s2_coverage_pct": (len(matched_s2_all) / len(train_s2_ids) * 100) if train_s2_ids else 0,
        "s3_coverage_pct": (len(matched_s3_all) / len(train_s3_ids) * 100) if train_s3_ids else 0,
        "s2_multi_mapped_count": len(s2_multi_mapped),
        "s3_multi_mapped_count": len(s3_multi_mapped),
        "match_count_distribution": {
            "0_singletons": singletons,
            "0_pct": float(singletons / total_rows * 100) if total_rows else 0,
            "1_match": one_match,
            "1_pct": float(one_match / total_rows * 100) if total_rows else 0,
            "2_matches": two_matches,
            "2_pct": float(two_matches / total_rows * 100) if total_rows else 0,
            "3_or_more": three_or_more,
            "3_or_more_pct": float(three_or_more / total_rows * 100) if total_rows else 0,
            "min": int(np.min(match_arr)),
            "max": int(np.max(match_arr)),
            "mean": float(np.mean(match_arr)),
            "std": float(np.std(match_arr)),
            "median": float(np.median(match_arr)),
            "p90": float(np.percentile(match_arr, 90)),
            "p95": float(np.percentile(match_arr, 95)),
            "p99": float(np.percentile(match_arr, 99)),
        }
    }
    
    print(f"  Total S1 rows: {total_rows:,} | Dup S1: {duplicate_s1}")
    print(f"  Singletons (0 matches): {singletons:,} ({results['match_count_distribution']['0_pct']:.2f}%)")
    print(f"  1 match: {one_match:,} ({results['match_count_distribution']['1_pct']:.2f}%)")
    print(f"  2 matches: {two_matches:,} ({results['match_count_distribution']['2_pct']:.2f}%)")
    print(f"  3+ matches: {three_or_more:,} ({results['match_count_distribution']['3_or_more_pct']:.2f}%)")
    print(f"  Max matches for a single S1: {int(np.max(match_arr))}")
    print(f"  Total links: {int(np.sum(match_arr)):,} (S2: {int(np.sum(s2_arr)):,}, S3: {int(np.sum(s3_arr)):,})")
    print(f"  Unique S2 matched: {len(matched_s2_all):,} / {len(train_s2_ids):,} ({results['s2_coverage_pct']:.2f}%) | Noise S2: {len(unmatched_s2):,}")
    print(f"  Unique S3 matched: {len(matched_s3_all):,} / {len(train_s3_ids):,} ({results['s3_coverage_pct']:.2f}%) | Noise S3: {len(unmatched_s3):,}")
    print(f"  Multi-mapped S2 to multiple S1: {len(s2_multi_mapped)}")
    print(f"  Multi-mapped S3 to multiple S1: {len(s3_multi_mapped)}")
    if s2_multi_mapped:
        sample_k = next(iter(s2_multi_mapped))
        print(f"    Example multi-mapped S2 {sample_k} -> S1 IDs: {s2_multi_mapped[sample_k]}")
    if s3_multi_mapped:
        sample_k = next(iter(s3_multi_mapped))
        print(f"    Example multi-mapped S3 {sample_k} -> S1 IDs: {s3_multi_mapped[sample_k]}")
    print()
    return results

def main():
    os.makedirs("artifacts", exist_ok=True)
    all_audit = {}
    
    # Audit train sources
    t_s1_res, t_s1_ids = audit_source_file("train_source1", FILES["train_s1"])
    t_s2_res, t_s2_ids = audit_source_file("train_source2", FILES["train_s2"])
    t_s3_res, t_s3_ids = audit_source_file("train_source3", FILES["train_s3"])
    all_audit["train_source1"] = t_s1_res
    all_audit["train_source2"] = t_s2_res
    all_audit["train_source3"] = t_s3_res
    
    # Audit train ground truth
    gt_res = audit_ground_truth(FILES["train_gt"], t_s1_ids, t_s2_ids, t_s3_ids)
    all_audit["train_ground_truth"] = gt_res
    
    # Audit test sources
    test_s1_res, _ = audit_source_file("test_source1", FILES["test_s1"])
    test_s2_res, _ = audit_source_file("test_source2", FILES["test_s2"])
    test_s3_res, _ = audit_source_file("test_source3", FILES["test_s3"])
    all_audit["test_source1"] = test_s1_res
    all_audit["test_source2"] = test_s2_res
    all_audit["test_source3"] = test_s3_res
    
    with open("artifacts/audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_audit, f, indent=2)
    print("Audit finished successfully. Results saved to artifacts/audit_summary.json")

if __name__ == "__main__":
    main()
