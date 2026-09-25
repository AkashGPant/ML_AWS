#!/usr/bin/env python3
"""
scripts/fast_audit.py: DuckDB-powered comprehensive, streaming data audit.
Analyzes train and test files for ML Hackathon 2026 Business Entity Resolution.
"""

import os
import sys
import json
import duckdb

sys.stdout.reconfigure(encoding='utf-8')

FILES = {
    "train_source1": "dataset/train/train_source1.tsv",
    "train_source2": "dataset/train/train_source2.tsv",
    "train_source3": "dataset/train/train_source3.tsv",
    "train_ground_truth": "dataset/train/train_ground_truth.tsv",
    "test_source1": "dataset/test/test_source1.tsv",
    "test_source2": "dataset/test/test_source2.tsv",
    "test_source3": "dataset/test/test_source3.tsv",
}


def main():
    os.makedirs("artifacts", exist_ok=True)
    conn = duckdb.connect()

    print("Initializing DuckDB configuration...")
    conn.execute("SET memory_limit = '4GB';")
    conn.execute("SET threads = 4;")

    audit_results = {}

    # 1. Profile each source file
    for name, path in FILES.items():
        if name == "train_ground_truth":
            continue
        print(f"\n================ Profiling {name} ({path}) ================")
        size_bytes = os.path.getsize(path)

        # Basic counts, nulls, duplicates
        query_stats = f"""
        WITH raw AS (
            SELECT 
                entity_id,
                business_name,
                business_address,
                country,
                length(business_name) as name_len,
                length(business_address) as addr_len,
                array_length(str_split(trim(regexp_replace(business_name, '\\s+', ' ', 'g')), ' ')) as name_words,
                array_length(str_split(trim(regexp_replace(business_address, '\\s+', ' ', 'g')), ' ')) as addr_words,
                CASE WHEN regexp_matches(business_name, '[^\\x00-\\x7F]') THEN 1 ELSE 0 END as non_ascii_name,
                CASE WHEN regexp_matches(business_address, '[^\\x00-\\x7F]') THEN 1 ELSE 0 END as non_ascii_addr,
                CASE WHEN lower(trim(business_name)) IN ('null', 'none', 'nan', 'n/a', 'na', '-', '', 'undefined') THEN 1 ELSE 0 END as lit_null_name,
                CASE WHEN lower(trim(business_address)) IN ('null', 'none', 'nan', 'n/a', 'na', '-', '', 'undefined') THEN 1 ELSE 0 END as lit_null_addr
            FROM read_csv('{path}', delim='\t', header=true, all_varchar=true, quote='', escape='')
        )
        SELECT 
            count(*) as total_rows,
            count(distinct entity_id) as unique_entity_ids,
            count(case when entity_id is null or entity_id = '' then 1 end) as empty_entity_id,
            count(case when business_name is null or business_name = '' then 1 end) as empty_name,
            sum(lit_null_name) as lit_null_name,
            count(case when business_address is null or business_address = '' then 1 end) as empty_addr,
            sum(lit_null_addr) as lit_null_addr,
            count(case when country is null or country = '' then 1 end) as empty_country,
            sum(non_ascii_name) as non_ascii_name_count,
            sum(non_ascii_addr) as non_ascii_addr_count,
            min(name_len) as name_len_min,
            quantile_cont(name_len, 0.25) as name_len_p25,
            quantile_cont(name_len, 0.50) as name_len_median,
            quantile_cont(name_len, 0.75) as name_len_p75,
            quantile_cont(name_len, 0.95) as name_len_p95,
            quantile_cont(name_len, 0.99) as name_len_p99,
            max(name_len) as name_len_max,
            avg(name_len) as name_len_mean,
            stddev(name_len) as name_len_std,
            min(addr_len) as addr_len_min,
            quantile_cont(addr_len, 0.25) as addr_len_p25,
            quantile_cont(addr_len, 0.50) as addr_len_median,
            quantile_cont(addr_len, 0.75) as addr_len_p75,
            quantile_cont(addr_len, 0.95) as addr_len_p95,
            quantile_cont(addr_len, 0.99) as addr_len_p99,
            max(addr_len) as addr_len_max,
            avg(addr_len) as addr_len_mean,
            stddev(addr_len) as addr_len_std,
            min(name_words) as name_words_min,
            quantile_cont(name_words, 0.50) as name_words_median,
            max(name_words) as name_words_max,
            avg(name_words) as name_words_mean,
            min(addr_words) as addr_words_min,
            quantile_cont(addr_words, 0.50) as addr_words_median,
            max(addr_words) as addr_words_max,
            avg(addr_words) as addr_words_mean
        FROM raw
        """
        stats_df = conn.execute(query_stats).df()
        stats_dict = stats_df.to_dict(orient="records")[0]

        # Country distribution
        country_query = f"""
        SELECT country, count(*) as count, round(count(*) * 100.0 / sum(count(*)) over(), 4) as pct
        FROM read_csv('{path}', delim='\t', header=true, all_varchar=true, quote='', escape='')
        GROUP BY country
        ORDER BY count DESC
        """
        country_df = conn.execute(country_query).df()
        countries = country_df.to_dict(orient="records")

        stats_dict["size_bytes"] = size_bytes
        stats_dict["country_distribution"] = countries
        audit_results[name] = stats_dict

        print(f"  Rows: {stats_dict['total_rows']:,} | Unique IDs: {stats_dict['unique_entity_ids']:,}")
        print(f"  Empty fields: ID={stats_dict['empty_entity_id']}, Name={stats_dict['empty_name']}, Addr={stats_dict['empty_addr']}, Country={stats_dict['empty_country']}")
        print(f"  Literal nulls: Name={stats_dict['lit_null_name']}, Addr={stats_dict['lit_null_addr']}")
        print(f"  Non-ASCII: Name={stats_dict['non_ascii_name_count']:,} ({stats_dict['non_ascii_name_count']/stats_dict['total_rows']*100:.2f}%), Addr={stats_dict['non_ascii_addr_count']:,} ({stats_dict['non_ascii_addr_count']/stats_dict['total_rows']*100:.2f}%)")
        print(f"  Countries: {countries}")
        print(f"  Name Char Len: min={stats_dict['name_len_min']}, median={stats_dict['name_len_median']}, max={stats_dict['name_len_max']}, mean={stats_dict['name_len_mean']:.1f}")
        print(f"  Addr Char Len: min={stats_dict['addr_len_min']}, median={stats_dict['addr_len_median']}, max={stats_dict['addr_len_max']}, mean={stats_dict['addr_len_mean']:.1f}")

    # 2. Detailed Ground Truth Audit
    print("\n================ Auditing train_ground_truth ================")
    gt_path = FILES["train_ground_truth"]
    gt_size = os.path.getsize(gt_path)

    # Register ground truth view
    conn.execute(f"""
        CREATE OR REPLACE VIEW gt_raw AS 
        SELECT 
            source1_entity_id, 
            matched_entity_ids,
            CASE WHEN matched_entity_ids IS NULL OR trim(matched_entity_ids) = '' THEN 0
                 ELSE array_length(str_split(trim(matched_entity_ids), ',')) END as match_count
        FROM read_csv('{gt_path}', delim='\t', header=true, all_varchar=true, quote='', escape='')
    """)

    gt_overview = conn.execute("""
        SELECT 
            count(*) as total_rows,
            count(distinct source1_entity_id) as unique_s1_ids,
            count(case when source1_entity_id is null or source1_entity_id = '' then 1 end) as null_s1_id,
            count(case when match_count = 0 then 1 end) as singletons_count,
            round(count(case when match_count = 0 then 1 end) * 100.0 / count(*), 4) as singletons_pct,
            count(case when match_count = 1 then 1 end) as one_match_count,
            round(count(case when match_count = 1 then 1 end) * 100.0 / count(*), 4) as one_match_pct,
            count(case when match_count = 2 then 1 end) as two_matches_count,
            round(count(case when match_count = 2 then 1 end) * 100.0 / count(*), 4) as two_matches_pct,
            count(case when match_count >= 3 then 1 end) as three_plus_matches_count,
            round(count(case when match_count >= 3 then 1 end) * 100.0 / count(*), 4) as three_plus_matches_pct,
            min(match_count) as match_count_min,
            quantile_cont(match_count, 0.50) as match_count_median,
            quantile_cont(match_count, 0.90) as match_count_p90,
            quantile_cont(match_count, 0.95) as match_count_p95,
            quantile_cont(match_count, 0.99) as match_count_p99,
            max(match_count) as match_count_max,
            avg(match_count) as match_count_mean,
            sum(match_count) as total_links
        FROM gt_raw
    """).df().to_dict(orient="records")[0]
    gt_overview["size_bytes"] = gt_size
    print(f"Ground Truth Overview:")
    for k, v in gt_overview.items():
        print(f"  {k}: {v}")

    # Match count histogram
    hist_df = conn.execute("""
        SELECT match_count, count(*) as count, round(count(*) * 100.0 / sum(count(*)) over(), 4) as pct
        FROM gt_raw
        GROUP BY match_count
        ORDER BY match_count
    """).df()
    gt_overview["match_count_histogram"] = hist_df.to_dict(orient="records")

    # Explode matches into individual pairs to test cardinality, S2 vs S3 coverage, and multi-mapping
    print("\nUnnesting ground truth matches for cardinality and coverage checks...")
    conn.execute("""
        CREATE OR REPLACE VIEW unnested_matches AS
        SELECT 
            source1_entity_id,
            unnest(str_split(trim(matched_entity_ids), ',')) as matched_id
        FROM gt_raw
        WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) != ''
    """)

    coverage_check = conn.execute(f"""
        WITH unnested AS (
            SELECT 
                source1_entity_id,
                trim(matched_id) as matched_id,
                case when matched_id like 'S2-%' then 'S2'
                     when matched_id like 'S3-%' then 'S3'
                     else 'OTHER' end as target_src
            FROM unnested_matches
        ),
        s2_ids AS (
            SELECT entity_id FROM read_csv('{FILES["train_source2"]}', delim='\t', header=true, all_varchar=true, quote='', escape='')
        ),
        s3_ids AS (
            SELECT entity_id FROM read_csv('{FILES["train_source3"]}', delim='\t', header=true, all_varchar=true, quote='', escape='')
        ),
        s1_ids AS (
            SELECT entity_id FROM read_csv('{FILES["train_source1"]}', delim='\t', header=true, all_varchar=true, quote='', escape='')
        )
        SELECT
            (SELECT count(*) FROM unnested) as total_pairs,
            (SELECT count(*) FROM unnested WHERE target_src = 'S2') as total_s2_pairs,
            (SELECT count(*) FROM unnested WHERE target_src = 'S3') as total_s3_pairs,
            (SELECT count(*) FROM unnested WHERE target_src = 'OTHER') as total_invalid_pairs,
            (SELECT count(distinct matched_id) FROM unnested WHERE target_src = 'S2') as unique_s2_matched,
            (SELECT count(distinct matched_id) FROM unnested WHERE target_src = 'S3') as unique_s3_matched,
            (SELECT count(*) FROM s2_ids) as total_s2_entities,
            (SELECT count(*) FROM s3_ids) as total_s3_entities,
            (SELECT count(*) FROM s1_ids) as total_s1_entities,
            -- S2 entities not matched to any S1 (noise mass)
            (SELECT count(*) FROM s2_ids WHERE entity_id NOT IN (SELECT matched_id FROM unnested WHERE target_src = 'S2')) as unmatched_s2_count,
            -- S3 entities not matched to any S1 (noise mass)
            (SELECT count(*) FROM s3_ids WHERE entity_id NOT IN (SELECT matched_id FROM unnested WHERE target_src = 'S3')) as unmatched_s3_count,
            -- S1 entities in train_s1 missing from GT
            (SELECT count(*) FROM s1_ids WHERE entity_id NOT IN (SELECT source1_entity_id FROM gt_raw)) as missing_s1_in_gt,
            -- S1 entities in GT not in train_s1
            (SELECT count(*) FROM gt_raw WHERE source1_entity_id NOT IN (SELECT entity_id FROM s1_ids)) as extra_s1_in_gt
    """).df().to_dict(orient="records")[0]

    print("Coverage check results:")
    for k, v in coverage_check.items():
        print(f"  {k}: {v}")

    gt_overview["coverage"] = coverage_check

    # Check if any S2 or S3 maps to MULTIPLE S1 entities (crucial constraint check!)
    multi_map_query = """
        WITH multi AS (
            SELECT 
                matched_id,
                case when matched_id like 'S2-%' then 'S2'
                     when matched_id like 'S3-%' then 'S3'
                     else 'OTHER' end as target_src,
                count(distinct source1_entity_id) as num_s1_mapped,
                list(source1_entity_id) as s1_list
            FROM unnested_matches
            GROUP BY matched_id
            HAVING count(distinct source1_entity_id) > 1
        )
        SELECT target_src, count(*) as count
        FROM multi
        GROUP BY target_src
    """
    multi_df = conn.execute(multi_map_query).df()
    print("Multi-mapping check (S2/S3 mapping to multiple S1):")
    print(multi_df)
    gt_overview["multi_mapped_counts"] = multi_df.to_dict(orient="records")

    # Check for duplicate IDs within same S1 matched list
    intra_dup_query = """
        WITH pair_counts AS (
            SELECT source1_entity_id, matched_id, count(*) as cnt
            FROM unnested_matches
            GROUP BY source1_entity_id, matched_id
            HAVING count(*) > 1
        )
        SELECT count(*) as intra_list_duplicate_pairs FROM pair_counts
    """
    intra_dup = conn.execute(intra_dup_query).df().to_dict(orient="records")[0]
    print(f"Intra-list duplicates: {intra_dup}")
    gt_overview["intra_list_duplicates"] = intra_dup["intra_list_duplicate_pairs"]

    audit_results["train_ground_truth"] = gt_overview

    # Save to JSON
    with open("artifacts/audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2, default=str)

    print("\nAudit completed successfully! Saved to artifacts/audit_summary.json")


if __name__ == "__main__":
    main()
