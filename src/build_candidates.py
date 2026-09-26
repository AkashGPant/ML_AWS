"""
Phase 1: Build Candidate Pairs (simple deterministic blocking)

Streams train_source1 row groups against candidate row groups to bounded RAM.
"""

import sys
import os
import time
import gc
import pyarrow.parquet as pq
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
# removed

from blocking import build_keys

DATASET_DIR = 'dataset/normalized'
S1_FILE     = os.path.join(DATASET_DIR, 'train_source1.parquet')
CAND_FILES  = [
    os.path.join(DATASET_DIR, 'train_source2.parquet'),
    os.path.join(DATASET_DIR, 'train_source3.parquet'),
]
OUTPUT_DIR  = 'outputs'
OUT_TSV     = os.path.join(OUTPUT_DIR, 'candidate_pairs_train.tsv')

COLS = ['entity_id', 'country', 'norm_name_clean', 'norm_address', 'postal_code']

def main():
    wall0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    LIMIT_S1 = 10000
    OUT_TSV = os.path.join(OUTPUT_DIR, 'candidate_pairs_train_sample.tsv')

    print(f"Building keys for train_source1 (LIMIT: {LIMIT_S1})...")
    s1_df = pl.read_parquet(S1_FILE, columns=COLS)
    if LIMIT_S1:
        s1_df = s1_df.head(LIMIT_S1)
        
    # We also save the subset of s1_ids we processed for evaluation
    s1_ids_processed = s1_df["entity_id"].unique().to_list()
    with open(os.path.join(OUTPUT_DIR, 's1_sample_ids.txt'), 'w') as f:
        f.write("\n".join(s1_ids_processed))

    # Function to apply frequency limits
    def filter_keys(df_keys, pass_col="pass_name", limit_C=15, limit_B=50, limit_D=500):
        return df_keys.filter(
            (pl.col(pass_col) == "A") | 
            ((pl.col(pass_col) == "B") & (pl.len().over("block_key") <= limit_B)) |
            ((pl.col(pass_col) == "C") & (pl.len().over("block_key") <= limit_C)) |
            ((pl.col(pass_col).is_in(["D", "E"])) & (pl.len().over("block_key") <= limit_D))
        )

    s1_keys = build_keys(s1_df).rename({"entity_id": "s1_id"})
    s1_keys = filter_keys(s1_keys, pass_col="pass_name")
    
    del s1_df
    gc.collect()
    
    print(f"S1 keys built: {len(s1_keys):,} rows")

    total_pairs = 0
    chunks = []
    
    for fpath in CAND_FILES:
        print(f"Processing candidate file: {fpath}")
        cand_pf = pq.ParquetFile(fpath)
        for crg in range(cand_pf.num_row_groups):
            t_rg = time.time()
            cand_tbl = cand_pf.read_row_group(crg, columns=COLS)
            cand_df = pl.from_arrow(cand_tbl)
            
            cand_keys = build_keys(cand_df).rename({"entity_id": "cand_id", "pass_name": "cand_pass"})
            cand_keys = filter_keys(cand_keys, pass_col="cand_pass")
            
            raw_joined = (
                s1_keys.join(cand_keys, on="block_key", how="inner")
                .select(["s1_id", "cand_id", "pass_name"])
                .unique()
            )
            if len(raw_joined) > 0:
                chunks.append(raw_joined)
                
            n_pairs_raw = len(raw_joined)
            
            del cand_tbl, cand_df, cand_keys, raw_joined
            gc.collect()
            
            elapsed = time.time() - t_rg
            print(f"  RG {crg+1}/{cand_pf.num_row_groups} | pairs raw: {n_pairs_raw} | t={elapsed:.1f}s", flush=True)

    print("Aggregating candidates...")
    if chunks:
        all_joined = pl.concat(chunks)
        result = (
            all_joined.group_by(["s1_id", "cand_id"])
            .agg(
                pl.col("pass_name").unique().sort().alias("passes"),
                pl.col("pass_name").n_unique().cast(pl.Int32).alias("n_passes"),
            )
            .rename({"s1_id": "source1_entity_id", "cand_id": "candidate_entity_id"})
        )
        
        n_pairs = len(result)
        if n_pairs > 0:
            result = result.with_columns(
                pl.col("passes").list.join("|").alias("passes")
            )
            result.write_csv(OUT_TSV, separator='\t')
        total_pairs = n_pairs
    else:
        # Create empty
        with open(OUT_TSV, 'w', encoding='utf-8') as out_f:
            out_f.write("source1_entity_id\tcandidate_entity_id\tpasses\tn_passes\n")

    print(f"\nTotal unique pairs: {total_pairs}")

    print(f"\nDone in {time.time()-wall0:.1f}s")
    print(f"Output: {OUT_TSV}")

if __name__ == '__main__':
    main()