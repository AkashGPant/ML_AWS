import polars as pl
import pyarrow.parquet as pq
from collections import Counter
import time
import math
import psutil
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

print(f"Starting mem: {get_mem():.1f} MB")
t_start = time.time()

# 1. Step 1: Compute candidate pool token DF for US on a sample of candidate row groups
print("\n--- Step 1: Computing DF on Candidate Pool (US) ---")
cand_files = [
    'dataset/normalized/train_source2.parquet',
    'dataset/normalized/train_source3.parquet'
]

# We will count DF across candidate pool
counter = Counter()
n_pool = 0
t0 = time.time()

# Process row groups
for fpath in cand_files:
    pf = pq.ParquetFile(fpath)
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=['entity_id', 'country', 'norm_name_clean'])
        df = pl.from_arrow(tbl).filter(pl.col('country') == 'US')
        n_pool += len(df)
        tokens = (
            df.with_columns(pl.col('norm_name_clean').str.split(' '))
            .explode('norm_name_clean')
            .rename({'norm_name_clean': 'token'})
            .filter(pl.col('token').str.len_bytes() >= 2)
            .select(['token'])
            .unique()
        )
        counter.update(tokens['token'].to_list())

print(f"Candidate pool US records: {n_pool:,}, unique NM tokens: {len(counter):,} in {time.time()-t0:.2f}s, mem: {get_mem():.1f} MB")

# DF threshold: 1% or 2% (let's test 1% = 0.01 * n_pool)
df_thresh = int(0.01 * n_pool)
retained_tokens = {t: math.log(n_pool / cnt) for t, cnt in counter.items() if cnt <= df_thresh}
del counter
print(f"Retained tokens (df <= {df_thresh:,}): {len(retained_tokens):,}, mem: {get_mem():.1f} MB")

# Convert retained_tokens to a Polars IDF DataFrame
idf_df = pl.DataFrame({
    'token_key': [f"NM_{t}" for t in retained_tokens.keys()],
    'idf': list(retained_tokens.values())
})

# 2. Step 2: Build candidate token table (only retained tokens!) for candidate pool
print("\n--- Step 2: Building Candidate Token Table (retained only) ---")
t0 = time.time()
cand_token_chunks = []

for fpath in cand_files:
    pf = pq.ParquetFile(fpath)
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=['entity_id', 'country', 'norm_name_clean'])
        df = pl.from_arrow(tbl).filter(pl.col('country') == 'US')
        toks = (
            df.with_columns(pl.col('norm_name_clean').str.split(' '))
            .explode('norm_name_clean')
            .rename({'norm_name_clean': 'token'})
            .filter(pl.col('token').str.len_bytes() >= 2)
            .select([
                pl.col('entity_id').alias('cand_id'),
                pl.concat_str([pl.lit('NM_'), pl.col('token')]).alias('token_key')
            ])
            .unique()
            .join(idf_df, on='token_key', how='inner')
        )
        if len(toks) > 0:
            cand_token_chunks.append(toks)

cand_tokens = pl.concat(cand_token_chunks)
del cand_token_chunks
print(f"Built candidate tokens: {len(cand_tokens):,} rows in {time.time()-t0:.2f}s, mem: {get_mem():.1f} MB")

# 3. Step 3: Query S1 in a 10,000 entity chunk using Vectorized Hash Join!
print("\n--- Step 3: Querying 10,000 S1 Entities via Vectorized Hash Join ---")
s1 = pl.read_parquet('dataset/normalized/train_source1.parquet', columns=['entity_id', 'country', 'norm_name_clean']).filter(pl.col('country') == 'US').head(10000)

s1_tokens = (
    s1.with_columns(pl.col('norm_name_clean').str.split(' '))
    .explode('norm_name_clean')
    .rename({'norm_name_clean': 'token'})
    .filter(pl.col('token').str.len_bytes() >= 2)
    .select([
        pl.col('entity_id').alias('s1_id'),
        pl.concat_str([pl.lit('NM_'), pl.col('token')]).alias('token_key'),
        pl.lit('NM').alias('token_type')
    ])
    .unique()
    .join(idf_df, on='token_key', how='inner')
)
print(f"S1 tokens: {len(s1_tokens):,}, mem: {get_mem():.1f} MB")

# JOIN S1 tokens with Candidate tokens
t0 = time.time()
joined = s1_tokens.join(cand_tokens, on='token_key', how='inner')
print(f"Vectorized Join: {len(joined):,} matched token pairs in {time.time()-t0:.2f}s, mem: {get_mem():.1f} MB")

# AGGREGATE by (s1_id, cand_id)
t0 = time.time()
cand_pairs = (
    joined.group_by(['s1_id', 'cand_id'])
    .agg([
        pl.col('idf').sum().alias('blocking_score'),
        pl.col('token_type').n_unique().alias('n_passes'),
        pl.len().alias('n_tokens')
    ])
)
print(f"Vectorized Aggregation: {len(cand_pairs):,} unique candidate pairs in {time.time()-t0:.2f}s, mem: {get_mem():.1f} MB")

# Top-K ranking
ranked = cand_pairs.sort(['s1_id', 'n_passes', 'blocking_score'], descending=[False, True, True])
top_25 = ranked.group_by('s1_id').head(25)

# 4. Step 4: Evaluate Recall against Ground Truth
gt = pl.read_parquet('dataset/normalized/train_ground_truth_pairs.parquet').filter(pl.col('source1_entity_id').is_in(s1['entity_id'].to_list()))
n_gt = len(gt)

rec_all = gt.join(cand_pairs, left_on=['source1_entity_id', 'matched_id'], right_on=['s1_id', 'cand_id'], how='inner')
rec_top25 = gt.join(top_25, left_on=['source1_entity_id', 'matched_id'], right_on=['s1_id', 'cand_id'], how='inner')

print(f"\nGround Truth Pairs: {n_gt:,}")
print(f"All Candidates Recall: {len(rec_all):,} / {n_gt:,} ({len(rec_all)/n_gt*100:.2f}%)")
print(f"Top-25 Recall:         {len(rec_top25):,} / {n_gt:,} ({len(rec_top25)/n_gt*100:.2f}%)")
print(f"\nTotal elapsed time: {time.time()-t_start:.2f}s, peak memory: {get_mem():.1f} MB")
