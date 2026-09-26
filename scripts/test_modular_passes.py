import polars as pl
import time
import math
import re
import psutil
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

print(f"Initial Memory: {get_mem():.1f} MB")
t_start = time.time()

# 1. Load S1 Sample (5,000 US entities)
s1 = pl.read_parquet(
    'dataset/normalized/train_source1.parquet',
    columns=['entity_id', 'country', 'raw_name', 'norm_name', 'norm_name_clean', 'norm_address']
).filter(pl.col('country') == 'US').head(5000)

s1_ids = s1['entity_id'].to_list()
print(f"Loaded S1 sample: {len(s1)} entities, mem: {get_mem():.1f} MB")

# 2. Load Ground Truth for this sample
gt = pl.read_parquet(
    'dataset/normalized/train_ground_truth_pairs.parquet'
).filter(pl.col('source1_entity_id').is_in(s1_ids))
n_gt = len(gt)
print(f"Ground Truth pairs for sample: {n_gt:,}")

# 3. Load Candidate Pool (S2 + S3 for US)
pool = pl.concat([
    pl.read_parquet('dataset/normalized/train_source2.parquet', columns=['entity_id', 'country', 'raw_name', 'norm_name', 'norm_name_clean', 'norm_address']).filter(pl.col('country') == 'US'),
    pl.read_parquet('dataset/normalized/train_source3.parquet', columns=['entity_id', 'country', 'raw_name', 'norm_name', 'norm_name_clean', 'norm_address']).filter(pl.col('country') == 'US')
])
n_pool = len(pool)
df_thresh = int(0.02 * n_pool)
print(f"Loaded candidate pool: {n_pool:,} records, DF threshold (2%): {df_thresh:,}, mem: {get_mem():.1f} MB")

# =========================================================================
# PASS 1: NM (Core Name Tokens)
# =========================================================================
print("\n--- Running Pass 1: NM ---")
t0 = time.time()
# Candidate NM tokens
pool_nm = (
    pool.select(['entity_id', 'norm_name_clean'])
    .with_columns(pl.col('norm_name_clean').str.split(' '))
    .explode('norm_name_clean')
    .rename({'norm_name_clean': 'token'})
    .filter(pl.col('token').str.len_bytes() >= 2)
    .select([pl.col('entity_id').alias('cand_id'), pl.concat_str([pl.lit('NM_'), pl.col('token')]).alias('token_key')])
    .unique()
)
# Compute DF and IDF
nm_df = pool_nm.group_by('token_key').len().rename({'len': 'df'})
retained_nm = nm_df.filter(pl.col('df') <= df_thresh).with_columns(
    (math.log(n_pool) - pl.col('df').log()).alias('idf')
)
pool_nm = pool_nm.join(retained_nm.select(['token_key', 'idf']), on='token_key', how='inner')

# S1 NM tokens
s1_nm = (
    s1.select(['entity_id', 'norm_name_clean'])
    .with_columns(pl.col('norm_name_clean').str.split(' '))
    .explode('norm_name_clean')
    .rename({'norm_name_clean': 'token'})
    .filter(pl.col('token').str.len_bytes() >= 2)
    .select([pl.col('entity_id').alias('s1_id'), pl.concat_str([pl.lit('NM_'), pl.col('token')]).alias('token_key')])
    .unique()
)

# Join S1 and Pool on token_key
pairs_nm = (
    s1_nm.join(pool_nm, on='token_key', how='inner')
    .group_by(['s1_id', 'cand_id'])
    .agg([
        pl.col('idf').sum().alias('score_nm'),
        pl.lit('NM').alias('pass_name')
    ])
)
del pool_nm, s1_nm, nm_df, retained_nm
print(f"Pass 1 (NM) completed in {time.time()-t0:.2f}s, candidate pairs: {len(pairs_nm):,}, mem: {get_mem():.1f} MB")

# Measure NM recall
rec_nm = gt.join(pairs_nm, left_on=['source1_entity_id', 'matched_id'], right_on=['s1_id', 'cand_id'], how='inner')
print(f"NM Standalone Recall: {len(rec_nm):,} / {n_gt:,} ({len(rec_nm)/n_gt*100:.2f}%)")

# =========================================================================
# PASS 2: NMALT (DBA / Alternate Trade-Name Tokens)
# =========================================================================
print("\n--- Running Pass 2: NMALT ---")
t0 = time.time()
# Extract DBA prefix from candidate pool raw_name
pool_dba = (
    pool.select(['entity_id', 'raw_name'])
    .with_columns(
        pl.col('raw_name').str.extract(r'(?i)^(.*?)\s*[:\s]\b(?:dba|d/b/a|d\.b\.a\.|t/a|trading as|fka|f/k/a|aka|a/k/a)\b', 1).alias('dba_name')
    )
    .filter(pl.col('dba_name').is_not_null())
    .with_columns(pl.col('dba_name').str.to_lowercase().str.replace_all(r'[^\w\s]', ' ').str.split(' '))
    .explode('dba_name')
    .rename({'dba_name': 'token'})
    .filter(pl.col('token').str.len_bytes() >= 2)
    .select([pl.col('entity_id').alias('cand_id'), pl.concat_str([pl.lit('NMALT_'), pl.col('token')]).alias('token_key')])
    .unique()
)
if len(pool_dba) > 0:
    dba_df = pool_dba.group_by('token_key').len().rename({'len': 'df'})
    retained_dba = dba_df.filter(pl.col('df') <= df_thresh).with_columns(
        (math.log(n_pool) - pl.col('df').log()).alias('idf')
    )
    pool_dba = pool_dba.join(retained_dba.select(['token_key', 'idf']), on='token_key', how='inner')

    # Query: S1 emits NMALT tokens from its name
    s1_nmalt = (
        s1.select(['entity_id', 'norm_name_clean'])
        .with_columns(pl.col('norm_name_clean').str.split(' '))
        .explode('norm_name_clean')
        .rename({'norm_name_clean': 'token'})
        .filter(pl.col('token').str.len_bytes() >= 2)
        .select([pl.col('entity_id').alias('s1_id'), pl.concat_str([pl.lit('NMALT_'), pl.col('token')]).alias('token_key')])
        .unique()
    )

    pairs_nmalt = (
        s1_nmalt.join(pool_dba, on='token_key', how='inner')
        .group_by(['s1_id', 'cand_id'])
        .agg([
            pl.col('idf').sum().alias('score_nmalt'),
            pl.lit('NMALT').alias('pass_name')
        ])
    )
    del pool_dba, s1_nmalt, dba_df, retained_dba
else:
    pairs_nmalt = pl.DataFrame(schema={'s1_id': pl.String, 'cand_id': pl.String, 'score_nmalt': pl.Float64, 'pass_name': pl.String})

print(f"Pass 2 (NMALT) completed in {time.time()-t0:.2f}s, candidate pairs: {len(pairs_nmalt):,}, mem: {get_mem():.1f} MB")

# =========================================================================
# PASS 3: AD (Address Word Tokens)
# =========================================================================
print("\n--- Running Pass 3: AD ---")
t0 = time.time()
pool_ad = (
    pool.select(['entity_id', 'norm_address'])
    .with_columns(pl.col('norm_address').str.split(' '))
    .explode('norm_address')
    .rename({'norm_address': 'token'})
    .filter(pl.col('token').str.contains(r'^[a-z]{3,}$'))
    .select([pl.col('entity_id').alias('cand_id'), pl.concat_str([pl.lit('AD_'), pl.col('token')]).alias('token_key')])
    .unique()
)
ad_df = pool_ad.group_by('token_key').len().rename({'len': 'df'})
retained_ad = ad_df.filter(pl.col('df') <= df_thresh).with_columns(
    (math.log(n_pool) - pl.col('df').log()).alias('idf')
)
pool_ad = pool_ad.join(retained_ad.select(['token_key', 'idf']), on='token_key', how='inner')

s1_ad = (
    s1.select(['entity_id', 'norm_address'])
    .with_columns(pl.col('norm_address').str.split(' '))
    .explode('norm_address')
    .rename({'norm_address': 'token'})
    .filter(pl.col('token').str.contains(r'^[a-z]{3,}$'))
    .select([pl.col('entity_id').alias('s1_id'), pl.concat_str([pl.lit('AD_'), pl.col('token')]).alias('token_key')])
    .unique()
)

pairs_ad = (
    s1_ad.join(pool_ad, on='token_key', how='inner')
    .group_by(['s1_id', 'cand_id'])
    .agg([
        pl.col('idf').sum().alias('score_ad'),
        pl.lit('AD').alias('pass_name')
    ])
)
del pool_ad, s1_ad, ad_df, retained_ad
print(f"Pass 3 (AD) completed in {time.time()-t0:.2f}s, candidate pairs: {len(pairs_ad):,}, mem: {get_mem():.1f} MB")

# =========================================================================
# PASS 4: DG (Address Digit Tokens)
# =========================================================================
print("\n--- Running Pass 4: DG ---")
t0 = time.time()
pool_dg = (
    pool.select(['entity_id', 'norm_address'])
    .with_columns(pl.col('norm_address').str.extract_all(r'\b\d+\b').alias('token'))
    .explode('token')
    .filter(pl.col('token').is_not_null())
    .select([pl.col('entity_id').alias('cand_id'), pl.concat_str([pl.lit('DG_'), pl.col('token')]).alias('token_key')])
    .unique()
)
dg_df = pool_dg.group_by('token_key').len().rename({'len': 'df'})
retained_dg = dg_df.filter(pl.col('df') <= df_thresh).with_columns(
    (math.log(n_pool) - pl.col('df').log()).alias('idf')
)
pool_dg = pool_dg.join(retained_dg.select(['token_key', 'idf']), on='token_key', how='inner')

s1_dg = (
    s1.select(['entity_id', 'norm_address'])
    .with_columns(pl.col('norm_address').str.extract_all(r'\b\d+\b').alias('token'))
    .explode('token')
    .filter(pl.col('token').is_not_null())
    .select([pl.col('entity_id').alias('s1_id'), pl.concat_str([pl.lit('DG_'), pl.col('token')]).alias('token_key')])
    .unique()
)

pairs_dg = (
    s1_dg.join(pool_dg, on='token_key', how='inner')
    .group_by(['s1_id', 'cand_id'])
    .agg([
        pl.col('idf').sum().alias('score_dg'),
        pl.lit('DG').alias('pass_name')
    ])
)
del pool_dg, s1_dg, dg_df, retained_dg
print(f"Pass 4 (DG) completed in {time.time()-t0:.2f}s, candidate pairs: {len(pairs_dg):,}, mem: {get_mem():.1f} MB")

# =========================================================================
# PASS 5: PX (Prefix4 of name_nospace)
# =========================================================================
print("\n--- Running Pass 5: PX ---")
t0 = time.time()
pool_px = (
    pool.select(['entity_id', 'norm_name_clean'])
    .with_columns(pl.col('norm_name_clean').str.replace_all(r'[\s_]+', '').str.slice(0, 4).alias('token'))
    .filter(pl.col('token').str.len_bytes() >= 3)
    .select([pl.col('entity_id').alias('cand_id'), pl.concat_str([pl.lit('PX_'), pl.col('token')]).alias('token_key')])
    .unique()
)
px_df = pool_px.group_by('token_key').len().rename({'len': 'df'})
retained_px = px_df.filter(pl.col('df') <= df_thresh).with_columns(
    (math.log(n_pool) - pl.col('df').log()).alias('idf')
)
pool_px = pool_px.join(retained_px.select(['token_key', 'idf']), on='token_key', how='inner')

s1_px = (
    s1.select(['entity_id', 'norm_name_clean'])
    .with_columns(pl.col('norm_name_clean').str.replace_all(r'[\s_]+', '').str.slice(0, 4).alias('token'))
    .filter(pl.col('token').str.len_bytes() >= 3)
    .select([pl.col('entity_id').alias('s1_id'), pl.concat_str([pl.lit('PX_'), pl.col('token')]).alias('token_key')])
    .unique()
)

pairs_px = (
    s1_px.join(pool_px, on='token_key', how='inner')
    .group_by(['s1_id', 'cand_id'])
    .agg([
        pl.col('idf').sum().alias('score_px'),
        pl.lit('PX').alias('pass_name')
    ])
)
del pool_px, s1_px, px_df, retained_px
print(f"Pass 5 (PX) completed in {time.time()-t0:.2f}s, candidate pairs: {len(pairs_px):,}, mem: {get_mem():.1f} MB")

# =========================================================================
# MULTI-PASS AGGREGATION & EXPERIMENTS
# =========================================================================
print("\n--- Multi-Pass Aggregation & Incremental Analysis ---")
# Standardize columns to (s1_id, cand_id, score, pass_name)
p1 = pairs_nm.rename({'score_nm': 'score'})
p2 = pairs_nmalt.rename({'score_nmalt': 'score'})
p3 = pairs_ad.rename({'score_ad': 'score'})
p4 = pairs_dg.rename({'score_dg': 'score'})
p5 = pairs_px.rename({'score_px': 'score'})

# Cumulative configurations:
# Config 1: NM
c1 = p1.group_by(['s1_id', 'cand_id']).agg([pl.col('score').sum().alias('blocking_score'), pl.len().alias('n_passes')])
rec_c1 = gt.join(c1, left_on=['source1_entity_id', 'matched_id'], right_on=['s1_id', 'cand_id'], how='inner')
print(f"Config 1 (NM):            Recall = {len(rec_c1):,} / {n_gt:,} ({len(rec_c1)/n_gt*100:.2f}%), Pairs = {len(c1):,}")

# Config 2: NM + NMALT
c2 = pl.concat([p1, p2]).group_by(['s1_id', 'cand_id']).agg([pl.col('score').sum().alias('blocking_score'), pl.len().alias('n_passes')])
rec_c2 = gt.join(c2, left_on=['source1_entity_id', 'matched_id'], right_on=['s1_id', 'cand_id'], how='inner')
print(f"Config 2 (NM + NMALT):    Recall = {len(rec_c2):,} / {n_gt:,} ({len(rec_c2)/n_gt*100:.2f}%), Pairs = {len(c2):,}")

# Config 3: + AD
c3 = pl.concat([p1, p2, p3]).group_by(['s1_id', 'cand_id']).agg([pl.col('score').sum().alias('blocking_score'), pl.len().alias('n_passes')])
rec_c3 = gt.join(c3, left_on=['source1_entity_id', 'matched_id'], right_on=['s1_id', 'cand_id'], how='inner')
print(f"Config 3 (+ AD):          Recall = {len(rec_c3):,} / {n_gt:,} ({len(rec_c3)/n_gt*100:.2f}%), Pairs = {len(c3):,}")

# Config 4: + DG
c4 = pl.concat([p1, p2, p3, p4]).group_by(['s1_id', 'cand_id']).agg([pl.col('score').sum().alias('blocking_score'), pl.len().alias('n_passes')])
rec_c4 = gt.join(c4, left_on=['source1_entity_id', 'matched_id'], right_on=['s1_id', 'cand_id'], how='inner')
print(f"Config 4 (+ DG):          Recall = {len(rec_c4):,} / {n_gt:,} ({len(rec_c4)/n_gt*100:.2f}%), Pairs = {len(c4):,}")

# Config 5: + PX (ALL PASSES)
c5 = pl.concat([p1, p2, p3, p4, p5]).group_by(['s1_id', 'cand_id']).agg([pl.col('score').sum().alias('blocking_score'), pl.len().alias('n_passes')])
rec_c5 = gt.join(c5, left_on=['source1_entity_id', 'matched_id'], right_on=['s1_id', 'cand_id'], how='inner')
print(f"Config 5 (+ PX / ALL):    Recall = {len(rec_c5):,} / {n_gt:,} ({len(rec_c5)/n_gt*100:.4f}%), Pairs = {len(c5):,}")

# Top-K Evaluation on Config 5
print("\n--- Top-K Evaluation on Config 5 ---")
# Rank candidates within each s1_id by n_passes DESC, blocking_score DESC
ranked = c5.sort(['s1_id', 'n_passes', 'blocking_score'], descending=[False, True, True])

for k in [10, 25, 50, 100]:
    top_k = ranked.group_by('s1_id').head(k)
    rec_k = gt.join(top_k, left_on=['source1_entity_id', 'matched_id'], right_on=['s1_id', 'cand_id'], how='inner')
    
    # Also evaluate UNION with n_passes >= 2
    union_p2 = ranked.filter((pl.int_range(0, pl.len()).over('s1_id') < k) | (pl.col('n_passes') >= 2))
    rec_u = gt.join(union_p2, left_on=['source1_entity_id', 'matched_id'], right_on=['s1_id', 'cand_id'], how='inner')
    
    # S1 recall (entity recall): fraction of S1s that have >= 1 true candidate recalled
    gt_s1_total = gt['source1_entity_id'].n_unique()
    gt_s1_recalled = rec_k['source1_entity_id'].n_unique()
    
    cand_per_s1 = top_k.group_by('s1_id').len()['len']
    print(f"Top-{k:<3}: Pair Recall = {len(rec_k):,}/{n_gt:,} ({len(rec_k)/n_gt*100:.2f}%), S1 Recall = {gt_s1_recalled}/{gt_s1_total} ({gt_s1_recalled/gt_s1_total*100:.2f}%), Cand/S1 Mean = {cand_per_s1.mean():.1f}, Max = {cand_per_s1.max()}")
    print(f"       + UNION(n_passes>=2): Pair Recall = {len(rec_u):,}/{n_gt:,} ({len(rec_u)/n_gt*100:.2f}%), Total Pairs = {len(union_p2):,}")

print(f"\nTotal Elapsed Time: {time.time()-t_start:.2f}s, Peak Mem: {get_mem():.1f} MB")
