import polars as pl
from collections import Counter, defaultdict
import time
import math
import re
import psutil
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

print(f"Starting mem: {get_mem():.1f} MB")
t_start = time.time()

# 1. Load S1 Sample: 5,000 US entities
s1 = pl.read_parquet(
    'dataset/normalized/train_source1.parquet',
    columns=['entity_id', 'country', 'raw_name', 'norm_name', 'norm_name_clean', 'norm_address']
).filter(pl.col('country') == 'US').head(5000)

s1_ids = set(s1['entity_id'].to_list())
print(f"Loaded S1 sample: {len(s1)} entities")

# 2. Load GT for sample
gt = pl.read_parquet(
    'dataset/normalized/train_ground_truth_pairs.parquet'
).filter(pl.col('source1_entity_id').is_in(list(s1_ids)))
n_gt = len(gt)
gt_pairs_set = set(zip(gt['source1_entity_id'].to_list(), gt['matched_id'].to_list()))
print(f"GT pairs: {n_gt:,}")

# 3. Load Candidate Pool (S2 + S3 US)
pool = pl.concat([
    pl.read_parquet('dataset/normalized/train_source2.parquet', columns=['entity_id', 'country', 'raw_name', 'norm_name', 'norm_name_clean', 'norm_address']).filter(pl.col('country') == 'US'),
    pl.read_parquet('dataset/normalized/train_source3.parquet', columns=['entity_id', 'country', 'raw_name', 'norm_name', 'norm_name_clean', 'norm_address']).filter(pl.col('country') == 'US')
])
n_pool = len(pool)
print(f"Candidate pool: {n_pool:,} records")

# 4. Extract token counts on candidate pool with max_df threshold (e.g. 5,000)
max_df = 5000
t0 = time.time()
print(f"Counting token frequencies on candidate pool (max_df = {max_df:,})...")

token_counts = Counter()
re_dba = re.compile(r'(?i)\b(?:dba|d/b/a|d\.b\.a\.|t/a|trading as|fka|f/k/a|aka|a/k/a)\b[:\s]*')
re_num = re.compile(r'\b\d+\b')

for r in pool.iter_rows(named=True):
    # NM
    nm = (r['norm_name_clean'] or '').split()
    for w in set(nm):
        if len(w) >= 2: token_counts[f"NM_{w}"] += 1
    
    # NMALT
    raw = r.get('raw_name', '') or ''
    m_dba = re_dba.search(raw)
    if m_dba:
        prefix = raw[:m_dba.start()].strip().lower()
        for w in set(re.findall(r'[a-z0-9]+', prefix)):
            if len(w) >= 2: token_counts[f"NMALT_{w}"] += 1
            
    # AD
    ad = (r['norm_address'] or '').split()
    for w in set(ad):
        if len(w) >= 3 and w.isalpha(): token_counts[f"AD_{w}"] += 1
        
    # DG
    for d in set(re_num.findall(r['norm_address'] or '')):
        token_counts[f"DG_{d}"] += 1
        
    # PX
    px = (r['norm_name_clean'] or '').replace(' ', '')[:4]
    if len(px) >= 3:
        token_counts[f"PX_{px}"] += 1

# Filter retained tokens and compute IDF
token_idf = {}
for k, df in token_counts.items():
    if df <= max_df:
        token_idf[k] = math.log(n_pool / df)

del token_counts
print(f"Retained tokens: {len(token_idf):,} in {time.time()-t0:.2f}s, mem: {get_mem():.1f} MB")

# 5. Build S1 token dictionary
s1_token_dict = defaultdict(list)
for r in s1.iter_rows(named=True):
    s1_id = r['entity_id']
    # NM
    nm = (r['norm_name_clean'] or '').split()
    for w in set(nm):
        if len(w) >= 2:
            k = f"NM_{w}"
            if k in token_idf: s1_token_dict[s1_id].append((k, token_idf[k], 'NM'))
            # S1 also emits NMALT from its name
            k_alt = f"NMALT_{w}"
            if k_alt in token_idf: s1_token_dict[s1_id].append((k_alt, token_idf[k_alt], 'NMALT'))
    # AD
    ad = (r['norm_address'] or '').split()
    for w in set(ad):
        if len(w) >= 3 and w.isalpha():
            k = f"AD_{w}"
            if k in token_idf: s1_token_dict[s1_id].append((k, token_idf[k], 'AD'))
    # DG
    for d in set(re_num.findall(r['norm_address'] or '')):
        k = f"DG_{d}"
        if k in token_idf: s1_token_dict[s1_id].append((k, token_idf[k], 'DG'))
    # PX
    px = (r['norm_name_clean'] or '').replace(' ', '')[:4]
    if len(px) >= 3:
        k = f"PX_{px}"
        if k in token_idf: s1_token_dict[s1_id].append((k, token_idf[k], 'PX'))

# 6. Build index for candidate pool ONLY on tokens needed by S1!
s1_needed_keys = set()
for tok_list in s1_token_dict.values():
    for k, _, _ in tok_list:
        s1_needed_keys.add(k)

print(f"S1 unique needed keys: {len(s1_needed_keys):,}")

# Build candidate index for needed keys
cand_index = defaultdict(list) # key -> list of cand_ids
t0 = time.time()

for r in pool.iter_rows(named=True):
    cid = r['entity_id']
    # NM
    nm = (r['norm_name_clean'] or '').split()
    for w in set(nm):
        k = f"NM_{w}"
        if k in s1_needed_keys: cand_index[k].append(cid)
    # NMALT
    raw = r.get('raw_name', '') or ''
    m_dba = re_dba.search(raw)
    if m_dba:
        prefix = raw[:m_dba.start()].strip().lower()
        for w in set(re.findall(r'[a-z0-9]+', prefix)):
            k = f"NMALT_{w}"
            if k in s1_needed_keys: cand_index[k].append(cid)
    # AD
    ad = (r['norm_address'] or '').split()
    for w in set(ad):
        k = f"AD_{w}"
        if k in s1_needed_keys: cand_index[k].append(cid)
    # DG
    for d in set(re_num.findall(r['norm_address'] or '')):
        k = f"DG_{d}"
        if k in s1_needed_keys: cand_index[k].append(cid)
    # PX
    px = (r['norm_name_clean'] or '').replace(' ', '')[:4]
    k = f"PX_{px}"
    if k in s1_needed_keys: cand_index[k].append(cid)

print(f"Built candidate index in {time.time()-t0:.2f}s, indexed keys: {len(cand_index):,}, mem: {get_mem():.1f} MB")

# 7. For each S1, score candidates
t0 = time.time()
print("Scoring candidates for 5,000 S1 queries...")

# Results per S1: s1_id -> list of (cand_id, n_passes, score)
s1_candidates = {}

for s1_id, tok_list in s1_token_dict.items():
    cand_scores = defaultdict(float)
    cand_passes = defaultdict(set)
    for k, idf, ptype in tok_list:
        if k in cand_index:
            for cid in cand_index[k]:
                cand_scores[cid] += idf
                cand_passes[cid].add(ptype)
    
    # Sort candidates by (n_passes DESC, blocking_score DESC)
    scored_cands = [
        (cid, len(cand_passes[cid]), cand_scores[cid])
        for cid in cand_scores
    ]
    scored_cands.sort(key=lambda x: (x[1], x[2]), reverse=True)
    s1_candidates[s1_id] = scored_cands

print(f"Scored 5,000 S1 queries in {time.time()-t0:.2f}s, mem: {get_mem():.1f} MB")

# 8. Evaluate Top-K and UNION policy
print("\n--- Candidate Selection Policy Evaluation ---")
for k in [10, 25, 50, 100]:
    recalled_pairs_topk = 0
    recalled_pairs_union = 0
    total_candidates_topk = 0
    total_candidates_union = 0
    cands_per_s1_topk = []
    cands_per_s1_union = []

    for s1_id in s1_ids:
        cands = s1_candidates.get(s1_id, [])
        topk = cands[:k]
        topk_cids = set(c[0] for c in topk)
        
        # UNION with n_passes >= 2
        union_cids = set(c[0] for c in cands if c[1] >= 2) | topk_cids
        
        total_candidates_topk += len(topk_cids)
        total_candidates_union += len(union_cids)
        cands_per_s1_topk.append(len(topk_cids))
        cands_per_s1_union.append(len(union_cids))
        
        for cid in topk_cids:
            if (s1_id, cid) in gt_pairs_set:
                recalled_pairs_topk += 1
        for cid in union_cids:
            if (s1_id, cid) in gt_pairs_set:
                recalled_pairs_union += 1

    s_topk = pl.Series(cands_per_s1_topk)
    s_union = pl.Series(cands_per_s1_union)
    
    print(f"\nK = {k}:")
    print(f"  Top-{k} Only:")
    print(f"    Pair Recall:      {recalled_pairs_topk:,} / {n_gt:,} ({recalled_pairs_topk/n_gt*100:.2f}%)")
    print(f"    Candidates/S1:    mean={s_topk.mean():.1f}, median={s_topk.median()}, p95={s_topk.quantile(0.95):.0f}, max={s_topk.max()}")
    print(f"    Total Pairs:      {total_candidates_topk:,}")
    print(f"  Top-{k} + UNION(n_passes >= 2):")
    print(f"    Pair Recall:      {recalled_pairs_union:,} / {n_gt:,} ({recalled_pairs_union/n_gt*100:.2f}%)")
    print(f"    Candidates/S1:    mean={s_union.mean():.1f}, median={s_union.median()}, p95={s_union.quantile(0.95):.0f}, max={s_union.max()}")
    print(f"    Total Pairs:      {total_candidates_union:,}")

print(f"\nTotal elapsed time: {time.time()-t_start:.2f}s, Peak memory: {get_mem():.1f} MB")
