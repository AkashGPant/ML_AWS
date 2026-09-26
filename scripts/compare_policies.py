import polars as pl
from collections import defaultdict
import time, math, re, sys

sys.stdout.reconfigure(encoding='utf-8')

# Load 1000 S1 sample and GT
s1 = pl.read_parquet(
    'dataset/normalized/train_source1.parquet',
    columns=['entity_id', 'country', 'raw_name', 'norm_name', 'norm_name_clean', 'norm_address']
).filter(pl.col('country') == 'US').head(1000)

s1_ids = set(s1['entity_id'].to_list())
gt = pl.read_parquet(
    'dataset/normalized/train_ground_truth_pairs.parquet'
).filter(pl.col('source1_entity_id').is_in(list(s1_ids)))
n_gt = len(gt)
gt_pairs = set(zip(gt['source1_entity_id'].to_list(), gt['matched_id'].to_list()))
print(f"Sample S1: {len(s1)}, GT pairs: {n_gt:,}")

pool = pl.concat([
    pl.read_parquet('dataset/normalized/train_source2.parquet', columns=['entity_id', 'country', 'raw_name', 'norm_name', 'norm_name_clean', 'norm_address']).filter(pl.col('country') == 'US'),
    pl.read_parquet('dataset/normalized/train_source3.parquet', columns=['entity_id', 'country', 'raw_name', 'norm_name', 'norm_name_clean', 'norm_address']).filter(pl.col('country') == 'US')
])
n_pool = len(pool)

# DF threshold: 2% of pool = 123,737
# But let's check max_df: 5,000 vs 10,000 vs 20,000
re_num = re.compile(r'\b\d+\b')
re_dba = re.compile(r'(?i)\b(?:dba|d/b/a|d\.b\.a\.|t/a|trading as|fka|f/k/a|aka|a/k/a)\b[:\s]*')

# Count tokens
from collections import Counter
counts = Counter()
for r in pool.head(500000).iter_rows(named=True):
    for w in set((r['norm_name_clean'] or '').split()):
        if len(w) >= 2: counts[f"NM_{w}"] += 1
    for w in set((r['norm_address'] or '').split()):
        if len(w) >= 3 and w.isalpha(): counts[f"AD_{w}"] += 1
    for d in set(re_num.findall(r['norm_address'] or '')):
        counts[f"DG_{d}"] += 1
    p = (r['norm_name_clean'] or '').replace(' ', '')[:4]
    if len(p) >= 3: counts[f"PX_{p}"] += 1

max_df = 5000 * 2  # scaled to 500k
idf_map = {k: math.log(500000 / c) for k, c in counts.items() if c <= max_df}

# Build S1 tokens
s1_tokens = defaultdict(list)
for r in s1.iter_rows(named=True):
    sid = r['entity_id']
    for w in set((r['norm_name_clean'] or '').split()):
        k = f"NM_{w}"
        if k in idf_map: s1_tokens[sid].append((k, idf_map[k], 'NM'))
    for w in set((r['norm_address'] or '').split()):
        k = f"AD_{w}"
        if k in idf_map: s1_tokens[sid].append((k, idf_map[k], 'AD'))
    for d in set(re_num.findall(r['norm_address'] or '')):
        k = f"DG_{d}"
        if k in idf_map: s1_tokens[sid].append((k, idf_map[k], 'DG'))
    p = (r['norm_name_clean'] or '').replace(' ', '')[:4]
    k = f"PX_{p}"
    if k in idf_map: s1_tokens[sid].append((k, idf_map[k], 'PX'))

s1_needed = set()
for tlist in s1_tokens.values():
    for k, _, _ in tlist: s1_needed.add(k)

# Candidate index
cand_idx = defaultdict(list)
for r in pool.iter_rows(named=True):
    cid = r['entity_id']
    for w in set((r['norm_name_clean'] or '').split()):
        k = f"NM_{w}"
        if k in s1_needed: cand_idx[k].append(cid)
    for w in set((r['norm_address'] or '').split()):
        k = f"AD_{w}"
        if k in s1_needed: cand_idx[k].append(cid)
    for d in set(re_num.findall(r['norm_address'] or '')):
        k = f"DG_{d}"
        if k in s1_needed: cand_idx[k].append(cid)
    p = (r['norm_name_clean'] or '').replace(' ', '')[:4]
    k = f"PX_{p}"
    if k in s1_needed: cand_idx[k].append(cid)

print("Scoring candidates...")
# Compare Policy A (n_passes DESC, score DESC) vs Policy B (score DESC) vs Policy C (has_name DESC, score DESC)
for policy_name in ['A: n_passes DESC, score DESC', 'B: blocking_score DESC', 'C: name_score DESC, blocking_score DESC']:
    for k in [10, 25, 50, 100]:
        recalled = 0
        total_cands = 0
        for sid in s1_ids:
            cand_scores = defaultdict(float)
            cand_passes = defaultdict(set)
            cand_name_score = defaultdict(float)
            for key, idf, ptype in s1_tokens[sid]:
                if key in cand_idx:
                    for cid in cand_idx[key]:
                        cand_scores[cid] += idf
                        cand_passes[cid].add(ptype)
                        if ptype in ('NM', 'PX'):
                            cand_name_score[cid] += idf
            
            cands = []
            for cid in cand_scores:
                n_p = len(cand_passes[cid])
                sc = cand_scores[cid]
                n_sc = cand_name_score[cid]
                if 'A:' in policy_name:
                    sort_key = (n_p, sc)
                elif 'B:' in policy_name:
                    sort_key = (sc, n_p)
                else: # C
                    sort_key = (n_sc > 0, sc, n_p)
                cands.append((cid, sort_key))
            
            cands.sort(key=lambda x: x[1], reverse=True)
            top = [c[0] for c in cands[:k]]
            total_cands += len(top)
            for cid in top:
                if (sid, cid) in gt_pairs:
                    recalled += 1
        print(f"[{policy_name[:2]}] K={k:<3}: Recall = {recalled:,}/{n_gt:,} ({recalled/n_gt*100:.2f}%), Total cands = {total_cands:,}")
