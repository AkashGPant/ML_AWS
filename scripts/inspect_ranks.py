import polars as pl
from collections import defaultdict
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load 500 S1 sample and GT
s1 = pl.read_parquet('dataset/normalized/train_source1.parquet', columns=['entity_id', 'country', 'norm_name_clean', 'norm_address']).filter(pl.col('country') == 'US').head(200)
s1_ids = set(s1['entity_id'].to_list())
gt = pl.read_parquet('dataset/normalized/train_ground_truth_pairs.parquet').filter(pl.col('source1_entity_id').is_in(list(s1_ids)))

pool = pl.concat([
    pl.read_parquet('dataset/normalized/train_source2.parquet', columns=['entity_id', 'country', 'raw_name', 'norm_name_clean', 'norm_address']).filter(pl.col('country') == 'US'),
    pl.read_parquet('dataset/normalized/train_source3.parquet', columns=['entity_id', 'country', 'raw_name', 'norm_name_clean', 'norm_address']).filter(pl.col('country') == 'US')
])

gt_pairs = set(zip(gt['source1_entity_id'].to_list(), gt['matched_id'].to_list()))
re_num = re.compile(r'\b\d+\b')

for r1 in s1.head(8).iter_rows(named=True):
    s1_id = r1['entity_id']
    gt_cands = [m[1] for m in gt_pairs if m[0] == s1_id]
    print(f"\n=== S1: {s1_id} | Name: '{r1['norm_name_clean']}' | Addr: '{r1['norm_address']}'")
    for cid in gt_cands:
        cand_rows = pool.filter(pl.col('entity_id') == cid)
        if len(cand_rows) > 0:
            c = cand_rows.to_dicts()[0]
            print(f"  GT {cid}: name='{c['norm_name_clean']}', addr='{c['norm_address']}'")
