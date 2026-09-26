import polars as pl

gt_misses = pl.read_csv('outputs/hard_misses.csv')

s1 = pl.read_parquet('dataset/normalized/train_source1.parquet', columns=['entity_id', 'country', 'norm_name_clean', 'norm_address', 'postal_code'])
c2 = pl.read_parquet('dataset/normalized/train_source2.parquet', columns=['entity_id', 'country', 'norm_name_clean', 'norm_address', 'postal_code'])
c3 = pl.read_parquet('dataset/normalized/train_source3.parquet', columns=['entity_id', 'country', 'norm_name_clean', 'norm_address', 'postal_code'])
cands = pl.concat([c2, c3])

joined = gt_misses.join(s1, left_on='source1_entity_id', right_on='entity_id') \
                  .join(cands, left_on='matched_id', right_on='entity_id', suffix='_cand')

print(joined.select(['country', 'country_cand', 'norm_name_clean', 'norm_name_clean_cand', 'postal_code', 'postal_code_cand']).head(20).to_pandas().to_markdown())
