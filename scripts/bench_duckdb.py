import duckdb
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

import os
os.makedirs("artifacts/blocking", exist_ok=True)
db_path = "artifacts/blocking/bench.duckdb"
if os.path.exists(db_path):
    os.remove(db_path)

con = duckdb.connect(db_path)
con.execute("PRAGMA memory_limit = '3GB'")
con.execute("PRAGMA threads = 4")
con.execute("PRAGMA preserve_insertion_order = false")
con.execute("PRAGMA temp_directory = 'artifacts/duckdb_temp'")

t0 = time.time()
print("Testing DuckDB vectorized join on US NM pass...")

con.execute("""
    CREATE TABLE s1_sample AS
    SELECT entity_id as s1_id, norm_name_clean
    FROM read_parquet('dataset/normalized/train_source1.parquet')
    WHERE country = 'US'
    LIMIT 10000;
""")
print(f"Created S1 sample in {time.time()-t0:.2f}s")

t1 = time.time()
con.execute("""
    CREATE TABLE s1_tokens AS
    SELECT s1_id, 'NM_' || t as token_key, 'NM' as token_type
    FROM (
        SELECT s1_id, unnest(str_split(norm_name_clean, ' ')) as t
        FROM s1_sample
    )
    WHERE length(t) >= 2;
""")
n_s1_tok = con.execute("SELECT count(*) FROM s1_tokens").fetchone()[0]
print(f"Created S1 tokens ({n_s1_tok:,}) in {time.time()-t1:.2f}s")

t2 = time.time()
con.execute("""
    CREATE TABLE pool_tokens AS
    WITH pool AS (
        SELECT entity_id as cand_id, norm_name_clean
        FROM read_parquet('dataset/normalized/train_source2.parquet')
        WHERE country = 'US'
        UNION ALL
        SELECT entity_id as cand_id, norm_name_clean
        FROM read_parquet('dataset/normalized/train_source3.parquet')
        WHERE country = 'US'
    ),
    unnested AS (
        SELECT cand_id, unnest(str_split(norm_name_clean, ' ')) as t
        FROM pool
    )
    SELECT cand_id, 'NM_' || t as token_key, 'NM' as token_type
    FROM unnested
    WHERE length(t) >= 2;
""")
n_pool_tok = con.execute("SELECT count(*) FROM pool_tokens").fetchone()[0]
print(f"Created pool tokens ({n_pool_tok:,}) in {time.time()-t2:.2f}s")

t3 = time.time()
con.execute("""
    CREATE TABLE token_idf AS
    WITH counts AS (
        SELECT token_key, count(DISTINCT cand_id) as df
        FROM pool_tokens
        GROUP BY token_key
    )
    SELECT token_key, df, ln(6186873.0 / df) as idf
    FROM counts
    WHERE df <= 0.02 * 6186873;
""")
n_idf = con.execute("SELECT count(*) FROM token_idf").fetchone()[0]
print(f"Computed IDF table ({n_idf:,} tokens) in {time.time()-t3:.2f}s")

t4 = time.time()
con.execute("""
    CREATE TABLE candidate_pairs AS
    WITH active_tokens AS (
        SELECT s.s1_id, s.token_key, i.idf, s.token_type
        FROM s1_tokens s
        JOIN token_idf i ON s.token_key = i.token_key
    )
    SELECT 
        a.s1_id, 
        p.cand_id, 
        sum(a.idf) as blocking_score, 
        count(DISTINCT a.token_type) as n_passes
    FROM active_tokens a
    JOIN pool_tokens p ON a.token_key = p.token_key
    GROUP BY a.s1_id, p.cand_id;
""")
n_pairs = con.execute("SELECT count(*) FROM candidate_pairs").fetchone()[0]
print(f"Joined and aggregated ({n_pairs:,} pairs) in {time.time()-t4:.2f}s")

t5 = time.time()
res = con.execute("""
    WITH gt AS (
        SELECT source1_entity_id, matched_id
        FROM read_parquet('dataset/normalized/train_ground_truth_pairs.parquet')
        WHERE source1_entity_id IN (SELECT s1_id FROM s1_sample)
    ),
    matched AS (
        SELECT g.source1_entity_id, g.matched_id
        FROM gt g
        JOIN candidate_pairs c ON g.source1_entity_id = c.s1_id AND g.matched_id = c.cand_id
    )
    SELECT 
        (SELECT count(*) FROM gt) as total_gt,
        (SELECT count(*) FROM matched) as recalled_gt;
""").fetchall()
print(f"GT Recall: {res[0][1]:,} / {res[0][0]:,} ({res[0][1]/res[0][0]*100:.2f}%) in {time.time()-t5:.2f}s")
