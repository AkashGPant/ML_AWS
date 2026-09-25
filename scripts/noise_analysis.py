#!/usr/bin/env python3
"""
scripts/noise_analysis.py: Deep noise, script, and pattern analysis on ground-truth positive pairs.
"""

import os
import sys
import json
import re
import duckdb
from collections import Counter, defaultdict
import unicodedata

sys.stdout.reconfigure(encoding='utf-8')

conn = duckdb.connect()
conn.execute("SET memory_limit = '4GB';")
conn.execute("SET threads = 4;")

print("Sampling positive pairs from ground truth...")

# Sample 50,000 matched pairs (25,000 US, 25,000 India; half S2, half S3) for deep pattern analysis
query_sample = """
WITH unnested AS (
    SELECT 
        source1_entity_id,
        unnest(str_split(trim(matched_entity_ids), ',')) as matched_id
    FROM read_csv('dataset/train/train_ground_truth.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')
    WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) != ''
),
s1 AS (
    SELECT entity_id, business_name as s1_name, business_address as s1_addr, country
    FROM read_csv('dataset/train/train_source1.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')
),
s2 AS (
    SELECT entity_id, business_name as target_name, business_address as target_addr, country, 'S2' as target_src
    FROM read_csv('dataset/train/train_source2.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')
),
s3 AS (
    SELECT entity_id, business_name as target_name, business_address as target_addr, country, 'S3' as target_src
    FROM read_csv('dataset/train/train_source3.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')
),
target_combined AS (
    SELECT * FROM s2
    UNION ALL
    SELECT * FROM s3
),
matched_pairs AS (
    SELECT 
        u.source1_entity_id as s1_id,
        u.matched_id as target_id,
        s1.country,
        tc.target_src,
        s1.s1_name,
        tc.target_name,
        s1.s1_addr,
        tc.target_addr
    FROM unnested u
    JOIN s1 ON u.source1_entity_id = s1.entity_id
    JOIN target_combined tc ON u.matched_id = tc.entity_id
)
SELECT * FROM matched_pairs USING SAMPLE 100000 (reservoir);
"""

pairs_df = conn.execute(query_sample).df()
print(f"Loaded {len(pairs_df):,} sampled ground-truth positive pairs.")

# Unicode scripts analyzer
def detect_script(text):
    if not text:
        return "EMPTY"
    scripts = Counter()
    for ch in text:
        if ch.isspace() or ch in "0123456789!@#$%^&*()_+-=[]{}|;':\",./<>?`~\\":
            continue
        try:
            name = unicodedata.name(ch)
            if "DEVANAGARI" in name:
                scripts["Devanagari"] += 1
            elif "BENGALI" in name:
                scripts["Bengali"] += 1
            elif "TAMIL" in name:
                scripts["Tamil"] += 1
            elif "TELUGU" in name:
                scripts["Telugu"] += 1
            elif "GUJARATI" in name:
                scripts["Gujarati"] += 1
            elif "GURMUKHI" in name:
                scripts["Gurmukhi"] += 1
            elif "KANNADA" in name:
                scripts["Kannada"] += 1
            elif "MALAYALAM" in name:
                scripts["Malayalam"] += 1
            elif "LATIN" in name:
                scripts["Latin"] += 1
            else:
                scripts[name.split()[0]] += 1
        except ValueError:
            scripts["UNKNOWN"] += 1
    if not scripts:
        return "PUNCT_DIGIT"
    return scripts.most_common(1)[0][0]

# Metrics
total_pairs = len(pairs_df)
exact_name_match = 0
lower_name_match = 0
exact_addr_match = 0
lower_addr_match = 0

missing_addr = 0
non_ascii_name = 0
non_ascii_addr = 0
target_scripts = Counter()

# Pattern checks
dba_pattern = re.compile(r'\b(dba|d/b/a|d\.b\.a\.|t/a|trading as|fka|f/k/a|aka|a/k/a)\b', re.IGNORECASE)
domain_pattern = re.compile(r'\b([a-zA-Z0-9-]+\.(?:com|org|net|in|co\.in|io|biz|info))\b', re.IGNORECASE)
paren_pattern = re.compile(r'\(.*?\)')

dba_in_s1 = 0
dba_in_target = 0
domain_in_s1 = 0
domain_in_target = 0

weak_name_strong_addr = []
strong_name_weak_addr = []
missing_addr_pairs = []
devanagari_pairs = []
domain_pairs = []
dba_pairs = []

def tokenize(s):
    if not s:
        return set()
    return set(re.findall(r'[a-zA-Z0-9]+', s.lower()))

def jaccard(s1, s2):
    t1 = tokenize(s1)
    t2 = tokenize(s2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

for row in pairs_df.itertuples():
    s1_n = str(row.s1_name) if row.s1_name is not None and str(row.s1_name) != 'nan' else ""
    t_n = str(row.target_name) if row.target_name is not None and str(row.target_name) != 'nan' else ""
    s1_a = str(row.s1_addr) if row.s1_addr is not None and str(row.s1_addr) != 'nan' else ""
    t_a = str(row.target_addr) if row.target_addr is not None and str(row.target_addr) != 'nan' else ""
    
    if s1_n == t_n:
        exact_name_match += 1
    if s1_n.lower() == t_n.lower():
        lower_name_match += 1
    
    if not t_a:
        missing_addr += 1
        missing_addr_pairs.append(row)
    else:
        if s1_a == t_a:
            exact_addr_match += 1
        if s1_a.lower() == t_a.lower():
            lower_addr_match += 1
            
    t_script = detect_script(t_n)
    target_scripts[t_script] += 1
    if t_script != "Latin" and t_script not in ("EMPTY", "PUNCT_DIGIT"):
        non_ascii_name += 1
        if t_script == "Devanagari":
            devanagari_pairs.append(row)
            
    if dba_pattern.search(s1_n) or dba_pattern.search(s1_a):
        dba_in_s1 += 1
    if dba_pattern.search(t_n) or dba_pattern.search(t_a):
        dba_in_target += 1
        dba_pairs.append(row)
        
    if domain_pattern.search(s1_n):
        domain_in_s1 += 1
    if domain_pattern.search(t_n):
        domain_in_target += 1
        domain_pairs.append(row)
        
    # Similarity
    n_jac = jaccard(s1_n, t_n)
    a_jac = jaccard(s1_a, t_a) if t_a else 0.0
    
    if n_jac < 0.2 and a_jac > 0.6:
        weak_name_strong_addr.append((n_jac, a_jac, row))
    if n_jac > 0.7 and a_jac < 0.2 and t_a:
        strong_name_weak_addr.append((n_jac, a_jac, row))

print("\n--- Match Similarity Statistics on 100,000 Sampled Pairs ---")
print(f"Exact Name Match: {exact_name_match} ({exact_name_match/total_pairs*100:.2f}%)")
print(f"Case-insensitive Name Match: {lower_name_match} ({lower_name_match/total_pairs*100:.2f}%)")
print(f"Exact Address Match: {exact_addr_match} ({exact_addr_match/total_pairs*100:.2f}%)")
print(f"Case-insensitive Address Match: {lower_addr_match} ({lower_addr_match/total_pairs*100:.2f}%)")
print(f"Missing Target Address: {missing_addr} ({missing_addr/total_pairs*100:.2f}%)")
print(f"Non-Latin Target Name Script: {non_ascii_name} ({non_ascii_name/total_pairs*100:.2f}%)")
print(f"Target Scripts distribution: {dict(target_scripts.most_common(10))}")
print(f"DBA patterns in target: {dba_in_target} ({dba_in_target/total_pairs*100:.2f}%)")
print(f"Domain patterns in target: {domain_in_target} ({domain_in_target/total_pairs*100:.2f}%)")
print(f"Weak Name (<0.2) & Strong Addr (>0.6) pairs: {len(weak_name_strong_addr)} ({len(weak_name_strong_addr)/total_pairs*100:.2f}%)")
print(f"Strong Name (>0.7) & Weak Addr (<0.2) pairs: {len(strong_name_weak_addr)} ({len(strong_name_weak_addr)/total_pairs*100:.2f}%)")

print("\n--- Representative Samples: Devanagari / Transliteration Noise ---")
for r in devanagari_pairs[:5]:
    print(f"  S1 [{r.country}]: {r.s1_name} | Addr: {r.s1_addr}")
    print(f"  {r.target_src} [{r.target_id}]: {r.target_name} | Addr: {r.target_addr}")
    print()

print("\n--- Representative Samples: Domain Names as Business Names ---")
for r in domain_pairs[:5]:
    print(f"  S1 [{r.country}]: {r.s1_name} | Addr: {r.s1_addr}")
    print(f"  {r.target_src} [{r.target_id}]: {r.target_name} | Addr: {r.target_addr}")
    print()

print("\n--- Representative Samples: DBA / Trade Names ---")
for r in dba_pairs[:5]:
    print(f"  S1 [{r.country}]: {r.s1_name} | Addr: {r.s1_addr}")
    print(f"  {r.target_src} [{r.target_id}]: {r.target_name} | Addr: {r.target_addr}")
    print()

print("\n--- Representative Samples: Weak Name, Strong Address ---")
for n_j, a_j, r in weak_name_strong_addr[:5]:
    print(f"  Name Jaccard={n_j:.2f}, Addr Jaccard={a_j:.2f} [{r.country}]")
    print(f"  S1: {r.s1_name} | Addr: {r.s1_addr}")
    print(f"  {r.target_src}: {r.target_name} | Addr: {r.target_addr}")
    print()

print("\n--- Representative Samples: Strong Name, Weak Address ---")
for n_j, a_j, r in strong_name_weak_addr[:5]:
    print(f"  Name Jaccard={n_j:.2f}, Addr Jaccard={a_j:.2f} [{r.country}]")
    print(f"  S1: {r.s1_name} | Addr: {r.s1_addr}")
    print(f"  {r.target_src}: {r.target_name} | Addr: {r.target_addr}")
    print()

print("\n--- Representative Samples: Missing Target Address ---")
for r in missing_addr_pairs[:5]:
    print(f"  S1 [{r.country}]: {r.s1_name} | Addr: {r.s1_addr}")
    print(f"  {r.target_src}: {r.target_name} | Addr: (MISSING)")
    print()

# Save detailed analysis JSON
analysis_data = {
    "sampled_pairs": total_pairs,
    "exact_name_match_pct": exact_name_match / total_pairs * 100,
    "lower_name_match_pct": lower_name_match / total_pairs * 100,
    "exact_addr_match_pct": exact_addr_match / total_pairs * 100,
    "lower_addr_match_pct": lower_addr_match / total_pairs * 100,
    "missing_target_addr_pct": missing_addr / total_pairs * 100,
    "target_scripts": dict(target_scripts),
    "dba_in_target_pct": dba_in_target / total_pairs * 100,
    "domain_in_target_pct": domain_in_target / total_pairs * 100,
    "weak_name_strong_addr_pct": len(weak_name_strong_addr) / total_pairs * 100,
    "strong_name_weak_addr_pct": len(strong_name_weak_addr) / total_pairs * 100,
}
with open("artifacts/noise_analysis.json", "w", encoding="utf-8") as f:
    json.dump(analysis_data, f, indent=2)

print("Noise analysis completed and saved to artifacts/noise_analysis.json")
