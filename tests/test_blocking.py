import pytest
import polars as pl
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from blocking import build_keys, generate_candidates

def test_build_keys():
    df = pl.DataFrame({
        "entity_id": ["1", "2"],
        "country": ["US", "US"],
        "norm_name_clean": ["apple computer inc", "a"], # 'a' is too short
        "postal_code": ["12345", "12345"],
        "norm_address": ["123 main st", "456 side st"]
    })
    
    keys = build_keys(df)
    
    assert "entity_id" in keys.columns
    assert "block_key" in keys.columns
    assert "pass_name" in keys.columns
    
    # Pass A: exact name
    exact_keys = keys.filter(pl.col("pass_name") == "A")
    assert len(exact_keys) == 1
    assert exact_keys["block_key"][0] == "US|A|apple computer inc"

    # Pass B: prefix
    prefix_keys = keys.filter(pl.col("pass_name") == "B")
    assert len(prefix_keys) == 1
    assert prefix_keys["block_key"][0] == "US|B|appl"

    # Pass C: tokens ("apple", "computer")
    token_keys = keys.filter((pl.col("pass_name") == "C") & (pl.col("entity_id") == "1"))
    assert len(token_keys) == 2
    toks = set(token_keys["block_key"].to_list())
    assert toks == {"US|C|apple", "US|C|computer"}

def test_generate_candidates():
    s1 = pl.DataFrame({
        "entity_id": ["s1"],
        "country": ["US"],
        "norm_name_clean": ["apple computer inc"],
        "postal_code": ["12345"],
        "norm_address": ["123 main st"]
    })
    
    cand = pl.DataFrame({
        "entity_id": ["c1", "c2"],
        "country": ["US", "US"],
        "norm_name_clean": ["apple corp", "banana inc"],
        "postal_code": ["12345", "67890"],
        "norm_address": ["123 broadway", "456 side st"]
    })
    
    cands = generate_candidates(s1, cand)
    
    assert "source1_entity_id" in cands.columns
    assert "candidate_entity_id" in cands.columns
    assert "passes" in cands.columns
    
    # should match c1
    match_c1 = cands.filter(pl.col("candidate_entity_id") == "c1")
    assert len(match_c1) == 1
    
    # Should match on prefix B ("appl"), token C ("apple"), postal D ("12345" + "apple"), addr E ("123" + "apple")
    passes = match_c1["passes"][0]
    assert "B" in passes
    assert "C" in passes
    assert "D" in passes
    assert "E" in passes
    assert "A" not in passes # exact name doesn't match
    
    # banana inc has no tokens in common except "inc" which is a stop word
    match_c2 = cands.filter(pl.col("candidate_entity_id") == "c2")
    assert len(match_c2) == 0
