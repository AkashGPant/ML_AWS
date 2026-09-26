import polars as pl
import pytest
from src.features import build_features

def test_build_features_labels_and_types():
    df = pl.DataFrame({
        "source1_entity_id": ["A1", "A2"],
        "candidate_entity_id": ["B1", "B2"],
        "match": [1, 0],
        "norm_name_clean_s1": ["apple inc", "banana corp"],
        "norm_name_clean_cand": ["apple inc", "banana ltd"],
        "norm_address_s1": ["123 main st", "456 elm st"],
        "norm_address_cand": ["123 main st", "456 elm"],
        "country_s1": ["US", "US"],
        "country_cand": ["US", "UK"],
        "postal_code_s1": ["12345", "54321"],
        "postal_code_cand": ["12345", "54321"]
    })
    
    features = build_features(df)
    
    # Check expected feature columns
    expected_cols = {
        "name_exact_match", "address_exact_match", "country_agree", "postal_agree",
        "name_len_diff", "address_len_diff", "name_tok_overlap", "addr_tok_overlap",
        "name_tok_jaccard", "addr_tok_jaccard", "name_jaro_winkler", "addr_jaro_winkler"
    }
    
    for col in expected_cols:
        assert col in features.columns, f"Missing feature column: {col}"
    
    # Verify logical correctness for first row (exact match)
    row1 = features.filter(pl.col("source1_entity_id") == "A1").row(0, named=True)
    assert row1["name_exact_match"] == 1
    assert row1["address_exact_match"] == 1
    assert row1["country_agree"] == 1
    assert row1["postal_agree"] == 1
    assert row1["name_len_diff"] == 0
    assert row1["address_len_diff"] == 0
    assert row1["name_tok_overlap"] == 2
    assert row1["addr_tok_overlap"] == 3
    assert row1["name_tok_jaccard"] == 1.0
    assert row1["addr_tok_jaccard"] == 1.0
    assert row1["name_jaro_winkler"] == 1.0
    assert row1["addr_jaro_winkler"] == 1.0
    
    # Verify logical correctness for second row (partial match)
    row2 = features.filter(pl.col("source1_entity_id") == "A2").row(0, named=True)
    assert row2["name_exact_match"] == 0
    assert row2["address_exact_match"] == 0
    assert row2["country_agree"] == 0
    assert row2["postal_agree"] == 1
    assert row2["name_tok_overlap"] == 1  # banana
    assert row2["addr_tok_overlap"] == 2  # 456 elm
    assert row2["name_jaro_winkler"] < 1.0
    assert row2["addr_jaro_winkler"] < 1.0

def test_build_features_null_handling():
    df = pl.DataFrame({
        "source1_entity_id": ["A1"],
        "candidate_entity_id": ["B1"],
        "match": [0],
        "norm_name_clean_s1": [None],
        "norm_name_clean_cand": ["apple inc"],
        "norm_address_s1": ["123 main st"],
        "norm_address_cand": [None],
        "country_s1": [None],
        "country_cand": [None],
        "postal_code_s1": ["12345"],
        "postal_code_cand": [None]
    })
    
    features = build_features(df)
    row = features.row(0, named=True)
    
    assert row["name_exact_match"] == 0
    assert row["address_exact_match"] == 0
    assert row["country_agree"] == 0
    assert row["postal_agree"] == 0
    assert row["name_tok_overlap"] == 0
    assert row["addr_tok_overlap"] == 0
    assert row["name_tok_jaccard"] == 0.0
    assert row["name_jaro_winkler"] == 0.0
    assert row["name_len_diff"] == 999  # Fallback for nulls
    assert row["address_len_diff"] == 999
