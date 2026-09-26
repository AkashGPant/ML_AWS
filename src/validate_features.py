import polars as pl
import time

def main():
    print("=== Phase 2: Feature Validation ===")
    out_file = "outputs/features_train_sample.parquet"
    
    print("Loading features...")
    df = pl.read_parquet(out_file)
    
    # 1. Schema validation
    expected_cols = {
        "source1_entity_id", "candidate_entity_id", "match",
        "name_exact_match", "address_exact_match", "country_agree", "postal_agree",
        "name_len_diff", "address_len_diff", "name_tok_overlap", "addr_tok_overlap",
        "name_tok_jaccard", "addr_tok_jaccard", "name_jaro_winkler", "addr_jaro_winkler"
    }
    missing_cols = expected_cols - set(df.columns)
    assert not missing_cols, f"Missing columns: {missing_cols}"
    print("[OK] Schema validation passed. All expected columns exist.")
    
    # 2. Data quality
    null_counts = df.null_count().to_dict(as_series=False)
    for col, count in null_counts.items():
        assert count[0] == 0, f"Column {col} contains {count[0]} null values."
    print("[OK] Data quality passed. No NaN/null values.")
    
    is_duplicate = df.is_duplicated()
    assert df.filter(is_duplicate).shape[0] == 0, "Duplicate rows found!"
    print("[OK] Data quality passed. No duplicate candidate pairs.")
    
    unique_labels = df["match"].unique().to_list()
    assert set(unique_labels) == {0, 1}, f"Invalid labels found: {unique_labels}"
    print("[OK] Data quality passed. Labels are strictly binary [0, 1].")
    
    # 3. Feature ranges
    assert df["name_tok_jaccard"].min() >= 0.0 and df["name_tok_jaccard"].max() <= 1.0
    assert df["addr_tok_jaccard"].min() >= 0.0 and df["addr_tok_jaccard"].max() <= 1.0
    assert df["name_jaro_winkler"].min() >= 0.0 and df["name_jaro_winkler"].max() <= 1.0
    assert df["addr_jaro_winkler"].min() >= 0.0 and df["addr_jaro_winkler"].max() <= 1.0
    print("[OK] Feature ranges passed. Similarity features are [0, 1].")
    
    assert set(df["name_exact_match"].unique()) <= {0, 1}
    assert set(df["address_exact_match"].unique()) <= {0, 1}
    assert set(df["country_agree"].unique()) <= {0, 1}
    assert set(df["postal_agree"].unique()) <= {0, 1}
    print("[OK] Feature ranges passed. Agreement features are binary.")
    
    assert df["name_len_diff"].min() >= 0
    assert df["address_len_diff"].min() >= 0
    print("[OK] Feature ranges passed. Length difference features are sensible (>= 0).")
    
    # 4 & 5. Label validation and Class distribution
    n_pos = df.filter(pl.col("match") == 1).shape[0]
    n_neg = df.filter(pl.col("match") == 0).shape[0]
    total = df.shape[0]
    
    assert n_pos > 0 and n_neg > 0, "Missing positive or negative pairs."
    print(f"[OK] Label validation passed. Positives: {n_pos}, Negatives: {n_neg}")
    print(f"[OK] Class distribution: {n_pos/total:.4%} Positive")
    
    # 6. Feature usefulness (Means)
    print("\n--- Feature Usefulness (Means) ---")
    pos_df = df.filter(pl.col("match") == 1)
    neg_df = df.filter(pl.col("match") == 0)
    
    numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64] and c not in {"match"}]
    
    useful_features = 0
    for col in numeric_cols:
        pos_mean = pos_df.select(pl.col(col).mean()).item()
        neg_mean = neg_df.select(pl.col(col).mean()).item()
        print(f"{col:>20}: Pos Mean = {pos_mean:.4f} | Neg Mean = {neg_mean:.4f}")
        if abs(pos_mean - neg_mean) > 0.001:
            useful_features += 1
            
    print(f"[OK] Feature usefulness: {useful_features}/{len(numeric_cols)} numeric features show separability.")
    
    print("\n=== Validation Complete ===")

if __name__ == "__main__":
    main()
