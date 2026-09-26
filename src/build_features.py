import polars as pl
import os
import time

from features import build_features

def main():
    print("=== Phase 2: Feature Generation ===")
    t0 = time.time()
    
    cands_file = "outputs/candidate_pairs_train_sample.tsv"
    gt_file = "dataset/normalized/train_ground_truth_pairs.parquet"
    out_file = "outputs/features_train_sample.parquet"
    
    if not os.path.exists(cands_file):
        print(f"Error: {cands_file} not found. Run Phase 1 first.")
        return
        
    print("Loading candidate pairs...")
    cands = pl.read_csv(cands_file, separator='\t')
    
    print("Loading normalized datasets to attach attributes...")
    # Load S1
    s1 = pl.read_parquet("dataset/normalized/train_source1.parquet", columns=["entity_id", "country", "norm_name_clean", "norm_address", "postal_code"])
    s1 = s1.rename({
        col: f"{col}_s1" for col in ["country", "norm_name_clean", "norm_address", "postal_code"]
    })
    
    # Load Candidates (S2 and S3)
    c2 = pl.read_parquet("dataset/normalized/train_source2.parquet", columns=["entity_id", "country", "norm_name_clean", "norm_address", "postal_code"])
    c3 = pl.read_parquet("dataset/normalized/train_source3.parquet", columns=["entity_id", "country", "norm_name_clean", "norm_address", "postal_code"])
    c_all = pl.concat([c2, c3])
    c_all = c_all.rename({
        col: f"{col}_cand" for col in ["country", "norm_name_clean", "norm_address", "postal_code"]
    })
    
    print("Joining attributes...")
    df = cands.join(s1, left_on="source1_entity_id", right_on="entity_id", how="left")
    df = df.join(c_all, left_on="candidate_entity_id", right_on="entity_id", how="left")
    
    print("Generating labels from ground truth...")
    gt = pl.read_parquet(gt_file).with_columns(pl.lit(1).alias("match").cast(pl.Int8))
    
    df = df.join(
        gt.select(["source1_entity_id", "matched_id", "match"]),
        left_on=["source1_entity_id", "candidate_entity_id"],
        right_on=["source1_entity_id", "matched_id"],
        how="left"
    ).with_columns(pl.col("match").fill_null(0))
    
    print(f"Base data ready. Total rows: {len(df)}")
    
    print("Computing features...")
    df_features = build_features(df)
    
    # Drop original raw text columns to keep ML dataset clean
    drop_cols = [
        "country_s1", "norm_name_clean_s1", "norm_address_s1", "postal_code_s1",
        "country_cand", "norm_name_clean_cand", "norm_address_cand", "postal_code_cand"
    ]
    df_features = df_features.drop(drop_cols)
    
    print(f"Saving {len(df_features)} rows to {out_file}...")
    df_features.write_parquet(out_file)
    
    # Quick sanity check print
    n_pos = df_features.filter(pl.col("match") == 1).shape[0]
    n_neg = df_features.filter(pl.col("match") == 0).shape[0]
    
    print(f"Features generated in {time.time() - t0:.2f}s")
    print(f"Total Positives (match=1): {n_pos}")
    print(f"Total Negatives (match=0): {n_neg}")
    print(f"Shape: {df_features.shape}")

if __name__ == "__main__":
    main()
