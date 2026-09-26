import polars as pl
import json
import os
from src.predict_classifier import predict

def main():
    print("=== Phase 4: Final Entity Matching ===")
    
    with open("outputs/threshold.json", "r") as f:
        best_thresh = json.load(f)["best_threshold"]
    
    print(f"Loading full 10k feature dataset and scoring with threshold {best_thresh}...")
    
    # 1. Score the entire 10k candidates
    df = pl.read_parquet("outputs/features_train_sample.parquet")
    df_scored = predict("outputs/classifier.model", "outputs/feature_cols.json", df)
    
    # 2. Filter to predictions >= threshold
    matches = df_scored.filter(pl.col("match_prob") >= best_thresh)
    
    # We only need source1_entity_id and candidate_entity_id
    matches = matches.select(["source1_entity_id", "candidate_entity_id"])
    
    # 3. Group by source1_entity_id and aggregate candidates into a comma-separated list
    grouped_matches = matches.group_by("source1_entity_id").agg(
        pl.col("candidate_entity_id").alias("matched_ids")
    ).with_columns(
        pl.col("matched_ids").list.join(",")
    ).rename({"matched_ids": "matched_entity_ids"})
    
    # 4. We MUST include every S1 entity exactly once. Even those with 0 candidates or 0 matches.
    # The list of ALL 10,000 S1 IDs used is in outputs/s1_sample_ids.txt
    print("Enforcing exactly one row per S1 entity...")
    with open("outputs/s1_sample_ids.txt", "r") as f:
        s1_ids = [line.strip() for line in f if line.strip()]
        
    s1_df = pl.DataFrame({"source1_entity_id": s1_ids})
    
    final_matching = s1_df.join(grouped_matches, on="source1_entity_id", how="left").fill_null("")
    
    # 5. Format candidate_pairs.tsv correctly too
    cands_raw = pl.read_csv("outputs/candidate_pairs_train_sample.tsv", separator="\t")
    # Group candidates properly
    grouped_cands = cands_raw.group_by("source1_entity_id").agg(
        pl.col("candidate_entity_id").alias("candidate_ids")
    ).with_columns(
        pl.col("candidate_ids").list.join(",")
    ).rename({"candidate_ids": "candidate_entity_ids"})
    
    final_candidates = s1_df.join(grouped_cands, on="source1_entity_id", how="left").fill_null("")
    
    # 6. Save outputs
    print("Saving matching_results.tsv and candidate_pairs.tsv...")
    final_matching.write_csv("outputs/matching_results.tsv", separator="\t", quote_style="never")
    final_candidates.write_csv("outputs/candidate_pairs.tsv", separator="\t", quote_style="never")
    
    print("Done! You can validate using:")
    print("python utils/validate_submission.py --matching outputs/matching_results.tsv --candidate outputs/candidate_pairs.tsv --test-dir dataset/test_sample")

if __name__ == "__main__":
    main()
