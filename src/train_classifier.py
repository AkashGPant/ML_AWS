import polars as pl
import xgboost as xgb
import json
import time
import os
from sklearn.metrics import precision_score, recall_score, fbeta_score, average_precision_score, confusion_matrix

def split_by_s1(df: pl.DataFrame, val_ratio=0.2):
    """Splits the dataframe into train/val ensuring no source1_entity_id overlap."""
    unique_s1 = df.select("source1_entity_id").unique()
    # Randomly shuffle and split
    unique_s1 = unique_s1.sample(fraction=1.0, shuffle=True, seed=42)
    split_idx = int(unique_s1.shape[0] * (1 - val_ratio))
    train_s1 = unique_s1.slice(0, split_idx)
    val_s1 = unique_s1.slice(split_idx, unique_s1.shape[0] - split_idx)
    
    train_df = df.join(train_s1, on="source1_entity_id", how="inner")
    val_df = df.join(val_s1, on="source1_entity_id", how="inner")
    return train_df, val_df

def main():
    print("=== Phase 3: Training ML Classifier ===")
    t0 = time.time()
    
    # 1. Load data
    df = pl.read_parquet("outputs/features_train_sample.parquet")
    
    # 2. Split train/val
    train_df, val_df = split_by_s1(df, val_ratio=0.2)
    print(f"Train size: {train_df.shape[0]} | Val size: {val_df.shape[0]}")
    
    # Features
    feature_cols = [c for c in df.columns if df[c].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64] and c not in {"match"}]
    
    # Save feature list
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
        
    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["match"].to_numpy()
    
    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df["match"].to_numpy()
    
    # 3. Class imbalance
    n_pos = train_df.filter(pl.col("match") == 1).shape[0]
    n_neg = train_df.filter(pl.col("match") == 0).shape[0]
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Positives: {n_pos}, Negatives: {n_neg} -> scale_pos_weight: {scale_pos_weight:.2f}")
    
    # 4. Train lightweight model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    print("Evaluating on validation set...")
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f05 = fbeta_score(y_val, y_pred, beta=0.5)
    pr_auc = average_precision_score(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)
    
    print("\n--- Validation Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F0.5:      {f05:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Predicted Positives: {y_pred.sum()}")
    print("Confusion Matrix:")
    print(cm)
    
    # 6. Save model and predictions
    model.save_model("outputs/classifier.model")
    
    val_out = val_df.select(["source1_entity_id", "candidate_entity_id", "match"]).with_columns(
        pl.Series("predicted_prob", y_prob),
        pl.Series("predicted_class", y_pred)
    )
    val_out.write_parquet("outputs/val_predictions.parquet")
    print(f"\nModel and predictions saved. Total time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
