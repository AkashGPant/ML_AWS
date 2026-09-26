import polars as pl
import xgboost as xgb
import json

def predict(model_path: str, feature_cols_path: str, df: pl.DataFrame) -> pl.DataFrame:
    """Loads model and applies it to dataframe."""
    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    X = df.select(feature_cols).to_numpy()
    y_prob = model.predict_proba(X)[:, 1]
    
    return df.with_columns(
        pl.Series("match_prob", y_prob)
    )

if __name__ == "__main__":
    # Simple test run on sample data
    df = pl.read_parquet("outputs/features_train_sample.parquet").head(100)
    df_scored = predict("outputs/classifier.model", "outputs/feature_cols.json", df)
    print(df_scored.select(["source1_entity_id", "candidate_entity_id", "match_prob"]).head())
