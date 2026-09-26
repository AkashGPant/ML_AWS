import polars as pl
import xgboost as xgb
import json
import os
import pytest
from src.predict_classifier import predict

def test_model_can_be_loaded():
    assert os.path.exists("outputs/classifier.model"), "Model file missing"
    assert os.path.exists("outputs/feature_cols.json"), "Feature columns file missing"
    
    model = xgb.XGBClassifier()
    model.load_model("outputs/classifier.model")
    
    with open("outputs/feature_cols.json", "r") as f:
        feature_cols = json.load(f)
        
    assert len(feature_cols) > 0, "Feature columns list is empty"

def test_prediction_output():
    # Only read a few rows to test
    df = pl.read_parquet("outputs/features_train_sample.parquet").head(50)
    
    df_scored = predict("outputs/classifier.model", "outputs/feature_cols.json", df)
    
    assert "match_prob" in df_scored.columns, "match_prob column is missing from predictions"
    probs = df_scored["match_prob"].to_list()
    
    for p in probs:
        assert 0.0 <= p <= 1.0, f"Probability {p} is out of bounds [0, 1]"
