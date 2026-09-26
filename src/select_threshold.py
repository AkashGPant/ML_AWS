import polars as pl
from sklearn.metrics import precision_score, recall_score, fbeta_score
import json
import os

def main():
    print("=== Phase 4: Threshold Selection ===")
    
    val_preds = pl.read_parquet("outputs/val_predictions.parquet")
    
    y_true = val_preds["match"].to_numpy()
    y_probs = val_preds["predicted_prob"].to_numpy()
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    best_f05 = -1
    best_thresh = 0.5
    
    print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F0.5':>10}")
    print("-" * 50)
    
    for t in thresholds:
        y_pred_t = (y_probs >= t).astype(int)
        
        # Guard against zero-division warnings if precision is 0
        if y_pred_t.sum() == 0:
            p, r, f = 0.0, 0.0, 0.0
        else:
            p = precision_score(y_true, y_pred_t)
            r = recall_score(y_true, y_pred_t)
            f = fbeta_score(y_true, y_pred_t, beta=0.5)
            
        print(f"{t:10.2f} | {p:10.4f} | {r:10.4f} | {f:10.4f}")
        
        if f > best_f05:
            best_f05 = f
            best_thresh = t
            
    print("-" * 50)
    print(f"Best Threshold: {best_thresh} (F0.5 = {best_f05:.4f})")
    
    # Save the selected threshold
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/threshold.json", "w") as f:
        json.dump({"best_threshold": best_thresh, "val_f05": best_f05}, f)

if __name__ == "__main__":
    main()
