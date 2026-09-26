# Phase 3: ML Pair Classifier Report

## Objective
The goal of Phase 3 was to train a lightweight binary classifier to predict whether a given candidate pair is a true business match, utilizing the carefully curated features from Phase 2. The key challenges were safely handling memory on a ~9.8M row dataset and dealing with severe class imbalance (~0.3% positives).

## Methodology

### 1. Data Splitting
We performed an 80/20 train/validation split. Crucially, the split was performed grouped by `source1_entity_id` to strictly prevent data leakage (ensuring the same S1 entity never appears in both the training and validation sets).
- **Train Set:** 7,800,799 rows
- **Validation Set:** 2,002,226 rows

### 2. Model Selection & Configuration
We utilized `XGBoost (XGBClassifier)`, configured with the `hist` tree method for ultra-fast, memory-efficient histogram-based splits.
- **Hyperparameters:** `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`.
- **Class Imbalance:** Handled natively by supplying `scale_pos_weight = (Negatives / Positives)`. The computed ratio was heavily skewed at **332.67**.

### 3. Training Performance
Despite processing nearly 8 million candidate pairs and 13 numerical similarity/agreement features, the model trained in just **~19 seconds**. Local memory usage remained low and highly stable throughout execution.

## Validation Results

The classifier performed exceptionally well, capturing nearly all true matches while keeping false positives reasonably low (to be thresholded in Phase 4).

* **Recall:** 99.38% (Captured 5,915 out of 5,952 true matches in the validation set).
* **Precision:** 42.10% (For every true match found, it safely predicted ~1.3 false positives).
* **F0.5 Score:** 0.4758
* **PR-AUC (Precision-Recall Area):** 0.9352 (Demonstrates extremely strong ranking power).

### Confusion Matrix (Validation Set):
| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 1,988,139 | 8,135 |
| **Actual Positive** | 37 | 5,915 |

## Deliverables
- `src/train_classifier.py`: Trains the XGBoost model and calculates metrics.
- `src/predict_classifier.py`: Prediction utility for safely attaching probability scores.
- `tests/test_classifier.py`: Ensures the model can load and emit valid [0,1] probability columns.
- **Model Artifacts:** The fitted model (`outputs/classifier.model`) and ordered feature names (`outputs/feature_cols.json`) are successfully exported.
- **Validation Predictions:** `outputs/val_predictions.parquet` containing probabilities and raw predictions.

## Conclusion
Phase 3 is **COMPLETE**. We have successfully trained an extremely fast, high-recall binary classifier capable of distinguishing true business entity matches from hard negatives. The classifier's robust PR-AUC (0.935) indicates that a precise threshold can be applied in the final phase to hit target business metrics.

We are now ready for **Phase 4 (Thresholding & Final Entity Matching)**.
