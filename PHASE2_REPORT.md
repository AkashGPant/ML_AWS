# Phase 2: Similarity Features Report

## Objective
The goal of Phase 2 was to convert the Phase 1 candidate pairs (a highly reduced search space of ~9.8M pairs) into a clean, ML-ready feature set. The feature set must be memory-safe, deterministic, and rely strictly on computationally lightweight string similarity and exact-match logic.

## Methodology

We built a highly parallelized feature pipeline using `Polars` and `rapidfuzz`, processing 9.8 million candidate pairs extremely efficiently.

### Computed Features:
- **Exact Matches:**
  - `name_exact_match` (1 if exact name match, else 0)
  - `address_exact_match` (1 if exact address match, else 0)
- **Token Overlap:**
  - `name_tok_overlap` (Number of exact shared tokens in names)
  - `addr_tok_overlap` (Number of exact shared tokens in addresses)
  - `name_tok_jaccard` (Jaccard similarity coefficient for names)
  - `addr_tok_jaccard` (Jaccard similarity coefficient for addresses)
- **Character-Level Similarity:**
  - `name_jaro_winkler` (Jaro-Winkler ratio using RapidFuzz)
  - `addr_jaro_winkler` (Jaro-Winkler ratio using RapidFuzz)
- **Agreement & Metadata:**
  - `country_agree` (Country code equality)
  - `postal_agree` (Postal code equality)
  - `name_len_diff` (Absolute difference in name lengths)
  - `address_len_diff` (Absolute difference in address lengths)

### Labeling
Labels were attached deterministically by joining the candidate space against `train_ground_truth_pairs.parquet`.
- `match = 1` for true pairs existing in ground truth.
- `match = 0` otherwise.

## 10,000 S1 Benchmark Results

The pipeline successfully generated features for the entire Phase 1 candidate output (10k sample benchmark) in a single pass.

* **Total Candidates Processed:** 9,803,025
* **Execution Time:** ~28.3 seconds
* **Memory Status:** 100% Safe (processed completely in-memory using vectorized `polars` expressions).
* **Positive Examples (`match=1`):** 29,331
* **Negative Examples (`match=0`):** 9,773,694
* **Class Distribution:** ~0.299% Positive

## Feature Validation
A comprehensive validation script (`src/validate_features.py`) was executed on the generated dataset with the following results:

1. **Schema & Data Quality:**
   - All expected feature columns exist.
   - 0 missing/NaN/null values across all columns.
   - 0 duplicate candidate pairs.
   - Labels are strictly binary (`[0, 1]`).
2. **Feature Ranges:**
   - Jaccard and Jaro-Winkler similarity features are strictly bounded between `[0.0, 1.0]`.
   - Agreement features are strictly binary (`[0, 1]`).
   - Length difference features are sensible (non-negative integers).
3. **Feature Usefulness (Separability):**
   - 12 out of 13 numeric features show clear separability between positive and negative examples.
   - `name_tok_jaccard`: Pos Mean = 0.7716 | Neg Mean = 0.1965
   - `addr_tok_jaccard`: Pos Mean = 0.6429 | Neg Mean = 0.0254
   - `name_jaro_winkler`: Pos Mean = 0.9007 | Neg Mean = 0.5346
   - `addr_jaro_winkler`: Pos Mean = 0.7845 | Neg Mean = 0.3498
   - *Note: `country_agree` is constant (1.0) because the Phase 1 blocker enforces country equality across all passes.*

## Deliverables
- `src/features.py`: Feature calculation logic utilizing Polars expressions and batched RapidFuzz operations.
- `src/build_features.py`: Pipeline executor for constructing `outputs/features_train_sample.parquet`.
- `tests/test_features.py`: Comprehensive test suite verifying feature types, logical correctness, and null-handling.
- `src/validate_features.py`: Final validation script ensuring data quality and feature bounds.

## Conclusion
Phase 2 is **COMPLETE**. The machine-learning dataset is fully materialized and strictly validated. The features exhibit strong predictive signal (separability), and the entire generation pipeline runs in under 30 seconds on ~9.8M rows while remaining heavily optimized for memory safety.

We are ready to move to **Phase 3 (ML Classifier)** whenever you proceed.
