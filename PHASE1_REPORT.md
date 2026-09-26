# Phase 1: Blocking and Candidate Generation Report

## Objective
The primary goal of Phase 1 was to implement a lightweight, memory-efficient deterministic blocker capable of significantly reducing the 2.2M × 10M cross-product space while retaining high true match recall. A secondary strict goal was keeping laptop CPU and memory usage strictly safe without resorting to expensive multi-pass out-of-core systems or complex TF-IDF vectorization.

## Methodology

We transitioned from a heavy, recursive out-of-core nested loop architecture to a fast, vectorized single-pass Polars approach. 

To achieve optimal candidate reduction and recall, we used a multi-key deterministic blocker combined with aggressive frequency limitations for highly generic terms:
1. **Pass A**: Exact `norm_name_clean` matching (No limit).
2. **Pass B**: Prefix4 `norm_name_clean[:4]` (`max_key_freq = 50`).
3. **Pass C**: Single Name Token (`max_key_freq = 15`). Suppressed cross-join explosion on common words like "care", "health", and "services".
4. **Pass D**: Postal Code + Name Token (`max_key_freq = 500`).
5. **Pass E**: Address Digit + Name Token (`max_key_freq = 500`). Allows entities sharing an address to correctly match even when relying on frequent generic tokens.

## 10,000 S1 Benchmark Results

A representative benchmark using a sample of 10,000 `train_source1.parquet` records was run to rigorously measure performance.

### Efficacy Metrics
* **Total Sample S1 Records Processed:** 10,000
* **Total Candidates Generated:** 9,803,025
* **Average Candidates per S1:** ~980 
* **Recall Evaluated Against:** 34,752 (applicable GT pairs)
* **Final Blocking Recall:** **84.40%** (29,331 / 34,752)

### Performance Metrics
* **Total Execution Time:** ~36 seconds
* **Memory Usage:** Stable and strictly bounded. The laptop remains highly stable with no crashes or heavy disk thrashing. Row groups were processed in chunks, aggregating in ~0.3s per chunk.

## Conclusion

Phase 1 Blocking is **COMPLETE**. The current candidate pool provides an excellent balance: high recall (84.4%) with a manageable candidate volume (~980 pairs/S1). This pool will be fed directly into Phase 2 for feature generation.

We will NOT generate the 2.2M candidates locally during development; the current 10k candidate sample is sufficient for developing and testing Phase 2 (Features) and Phase 3 (Machine Learning).
