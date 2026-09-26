# Phase 4: Threshold Selection & Final Entity Matching Report

## Objective
The final phase of the pipeline aimed to map the pairwise match probabilities emitted by the Phase 3 XGBoost classifier into a final set of entity match lists. We required a robust probability threshold that maximizes F0.5 (weighting precision > recall to avoid false merges) and strict formatting enforcement according to the competition's submission constraints.

## Methodology

### 1. Threshold Selection (`src/select_threshold.py`)
To isolate the optimal decision boundary, we scanned probabilities between 0.1 and 0.99 purely on the **held-out validation set**. The model produced extremely confident predictions natively (owing to `scale_pos_weight`), shifting the ideal probability threshold significantly higher.

**Results on Validation Split (2M pairs):**
| Threshold | Precision | Recall | F0.5 Score |
| :--- | :--- | :--- | :--- |
| 0.10 | 0.2144 | 0.9987 | 0.2543 |
| 0.50 | 0.4210 | 0.9938 | 0.4758 |
| 0.90 | 0.5931 | 0.9834 | 0.6443 |
| 0.95 | 0.7135 | 0.9681 | 0.7531 |
| **0.99** | **0.8026** | **0.9399** | **0.8267** |

**Selection:** The highest F0.5 metric was realized at an extremely strict threshold of **0.99**. This correctly optimizes for precision (fewer false positive merges) without losing significant recall. 

### 2. Output Formatting (`src/generate_submission.py`)
Applying the 0.99 threshold to the entire 10k candidate development dataset, we enforced the exact challenge formatting constraints:
- Included exactly one row per S1 entity from the source block.
- Singletons/unmatched entities safely received an empty match field.
- Candidate pairs were successfully rolled up into a comma-separated column.
- Duplicates were omitted, maintaining strict compliance.

## Validation Results
We ran the official evaluation script `utils/validate_submission.py` across our generated outputs:

```text
ML Challenge 2026 — submission validator
  required S1 entities: 10000
  matching_results.tsv: 10000 rows (524 empty, 9476 non-empty).
  candidate_pairs.tsv: 10000 rows (6 empty, 9994 non-empty).

PASS — no blocking issues found. Safe to submit.
```
The output format perfectly matches the challenge specification. The prediction pipeline is highly deterministic and respects the local memory envelope.

## Deliverables
- `src/select_threshold.py`: Evaluates and selects the F0.5-optimal probability threshold.
- `src/generate_submission.py`: Generates the properly formatted TSV files.
- `outputs/matching_results.tsv`: The primary submission file containing final matched S2/S3 IDs.
- `outputs/candidate_pairs.tsv`: The candidate pool (required for diagnostic submission).

## Conclusion
Phase 4 is **COMPLETE**. The End-to-End Business Entity Resolution pipeline is successfully built, trained, scored, and validated on the 10,000 S1 entity benchmark while strictly adhering to safety limits on compute. 

The system achieves a robust validation F0.5 score of ~0.826 and produces competition-ready output formats natively.
