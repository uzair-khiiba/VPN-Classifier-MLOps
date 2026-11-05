# Phase 3: Baseline reproduction

**Command:** `python -m experiments.run_baseline`, which calls the unmodified `src.train.main()` with its outputs redirected to `reports/experiments/artifacts/baseline/`.

**Result: the documented results reproduced exactly.** Every metric for all 5 models matches `models/model_meta.json` to 4 decimal places, even though this run used Python 3.12 instead of 3.11. The committed model and metadata were not changed, which was confirmed by matching SHA-256 hashes before and after.

## Results on the held-out test set

Test set: 29,652 IPs, of which 578 are VPN/proxy.

| Model | CV F1 | Test F1 | Precision | Recall | PR-AUC | ROC-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`hgb_slow`** (selected) | **0.7694** | **0.7663** | 0.733 | 0.803 | 0.797 | 0.988 | 0.901 |
| `hgb` | 0.7661 | 0.7608 | 0.732 | 0.792 | 0.781 | 0.989 | 0.915 |
| `rf` | 0.7607 | 0.7502 | 0.708 | 0.798 | 0.804 | 0.987 | 0.340 |
| `rf_shallow` | 0.7291 | 0.7085 | 0.704 | 0.713 | 0.729 | 0.987 | 0.827 |
| `logreg` | 0.5936 | 0.5968 | 0.574 | 0.621 | 0.551 | 0.973 | 0.939 |

## Confusion matrices

These were not recorded by the original pipeline.

| Model | True positives | False positives | False negatives | True negatives |
|---|---:|---:|---:|---:|
| `hgb_slow` | 464 | 169 | 114 | 28,905 |
| `hgb` | 458 | 168 | 120 | 28,906 |
| `rf` | 461 | 190 | 117 | 28,884 |
| `rf_shallow` | 412 | 173 | 166 | 28,901 |
| `logreg` | 359 | 266 | 219 | 28,808 |

In plain terms, `hgb_slow` flags 633 IPs. 464 of those are correct and 169 are false alarms, and it misses 114 real VPN/proxy IPs.

## Cost

| Model | CV + final fit, wall time | Model size | Prediction time per IP | Trees built |
|---|---:|---:|---:|---:|
| `hgb_slow` | 132 s | 287 KB | 0.007 ms | 94 |
| `hgb` | 149 s | 130 KB | 0.004 ms | 77 |
| `rf` | 95 s | 10,355 KB | 0.011 ms | 200 |
| `rf_shallow` | 138 s | 6,092 KB | 0.017 ms | 300 |
| `logreg` | 9 s | 2 KB | 0.002 ms | n/a |

Wall time includes local MLflow logging. Size is the model alone, compressed. With its metadata, the saved `hgb_slow` bundle is 294 KB, matching the README. Times were measured on a 4-core i5-8250U.

## Hyperparameters

| Model | Settings from `src/train.py` |
|---|---|
| `hgb_slow` | `learning_rate=0.05, max_iter=600, max_leaf_nodes=63, l2_regularization=1.0, class_weight="balanced"` |
| `hgb` | `learning_rate=0.1, max_iter=300, class_weight="balanced"` |
| `rf` | `n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample"` |
| `rf_shallow` | `n_estimators=300, max_depth=12, min_samples_leaf=5, class_weight="balanced_subsample"` |
| `logreg` | `log1p` then `StandardScaler`, `C=1.0, class_weight="balanced", max_iter=2000` |

## Verification

- **Positive-class metric:** `f1_score` is called with its default `pos_label=1`, and label 1 is VPN/proxy. A hand check from the matrix gives 2 × 464 / (2 × 464 + 169 + 114) = 0.7663, which matches.
- **Accuracy is not the main metric.** `logreg` scores 98.36% accuracy with F1 0.597, against 98.05% for a model that never flags anything.
- **Selection:** `hgb_slow` has the highest CV F1. The selection rule never reads test scores.

## Differences from the documentation

| Topic | Documentation | Actual |
|---|---|---|
| Metric values | As listed | Identical |
| `hgb_slow` iterations | 600 | **94 trees.** `max_iter` is only a ceiling. Scikit-learn's `early_stopping="auto"` turns on above 10,000 rows, so training stopped when the internal validation loss stopped improving |
| Training data used | All training IPs | Early stopping sets aside 10% of each fit as internal validation, so `hgb` models train on about 90% |
| `hgb` iterations | 300 | **77 trees**, for the same reason |

Experiment C measures whether this matters.
