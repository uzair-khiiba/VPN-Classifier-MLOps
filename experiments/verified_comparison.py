"""Rebuild the model comparison from saved models and predictions, and stop if anything disagrees.

Nothing trains. The test set was held back from every model below, and no choice here uses a test score:
ordering follows the metric each model was actually selected on.

Writes:
  reports/experiments/results/verified_comparison.json
  reports/experiments/results/objective_comparison.json   which validation metric best tracks F1 (CV results only)
  reports/experiments/verified_model_comparison.md

Usage: python -m experiments.verified_comparison
"""

import json

import numpy as np
import pandas as pd

from experiments.common import (
    ARTIFACT_DIR,
    REPORT_DIR,
    RESULTS_DIR,
    load_records,
    load_split,
    split_fingerprint,
    test_summary,
)
from experiments.package_candidate import CANDIDATE_DIR, CANDIDATE_RUN
from src.config import ID_COLUMN, MODEL_PATH, MODELS_DIR, TARGET
from src.features import FEATURES
from src.model_io import load_bundle

TOLERANCE = 1e-4  # model_meta.json rounds to 4 decimals
OBJECTIVE_TRACKS = ("exp_b", "exp_c", "exp_c2", "exp_d", "exp_e")
METRICS = ("test_precision", "test_recall", "test_f1", "test_pr_auc", "test_roc_auc", "test_accuracy", "test_tp",
           "test_fp", "test_fn", "test_tn")


def _row(name: str, status: str, source: str, selection: str, threshold: float, scored: dict, cv_f1=None, val_f1=None,
         n_train=None) -> dict:
    return {"model": name, "status": status, "source": source, "selected_on": selection, "n_train": n_train,
            "cv_f1": cv_f1, "val_f1": val_f1, "threshold": threshold, **{k: scored[k] for k in METRICS}}


def _require(label: str, value: float, expected: float, tolerance: float = TOLERANCE) -> None:
    if abs(float(value) - float(expected)) > tolerance:
        raise RuntimeError(f"{label}: computed {value}, saved {expected}")


def build_rows(test: pd.DataFrame, n_train: int) -> list[dict]:
    X, y = test[FEATURES], test[TARGET].to_numpy()
    rows = []

    bundle = load_bundle(MODEL_PATH)
    meta = bundle["metadata"]
    scored = test_summary(y, bundle["model"].predict_proba(X)[:, 1], bundle["threshold"])
    _require("served test F1", scored["test_f1"], meta["metrics"]["test_f1"])
    rows.append(_row(f"{meta['model_name']} ({meta['version']})", "DEPLOYED", "models/model.joblib, rescored now",
                     "5-fold CV F1", float(bundle["threshold"]), scored, meta["metrics"]["cv_f1"],
                     n_train=meta["n_train"]))

    saved_runs = json.loads((MODELS_DIR / "model_meta.json").read_text())["all_runs"]
    for record in load_records("baseline"):
        metrics = record["metrics"]
        name = record.get("model") or record["name"]
        _require(f"{name} matches model_meta.json", metrics["test_f1"], saved_runs[name]["test_f1"])
        if name != meta["model_name"]:
            rows.append(_row(name, "baseline candidate", "results/baseline.json, reproduction of src/train.py",
                             "5-fold CV F1", metrics["threshold"], metrics, metrics["cv_f1"], n_train=n_train))

    candidate_path = CANDIDATE_DIR / f"{CANDIDATE_RUN}.joblib"
    record = next(r for r in load_records("exp_c") if r["run_name"] == CANDIDATE_RUN)
    if candidate_path.exists():
        candidate = load_bundle(candidate_path)
        scored = test_summary(y, candidate["model"].predict_proba(X)[:, 1], candidate["threshold"])
        source = "models/candidates, rescored now"
    else:
        scored, source = record["metrics"], "results/exp_c.json"
    _require("candidate test F1", scored["test_f1"], record["metrics"]["test_f1"], 1e-9)
    rows.append(_row(CANDIDATE_RUN, "best experimental, NOT deployed", source, "5-fold CV F1, simplest within margin",
                     record["metrics"]["threshold"], scored, record["metrics"]["cv_f1"], n_train=n_train))

    ludwig = load_records("exp_f")[0]
    proba = np.load(ARTIFACT_DIR / "test_proba_ludwig.npy")
    if len(proba) != len(y):
        raise RuntimeError("Ludwig predictions don't cover the same test rows")
    scored = test_summary(y, proba, ludwig["metrics"]["threshold"])
    _require("Ludwig test F1", scored["test_f1"], ludwig["metrics"]["test_f1"], 1e-3)
    rows.append(_row("ludwig_automl (first search, 20 trials)", "experiment, NOT deployed",
                     "saved predictions, rescored now", "validation ROC-AUC, one split", ludwig["metrics"]["threshold"],
                     scored, val_f1=ludwig["metrics"].get("val_f1"), n_train=int(ludwig["params"]["n_train"])))
    return rows


def objective_comparison() -> dict:
    """Across runs scored on the same folds: which threshold-free metric best predicts the F1 we care about?

    Uses cross-validation results only, never the test set. These are tree models, so this is evidence about the
    metrics on this data, rather than a measurement of Ludwig itself.
    """
    rows = []
    for track in OBJECTIVE_TRACKS:
        for record in load_records(track):
            metrics = record.get("metrics", {})
            if all(metrics.get(k) is not None for k in ("cv_f1", "cv_pr_auc", "cv_roc_auc")):
                rows.append({"run": record["run_name"],
                             **{k: metrics[k] for k in ("cv_f1", "cv_pr_auc", "cv_roc_auc")}})
    frame = pd.DataFrame(rows)
    top = frame.nlargest(10, "cv_f1")["run"]
    picks = {metric: frame.nlargest(1, metric).iloc[0] for metric in ("cv_pr_auc", "cv_roc_auc")}
    return {
        "runs_compared": len(frame), "data": "5-fold cross-validation on training IPs; test set not used",
        "spearman_with_cv_f1": {metric: float(frame[metric].corr(frame["cv_f1"], method="spearman"))
                                for metric in ("cv_pr_auc", "cv_roc_auc")},
        "top10_by_f1_also_in_top10_by": {metric: int(frame.nlargest(10, metric)["run"].isin(top).sum())
                                         for metric in ("cv_pr_auc", "cv_roc_auc")},
        "cv_f1_of_run_each_metric_would_pick": {metric: {"run": row["run"], "cv_f1": float(row["cv_f1"])}
                                                for metric, row in picks.items()},
        "best_cv_f1_available": float(frame["cv_f1"].max()),
        "recommendation": "Validation PR-AUC as the search objective. Fixed-threshold F1 was rejected: Ludwig would "
                          "measure it at 0.5, while the project tunes the threshold afterwards, and class weights "
                          "change what 0.5 means from trial to trial. PR-AUC needs no threshold and focuses on the "
                          "rare class. ROC-AUC stays as a second recorded metric.",
    }


def _fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_markdown(rows: list[dict], context: dict, objective: dict) -> None:
    lines = [
        "# Verified model comparison", "",
        "Generated by `python -m experiments.verified_comparison`, which stops if a score doesn't match its saved "
        "record. Models are listed by status, not ranked by test score.", "",
        f"All models were scored on the same {context['n_test']:,} test IPs (fingerprint `{context['fingerprint']}`), "
        f"using the same {len(FEATURES)} features and the same evaluation code.", "",
        "| Model | Status | Selected on | CV F1 | Val F1 | Threshold | Precision | Recall | Test F1 | PR-AUC | "
        "ROC-AUC | TP | FP | FN | TN |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['status']} | {r['selected_on']} | {_fmt(r['cv_f1'])} | "
                     f"{_fmt(r['val_f1'])} | {_fmt(r['threshold'])} | "
                     f"{_fmt(r['test_precision'], 3)} | {_fmt(r['test_recall'], 3)} | "
                     f"**{_fmt(r['test_f1'])}** | {_fmt(r['test_pr_auc'], 3)} | {_fmt(r['test_roc_auc'], 3)} | "
                     f"{r['test_tp']:,} | {r['test_fp']:,} | {r['test_fn']:,} | {r['test_tn']:,} |")
    rate = context["positive_rate"]
    lines += [
        "",
        "## How to read this", "",
        "- **CV F1** comes from 5-fold cross-validation on training IPs, and **Val F1** from one validation split. "
        "They were available when models were chosen. **Test** columns weren't, and only report the result.",
        "- **The deployed model and the best experimental model are different.** The experimental model scores higher, "
        "but it stays undeployed until someone approves it through `experiments.promote_model`.",
        f"- **Class imbalance.** Only {rate:.2%} of test IPs are VPN/proxy, so labelling everything \"not VPN\" scores "
        f"{1 - rate:.2%} accuracy with F1 0 and PR-AUC about {rate:.3f}. That's why accuracy isn't used here, and why "
        "ROC-AUC can look excellent while F1 is modest.",
        "- **Ludwig's comparison isn't like for like.** It trained on 94,885 IPs rather than 118,607, used one "
        "validation split rather than 5 folds, and optimised ROC-AUC in an effectively random search. This shows how "
        "one configuration did on one split. It doesn't show that Ludwig, or neural networks, are worse in general.",
        "- **The README's earlier Ludwig result (test F1 0.711)** has no saved run or predictions, so it's "
        "excluded.", "",
        "## Choosing a search objective for Ludwig", "",
        f"Across {objective['runs_compared']} runs on the same folds, using cross-validation only:", "",
        "| Metric | Spearman with CV F1 | Of the top 10 by F1, also top 10 by this metric | "
        "CV F1 of the run it would pick |",
        "|---|---:|---:|---:|",
    ]
    for metric, label in (("cv_pr_auc", "PR-AUC"), ("cv_roc_auc", "ROC-AUC")):
        lines.append(f"| {label} | {objective['spearman_with_cv_f1'][metric]:.3f} | "
                     f"{objective['top10_by_f1_also_in_top10_by'][metric]} | "
                     f"{objective['cv_f1_of_run_each_metric_would_pick'][metric]['cv_f1']:.4f} |")
    lines += ["", objective["recommendation"], ""]
    (REPORT_DIR / "verified_model_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train, test = load_split()
    baseline_meta = json.loads((RESULTS_DIR / "baseline_meta.json").read_text())
    fingerprint = split_fingerprint(train, test)
    recorded = baseline_meta.get("split_fingerprint")
    if recorded and recorded != fingerprint:
        raise RuntimeError(f"test split changed: {fingerprint}, recorded {recorded}")
    if set(train[ID_COLUMN]) & set(test[ID_COLUMN]):
        raise RuntimeError("IP overlap between train and test")
    rows = build_rows(test, len(train))
    objective = objective_comparison()
    context = {"n_test": len(test), "fingerprint": fingerprint, "positive_rate": float(test[TARGET].mean()),
               "features": FEATURES}
    (RESULTS_DIR / "verified_comparison.json").write_text(json.dumps({**context, "models": rows}, indent=2,
                                                                     default=float))
    (RESULTS_DIR / "objective_comparison.json").write_text(json.dumps(objective, indent=2))
    write_markdown(rows, context, objective)
    print(pd.DataFrame(rows)[["model", "status", "cv_f1", "val_f1", "test_precision", "test_recall", "test_f1",
                              "test_pr_auc", "test_roc_auc", "threshold"]].round(4).to_string(index=False))
    print(json.dumps(objective, indent=1))


if __name__ == "__main__":
    main()
