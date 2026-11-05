"""Phase 7: build the comparison tables from saved results. Nothing is refitted, and no choice is made here.

A verdict compares test F1 with the baseline using the paired bootstrap interval. Test scores of
runs that were not selected are retrospective: use them to learn, never to pick a model.

Usage: python -m experiments.build_comparison
"""

import csv
import json

import numpy as np
from sklearn.metrics import precision_recall_curve

from experiments.common import (
    ARTIFACT_DIR,
    REPORT_DIR,
    RESULTS_DIR,
    load_records,
    load_split,
    save_records,
    test_summary,
)
from src.config import TARGET

ORDER = [("exp_e", "E: models"), ("exp_b", "B: imbalance"), ("exp_c", "C: hyperparameters"),
         ("exp_c2", "C2: follow-up"), ("exp_d", "D: features")]
COLUMNS = ["Experiment", "Run", "Precision", "Recall", "Test F1", "CV F1", "PR-AUC", "Threshold", "Fit (s)",
           "Size (KB)",
           "F1 vs baseline [95% interval]", "Verdict"]
BASELINE = "exp_e/hgb_slow"
PICKS = "selected"


def verdict(key: str, comparison: dict) -> str:
    if key == BASELINE:
        return "baseline"
    c = comparison.get(key)
    if c is None or (c["diff_low"] == 0 and c["diff_high"] == 0):
        return "same model as baseline"
    if c["diff_low"] > 0:
        return "better"
    if c["diff_high"] < 0:
        return "worse"
    return "no clear change"


def table_rows() -> list[dict]:
    comparison = json.loads((RESULTS_DIR / "test_comparison.json").read_text())["vs_baseline"]
    rows = []
    for experiment, label in ORDER:
        for rec in load_records(experiment):
            if rec.get("screening_only"):
                continue
            m, key = rec["metrics"], f"{experiment}/{rec['name']}"
            c = comparison.get(key)
            interval = f"{c['diff']:+.4f} [{c['diff_low']:+.3f}, {c['diff_high']:+.3f}]" if c else "n/a"
            rows.append({"Experiment": label, "Run": rec["name"], "Precision": f"{m['test_precision']:.3f}",
                         "Recall": f"{m['test_recall']:.3f}", "Test F1": f"{m['test_f1']:.4f}",
                         "CV F1": f"{m['cv_f1']:.4f}", "PR-AUC": f"{m['test_pr_auc']:.3f}",
                         "Threshold": f"{m['threshold']:.3f}",
                         "Fit (s)": f"{m['refit_seconds']:.0f}" if m.get("refit_seconds") else "n/a",
                         "Size (KB)": f"{m['model_size_kb']:.0f}" if m.get("model_size_kb") else "n/a",
                         "F1 vs baseline [95% interval]": interval, "Verdict": verdict(key, comparison)})
    extra = RESULTS_DIR / "exp_f.json"
    if extra.exists():
        for rec in json.loads(extra.read_text()):
            m = rec["metrics"]
            rows.append({"Experiment": "F: Ludwig", "Run": rec["name"], "Precision": f"{m['test_precision']:.3f}",
                         "Recall": f"{m['test_recall']:.3f}", "Test F1": f"{m['test_f1']:.4f}",
                         "CV F1": f"{m['val_f1']:.4f} (validation)", "PR-AUC": f"{m['test_pr_auc']:.3f}",
                         "Threshold": f"{m['threshold']:.3f}", "Fit (s)": "see report", "Size (KB)": "n/a",
                         "F1 vs baseline [95% interval]": "n/a", "Verdict": rec.get("verdict", "n/a")})
    return rows


def threshold_rows() -> list[dict]:
    rows = []
    for rec in load_records("exp_a"):
        m = rec["metrics"]
        rows.append({"Threshold": f"{m['threshold']:.3f}", "CV precision": f"{m['cv_precision']:.3f}",
                     "CV recall": f"{m['cv_recall']:.3f}", "CV F1": f"{m['cv_f1']:.4f}",
                     "CV false alarms": f"{m['cv_fp']:.0f}", "CV missed": f"{m['cv_fn']:.0f}",
                     "Test F1": f"{m['test_f1']:.4f}", "Test false alarms": f"{m['test_fp']:.0f}",
                     "Test missed": f"{m['test_fn']:.0f}"})
    return rows


def operating_points() -> list[dict]:
    """The selected model at thresholds chosen on OOF for other goals, then checked on test."""
    train, test = load_split()
    y, y_test = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    rec = next(r for r in load_records("exp_c") if r["name"] == "trees_400_no_early_stop")
    oof = np.load(ARTIFACT_DIR / f"oof_{rec['run_name']}.npz")["oof"]
    proba = np.load(ARTIFACT_DIR / f"test_proba_{rec['test_proba_key']}.npy")
    precision, recall, thresholds = precision_recall_curve(y, oof)
    precision, recall = precision[:-1], recall[:-1]
    goals = {"Best F1 (selected)": rec["metrics"]["threshold"],
             "Catch at least 90% (recall >= 0.90 on OOF)": thresholds[np.flatnonzero(recall >= 0.90)[-1]],
             "Few false alarms (precision >= 0.90 on OOF)": thresholds[np.flatnonzero(precision >= 0.90)[0]]}
    rows = []
    for goal, threshold in goals.items():
        m = test_summary(y_test, proba, float(threshold))
        rows.append({"Goal": goal, "Threshold": f"{threshold:.3f}", "Test precision": f"{m['test_precision']:.3f}",
                     "Test recall": f"{m['test_recall']:.3f}", "Test F1": f"{m['test_f1']:.4f}",
                     "False alarms": str(m["test_fp"]), "Missed": str(m["test_fn"])})
    return rows


def markdown(rows: list[dict]) -> str:
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    rows = table_rows()
    with open(REPORT_DIR / "comparison_table.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    operating = operating_points()
    (REPORT_DIR / "comparison_tables.md").write_text(
        "# Comparison tables\n\nGenerated by `python -m experiments.build_comparison`.\n\n"
        "Verdicts compare test F1 with the baseline. Only the selected run's test score is a fair estimate. "
        "With this many comparisons, a few \"better\" or \"worse\" verdicts are expected by chance.\n\n"
        "## All scored runs\n\n" + markdown(rows) +
        "\n\n## Experiment A: thresholds on one model\n\n" + markdown(threshold_rows()) +
        "\n\n## Selected model at other operating points\n\n" + markdown(operating) + "\n", encoding="utf-8")
    print(markdown(operating))
    for name in ("better", "worse", "no clear change"):
        print(f"{name}: {[r['Run'] for r in rows if r['Verdict'] == name]}")

    project = REPORT_DIR.parent.parent
    figures = sorted(p.relative_to(project).as_posix() for p in (REPORT_DIR / "figures").glob("*.png"))
    base = "reports/experiments"
    selected = next(r for r in load_records("exp_c") if r["name"] == "trees_400_no_early_stop")["metrics"]
    save_records("summary", [{
        "run_name": "summary_selected_model_and_figures", "experiment": "summary",
        "change": "selected model, figures and comparison tables",
        "params": {"selected_run": json.loads((RESULTS_DIR / "selections.json").read_text())["overall"]
                   ["selected_run"]},
        "metrics": {k: selected[k] for k in ("cv_f1", "threshold", "test_f1", "test_precision", "test_recall",
                                             "test_pr_auc", "model_size_kb")},
        "artifacts": [*figures, f"{base}/comparison_table.csv", f"{base}/comparison_tables.md",
                      f"{base}/results/selections.json", f"{base}/results/test_comparison.json",
                      f"{base}/results/selection_audit.json",
                      f"{base}/artifacts/selected_models/exp_c_trees_400_no_early_stop.joblib"],
    }])


if __name__ == "__main__":
    main()
