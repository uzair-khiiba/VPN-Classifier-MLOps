"""Global SHAP report for a model bundle: summary plot, bar plot, ranking and two local examples.

Explains every row of the held-out test set. SHAP doesn't use the labels here, and nothing is selected from these
results. Writes to reports/figures/shap/<model name>/.

Usage:
    python -m src.shap_report                                  # the served model
    python -m src.shap_report --model models/candidates/x.joblib
"""

import argparse
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODEL_PATH, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, ROOT
from src.model_io import load_bundle
from src.shap_explain import LIMITATION, POSITIVE_CLASS, ShapExplainer, global_ranking

FIGURE_DIR = ROOT / "reports" / "figures" / "shap"
BACKGROUND = "#fcfcfb"
BAR = "#2a78d6"
MAX_BEESWARM_ROWS = 5000


def report_dir(bundle: dict, base: Path = FIGURE_DIR) -> Path:
    return base / bundle["metadata"]["model_name"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_bar(ranking: pd.DataFrame, path: Path, top_k: int = 15) -> None:
    plt = _plt()
    top = ranking.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(top) + 1.3), dpi=130, facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.barh(top["feature"], top["mean_abs_shap"], height=0.62, color=BAR, edgecolor=BACKGROUND, linewidth=2)
    pad = top["mean_abs_shap"].max() * 0.015
    for i, value in enumerate(top["mean_abs_shap"]):
        ax.text(value + pad, i, f"{value:.3f}", va="center", fontsize=9, color="#0b0b0b")
    ax.set_title("SHAP importance: average size of each feature's effect", fontsize=11, loc="left", color="#0b0b0b")
    ax.set_xlabel("Mean |SHAP value| (log-odds of VPN/proxy)", fontsize=9, color="#52514e")
    ax.tick_params(length=0, labelsize=9, colors="#52514e")
    ax.grid(axis="x", color="#e9e8e4", lw=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=BACKGROUND)
    plt.close(fig)


def plot_summary(values: np.ndarray, X: pd.DataFrame, path: Path) -> None:
    import shap

    plt = _plt()
    rows = np.random.default_rng(RANDOM_STATE).permutation(len(X))[:MAX_BEESWARM_ROWS]
    shap.summary_plot(values[rows], X.iloc[rows], max_display=15, show=False, plot_size=(9, 6.5))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP summary: right of 0 pushes toward VPN/proxy, left pushes away", fontsize=11, loc="left")
    fig.axes[0].set_xlabel("SHAP value (log-odds of VPN/proxy)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_waterfall(explainer: ShapExplainer, row: dict, title: str, path: Path) -> None:
    import shap

    plt = _plt()
    frame = pd.DataFrame([row])[explainer.features]
    values, base = explainer.values(frame)
    explanation = shap.Explanation(values=values[0], base_values=base, data=frame.iloc[0].to_numpy(),
                                   feature_names=explainer.features)
    shap.plots.waterfall(explanation, max_display=12, show=False)
    fig = plt.gcf()
    fig.axes[0].set_title(title, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _permutation_agreement(ranking: pd.DataFrame, model_id: str) -> float | None:
    """Rank correlation with the saved permutation importance, when it describes this same model."""
    path = MODELS_DIR / "feature_importance.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if saved.get("model") != model_id:
        return None
    joined = ranking.merge(pd.DataFrame(saved["importance"]), on="feature")
    return float(joined["mean_abs_shap"].corr(joined["f1_drop"], method="spearman"))


def build_report(bundle: dict, X: pd.DataFrame, out_dir: Path, examples: dict[str, dict] | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    explainer = ShapExplainer(bundle)
    X = X[explainer.features]
    values, base = explainer.values(X)
    ranking = global_ranking(values, explainer.features)
    ranking.to_csv(out_dir / "shap_ranking.csv", index=False)
    plot_bar(ranking, out_dir / "shap_bar.png")
    plot_summary(values, X, out_dir / "shap_summary.png")
    for name, row in (examples or {}).items():
        plot_waterfall(explainer, row, f"Why the model scored this record: {name}", out_dir / f"local_{name}.png")
    meta = bundle["metadata"]
    model_id = f"{meta['model_name']}-{meta['version']}"
    report = {"model": model_id, "explainer": explainer.explainer_name, "method": explainer.detail,
              "positive_class": POSITIVE_CLASS, "units": "log-odds of VPN/proxy", "base_value": base,
              "rows_explained": len(X), "data": "held-out test set, labels not used",
              "additivity_checked": True, "shap_version": version("shap"),
              "scikit_learn_version": version("scikit-learn"),
              "generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "limitation": LIMITATION,
              "spearman_with_permutation_importance": _permutation_agreement(ranking, model_id),
              "ranking": ranking.round(6).to_dict(orient="records")}
    (out_dir / "shap_global.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    bundle = load_bundle(args.model)
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    examples = {path.stem: json.loads(path.read_text()) for path in sorted((ROOT / "examples").glob("*.json"))}
    out_dir = args.out_dir or report_dir(bundle)
    report = build_report(bundle, test, out_dir, examples)
    print(f"{report['model']}: {report['explainer']} on {report['rows_explained']:,} rows -> {out_dir}")
    print(pd.DataFrame(report["ranking"]).head(10).to_string(index=False))
    print("Spearman with permutation importance:", report["spearman_with_permutation_importance"])


if __name__ == "__main__":
    main()
