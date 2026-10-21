"""Global explainability: which features the served model relies on most.

Permutation importance on the held-out test set: shuffle one feature at a time
(breaking its link to the label) and measure how much test F1 drops, using the
model's own tuned threshold. A big drop means the model depends on that feature.
This complements /explain in the API, which explains a single prediction.

Usage:
    python -m src.explain     # writes models/feature_importance.json and docs/feature_importance.png
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from src.config import MODEL_PATH, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, ROOT, TARGET
from src.model_io import load_bundle


def f1_at_threshold(threshold: float):
    """sklearn scorer: F1 of the positive class at a fixed probability threshold."""

    def scorer(estimator, X, y):
        return f1_score(y, estimator.predict_proba(X)[:, 1] >= threshold, zero_division=0)

    return scorer


def permutation_f1_importance(bundle: dict, X: pd.DataFrame, y, n_repeats: int = 5,
                              random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Mean and std of the F1 drop per feature, most important first."""
    X = X[bundle["features"]]
    result = permutation_importance(bundle["model"], X, y, scoring=f1_at_threshold(bundle["threshold"]),
                                    n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    return (pd.DataFrame({"feature": X.columns, "f1_drop": result.importances_mean,
                          "f1_drop_std": result.importances_std})
            .sort_values("f1_drop", ascending=False, ignore_index=True))


def plot_importance(importance: pd.DataFrame, path: Path, top_k: int = 12) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = importance.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(top) + 1.2), dpi=130, facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.barh(top["feature"], top["f1_drop"], xerr=top["f1_drop_std"], height=0.62, color="#2a78d6",
            edgecolor="#fcfcfb", linewidth=2, error_kw={"ecolor": "#8a8984", "lw": 1, "capsize": 2})
    pad = top["f1_drop"].max() * 0.02
    for i, (v, sd) in enumerate(zip(top["f1_drop"], top["f1_drop_std"], strict=True)):
        ax.text(v + sd + pad, i, f"{v:.3f}", va="center", fontsize=9, color="#0b0b0b")
    ax.set_title("Permutation importance: drop in test F1 when a feature is shuffled",
                 fontsize=11, loc="left", color="#0b0b0b")
    ax.tick_params(length=0, labelsize=9, colors="#52514e")
    ax.grid(axis="x", color="#e9e8e4", lw=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-repeats", type=int, default=5)
    args = parser.parse_args()

    bundle = load_bundle(MODEL_PATH)
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    importance = permutation_f1_importance(bundle, test, test[TARGET].to_numpy(), n_repeats=args.n_repeats)

    meta = bundle["metadata"]
    out = {"model": f"{meta['model_name']}-{meta['version']}", "metric": "test F1 drop", "n_test": len(test),
           "n_repeats": args.n_repeats,
           "importance": importance.round(4).to_dict(orient="records")}
    (MODELS_DIR / "feature_importance.json").write_text(json.dumps(out, indent=2))
    plot_importance(importance, ROOT / "docs" / "feature_importance.png")
    print(importance.head(10).to_string(index=False))
    print(f"Saved {MODELS_DIR / 'feature_importance.json'} and docs/feature_importance.png")


if __name__ == "__main__":
    main()
