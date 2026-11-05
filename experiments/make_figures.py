"""Draw the report figures from saved results. Nothing is refitted.

Colours are the validated categorical palette already used by src/explain.py (blue first).
Every series is labelled directly, so identity never depends on colour alone.

Usage: python -m experiments.make_figures
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from experiments.common import ARTIFACT_DIR, FIGURES_DIR, RESULTS_DIR, load_records, load_split  # noqa: E402
from src.config import TARGET  # noqa: E402
from src.explain import plot_importance  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e9e8e4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PALE = "#a6c8ee"


def new_axes(width=8.0, height=4.5):
    fig, ax = plt.subplots(figsize=(width, height), dpi=130, facecolor=SURFACE)
    style(ax)
    return fig, ax


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(length=0, labelsize=9, colors=MUTED)
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def finish(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", FIGURES_DIR / name)


def bars(rows: list[tuple[str, float, bool]], title: str, xlabel: str, name: str) -> None:
    """Horizontal ranked bars; highlighted rows are solid, the rest pale."""
    labels, values, marks = zip(*rows[::-1], strict=True)
    fig, ax = new_axes(height=0.42 * len(rows) + 1.4)
    ax.barh(labels, values, height=0.62, color=[BLUE if m else PALE for m in marks], edgecolor=SURFACE, linewidth=2)
    for i, value in enumerate(values):
        ax.text(value + 0.004, i, f"{value:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(values) * 1.1)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_title(title, fontsize=11, loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    finish(fig, name)


def threshold_figure(y: np.ndarray) -> None:
    oof = np.load(ARTIFACT_DIR / "oof_hgb_slow_base.npz")["oof"]
    precision, recall, thresholds = precision_recall_curve(y, oof)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    tuned = thresholds[np.argmax(f1)]
    fig, ax = new_axes()
    for values, color, label, x in ((precision, BLUE, "Precision", 0.36), (recall, ORANGE, "Recall", 0.36),
                                    (f1, AQUA, "F1", 0.25)):
        ax.plot(thresholds, values, color=color, lw=2, label=label)
        ax.text(x, values[np.searchsorted(thresholds, x)] + 0.025, label, color=INK, fontsize=9)
    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(0.505, 0.04, "default 0.5", fontsize=9, color=MUTED)
    ax.axvline(tuned, color=MUTED, lw=1, ls="--")
    ax.text(tuned - 0.01, 0.04, f"tuned {tuned:.2f}", fontsize=9, color=MUTED, ha="right")
    ax.set(xlim=(0, 1), ylim=(0, 1.02))
    ax.set_xlabel("Decision threshold", fontsize=9, color=MUTED)
    ax.set_title("Experiment A: raising the threshold trades recall for precision (out-of-fold, hgb_slow)",
                 fontsize=11, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1, -0.04), ncol=3, labelcolor=MUTED)
    finish(fig, "exp_a_threshold_tradeoff.png")


def imbalance_figure() -> None:
    labels = {"oversample_1to1": "Oversample 1:1 (ran 600 trees)",
              "oversample_1to10": "Oversample 1:10 (ran 600 trees)",
              "no_weighting": "No weighting (136 trees)",
              "oversample_1to1_no_early_stop": "Oversample 1:1, held to 94 trees",
              "undersample_1to10": "Undersample 1:10 (115 trees)",
              "class_weight_balanced": "Class weights, baseline (97 trees)",
              "undersample_1to1": "Undersample 1:1 (ran 600 trees)"}
    records = sorted(load_records("exp_b"), key=lambda r: -r["metrics"]["cv_f1"])
    rows = [(labels[r["name"]], r["metrics"]["cv_f1"], r["name"] == "class_weight_balanced") for r in records]
    bars(rows, "Experiment B: cross-validated F1 by imbalance method (solid = baseline)", "CV F1",
         "exp_b_imbalance.png")


def trees_figure() -> None:
    results = {r["name"]: r["metrics"] for e in ("exp_c", "exp_c2") for r in load_records(e)}
    baseline = results["baseline"]["cv_f1"]
    series = {
        "Class weights": ([100, 200, 400, 600, 800], ["trees_100_no_early_stop", "trees_200_no_early_stop",
                                                      "trees_400_no_early_stop", "balanced_trees_600",
                                                      "balanced_trees_800"], BLUE, "o"),
        "No weighting": ([200, 400, 600], ["unweighted_trees_200", "unweighted_trees_400", "unweighted_trees_600"],
                         ORANGE, "s"),
        "Oversample 1:1": ([200, 400, 600], ["oversample_1to1_trees_200", "oversample_1to1_trees_400",
                                             "oversample_1to1_trees_600"], AQUA, "^"),
    }
    fig, ax = new_axes()
    for label, (trees, names, color, marker) in series.items():
        ax.plot(trees, [results[n]["cv_f1"] for n in names], color=color, marker=marker, markersize=8, lw=2,
                label=label, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axhline(baseline, color=MUTED, lw=1, ls="--")
    ax.text(250, baseline + 0.0008, f"baseline with early stopping (97 trees): {baseline:.4f}", color=MUTED, fontsize=9)
    ax.set_xlabel("Trees (early stopping off)", fontsize=9, color=MUTED)
    ax.set_ylabel("CV F1", fontsize=9, color=MUTED)
    ax.set_title("Follow-up: with enough trees, all three imbalance methods reach about the same F1", fontsize=11,
                 loc="left",
                 color=INK)
    ax.legend(frameon=False, fontsize=9, loc="center right", labelcolor=MUTED)
    finish(fig, "exp_c_trees_vs_f1.png")


def hyperparameter_figure() -> None:
    records = [r for r in load_records("exp_c") if not r.get("screening_only")]
    records.sort(key=lambda r: -r["metrics"]["cv_f1"])
    rows = [(r["name"], r["metrics"]["cv_f1"], r["name"] == "baseline") for r in records]
    bars(rows, "Experiment C: cross-validated F1, each configuration (solid = baseline)", "CV F1",
         "exp_c_hyperparameters.png")


def ablation_figure() -> None:
    labels = {"all_features": "All 28 features (baseline)", "no_has_shodan": "Without has_shodan",
              "no_n_tcp_ports": "Without n_tcp_ports", "inductive_aligned": "Counts from training IPs, frozen table",
              "no_asn_unique_ips": "Without asn_unique_ips", "no_shodan_ports": "Without Shodan port features",
              "no_headers_hash_shared": "Without headers_hash_shared",
              "no_fingerprint_counts": "Without fingerprint counts",
              "inductive_counts": "Counts from training IPs, off by one", "only_shodan": "Only the 7 Shodan features",
              "no_population_counts": "Without all 4 population counts", "no_shodan": "Without all 7 Shodan features"}
    records = sorted(load_records("exp_d"), key=lambda r: -r["metrics"]["cv_f1"])
    rows = [(labels[r["name"]], r["metrics"]["cv_f1"], r["name"] == "all_features") for r in records]
    bars(rows, "Experiment D: cross-validated F1 by feature set (solid = baseline)", "CV F1",
         "exp_d_feature_ablation.png")


def model_figure() -> None:
    records = sorted(load_records("exp_e"), key=lambda r: r["metrics"]["cv_f1"])
    rows = [(r["name"], r["metrics"]["cv_f1"], r["metrics"]["test_f1"]) for r in records]
    extra = RESULTS_DIR / "exp_f.json"
    if extra.exists():
        for r in json.loads(extra.read_text()):
            if r["metrics"].get("test_f1") is not None:
                rows.append((r["name"], r["metrics"]["val_f1"], r["metrics"]["test_f1"]))
    selected = next(r for r in load_records("exp_c") if r["name"] == "trees_400_no_early_stop")
    rows.append(("hgb_slow, 400 trees, no early stopping", selected["metrics"]["cv_f1"],
                 selected["metrics"]["test_f1"]))
    rows.sort(key=lambda row: row[1])
    fig, ax = new_axes(height=0.62 * len(rows) + 1.4)
    positions = np.arange(len(rows))
    ax.barh(positions + 0.19, [r[1] for r in rows], height=0.36, color=PALE, edgecolor=SURFACE, linewidth=2,
            label="Cross-validated F1 (Ludwig: validation F1)")
    ax.barh(positions - 0.19, [r[2] for r in rows], height=0.36, color=BLUE, edgecolor=SURFACE, linewidth=2,
            label="Test F1 (reported after selection)")
    for i, (_, cv, test) in enumerate(rows):
        ax.text(cv + 0.005, i + 0.19, f"{cv:.3f}", va="center", fontsize=9, color=INK)
        ax.text(test + 0.005, i - 0.19, f"{test:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(positions, [r[0] for r in rows])
    ax.set_xlim(0, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_title("Model comparison: F1 of the VPN/proxy class", fontsize=11, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.4, -0.05), ncol=2, labelcolor=MUTED)
    finish(fig, "model_comparison.png")


def pr_figure(y: np.ndarray) -> None:
    fig, ax = new_axes(height=5.0)
    for path, label, color in (("oof_hgb_slow_base.npz", "Baseline, 97 trees", PALE),
                               ("oof_exp_c_trees_400_no_early_stop.npz", "400 trees, no early stopping", BLUE)):
        precision, recall, _ = precision_recall_curve(y, np.load(ARTIFACT_DIR / path)["oof"])
        ax.plot(recall, precision, color=color, lw=2, label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1.02))
    ax.set_xlabel("Recall", fontsize=9, color=MUTED)
    ax.set_ylabel("Precision", fontsize=9, color=MUTED)
    ax.set_title("Precision-recall curves, out-of-fold on the training IPs", fontsize=11, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=MUTED)
    finish(fig, "pr_curves.png")


def explain_figures() -> None:
    data = json.loads((RESULTS_DIR / "explainability.json").read_text())
    importance = pd.DataFrame(data["permutation"])
    plot_importance(importance, FIGURES_DIR / "permutation_importance.png")
    print("wrote", FIGURES_DIR / "permutation_importance.png")
    groups = sorted(data["groups"], key=lambda g: -g["f1_drop"])
    bars([(f"{g['group']} ({g['n_features']})", g["f1_drop"], False) for g in groups],
         "Shuffling a whole feature group: drop in test F1 (features in group)", "Drop in test F1",
         "group_permutation_importance.png")
    sweeps = data["what_if"]["sweeps"]
    fig, axes = plt.subplots(1, len(sweeps), figsize=(11, 3), dpi=130, facecolor=SURFACE, sharey=True)
    for ax, (feature, sweep) in zip(axes, sweeps.items(), strict=True):
        style(ax)
        positions = range(len(sweep))
        ax.plot(positions, [s["probability"] for s in sweep], color=BLUE, lw=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.axhline(data["threshold"], color=MUTED, lw=1, ls="--")
        ax.set_xticks(list(positions), [str(s["value"]) for s in sweep])
        ax.set_title(feature, fontsize=9, loc="left", color=INK)
    axes[0].set_ylabel("Probability (dashed = threshold)", fontsize=9, color=MUTED)
    fig.suptitle("Changing one feature at a time, for one typical VPN/proxy IP", fontsize=11, x=0.01, ha="left",
                 color=INK)
    finish(fig, "what_if_one_feature.png")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train, _ = load_split()
    y = train[TARGET].to_numpy()
    threshold_figure(y)
    imbalance_figure()
    trees_figure()
    hyperparameter_figure()
    ablation_figure()
    model_figure()
    pr_figure(y)
    explain_figures()


if __name__ == "__main__":
    main()
