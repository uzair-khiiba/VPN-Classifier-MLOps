"""SHAP explanations of the VPN/proxy class, for any model bundle.

Values are in log-odds, the units the tree model adds up internally:
    base_value + sum(shap values) == log-odds of VPN/proxy, and probability = 1 / (1 + exp(-log_odds))

TreeExplainer computes exact values from the trees, and is used whenever SHAP supports the model. Otherwise this
falls back to PermutationExplainer, which estimates them by masking features with training medians. The name of the
explainer that actually ran is always reported.

SHAP values describe how this model reached a score. They don't show what causes an IP to be a VPN or proxy.
"""

import numpy as np
import pandas as pd
from scipy.special import expit

ADDITIVITY_TOLERANCE = 1e-6
POSITIVE_CLASS = 1
LIMITATION = ("SHAP values show how this model reached its score. They describe the model, not the real world, "
              "so they don't show that a feature causes an IP to be a VPN or proxy.")


class ShapUnavailableError(RuntimeError):
    """SHAP isn't installed, or can't explain this model correctly."""


def positive_index(model) -> int:
    """Position of the VPN/proxy class, checked rather than assumed."""
    classes = [int(c) for c in model.classes_]
    if classes != [0, 1]:
        raise ShapUnavailableError(f"expected a binary model with classes [0, 1], found {classes}")
    return classes.index(POSITIVE_CLASS)


def log_odds(model, X: pd.DataFrame) -> np.ndarray:
    """The model's output for the VPN/proxy class, on the scale SHAP values add up to."""
    positive_index(model)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    proba = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return np.log(proba / (1 - proba))


def _sigmoid(value: float) -> float:
    return float(expit(value))


class ShapExplainer:
    """Explains one bundle. Building it reads the trees once, so keep it and reuse it."""

    def __init__(self, bundle: dict):
        try:
            import shap
        except ImportError as exc:
            raise ShapUnavailableError("the shap package is not installed") from exc
        self.model = bundle["model"]
        self.features = list(bundle["features"])
        self.positive = positive_index(self.model)
        self._shap = shap
        try:
            self._explainer = shap.TreeExplainer(self.model)
            self.explainer_name = "TreeExplainer"
            self.detail = (f"shap.TreeExplainer, {self._explainer.feature_perturbation}, "
                           f"log-odds of class {POSITIVE_CLASS} (VPN/proxy)")
        except Exception:  # SHAP raises several types for models it can't read as trees
            background = pd.DataFrame([bundle["reference"]])[self.features]
            self._explainer = shap.PermutationExplainer(lambda data: log_odds(self.model, self._as_frame(data)),
                                                        background)
            self.explainer_name = "PermutationExplainer"
            self.detail = (f"shap.PermutationExplainer, masking with training medians, "
                           f"log-odds of class {POSITIVE_CLASS} (VPN/proxy)")

    def _as_frame(self, data) -> pd.DataFrame:
        return pd.DataFrame(data, columns=self.features)

    def values(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        """(n_rows x n_features SHAP values for the VPN/proxy class, base value). Checks they add up to the model."""
        X = X[self.features]
        if self.explainer_name == "TreeExplainer":
            raw = np.asarray(self._explainer.shap_values(X), dtype=float)
            base = np.ravel(self._explainer.expected_value)
            base = float(base[self.positive] if base.size > 1 else base[0])
        else:
            result = self._explainer(X.to_numpy(dtype=float), silent=True)
            raw, base = np.asarray(result.values, dtype=float), float(np.ravel(result.base_values)[0])
        if raw.ndim == 3:  # some SHAP versions return one matrix per class
            raw = raw[:, :, self.positive]
        error = np.abs(base + raw.sum(axis=1) - log_odds(self.model, X)).max()
        if error > ADDITIVITY_TOLERANCE:
            raise ShapUnavailableError(f"SHAP values don't add up to the model's output (off by {error:.2e})")
        return raw, base

    def explain_row(self, row: dict, top_k: int | None = None) -> dict:
        frame = pd.DataFrame([row])[self.features]
        values, base = self.values(frame)
        contributions = [
            {"feature": name, "value": float(frame.iloc[0][name]), "shap_value": float(value),
             "direction": "toward VPN/proxy" if value > 0 else "away from VPN/proxy" if value < 0 else "no effect"}
            for name, value in zip(self.features, values[0], strict=True)
        ]
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
        output = base + float(values[0].sum())
        return {"explainer": self.explainer_name, "method": self.detail, "positive_class": POSITIVE_CLASS,
                "units": "log-odds of VPN/proxy", "base_value": base, "base_probability": _sigmoid(base),
                "output_value": output, "output_probability": _sigmoid(output), "limitation": LIMITATION,
                "contributions": contributions[:top_k] if top_k else contributions}


def global_ranking(values: np.ndarray, features: list[str]) -> pd.DataFrame:
    """Features by mean absolute SHAP value, with the mean signed value showing the usual direction."""
    return (pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(values).mean(axis=0),
                          "mean_shap": values.mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False, ignore_index=True))
