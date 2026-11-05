"""Adds PR-AUC to Ludwig, which only offers ROC-AUC for binary outputs. Runs in .venv-ludwig only.

Importing this module registers the metric, and adds its name to Ludwig's list of valid hyperopt metrics. Ludwig
copies that list once while loading, so it has to be updated directly, whichever module was imported first.

Ray runs trials in other processes, so RegisterMetrics is passed to the search as a callback. Unpickling it in a
worker imports this module, which does all of this there too.
"""

import torch
from hyperopt import rand
from ludwig.api import LudwigModel  # noqa: F401  (loads all of Ludwig, so the schema below exists)
from ludwig.callbacks import Callback
from ludwig.config_validation.validation import get_schema
from ludwig.constants import BINARY, MAXIMIZE, MODEL_ECD, PROBABILITIES
from ludwig.hyperopt import execution
from ludwig.modules.metric_modules import LudwigMetric
from ludwig.modules.metric_registry import register_metric
from ludwig.schema.hyperopt import HyperoptConfig
from marshmallow.validate import OneOf
from torchmetrics.classification import BinaryAveragePrecision

PR_AUC = "pr_auc"


@register_metric(PR_AUC, [BINARY], MAXIMIZE, PROBABILITIES)
class BinaryPrAucMetric(BinaryAveragePrecision, LudwigMetric):
    """Average precision, the same definition as sklearn's average_precision_score used everywhere else here."""

    def __init__(self, **kwargs):
        super().__init__()

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        super().update(preds, target.type(torch.int8))


def _allow_in_schema() -> None:
    """get_schema caches its result, so changing the returned schema changes what later validation accepts.
    The cache treats keyword and positional calls separately, so this must call it the way check_schema does."""
    allowed = get_schema(model_type=MODEL_ECD)["properties"]["hyperopt"]["properties"]["metric"]["enum"]
    if PR_AUC not in allowed:
        allowed.append(PR_AUC)
    # Ludwig validates a second time when it loads the config, using a separate copy of the names.
    field = HyperoptConfig.Schema._declared_fields["metric"]
    for validator in field.validators:
        if isinstance(validator, OneOf) and PR_AUC not in validator.choices:
            validator.choices = [*validator.choices, PR_AUC]


_allow_in_schema()


class RegisterMetrics(Callback):
    """Does nothing itself. Its presence makes Ray workers import this module."""


class SearchPhaseRecorder:
    """Makes the startup setting take effect, and records what each suggestion actually was.

    Ludwig 0.11 drops n_initial_points when it loads a config, so the searcher would keep its default of 20. This
    passes the value directly when Ludwig creates the searcher, and stops if the searcher doesn't end up with it.
    Hyperopt falls back to rand.suggest until enough trials have finished, so counting those calls shows which
    suggestions were random and which came from TPE. It's measured, not inferred from the trial's position.
    """

    def __init__(self, n_initial_points: int, seed: int):
        self.n_initial_points, self.seed = n_initial_points, seed
        self.phases: list[str] = []
        self.n_startup_jobs: int | None = None
        self._random_calls = 0

    def install(self) -> None:
        original_create, original_random = execution.tune.create_searcher, rand.suggest

        def counted_random(*args, **kwargs):
            self._random_calls += 1
            return original_random(*args, **kwargs)

        def create_searcher(kind, **kwargs):
            kwargs.update(n_initial_points=self.n_initial_points, random_state_seed=self.seed)
            searcher = original_create(kind, **kwargs)
            self.n_startup_jobs = searcher.algo.keywords["n_startup_jobs"]
            if self.n_startup_jobs != self.n_initial_points:
                raise RuntimeError(f"searcher has n_startup_jobs={self.n_startup_jobs}, not {self.n_initial_points}")
            searcher.algo = self._counted(searcher.algo)
            return searcher

        rand.suggest = counted_random
        execution.tune.create_searcher = create_searcher

    def _counted(self, algorithm):
        def suggest(*args, **kwargs):
            before = self._random_calls
            result = algorithm(*args, **kwargs)
            self.phases.append("random startup" if self._random_calls > before else "adaptive TPE")
            return result

        return suggest
