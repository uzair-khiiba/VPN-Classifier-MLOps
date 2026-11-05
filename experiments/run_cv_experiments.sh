#!/usr/bin/env bash
# Runs the cross-validation stage of experiments A-E, one after another. Never touches the test set.
# Usage (Git Bash, from the project root): bash experiments/run_cv_experiments.sh
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8 MLFLOW_DISABLE_AGENT_HINT=1
for module in exp_a_threshold exp_e_models exp_b_imbalance exp_d_features exp_c_hyperparameters exp_c2_followup; do
  echo "=== $module $(date +%T)"
  "$PY" -m "experiments.$module" || echo "!!! $module FAILED"
done
echo "=== finished $(date +%T)"
