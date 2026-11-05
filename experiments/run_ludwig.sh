#!/usr/bin/env bash
# Experiment F: run the project's unmodified src.automl_ludwig on Windows, logging locally.
# A 2-trial check runs first, in its own MLflow experiment. The full search runs only if it passes.
# Usage (Git Bash, from the project root): bash experiments/run_ludwig.sh
PY=.venv-ludwig/Scripts/python
LOGS=reports/experiments/logs
export PYTHONIOENCODING=utf-8 MLFLOW_DISABLE_AGENT_HINT=1
export MLFLOW_TRACKING_URI="sqlite:///$(pwd -W)/reports/experiments/artifacts/zainab_mlflow.db"

"$PY" experiments/patch_ludwig_windows.py || exit 1

rm -rf ludwig_results  # earlier hyperopt state would make Ray try to resume old trials
export MLFLOW_EXPERIMENT=zainab_ludwig_check
echo "check started $(date +%T)"
"$PY" -m src.automl_ludwig --num-samples 2 > "$LOGS/ludwig_check.log" 2>&1
status=$?
echo "check finished $(date +%T) exit=$status"
grep -E "^Best trial|^Ludwig best" "$LOGS/ludwig_check.log" | cut -c1-300
[ $status -eq 0 ] || exit $status

rm -rf ludwig_results
export MLFLOW_EXPERIMENT=zainab_experiments
echo "full search started $(date +%T)"
"$PY" -m src.automl_ludwig > "$LOGS/ludwig_full.log" 2>&1
status=$?
echo "full search finished $(date +%T) exit=$status"
grep -E "^Ludwig AutoML|^Best trial|^Ludwig best" "$LOGS/ludwig_full.log" | cut -c1-400
exit $status
