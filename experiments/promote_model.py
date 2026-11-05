"""The only supported way to change the deployed model, and it needs an explicit decision.

Without --approve this just reports the checks and changes nothing. With --approve and a --reason, it:
  1. refuses a bundle the API couldn't serve, or with different features, or with a lower cross-validated F1
  2. archives the current model to models/archive/
  3. copies the candidate to models/model.joblib, and rewrites models/model_meta.json
  4. appends the decision to models/promotion_log.jsonl

Test scores are shown for information, and never decide anything. Restart the API afterwards to load the new model.

Usage:
    python -m experiments.promote_model models/candidates/exp_c_trees_400_no_early_stop.joblib
    python -m experiments.promote_model models/candidates/... --approve --reason "why this is being deployed"
"""

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.features import FEATURES
from src.model_io import ModelLoadError, load_bundle


def review(candidate: dict, current: dict | None) -> dict:
    """What's required to deploy, and what's only for information."""
    new = candidate["metadata"].get("metrics", {})
    old = current["metadata"].get("metrics", {}) if current else {}
    checks = {
        "features match src/features.py, in order": candidate["features"] == FEATURES,
        "threshold is between 0 and 1": 0 < float(candidate["threshold"]) < 1,
        "has cross-validated F1": "cv_f1" in new,
        "cross-validated F1 is at least the current model's": "cv_f1" in new and new["cv_f1"] >= old.get("cv_f1", 0),
    }
    return {"checks": checks, "passed": all(checks.values()),
            "information_only": {"current": {k: old.get(k) for k in ("cv_f1", "test_f1")},
                                 "candidate": {k: new.get(k) for k in ("cv_f1", "test_f1")}}}


def promote(candidate_path: Path, model_path: Path, approve: bool = False, reason: str | None = None) -> dict:
    candidate = load_bundle(candidate_path)
    current = load_bundle(model_path) if model_path.exists() else None
    result = review(candidate, current)
    result["promoted"] = False
    if not approve:
        result["message"] = "Nothing changed. Add --approve and --reason to deploy."
        return result
    if not reason or not reason.strip():
        raise ValueError("--approve needs a --reason, which is kept in the promotion log")
    if not result["passed"]:
        failed = [name for name, ok in result["checks"].items() if not ok]
        raise ValueError(f"not promoted, checks failed: {failed}")

    models_dir = model_path.parent
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    entry = {"at": stamp, "reason": reason, "to": candidate["metadata"]["model_name"],
             "to_version": candidate["metadata"]["version"]}
    if current:
        old = current["metadata"]
        archive = models_dir / "archive" / f"{old['model_name']}_{old['version']}.joblib"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            shutil.copy2(model_path, archive)
        meta_path = models_dir / "model_meta.json"
        if meta_path.exists():
            shutil.copy2(meta_path, archive.with_suffix(".meta.json"))
        entry.update({"from": current["metadata"]["model_name"], "from_version": current["metadata"]["version"],
                      "archived_to": archive.name})
    shutil.copy2(candidate_path, model_path)
    load_bundle(model_path)
    (models_dir / "model_meta.json").write_text(json.dumps(candidate["metadata"], indent=2, default=str))
    with (models_dir / "promotion_log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry) + "\n")
    result.update(promoted=True, message="Deployed. Restart the API to load it.", entry=entry)
    return result


def main() -> None:
    from src.config import MODEL_PATH

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--reason")
    args = parser.parse_args()
    try:
        result = promote(args.candidate, MODEL_PATH, args.approve, args.reason)
    except (ValueError, ModelLoadError) as exc:
        sys.exit(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
