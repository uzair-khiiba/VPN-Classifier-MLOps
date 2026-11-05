"""Audit: could any test-set result have influenced a selection? Also checks results are complete.

Read-only apart from writing its report and a backup archive. Any failed check is printed and
makes the script exit with an error.

Checks:
  * Timing: every cached prediction predates selections.json, which predates every test prediction.
  * Independence: selections are recomputed with all test metrics removed, then with them scrambled.
  * Provenance: every CV metric and threshold is recomputed from its cached out-of-fold file.
  * Code: no CV script reads the test table.
  * Completeness: every expected run exists, with CV metrics, and test metrics where expected.

Usage: python -m experiments.audit_selection
"""

import ast
import hashlib
import json
import random
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np

import experiments.final_test_eval as final
from experiments.common import ARTIFACT_DIR, REPORT_DIR, RESULTS_DIR, load_records, load_split, split_fingerprint
from experiments.runner import build_model
from src.config import ROOT, TARGET
from src.train import best_f1_threshold

CV_SCRIPTS = ["exp_a_threshold", "exp_b_imbalance", "exp_c_hyperparameters", "exp_c2_followup", "exp_d_features",
              "exp_e_models"]
EXPECTED_RUNS = {"exp_a": 10, "exp_b": 7, "exp_c": 40, "exp_c2": 8, "exp_d": 12, "exp_e": 5, "exp_f": 1, "baseline": 5}
EXPERIMENTS = ["exp_a", "exp_b", "exp_c", "exp_c2", "exp_d", "exp_e"]
SHARED_OOF = {"exp_a_tuned_threshold", "exp_b_class_weight_balanced", "exp_c_baseline", "exp_d_all_features",
              "exp_e_hgb_slow"}
results: list[tuple[bool, str]] = []


def check(passed: bool, message: str) -> None:
    results.append((bool(passed), message))
    print(f"[{'PASS' if passed else 'FAIL'}] {message}", flush=True)


def stamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")


def check_timing() -> None:
    locked = RESULTS_DIR / "selections.json"
    oof_files = list(ARTIFACT_DIR.glob("oof_*.npz"))
    test_files = list(ARTIFACT_DIR.glob("test_proba_*.npy"))
    newest_oof = max(oof_files, key=lambda p: p.stat().st_mtime)
    oldest_test = min(test_files, key=lambda p: p.stat().st_mtime)
    check(newest_oof.stat().st_mtime < locked.stat().st_mtime,
          f"newest of {len(oof_files)} OOF files ({stamp(newest_oof)}) predates the lock ({stamp(locked)})")
    check(locked.stat().st_mtime <= oldest_test.stat().st_mtime,
          f"the lock ({stamp(locked)}) predates the first of {len(test_files)} test predictions ({stamp(oldest_test)})")
    log = (REPORT_DIR / "logs" / "final_test_eval.log").read_text(encoding="utf-8", errors="replace")
    check(log.count("selections locked before test scoring") == 1, "the test stage log shows exactly one run")


def strip_test(records):
    if not isinstance(records, list):
        return records
    clean = json.loads(json.dumps(records))
    for rec in clean:
        rec["metrics"] = {k: v for k, v in rec["metrics"].items() if not k.startswith("test_")}
        rec.pop("test_proba_key", None)
    return clean


def scramble_test(records):
    clean = json.loads(json.dumps(records))
    rng = random.Random(0)
    for rec in clean:
        for key in rec["metrics"]:
            if key.startswith("test_"):
                rec["metrics"][key] = rng.random()
    return clean


def check_independence() -> None:
    locked = json.loads((RESULTS_DIR / "selections.json").read_text())
    with mock.patch.object(final, "load_records", lambda e: strip_test(load_records(e))):
        without_test = final.selections()
    check(without_test == locked, "selections are identical when every test metric is removed")
    with mock.patch.object(final, "load_records", lambda e: scramble_test(load_records(e))):
        scrambled = final.selections()
    check(scrambled == locked, "selections are identical when every test metric is replaced with random numbers")

    pool = [r for e in ("exp_b", "exp_c", "exp_c2") for r in load_records(e) if not r.get("screening_only")]
    by_test = max(pool, key=lambda r: r["metrics"]["test_f1"])
    chosen = next(r for r in pool if r["run_name"] == locked["overall"]["selected_run"])
    check(by_test["run_name"] != chosen["run_name"],
          f"the selected run ({chosen['run_name']}, test F1 {chosen['metrics']['test_f1']:.4f}) is not the best on "
          f"test ({by_test['run_name']}, {by_test['metrics']['test_f1']:.4f})")


def check_provenance(y: np.ndarray) -> None:
    mismatches, verified = [], 0
    for experiment in EXPERIMENTS:
        for rec in load_records(experiment):
            if experiment == "exp_a" and rec["name"].startswith("threshold_"):
                continue
            name = "hgb_slow_base" if rec["run_name"] in SHARED_OOF else rec["run_name"]
            oof = np.load(ARTIFACT_DIR / f"oof_{name}.npz")["oof"]
            threshold, f1 = best_f1_threshold(y, oof)
            good = (len(oof) == len(y) and oof.min() >= 0 and oof.max() <= 1
                    and abs(threshold - rec["metrics"]["threshold"]) < 1e-12
                    and abs(f1 - rec["metrics"]["cv_f1"]) < 1e-12)
            verified += 1
            if not good:
                mismatches.append(rec["run_name"])
    check(not mismatches, f"{verified} thresholds and CV F1 scores recomputed from training labels and OOF files "
                          f"match exactly {mismatches or ''}")

    specs = {e: next(r for r in load_records(e) if r["run_name"] in SHARED_OOF)["spec"] for e in EXPERIMENTS
             if e != "exp_c2"}
    params = [build_model(spec).get_params() for spec in specs.values()]
    check(all(p == params[0] for p in params), f"the 5 runs sharing one OOF file have identical model settings "
                                               f"({', '.join(specs)})")


def check_code() -> None:
    for script in CV_SCRIPTS:
        source = (ROOT / "experiments" / f"{script}.py").read_text(encoding="utf-8")
        names = {n.id for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Name)}
        check("test" not in names and "y_test" not in names and "X_test" not in names,
              f"{script}.py never holds the test table in a variable")


def check_split(train, test) -> None:
    fingerprint = split_fingerprint(train, test)
    stored = json.loads((RESULTS_DIR / "baseline_meta.json").read_text())["split_fingerprint"]
    check(fingerprint == stored == "7a8a6674be67b514", f"the split is unchanged ({fingerprint})")


def check_completeness() -> None:
    for experiment, expected in EXPECTED_RUNS.items():
        records = load_records(experiment)
        check(len(records) == expected, f"{experiment}: {len(records)} of {expected} runs present")
        check(len({r["run_name"] for r in records}) == len(records), f"{experiment}: run names are unique")
        needs_test = [r for r in records if not r.get("screening_only")]
        missing = [r["run_name"] for r in needs_test if r["metrics"].get("test_f1") is None]
        check(not missing, f"{experiment}: all {len(needs_test)} scored runs have test metrics {missing or ''}")
        bad = [r["run_name"] for r in records for v in r["metrics"].values()
               if isinstance(v, float) and not np.isfinite(v)]
        check(not bad, f"{experiment}: no NaN or infinite metrics {bad or ''}")


def make_backup() -> dict:
    """A verified archive of everything needed to rebuild the reports. It stays on this disk."""
    backup_dir = ARTIFACT_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    name = backup_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    files = [*RESULTS_DIR.glob("*.json"), *ARTIFACT_DIR.glob("oof_*.npz"), *ARTIFACT_DIR.glob("test_proba_*.npy"),
             *(ROOT / "configs" / "experiments").glob("*.yaml")]
    digests = {}
    with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            arcname = path.relative_to(ROOT).as_posix()
            archive.write(path, arcname)
            digests[arcname] = hashlib.sha256(path.read_bytes()).hexdigest()
        archive.writestr("MANIFEST.json", json.dumps(digests, indent=2))
    with zipfile.ZipFile(name) as archive:
        same = all(hashlib.sha256(archive.read(a)).hexdigest() == d for a, d in digests.items())
    check(same, f"backup verified: {len(digests)} files in {name.relative_to(ROOT).as_posix()} "
                f"({name.stat().st_size // 1024} KB)")
    return {"path": name.relative_to(ROOT).as_posix(), "n_files": len(digests)}


def main() -> None:
    train, test = load_split()
    y = train[TARGET].to_numpy()
    check_split(train, test)
    check_timing()
    check_independence()
    check_provenance(y)
    check_code()
    check_completeness()
    backup = make_backup()
    failed = [m for ok, m in results if not ok]
    report = {"passed": len(results) - len(failed), "failed": failed, "backup": backup,
              "checks": [{"passed": ok, "check": m} for ok, m in results]}
    (RESULTS_DIR / "selection_audit.json").write_text(json.dumps(report, indent=2))
    print(f"\n{report['passed']} passed, {len(failed)} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
