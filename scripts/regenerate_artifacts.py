#!/usr/bin/env python3
"""
Regenerate results/*.json and figures/*.png from the current processed data and models.

Runs (from repo root, with PYTHONPATH=.):
  1. XGBoost ablations → full_model_comparison.json, combined_model_results.json, kams_only_model_results.json
  2. Multisource model benchmark → multisource_model_comparison.json + PNG
  3. SHAP → shap_*.png, shap_report.json
  4. Pipeline figures + multiclass error_analysis.json (confusion, PCA, confidence, etc.)
  5. LLM verdicts (template mode; no GPU/Ollama required) → results/verdicts/*

Usage:
  python scripts/regenerate_artifacts.py
  python scripts/regenerate_artifacts.py --skip-verdicts
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def python_exe() -> str:
    v = ROOT / ".venv" / "bin" / "python"
    if v.exists():
        return str(v)
    return sys.executable


def run_step(label: str, argv: list[str]) -> None:
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
    r = subprocess.run([python_exe(), *argv], cwd=str(ROOT), env=env)
    if r.returncode != 0:
        raise SystemExit(f"Step failed ({label}): exit {r.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-verdicts", action="store_true", help="Skip LLM verdict generation")
    args = ap.parse_args()

    steps = [
        ("XGBoost KAMs only", ["models/xgboost_kams_only.py"]),
        ("XGBoost with KAMs", ["models/xgboost_with_kams.py"]),
        ("XGBoost full (14 features)", ["models/xgboost_full.py"]),
        ("Multisource model benchmark", ["models/evaluate_multisource_models.py"]),
        ("All-features benchmark (21 features)", ["models/xgboost_all_features.py"]),
        ("SHAP explainability", ["models/shap_explainability.py"]),
        ("Pipeline figures + error_analysis.json", ["models/generate_pipeline_figures.py"]),
    ]
    if not args.skip_verdicts:
        steps.append(("LLM verdicts (template)", ["models/llm_verdict.py", "template"]))

    for label, argv in steps:
        run_step(label, argv)

    print("\nDone. Check figures/ and results/ (and results/verdicts/ if not skipped).")


if __name__ == "__main__":
    main()
