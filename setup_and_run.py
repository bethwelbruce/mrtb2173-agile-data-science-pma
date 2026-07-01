#!/usr/bin/env python3
"""
setup_and_run.py
================
One-click setup and test runner for MRTB 2173 Agile Data Science PMA.
Run from the project root:  python setup_and_run.py
"""
import subprocess, sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))

def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] '{label}' failed (exit {result.returncode}). Fix above errors before continuing.")
        sys.exit(result.returncode)
    print(f"[OK] {label}")

def step(n, total, msg):
    print(f"\n\033[1;34m[{n}/{total}] {msg}\033[0m")

STEPS = [
    ("pip install -r requirements.txt",           "Installing Python dependencies"),
    ("python src/generate_data.py",               "Generating synthetic Telco dataset"),
    ("python src/eda.py",                         "Running EDA — saving 4 charts to outputs/"),
    ("python src/data_quality.py",                "Running data quality checks & cleaning"),
    ("python src/validate.py",                    "Automated validation (9 checks)"),
    ("python src/train_models.py",                "Training baseline + improved models"),
    ("python -m pytest tests/ -v",                "Running 8 automated pytest tests"),
    ("python src/monitoring.py",                  "Running monitoring & drift analysis"),
]

print("\n\033[1;32m MRTB 2173 Agile Data Science — Full Pipeline\033[0m")
print(" All steps run from:", ROOT)

for i, (cmd, label) in enumerate(STEPS, 1):
    step(i, len(STEPS), label)
    run(cmd, label)

print(f"\n{'='*60}")
print("  ALL STEPS COMPLETED SUCCESSFULLY")
print(f"{'='*60}")
print("\nOutputs saved:")
print("  outputs/   — 6 charts + model_metrics.json + monitoring_results.json")
print("  models/    — baseline_logistic.pkl, improved_gb.pkl, scaler.pkl")
print("  data/      — telco_churn.csv, telco_churn_clean.csv")
print("\nTo launch the dashboard:")
print("  streamlit run src/dashboard.py")
print("\nTo run a single step:")
print("  python src/eda.py")
print("  python src/validate.py")
print("  python -m pytest tests/ -v")
