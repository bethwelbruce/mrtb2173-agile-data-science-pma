import os
"""
Q3(a) – Automated Data Validation Script
Checks: missing values, duplicates, data types, value ranges.
Exit code 0 = all checks pass; non-zero = issues found.
"""
import pandas as pd
import sys

RULES = {
    'MonthlyCharges': (18.0, 120.0),
    'tenure':         (0, 72),
    'TotalCharges':   (-150.0, 9000.0),
}

def validate(path: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  AUTOMATED DATA VALIDATION: {path}")
    print(f"{'='*60}")
    df = pd.read_csv(path)
    passed = True

    # Check 1 – Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("[PASS] No missing values detected.")
    else:
        print(f"[FAIL] Missing values found:\n{missing}")
        passed = False

    # Check 2 – Duplicate rows
    dupes = df.duplicated().sum()
    if dupes == 0:
        print("[PASS] No duplicate rows.")
    else:
        print(f"[FAIL] {dupes} duplicate rows detected.")
        passed = False

    # Check 3 – Expected data types
    expected_types = {
        'SeniorCitizen': 'int64',
        'tenure': 'int64',
        'MonthlyCharges': 'float64',
    }
    for col, expected in expected_types.items():
        if col not in df.columns:
            print(f"[WARN] Column '{col}' not found; skipping type check.")
            continue
        actual = str(df[col].dtype)
        if actual == expected:
            print(f"[PASS] {col} dtype = {actual}")
        else:
            print(f"[FAIL] {col} dtype expected {expected}, got {actual}")
            passed = False

    # Check 4 – Value ranges
    for col, (lo, hi) in RULES.items():
        if col not in df.columns:
            continue
        out_of_range = df[(df[col] < lo) | (df[col] > hi)]
        if out_of_range.empty:
            print(f"[PASS] {col} values within [{lo}, {hi}].")
        else:
            print(f"[FAIL] {col}: {len(out_of_range)} rows outside [{lo}, {hi}].")
            passed = False

    # Check 5 – Churn column values
    if 'Churn' in df.columns:
        unique_vals = set(df['Churn'].unique())
        valid_vals  = {'Yes','No','0','1',0,1}
        unexpected  = unique_vals - valid_vals
        if not unexpected:
            print(f"[PASS] Churn column values are valid: {unique_vals}")
        else:
            print(f"[FAIL] Unexpected Churn values: {unexpected}")
            passed = False

    print(f"\n{'='*60}")
    if passed:
        print("  RESULT: ALL CHECKS PASSED")
    else:
        print("  RESULT: ONE OR MORE CHECKS FAILED")
    print(f"{'='*60}\n")
    return passed


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, 'data', 'telco_churn_clean.csv')
    ok = validate(path)
    sys.exit(0 if ok else 1)
