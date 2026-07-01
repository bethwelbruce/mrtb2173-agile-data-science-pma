import os
"""Sprint 1 - Data Quality Checks (Q1c) and Cleaning.
Reads raw data, identifies and fixes issues, saves clean file.
"""
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW   = os.path.join(BASE_DIR, 'data', 'telco_churn.csv')
CLEAN = os.path.join(BASE_DIR, 'data', 'telco_churn_clean.csv')

df = pd.read_csv(RAW)
issues = []

print("=" * 70)
print("Q1(c) DATA QUALITY CHECKS")
print("=" * 70)

# Check 1: Missing / blank values in TotalCharges
print("\n-- CHECK 1: Missing / Blank Values --")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing)
issues.append({'Issue': 'Missing Values (TotalCharges)', 'Count': int(df['TotalCharges'].isna().sum()),
               'Impact': 'Causes model training errors; distorts revenue-based features'})

# Check 2: Duplicate rows
print("\n-- CHECK 2: Duplicate Rows --")
dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes}")
issues.append({'Issue': 'Duplicate Rows', 'Count': int(dupes),
               'Impact': 'Biases model toward over-represented patterns'})

# Check 3: Data type
print("\n-- CHECK 3: TotalCharges dtype (before fix):", df['TotalCharges'].dtype)

# Check 4: Outliers
print("\n-- CHECK 4: MonthlyCharges outliers --")
q1, q3 = df['MonthlyCharges'].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
outs = df[(df['MonthlyCharges'] < low) | (df['MonthlyCharges'] > high)]
print(f"IQR bounds: [{low:.2f}, {high:.2f}] | Outliers: {len(outs)}")

# Summary
import pandas as pd
summary = pd.DataFrame(issues)
print("\n-- SUMMARY TABLE --")
print(summary.to_string(index=False))

# Fixes
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df = df.drop_duplicates()
df = df.drop(columns=['customerID'])
print(f"\nCleaned shape: {df.shape} | Missing: {df.isnull().sum().sum()}")
df.to_csv(CLEAN, index=False)
print(f"Saved: {CLEAN}")
