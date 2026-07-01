"""
Q3(c) – Automated Tests (pytest)
Tests data quality validation functions.
"""
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pathlib
DATA_CLEAN = str(pathlib.Path(__file__).parent.parent / 'data' / 'telco_churn_clean.csv')

@pytest.fixture(scope='module')
def df():
    return pd.read_csv(DATA_CLEAN)

def test_no_missing_values(df):
    """Cleaned dataset must have zero missing values."""
    missing = df.isnull().sum().sum()
    assert missing == 0, f"Found {missing} missing values in cleaned data."

def test_no_duplicate_rows(df):
    """Cleaned dataset must have no duplicate rows."""
    dupes = df.duplicated().sum()
    assert dupes == 0, f"Found {dupes} duplicate rows."

def test_monthly_charges_range(df):
    """MonthlyCharges must be within [18, 120]."""
    assert df['MonthlyCharges'].between(18, 120).all(), \
        "MonthlyCharges contains out-of-range values."

def test_tenure_non_negative(df):
    """Tenure must be >= 0."""
    assert (df['tenure'] >= 0).all(), "Tenure contains negative values."

def test_churn_binary_values(df):
    """Churn must contain only 'Yes' and 'No'."""
    valid = {'Yes', 'No'}
    actual = set(df['Churn'].unique())
    assert actual.issubset(valid), f"Unexpected Churn values: {actual - valid}"

def test_dataset_size(df):
    """Dataset should have at least 5000 records after cleaning."""
    assert len(df) >= 5000, f"Dataset too small: {len(df)} rows."

def test_senior_citizen_binary(df):
    """SeniorCitizen must be 0 or 1 only."""
    valid = {0, 1}
    actual = set(df['SeniorCitizen'].unique())
    assert actual.issubset(valid), f"Unexpected SeniorCitizen values: {actual}"

def test_required_columns_present(df):
    """All required columns must be present."""
    required = ['gender','SeniorCitizen','tenure','MonthlyCharges',
                'TotalCharges','Contract','Churn']
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"
