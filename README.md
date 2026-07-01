# MRTB 2173 – Agile Data Science PMA
**Telco Customer Churn Prediction | Semester 2, Session 2025/2026**

## Project Overview
An end-to-end Agile Data Science solution predicting customer churn for a
telecommunications company. Built across 4 sprints following Scrum principles.

| Item | Detail |
|------|--------|
| Dataset | Telco Customer Churn (synthetic replica of IBM dataset) |
| Records | 7,043 |
| Target | Churn (Yes / No) |
| Best Model | Gradient Boosting (AUC 0.7416) |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |

## Repository Structure
```
├── data/
│   ├── telco_churn.csv          # Raw dataset
│   └── telco_churn_clean.csv    # Cleaned dataset
├── src/
│   ├── generate_data.py         # Dataset generation
│   ├── eda.py                   # Sprint 1 – EDA
│   ├── data_quality.py          # Sprint 1 – Quality checks & cleaning
│   ├── train_models.py          # Sprint 2 & 3 – Feature eng + models
│   ├── validate.py              # Automated validation script
│   ├── dashboard.py             # Sprint 3 – Streamlit dashboard
│   └── monitoring.py            # Sprint 4 – Monitoring & drift
├── tests/
│   └── test_data_quality.py     # pytest unit tests (8 tests)
├── models/                      # Trained model artifacts (.pkl)
├── outputs/                     # Charts and JSON metrics
├── .github/workflows/ci.yml     # GitHub Actions CI/CD pipeline
└── requirements.txt
```

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python src/generate_data.py

# 3. Run EDA
python src/eda.py

# 4. Clean data
python src/data_quality.py

# 5. Train models
python src/train_models.py

# 6. Run tests
pytest tests/ -v

# 7. Launch dashboard
streamlit run src/dashboard.py

# 8. Run monitoring
python src/monitoring.py
```

## Sprint Summary
| Sprint | Focus | Key Deliverable |
|--------|-------|-----------------|
| Sprint 1 | EDA + Data Quality | Cleaned dataset, 4 visualisations |
| Sprint 2 | Baseline Model | Logistic Regression (AUC 0.709) |
| Sprint 3 | Improved Model | Gradient Boosting (AUC 0.742) |
| Sprint 4 | Monitoring | Drift detection + KPI dashboard |

## CI/CD Pipeline
On every `push` to `main` or `develop`:
1. **Validate** – automated data quality script
2. **Test** – 8 pytest unit tests
3. **Train** – model training and evaluation
4. **Artifact Upload** – models and outputs stored in GitHub Actions
