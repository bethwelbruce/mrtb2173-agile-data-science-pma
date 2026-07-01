import os
"""Generate synthetic Telco Customer Churn dataset (mirrors IBM Telco structure)."""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 7043

def synthetic_telco(n):
    genders = np.random.choice(['Male','Female'], n)
    senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(['Yes','No'], n, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes','No'], n, p=[0.30, 0.70])
    tenure = np.random.randint(0, 73, n)
    phone = np.random.choice(['Yes','No'], n, p=[0.90, 0.10])
    multiple_lines = np.where(phone=='No', 'No phone service',
                      np.random.choice(['Yes','No'], n, p=[0.42, 0.58]))
    internet = np.random.choice(['DSL','Fiber optic','No'], n, p=[0.34, 0.44, 0.22])
    online_sec = np.where(internet=='No','No internet service',
                   np.random.choice(['Yes','No'], n, p=[0.29, 0.71]))
    online_bk  = np.where(internet=='No','No internet service',
                   np.random.choice(['Yes','No'], n, p=[0.28, 0.72]))
    device_prot= np.where(internet=='No','No internet service',
                   np.random.choice(['Yes','No'], n, p=[0.29, 0.71]))
    tech_supp  = np.where(internet=='No','No internet service',
                   np.random.choice(['Yes','No'], n, p=[0.29, 0.71]))
    streaming_tv=np.where(internet=='No','No internet service',
                   np.random.choice(['Yes','No'], n, p=[0.38, 0.62]))
    streaming_m =np.where(internet=='No','No internet service',
                   np.random.choice(['Yes','No'], n, p=[0.39, 0.61]))
    contract = np.random.choice(['Month-to-month','One year','Two year'], n, p=[0.55, 0.21, 0.24])
    paperless = np.random.choice(['Yes','No'], n, p=[0.59, 0.41])
    payment   = np.random.choice(['Electronic check','Mailed check',
                                  'Bank transfer (automatic)','Credit card (automatic)'],
                                  n, p=[0.34, 0.23, 0.22, 0.21])
    monthly_charges = np.round(np.random.uniform(18.25, 118.75, n), 2)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 20, n), 2)
    total_charges = total_charges.astype(object)
    # Introduce 11 blank strings (like real IBM dataset)
    nan_idx = np.random.choice(n, 11, replace=False)
    for i in nan_idx:
        total_charges[i] = ''

    churn_prob = (
        0.05
        + 0.30 * (contract == 'Month-to-month')
        + 0.10 * (internet == 'Fiber optic')
        - 0.02 * (tenure / 72)
        + 0.05 * (monthly_charges > 80)
        + 0.08 * (senior == 1)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.85)
    churn = np.where(np.random.uniform(0, 1, n) < churn_prob, 'Yes', 'No')

    customer_id = ['CUS' + str(i).zfill(5) for i in range(1, n+1)]

    df = pd.DataFrame({
        'customerID': customer_id,
        'gender': genders,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiple_lines,
        'InternetService': internet,
        'OnlineSecurity': online_sec,
        'OnlineBackup': online_bk,
        'DeviceProtection': device_prot,
        'TechSupport': tech_supp,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_m,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    })
    # Add 5 duplicate rows
    dup_rows = df.sample(5, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)
    return df

if __name__ == '__main__':
    df = synthetic_telco(n)
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'telco_churn.csv')
    df.to_csv(out_path, index=False)
    print(f"Dataset shape: {df.shape}")
    print(df.head(3).to_string())
    print(df.dtypes)
