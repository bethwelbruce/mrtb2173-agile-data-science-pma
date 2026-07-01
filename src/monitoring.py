"""
Q5 – Monitoring: Data Drift, Model Performance Degradation, and Metrics.
Simulates a production monitoring scenario by comparing reference vs current data.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT   = f'{BASE}/outputs'
os.makedirs(OUT, exist_ok=True)

np.random.seed(99)

# ── Load reference data ───────────────────────────────────────────────────────
df_ref = pd.read_csv(f'{BASE}/data/telco_churn_clean.csv')

# ── Simulate current (drifted) data ──────────────────────────────────────────
# In production the current period would come from a live pipeline.
# Here we simulate drift: MonthlyCharges shifted up by ~15 USD (price increase)
# and higher proportion of Fiber optic customers.
df_cur = df_ref.copy()
df_cur['MonthlyCharges'] += np.random.normal(15, 5, len(df_cur))  # drift: mean shift
df_cur['MonthlyCharges'] = df_cur['MonthlyCharges'].clip(18, 140)
# Simulate some unknown customers having null tenure
random_idx = np.random.choice(len(df_cur), 20, replace=False)
df_cur.loc[random_idx, 'tenure'] = np.nan

print("=" * 65)
print("  Q5 – MONITORING ANALYSIS REPORT")
print("=" * 65)

# ── METRIC 1: Monthly Charges Distribution Drift ─────────────────────────────
print("\n── Metric 1: MonthlyCharges Distribution Shift ─────────────────")
ref_mean = df_ref['MonthlyCharges'].mean()
cur_mean = df_cur['MonthlyCharges'].mean()
ref_std  = df_ref['MonthlyCharges'].std()
cur_std  = df_cur['MonthlyCharges'].std()
print(f"Reference period  → Mean: {ref_mean:.2f}, Std: {ref_std:.2f}")
print(f"Current period    → Mean: {cur_mean:.2f}, Std: {cur_std:.2f}")
print(f"Mean shift        → {cur_mean - ref_mean:+.2f} USD ({(cur_mean-ref_mean)/ref_mean*100:+.1f}%)")

# PSI (Population Stability Index)
def psi(expected, actual, buckets=10):
    bins = np.linspace(min(expected.min(), actual.min()),
                       max(expected.max(), actual.max()), buckets+1)
    e_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    a_pct = np.histogram(actual,   bins=bins)[0] / len(actual)
    e_pct = np.where(e_pct == 0, 0.0001, e_pct)
    a_pct = np.where(a_pct == 0, 0.0001, a_pct)
    return np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))

psi_val = psi(df_ref['MonthlyCharges'], df_cur['MonthlyCharges'])
alert = 'ALERT: Significant drift!' if psi_val > 0.2 else ('WARNING: Moderate drift' if psi_val > 0.1 else 'STABLE')
print(f"PSI (MonthlyCharges): {psi_val:.4f}  →  {alert}")

# ── METRIC 2: Missing Data Rate in Current Period ────────────────────────────
print("\n── Metric 2: Operational Data Quality Metric ───────────────────")
ref_miss_rate = df_ref.isnull().sum().sum() / df_ref.size
cur_miss_rate = df_cur.isnull().sum().sum() / df_cur.size
print(f"Reference missing rate: {ref_miss_rate:.4%}")
print(f"Current missing rate  : {cur_miss_rate:.4%}")
if cur_miss_rate > ref_miss_rate * 2:
    print("WARNING: Missing rate has more than doubled – data pipeline issue suspected.")
else:
    print("STABLE: Missing rate within acceptable threshold.")

# ── METRIC 3: Simulated Model Performance Degradation ────────────────────────
print("\n── Metric 3: Simulated Model ROC-AUC Over Time ─────────────────")
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

model  = joblib.load(f'{BASE}/models/improved_gb.pkl')
scaler = joblib.load(f'{BASE}/models/scaler.pkl')
feat   = joblib.load(f'{BASE}/models/feature_names.pkl')

def prepare(df_in):
    d = df_in.copy().dropna()
    d['Churn'] = d['Churn'].apply(lambda x: 1 if str(x).strip()=='Yes' else 0)
    cat = [c for c in d.select_dtypes(include=['object','string']).columns if c!='Churn']
    le  = LabelEncoder()
    for col in cat:
        d[col] = le.fit_transform(d[col].astype(str))
    d['TenureGroup'] = pd.cut(d['tenure'], bins=[0,12,24,48,72],
                               labels=[0,1,2,3], include_lowest=True).astype(int)
    X = d[[c for c in feat if c in d.columns]].reindex(columns=feat, fill_value=0)
    y = d['Churn']
    num = ['MonthlyCharges','TotalCharges','tenure']
    X[num] = scaler.transform(X[num])
    return X, y

X_ref, y_ref = prepare(df_ref)
X_cur, y_cur = prepare(df_cur)
auc_ref = roc_auc_score(y_ref, model.predict_proba(X_ref)[:,1])
auc_cur = roc_auc_score(y_cur, model.predict_proba(X_cur)[:,1])
print(f"AUC on Reference data : {auc_ref:.4f}")
print(f"AUC on Current data   : {auc_cur:.4f}")
print(f"Performance drop      : {auc_ref - auc_cur:+.4f}")
if auc_ref - auc_cur > 0.05:
    print("ALERT: Model degradation exceeds 5% threshold – retraining recommended.")
else:
    print("STABLE: Model performance within acceptable range.")

# ── Visualisations ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Plot 1 – Distribution drift
axes[0].hist(df_ref['MonthlyCharges'], bins=25, alpha=0.6, label='Reference', color='steelblue')
axes[0].hist(df_cur['MonthlyCharges'], bins=25, alpha=0.6, label='Current',   color='#E74C3C')
axes[0].set_title(f'MonthlyCharges Drift\nPSI = {psi_val:.3f}')
axes[0].set_xlabel('Monthly Charges (USD)'); axes[0].legend()

# Plot 2 – Missing rate bar
axes[1].bar(['Reference','Current'], [ref_miss_rate*100, cur_miss_rate*100],
            color=['steelblue','#E74C3C'])
axes[1].set_title('Missing Data Rate (%)')
axes[1].set_ylabel('Missing %')
for i, v in enumerate([ref_miss_rate*100, cur_miss_rate*100]):
    axes[1].text(i, v+0.001, f'{v:.3f}%', ha='center', fontweight='bold')

# Plot 3 – AUC degradation over simulated sprints
sprints   = ['Sprint 3','Sprint 4 (Sim)','Sprint 5 (Sim)','Sprint 6 (Sim)']
auc_trend = [auc_ref, auc_ref*0.99, auc_ref*0.97, auc_cur]
colors    = ['steelblue','steelblue','orange','#E74C3C']
axes[2].plot(sprints, auc_trend, marker='o', linewidth=2, color='navy')
for s, a in zip(sprints, auc_trend):
    axes[2].annotate(f'{a:.3f}', (s, a), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
axes[2].axhline(y=auc_ref*0.95, color='red', linestyle='--', linewidth=1, label='Alert threshold')
axes[2].set_title('ROC-AUC Degradation Over Time')
axes[2].set_ylabel('ROC-AUC'); axes[2].legend()
axes[2].set_ylim(0.60, 0.85)
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=15)

plt.suptitle('Q5 – Model Monitoring Dashboard', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_monitoring.png', dpi=150)
plt.close()
print("\n[Saved] fig6_monitoring.png")

# Save monitoring results
monitor_results = {
    'monthly_charges_psi': round(psi_val, 4),
    'psi_status': alert,
    'ref_missing_rate': round(ref_miss_rate, 6),
    'cur_missing_rate': round(cur_miss_rate, 6),
    'auc_reference': round(auc_ref, 4),
    'auc_current': round(auc_cur, 4),
    'auc_drop': round(auc_ref - auc_cur, 4)
}
with open(f'{OUT}/monitoring_results.json', 'w') as f:
    json.dump(monitor_results, f, indent=2)
print("[Saved] monitoring_results.json")
print("\n── MONITORING COMPLETE ───────────────────────────────────────────")
