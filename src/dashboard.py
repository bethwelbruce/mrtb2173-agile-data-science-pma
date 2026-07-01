"""
Q4 – Streamlit Dashboard: Telco Churn Analytics & Predictor
Run: streamlit run src/dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, json, os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title='Telco Churn Dashboard', layout='wide', page_icon='📊')
st.title('📊 Telco Customer Churn — Agile Data Science Dashboard')
st.caption('MRTB 2173 | Sprint 3 Deliverable | Gradient Boosting Model')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    df = pd.read_csv(f'{BASE}/data/telco_churn_clean.csv')
    return df

@st.cache_resource
def load_model():
    return joblib.load(f'{BASE}/models/improved_gb.pkl')

@st.cache_resource
def load_scaler():
    return joblib.load(f'{BASE}/models/scaler.pkl')

@st.cache_resource
def load_features():
    return joblib.load(f'{BASE}/models/feature_names.pkl')

df_raw = load_data()
model  = load_model()
scaler = load_scaler()
feat_names = load_features()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header('🔍 Dashboard Filters')
contract_filter = st.sidebar.multiselect(
    'Contract Type',
    options=df_raw['Contract'].unique().tolist(),
    default=df_raw['Contract'].unique().tolist()
)
internet_filter = st.sidebar.multiselect(
    'Internet Service',
    options=df_raw['InternetService'].unique().tolist(),
    default=df_raw['InternetService'].unique().tolist()
)
monthly_range = st.sidebar.slider(
    'Monthly Charges Range (USD)',
    float(df_raw['MonthlyCharges'].min()),
    float(df_raw['MonthlyCharges'].max()),
    (float(df_raw['MonthlyCharges'].min()), float(df_raw['MonthlyCharges'].max()))
)

df = df_raw[
    df_raw['Contract'].isin(contract_filter) &
    df_raw['InternetService'].isin(internet_filter) &
    df_raw['MonthlyCharges'].between(*monthly_range)
].copy()

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric('Total Customers', f'{len(df):,}')
k2.metric('Churned', f'{(df["Churn"]=="Yes").sum():,}')
churn_rate = (df['Churn']=='Yes').mean()
k3.metric('Churn Rate', f'{churn_rate:.1%}')
k4.metric('Avg Monthly Charge', f'${df["MonthlyCharges"].mean():.2f}')

st.divider()

# ── Visualizations ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['📈 EDA Insights', '🤖 Churn Predictor', '📡 Model Metrics'])

with tab1:
    col1, col2 = st.columns(2)

    # Viz 1 – Churn by Contract
    with col1:
        st.subheader('Churn Rate by Contract Type')
        churn_contract = (df.groupby('Contract')['Churn']
                            .apply(lambda x: (x=='Yes').mean())
                            .reset_index()
                            .rename(columns={'Churn':'ChurnRate'}))
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        colours = ['#E74C3C','#F39C12','#27AE60'][:len(churn_contract)]
        bars = ax1.bar(churn_contract['Contract'], churn_contract['ChurnRate'], color=colours)
        ax1.set_ylim(0, 0.7)
        for bar, val in zip(bars, churn_contract['ChurnRate']):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                     f'{val:.1%}', ha='center', fontsize=9, fontweight='bold')
        ax1.set_ylabel('Churn Rate')
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

    # Viz 2 – Monthly Charges distribution
    with col2:
        st.subheader('Monthly Charges: Churned vs Retained')
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        for label, color in [('Yes','#E74C3C'),('No','#27AE60')]:
            subset = df[df['Churn']==label]['MonthlyCharges']
            ax2.hist(subset, bins=25, alpha=0.6, label=label, color=color)
        ax2.legend(title='Churn')
        ax2.set_xlabel('Monthly Charges (USD)')
        ax2.set_ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Viz 3 – Internet Service count
    st.subheader('Customer Count by Internet Service & Churn')
    fig3, ax3 = plt.subplots(figsize=(9, 3.5))
    pivot = df.groupby(['InternetService','Churn']).size().unstack(fill_value=0)
    pivot.plot(kind='bar', ax=ax3, color=['#27AE60','#E74C3C'], edgecolor='white')
    ax3.set_xlabel('Internet Service')
    ax3.set_ylabel('Customer Count')
    ax3.legend(title='Churn')
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with tab2:
    st.subheader('🔮 Predict Individual Customer Churn Probability')
    st.info('Adjust inputs below and click **Predict** to get the model output.')

    c1, c2, c3 = st.columns(3)
    tenure_in    = c1.slider('Tenure (months)', 0, 72, 12)
    monthly_in   = c2.slider('Monthly Charges (USD)', 18.0, 120.0, 65.0, step=0.5)
    contract_in  = c3.selectbox('Contract', ['Month-to-month','One year','Two year'])
    internet_in  = c1.selectbox('Internet Service', ['DSL','Fiber optic','No'])
    senior_in    = c2.selectbox('Senior Citizen', ['No','Yes'])
    paperless_in = c3.selectbox('Paperless Billing', ['Yes','No'])

    if st.button('🔍 Predict Churn'):
        from sklearn.preprocessing import LabelEncoder
        import warnings; warnings.filterwarnings('ignore')

        df_input_raw = df_raw.copy()
        df_input_raw['TotalCharges'] = pd.to_numeric(df_input_raw['TotalCharges'], errors='coerce').fillna(0)
        cat_cols = [c for c in df_input_raw.select_dtypes(include=['object','string']).columns if c != 'Churn']
        le = LabelEncoder()
        encoders = {}
        for col in cat_cols:
            le.fit(df_input_raw[col].astype(str))
            encoders[col] = {v: i for i, v in enumerate(le.classes_)}

        sample = {col: 0 for col in feat_names}
        # Fill known inputs
        sample['tenure']         = tenure_in
        sample['MonthlyCharges'] = monthly_in
        sample['TotalCharges']   = monthly_in * tenure_in
        sample['SeniorCitizen']  = 1 if senior_in == 'Yes' else 0
        sample['Contract']       = encoders.get('Contract', {}).get(contract_in, 0)
        sample['InternetService']= encoders.get('InternetService', {}).get(internet_in, 0)
        sample['PaperlessBilling']= encoders.get('PaperlessBilling', {}).get(paperless_in, 0)
        sample['TenureGroup']    = min(3, tenure_in // 12)

        row = pd.DataFrame([sample])[feat_names]
        num_cols = ['MonthlyCharges','TotalCharges','tenure']
        row[num_cols] = scaler.transform(row[num_cols])

        prob = model.predict_proba(row)[0][1]
        pred = 'WILL CHURN' if prob >= 0.5 else 'WILL NOT CHURN'
        color = 'red' if prob >= 0.5 else 'green'
        st.markdown(f'### Prediction: :{color}[{pred}]')
        st.progress(float(prob), text=f'Churn Probability: {prob:.1%}')

with tab3:
    st.subheader('Model Performance Comparison (Sprint 2 vs Sprint 3)')
    metrics_path = f'{BASE}/outputs/model_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            mets = json.load(f)
        comp_df = pd.DataFrame({
            'Metric': ['Accuracy','ROC-AUC','F1-Score'],
            'Sprint 2 (Logistic Reg)': [
                mets['baseline']['accuracy'],
                mets['baseline']['roc_auc'],
                mets['baseline']['f1']
            ],
            'Sprint 3 (Gradient Boost)': [
                mets['improved']['accuracy'],
                mets['improved']['roc_auc'],
                mets['improved']['f1']
            ]
        })
        st.dataframe(comp_df.set_index('Metric'), use_container_width=True)

        fig4, ax4 = plt.subplots(figsize=(7, 3.5))
        x = np.arange(3)
        w = 0.35
        ax4.bar(x - w/2, comp_df['Sprint 2 (Logistic Reg)'], w, label='Sprint 2', color='steelblue')
        ax4.bar(x + w/2, comp_df['Sprint 3 (Gradient Boost)'], w, label='Sprint 3', color='#E74C3C')
        ax4.set_xticks(x); ax4.set_xticklabels(['Accuracy','ROC-AUC','F1'])
        ax4.legend(); ax4.set_ylim(0, 1); ax4.set_title('Sprint 2 vs Sprint 3 Metrics')
        plt.tight_layout(); st.pyplot(fig4); plt.close()
    else:
        st.warning('Model metrics file not found. Run train_models.py first.')

st.divider()
st.caption('MRTB 2173 Agile Data Science PMA | Semester 2, Session 2025/2026 | UTM')
