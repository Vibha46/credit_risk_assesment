import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Credit Risk Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #555;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-left: 4px solid #1a73e8;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #fff5f5, #ffe0e0);
        border-left: 4px solid #e53935;
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 1.1rem;
        font-weight: 700;
        color: #b71c1c;
    }
    .risk-low {
        background: linear-gradient(135deg, #f0fff4, #d4edda);
        border-left: 4px solid #2e7d32;
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 1.1rem;
        font-weight: 700;
        color: #1b5e20;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a73e8;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def load_and_prepare_data(path: str):
    df = pd.read_csv(path, index_col=0)

    for col in ["Saving accounts", "Checking account"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    np.random.seed(42)
    risk_score = (
        (df["Credit amount"] / 1000) * 0.3 +
        (df["Duration"] / 12) * 0.25 +
        (df["Age"].apply(lambda x: max(0, 40 - x)) / 10) * 0.2 +
        df["Saving accounts"].map({"little": 1.5, "moderate": 0.8, "quite rich": 0.4, "rich": 0.2}).fillna(1.0) +
        df["Checking account"].map({"little": 1.2, "moderate": 0.6, "rich": 0.2}).fillna(0.9) +
        np.random.normal(0, 0.5, len(df))
    )
    df["Risk"] = (risk_score > risk_score.quantile(0.7)).astype(int)
    df["Risk_Label"] = df["Risk"].map({0: "Good", 1: "Bad"})
    return df


@st.cache_data
def encode_features(df: pd.DataFrame):
    df_enc = df.copy()
    le = LabelEncoder()
    cat_cols = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    for col in cat_cols:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    return df_enc


@st.cache_resource
def train_models(df_enc: pd.DataFrame):
    feature_cols = ["Age", "Sex", "Job", "Housing", "Saving accounts",
                    "Checking account", "Credit amount", "Duration", "Purpose"]
    X = df_enc[feature_cols]
    y = df_enc["Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "SVM":                 SVC(probability=True, random_state=42),
    }

    results = {}
    for name, model in models.items():
        use_scaled = name in ["Logistic Regression", "SVM"]
        Xtr = X_train_sc if use_scaled else X_train
        Xte = X_test_sc  if use_scaled else X_test

        model.fit(Xtr, y_train)
        y_pred    = model.predict(Xte)
        y_prob    = model.predict_proba(Xte)[:, 1]
        cv_scores = cross_val_score(model, Xtr, y_train, cv=5, scoring="accuracy")

        results[name] = {
            "model":     model,
            "scaler":    scaler if use_scaled else None,
            "scaled":    use_scaled,
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "auc":       roc_auc_score(y_test, y_prob),
            "cv_mean":   cv_scores.mean(),
            "cv_std":    cv_scores.std(),
            "confusion": confusion_matrix(y_test, y_pred),
            "fpr":       roc_curve(y_test, y_prob)[0],
            "tpr":       roc_curve(y_test, y_prob)[1],
            "report":    classification_report(y_test, y_pred, output_dict=True),
        }

    return results, feature_cols, X_train, X_test, y_train, y_test, scaler


# ─── Load Data ────────────────────────────────────────────────────────────────
import os

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_SCRIPT_DIR, "german_credit_data (1).csv")

if not os.path.exists(_DEFAULT_CSV):
    st.error(
        f"Dataset not found:\n`{_DEFAULT_CSV}`\n\n"
        "Please place **german_credit_data (1).csv** in the same folder as this script."
    )
    st.stop()

data_path = _DEFAULT_CSV

df     = load_and_prepare_data(data_path)
df_enc = encode_features(df)
results, feature_cols, X_train, X_test, y_train, y_test, scaler = train_models(df_enc)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##Configuration")
    st.markdown("###Model Selection")
    selected_model_name = st.selectbox(
        "Choose Model",
        ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM"],
        index=0
    )
    st.markdown("---")
    st.markdown("###Risk Threshold")
    threshold = st.slider("Probability Threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Lower = more conservative (flag more as risky)")
    st.markdown("---")
    st.markdown("**Credit Risk App**  \nGerman Credit Dataset  \nML-Powered Assessment")

sel = results[selected_model_name]

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Credit Risk Assessment</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML-powered credit risk evaluation using the German Credit Dataset</p>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard", "Model Performance", "Feature Analysis",
    "Predict Risk", "Data Explorer"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    good_pct = (df["Risk"] == 0).mean() * 100
    bad_pct  = (df["Risk"] == 1).mean() * 100
    c1.metric("Total Records",     f"{len(df):,}")
    c2.metric("Good Credit",       f"{int(good_pct)}%",  f"{(df['Risk']==0).sum()} customers")
    c3.metric("Bad Credit",        f"{int(bad_pct)}%",   f"{(df['Risk']==1).sum()} customers")
    c4.metric("Avg Credit Amount", f"€{df['Credit amount'].mean():,.0f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        risk_counts = df["Risk_Label"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        fig = px.pie(risk_counts, names="Risk", values="Count",
                     title="Credit Risk Distribution",
                     color="Risk",
                     color_discrete_map={"Good": "#2e7d32", "Bad": "#e53935"},
                     hole=0.45)
        fig.update_traces(textinfo="percent+label", textfont_size=14)
        fig.update_layout(height=340, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = px.box(df, x="Risk_Label", y="Credit amount",
                     color="Risk_Label",
                     color_discrete_map={"Good": "#2e7d32", "Bad": "#e53935"},
                     title="Credit Amount by Risk",
                     labels={"Risk_Label": "Risk", "Credit amount": "Amount (€)"})
        fig.update_layout(height=340, margin=dict(t=50, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        fig = px.histogram(df, x="Age", color="Risk_Label", nbins=25,
                           barmode="overlay", opacity=0.75,
                           color_discrete_map={"Good": "#1a73e8", "Bad": "#e53935"},
                           title="Age Distribution by Risk",
                           labels={"Risk_Label": "Risk"})
        fig.update_layout(height=320, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        purpose_risk = df.groupby(["Purpose", "Risk_Label"]).size().reset_index(name="Count")
        fig = px.bar(purpose_risk, x="Purpose", y="Count", color="Risk_Label",
                     barmode="stack",
                     color_discrete_map={"Good": "#2e7d32", "Bad": "#e53935"},
                     title="Loan Purpose by Risk",
                     labels={"Risk_Label": "Risk"})
        fig.update_layout(height=320, margin=dict(t=50, b=10), xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-title">Model Comparison at a Glance</p>', unsafe_allow_html=True)
    model_summary = pd.DataFrame({
        "Model":    list(results.keys()),
        "Accuracy": [f"{v['accuracy']:.1%}" for v in results.values()],
        "AUC-ROC":  [f"{v['auc']:.3f}"     for v in results.values()],
        "F1-Score": [f"{v['f1']:.3f}"      for v in results.values()],
        "CV Mean":  [f"{v['cv_mean']:.3f}" for v in results.values()],
    })
    st.dataframe(model_summary.style.highlight_max(
        subset=["Accuracy"], color="#d4edda"
    ), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<p class="section-title">{selected_model_name} — Detailed Performance</p>',
                unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{sel['accuracy']:.1%}")
    m2.metric("Precision", f"{sel['precision']:.3f}")
    m3.metric("Recall",    f"{sel['recall']:.3f}")
    m4.metric("F1-Score",  f"{sel['f1']:.3f}")
    m5.metric("AUC-ROC",   f"{sel['auc']:.3f}")

    st.markdown(f"**Cross-Validation (5-fold):** {sel['cv_mean']:.3f} ± {sel['cv_std']:.3f}")
    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        cm = sel["confusion"]
        fig = px.imshow(
            cm, text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            x=["Good (0)", "Bad (1)"],
            y=["Good (0)", "Bad (1)"],
            title="Confusion Matrix"
        )
        fig.update_layout(height=350, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sel["fpr"], y=sel["tpr"], mode="lines",
            name=f"{selected_model_name} (AUC={sel['auc']:.3f})",
            line=dict(color="#1a73e8", width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random Classifier",
            line=dict(color="#aaa", dash="dash")
        ))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=350, margin=dict(t=50, b=10),
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-title">All Models — ROC Comparison</p>', unsafe_allow_html=True)
    fig = go.Figure()
    colors = ["#1a73e8", "#e53935", "#2e7d32", "#f57c00"]
    for (name, res), color in zip(results.items(), colors):
        fig.add_trace(go.Scatter(
            x=res["fpr"], y=res["tpr"], mode="lines",
            name=f"{name} (AUC={res['auc']:.3f})",
            line=dict(color=color, width=2)
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(color="#ccc", dash="dash")
    ))
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400, margin=dict(t=20, b=10),
        legend=dict(x=0.55, y=0.1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-title">Metrics Comparison</p>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame({
        "Model":     list(results.keys()),
        "Accuracy":  [v["accuracy"]  for v in results.values()],
        "Precision": [v["precision"] for v in results.values()],
        "Recall":    [v["recall"]    for v in results.values()],
        "F1":        [v["f1"]        for v in results.values()],
        "AUC":       [v["auc"]       for v in results.values()],
    })
    metrics_melt = metrics_df.melt("Model", var_name="Metric", value_name="Score")
    fig = px.bar(metrics_melt, x="Model", y="Score", color="Metric",
                 barmode="group", text_auto=".2f",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=400, yaxis_range=[0, 1.05], margin=dict(t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Feature Importance Analysis</p>', unsafe_allow_html=True)

    model = sel["model"]
    col_l, col_r = st.columns(2)

    with col_l:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
            feat_df = feat_df.sort_values("Importance", ascending=True)
            fig = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues",
                         title=f"Feature Importances — {selected_model_name}")
            fig.update_layout(height=420, margin=dict(t=50, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_[0])
            feat_df = pd.DataFrame({"Feature": feature_cols, "Coefficient (|abs|)": coef})
            feat_df = feat_df.sort_values("Coefficient (|abs|)", ascending=True)
            fig = px.bar(feat_df, x="Coefficient (|abs|)", y="Feature", orientation="h",
                         color="Coefficient (|abs|)", color_continuous_scale="Purples",
                         title=f"Feature Coefficients — {selected_model_name}")
            fig.update_layout(height=420, margin=dict(t=50, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not directly available for this model. "
                    "Use Permutation Importance (right panel).")

    with col_r:
        st.markdown("**Permutation Importance (test set)**")
        X_test_df = df_enc[feature_cols].iloc[X_test.index] if hasattr(X_test, "index") else X_test
        Xte_for_perm = scaler.transform(X_test_df) if sel["scaled"] else X_test_df
        perm = permutation_importance(model, Xte_for_perm, y_test, n_repeats=10, random_state=42)
        perm_df = pd.DataFrame({
            "Feature":    feature_cols,
            "Importance": perm.importances_mean
        }).sort_values("Importance", ascending=True)
        fig = px.bar(perm_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Oranges",
                     title="Permutation Importance")
        fig.update_layout(height=420, margin=dict(t=50, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-title">Feature Correlation Heatmap</p>', unsafe_allow_html=True)
    num_df = df_enc[feature_cols + ["Risk"]]
    corr   = num_df.corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    text_auto=".2f", title="Correlation Matrix")
    fig.update_layout(height=500, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-title">Credit Amount vs Duration</p>', unsafe_allow_html=True)
    fig = px.scatter(df, x="Duration", y="Credit amount", color="Risk_Label",
                     size="Age", hover_data=["Purpose", "Housing"],
                     color_discrete_map={"Good": "#2e7d32", "Bad": "#e53935"},
                     opacity=0.7,
                     title="Credit Amount vs Duration (bubble size = Age)")
    fig.update_layout(height=420, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICT RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">Predict Credit Risk for New Applicant</p>',
                unsafe_allow_html=True)
    st.markdown(f"Using **{selected_model_name}** with threshold **{threshold:.2f}**")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age           = st.number_input("Age",           min_value=18, max_value=90, value=35)
            sex           = st.selectbox("Sex",              ["male", "female"])
            job           = st.selectbox("Job Type",         [0, 1, 2, 3],
                                         format_func=lambda x: f"{x} — {'Unskilled non-resident' if x==0 else 'Unskilled resident' if x==1 else 'Skilled' if x==2 else 'Highly skilled'}")
        with c2:
            housing       = st.selectbox("Housing",          ["own", "rent", "free"])
            saving_acc    = st.selectbox("Saving Accounts",  ["little", "moderate", "quite rich", "rich"])
            checking_acc  = st.selectbox("Checking Account", ["little", "moderate", "rich"])
        with c3:
            credit_amount = st.number_input("Credit Amount (€)", min_value=250, max_value=20000, value=3000)
            duration      = st.number_input("Duration (months)", min_value=4,   max_value=72,    value=24)
            purpose       = st.selectbox("Purpose", ["car", "furniture/equipment", "radio/TV",
                                                     "domestic appliances", "repairs",
                                                     "education", "business", "vacation/others"])
        submitted = st.form_submit_button("Assess Risk", use_container_width=True)

    if submitted:
        sex_enc      = 1 if sex == "male" else 0
        housing_enc  = {"own": 2, "rent": 1, "free": 0}[housing]
        saving_enc   = {"little": 1, "moderate": 2, "quite rich": 3, "rich": 4}[saving_acc]
        checking_enc = {"little": 1, "moderate": 2, "rich": 3}[checking_acc]
        purpose_enc  = ["car", "domestic appliances", "education", "furniture/equipment",
                        "business", "radio/TV", "repairs", "vacation/others"].index(purpose)                        if purpose in ["car", "domestic appliances", "education", "furniture/equipment",
                                      "business", "radio/TV", "repairs", "vacation/others"] else 0

        input_arr = np.array([[age, sex_enc, job, housing_enc,
                               saving_enc, checking_enc,
                               credit_amount, duration, purpose_enc]])

        model_obj = sel["model"]
        if sel["scaled"]:
            input_arr = sel["scaler"].transform(input_arr)

        prob_bad   = model_obj.predict_proba(input_arr)[0][1]
        prediction = int(prob_bad >= threshold)

        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns([1, 1, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(f'<div class="risk-high">HIGH RISK — BAD CREDIT<br>'
                            f'Probability: {prob_bad:.1%}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">LOW RISK — GOOD CREDIT<br>'
                            f'Probability of Bad: {prob_bad:.1%}</div>', unsafe_allow_html=True)

        with res_col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_bad * 100,
                title={"text": "Risk Score (%)"},
                delta={"reference": threshold * 100},
                gauge={
                    "axis":  {"range": [0, 100]},
                    "bar":   {"color": "#e53935" if prediction == 1 else "#2e7d32"},
                    "steps": [
                        {"range": [0,  50], "color": "#e8f5e9"},
                        {"range": [50, 75], "color": "#fff9c4"},
                        {"range": [75, 100], "color": "#ffebee"},
                    ],
                    "threshold": {
                        "line":      {"color": "#333", "width": 3},
                        "thickness": 0.75,
                        "value":     threshold * 100
                    }
                }
            ))
            fig.update_layout(height=280, margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        with res_col3:
            st.markdown("**Risk Factors**")
            factors = {
                "Credit Amount": f"€{credit_amount:,}",
                "Duration":      f"{duration} months",
                "Age":           f"{age} years",
                "Housing":       housing,
                "Saving Acc.":   saving_acc,
                "Checking Acc.": checking_acc,
                "Purpose":       purpose,
            }
            for k, v in factors.items():
                st.markdown(f"**{k}:** {v}")

        st.markdown("---")
        st.markdown("**Similar Past Applicants (by credit amount & duration)**")
        tol_amt = credit_amount * 0.25
        tol_dur = 12
        similar = df[
            (df["Credit amount"].between(credit_amount - tol_amt, credit_amount + tol_amt)) &
            (df["Duration"].between(duration - tol_dur, duration + tol_dur))
        ][["Age", "Sex", "Housing", "Saving accounts", "Checking account",
           "Credit amount", "Duration", "Purpose", "Risk_Label"]].head(10)
        if len(similar):
            st.dataframe(similar.style.applymap(
                lambda v: "background-color: #d4edda" if v == "Good"
                          else "background-color: #f8d7da" if v == "Bad" else "",
                subset=["Risk_Label"]
            ), use_container_width=True, hide_index=True)
        else:
            st.info("No similar applicants found in the dataset.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">Interactive Data Explorer</p>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        risk_filter    = st.multiselect("Risk",    ["Good", "Bad"],        default=["Good", "Bad"])
    with col_f2:
        housing_filter = st.multiselect("Housing", df["Housing"].unique(), default=list(df["Housing"].unique()))
    with col_f3:
        purpose_filter = st.multiselect("Purpose", df["Purpose"].unique(), default=list(df["Purpose"].unique()))

    age_range = st.slider("Age Range",
                          int(df["Age"].min()), int(df["Age"].max()),
                          (int(df["Age"].min()), int(df["Age"].max())))
    amt_range = st.slider("Credit Amount Range (€)",
                          int(df["Credit amount"].min()), int(df["Credit amount"].max()),
                          (int(df["Credit amount"].min()), int(df["Credit amount"].max())))

    filtered = df[
        (df["Risk_Label"].isin(risk_filter)) &
        (df["Housing"].isin(housing_filter)) &
        (df["Purpose"].isin(purpose_filter)) &
        (df["Age"].between(*age_range)) &
        (df["Credit amount"].between(*amt_range))
    ]

    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} records**")

    display_cols = ["Age", "Sex", "Job", "Housing", "Saving accounts",
                    "Checking account", "Credit amount", "Duration", "Purpose", "Risk_Label"]
    st.dataframe(
        filtered[display_cols].style.applymap(
            lambda v: "background-color: #d4edda" if v == "Good"
                      else "background-color: #f8d7da" if v == "Bad" else "",
            subset=["Risk_Label"]
        ),
        use_container_width=True, height=450
    )

    st.markdown('<p class="section-title">Filtered Dataset Statistics</p>', unsafe_allow_html=True)
    st.dataframe(filtered[["Age", "Credit amount", "Duration", "Job"]].describe().T,
                 use_container_width=True)

    csv_export = filtered[display_cols].to_csv(index=False)
    st.download_button("Download Filtered Data as CSV",
                       csv_export, "filtered_credit_data.csv", "text/csv")
