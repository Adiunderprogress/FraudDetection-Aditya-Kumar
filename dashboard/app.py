"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         Real-Time Fraud Detection — Streamlit Operations Dashboard           ║
║         Pages: Overview · Transaction Explorer · SHAP Explainer             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import json
import os

# ─── Optional Plotly import ───────────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.kpi-label { font-size: 13px; color: #94a3b8; margin-bottom: 6px; letter-spacing: .05em; }
.kpi-value { font-size: 28px; font-weight: 700; color: #f8fafc; }
.kpi-sub   { font-size: 12px; color: #64748b; margin-top: 4px; }

/* ── Risk badges ── */
.badge-critical  { background:#fee2e2; color:#b91c1c; border-radius:9999px; padding:3px 12px; font-size:12px; font-weight:600; }
.badge-suspicious{ background:#fef9c3; color:#a16207; border-radius:9999px; padding:3px 12px; font-size:12px; font-weight:600; }
.badge-clear     { background:#dcfce7; color:#166534; border-radius:9999px; padding:3px 12px; font-size:12px; font-weight:600; }

/* ── Section headers ── */
.section-header {
    font-size: 18px; font-weight: 700; color: #1e293b;
    border-left: 4px solid #3b82f6;
    padding-left: 12px; margin: 20px 0 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING  (cached so they load once)
# ══════════════════════════════════════════════════════════════════════════════
BASE = os.path.dirname(__file__)   # dashboard/ directory

@st.cache_resource(show_spinner="Loading ML model …")
def load_model():
    path = os.path.join(BASE, "model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource(show_spinner="Loading scaler …")
def load_scaler():
    path = os.path.join(BASE, "scaler.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data(show_spinner="Loading predictions …")
def load_predictions():
    path = os.path.join(BASE, "test_predictions.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

@st.cache_data(show_spinner="Loading feature metadata …")
def load_meta():
    path = os.path.join(BASE, "feature_meta.json")
    if not os.path.exists(path):
        return {"feature_cols": [], "optimal_threshold": 0.5, "fraud_rate": 0.035}
    with open(path) as f:
        return json.load(f)

model       = load_model()
scaler      = load_scaler()
df_pred     = load_predictions()
meta        = load_meta()
THRESHOLD   = meta.get("optimal_threshold", 0.50)
FEAT_COLS   = meta.get("feature_cols", [])

# Fallback demo data when model artefacts are not yet generated
def _demo_data(n=5000):
    rng = np.random.default_rng(42)
    fraud_prob = rng.beta(0.5, 8, n)
    is_fraud   = (fraud_prob > 0.55).astype(int)
    data = {
        "TransactionID" : np.arange(1_000_000, 1_000_000 + n),
        "TransactionAmt": rng.exponential(100, n).round(2),
        "HourOfDay"     : rng.integers(0, 24, n),
        "FraudProb"     : fraud_prob.round(4),
        "ActualFraud"   : is_fraud,
        "DeviceRisk"    : rng.integers(0, 2, n),
        "ProductCD"     : rng.choice([0,1,2,3,4], n),
    }
    df = pd.DataFrame(data)
    def tier(p):
        if p >= 0.75: return "Critical"
        if p >= 0.40: return "Suspicious"
        return "Clear"
    df["RiskTier"] = df["FraudProb"].map(tier)
    return df

if df_pred is None:
    df_pred = _demo_data()
    DEMO_MODE = True
else:
    DEMO_MODE = False

# Ensure RiskTier column exists
if "RiskTier" not in df_pred.columns:
    def _tier(p):
        if p >= 0.75: return "Critical"
        if p >= 0.40: return "Suspicious"
        return "Clear"
    df_pred["RiskTier"] = df_pred["FraudProb"].map(_tier)

TIER_COLORS = {"Critical": "#ef4444", "Suspicious": "#f59e0b", "Clear": "#22c55e"}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔐 FraudGuard AI")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Transaction Explorer", "🧠 SHAP Explainer"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # ── Global sidebar filters ─────────────────────────────────────────────
    st.markdown("### 🎛️ Global Filters")

    risk_filter = st.multiselect(
        "Risk Tier",
        options=["Critical", "Suspicious", "Clear"],
        default=["Critical", "Suspicious", "Clear"],
    )

    amt_min = float(df_pred["TransactionAmt"].min())
    amt_max = float(df_pred["TransactionAmt"].max())
    amt_range = st.slider(
        "Transaction Amount ($)",
        min_value=amt_min,
        max_value=min(amt_max, 5000.0),
        value=(amt_min, min(amt_max, 5000.0)),
        format="$%.0f",
    )

    if "HourOfDay" in df_pred.columns:
        hour_range = st.slider("Hour of Day", 0, 23, (0, 23))
    else:
        hour_range = (0, 23)

    st.markdown("---")
    st.markdown(f"**Optimal Threshold** `{THRESHOLD:.3f}`")
    if DEMO_MODE:
        st.warning("⚠️ Demo mode — run the notebook first to load real model artefacts.")
    else:
        st.success("✅ Live model loaded")
    st.markdown("---")
    st.caption("Built with LightGBM + SHAP + Streamlit")

# ── Apply filters ──────────────────────────────────────────────────────────
df_filtered = df_pred[
    df_pred["RiskTier"].isin(risk_filter) &
    df_pred["TransactionAmt"].between(amt_range[0], amt_range[1])
].copy()
if "HourOfDay" in df_filtered.columns:
    df_filtered = df_filtered[
        df_filtered["HourOfDay"].between(hour_range[0], hour_range[1])
    ]

# ══════════════════════════════════════════════════════════════════════════════
# ███  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 📊 Fraud Operations — Overview")
    if DEMO_MODE:
        st.info("Running on demo data. Execute **analysis.ipynb** to load real predictions.")

    # ── KPI Row ───────────────────────────────────────────────────────────
    total     = len(df_filtered)
    fraud_cnt = int(df_filtered["ActualFraud"].sum()) if "ActualFraud" in df_filtered.columns \
                else int((df_filtered["FraudProb"] >= THRESHOLD).sum())
    det_rate  = fraud_cnt / total * 100 if total > 0 else 0
    avg_fraud_amt = df_filtered.loc[
        df_filtered["FraudProb"] >= THRESHOLD, "TransactionAmt"
    ].mean() if (df_filtered["FraudProb"] >= THRESHOLD).any() else 0

    critical_cnt   = (df_filtered["RiskTier"] == "Critical").sum()
    suspicious_cnt = (df_filtered["RiskTier"] == "Suspicious").sum()
    clear_cnt      = (df_filtered["RiskTier"] == "Clear").sum()

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, "🔢 Total Transactions",  f"{total:,}",         "Filtered view"),
        (c2, "🚨 Fraud Detected",       f"{fraud_cnt:,}",     f"{det_rate:.2f}% detection rate"),
        (c3, "💰 Avg Fraud Amount",     f"${avg_fraud_amt:,.2f}", "Per flagged transaction"),
        (c4, "🔴 Critical Risk",        f"{critical_cnt:,}",  "Prob ≥ 0.75 — block immediately"),
    ]
    for col, label, value, sub in kpis:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Charts ─────────────────────────────────────────────────────
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="section-header">Risk Tier Distribution</div>',
                    unsafe_allow_html=True)
        tier_counts = df_filtered["RiskTier"].value_counts()
        if PLOTLY_OK:
            fig_donut = go.Figure(go.Pie(
                labels=tier_counts.index,
                values=tier_counts.values,
                hole=0.55,
                marker_colors=[TIER_COLORS.get(t, "#94a3b8") for t in tier_counts.index],
                textinfo="label+percent",
            ))
            fig_donut.update_layout(
                showlegend=True, height=320,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.bar_chart(tier_counts)

    with col_b:
        st.markdown('<div class="section-header">Fraud Probability Distribution</div>',
                    unsafe_allow_html=True)
        if PLOTLY_OK:
            fig_hist = px.histogram(
                df_filtered, x="FraudProb", nbins=60,
                color_discrete_sequence=["#3b82f6"],
                labels={"FraudProb": "Fraud Probability"},
            )
            fig_hist.add_vline(x=THRESHOLD, line_dash="dash",
                               line_color="#ef4444",
                               annotation_text=f"Threshold {THRESHOLD:.2f}",
                               annotation_position="top right")
            fig_hist.update_layout(
                height=320, margin=dict(t=10, b=30, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.area_chart(df_filtered["FraudProb"].value_counts().sort_index())

    # ── Row 3: Hour of Day + Amount by Tier ───────────────────────────────
    col_c, col_d = st.columns([1, 1])

    with col_c:
        st.markdown('<div class="section-header">Fraud Rate by Hour of Day</div>',
                    unsafe_allow_html=True)
        if "HourOfDay" in df_filtered.columns and PLOTLY_OK:
            hourly = df_filtered.groupby("HourOfDay")["FraudProb"].mean().reset_index()
            fig_hr = px.bar(hourly, x="HourOfDay", y="FraudProb",
                            color="FraudProb", color_continuous_scale="Reds",
                            labels={"FraudProb":"Avg Fraud Prob","HourOfDay":"Hour"})
            fig_hr.update_layout(height=300, margin=dict(t=10,b=30,l=10,r=10),
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="#f8fafc", coloraxis_showscale=False)
            st.plotly_chart(fig_hr, use_container_width=True)
        else:
            st.info("HourOfDay feature not available in current data.")

    with col_d:
        st.markdown('<div class="section-header">Avg Transaction Amount by Risk Tier</div>',
                    unsafe_allow_html=True)
        if PLOTLY_OK:
            amt_by_tier = df_filtered.groupby("RiskTier")["TransactionAmt"].mean().reset_index()
            fig_amt = px.bar(
                amt_by_tier, x="RiskTier", y="TransactionAmt",
                color="RiskTier",
                color_discrete_map=TIER_COLORS,
                labels={"TransactionAmt":"Avg Amount ($)", "RiskTier":"Risk Tier"},
            )
            fig_amt.update_layout(height=300, margin=dict(t=10,b=30,l=10,r=10),
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="#f8fafc", showlegend=False)
            st.plotly_chart(fig_amt, use_container_width=True)

    # ── Row 4: Tier Summary Table ──────────────────────────────────────────
    st.markdown('<div class="section-header">Risk Tier Summary</div>',
                unsafe_allow_html=True)
    tier_tbl = df_filtered.groupby("RiskTier").agg(
        Transactions  = ("FraudProb", "count"),
        AvgFraudProb  = ("FraudProb", "mean"),
        AvgAmount     = ("TransactionAmt", "mean"),
    ).round(4).reset_index()
    tier_tbl["Transactions"] = tier_tbl["Transactions"].apply(lambda x: f"{x:,}")
    tier_tbl["AvgFraudProb"] = tier_tbl["AvgFraudProb"].apply(lambda x: f"{x:.4f}")
    tier_tbl["AvgAmount"]    = tier_tbl["AvgAmount"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(tier_tbl, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# ███  PAGE 2 — TRANSACTION EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Transaction Explorer":
    st.markdown("# 🔍 Transaction Explorer")
    st.caption("Search, filter, and inspect individual transactions with live risk scores.")

    # ── Search bar ────────────────────────────────────────────────────────
    search_id = st.text_input(
        "🔎 Search by TransactionID",
        placeholder="e.g. 2987004",
    )

    show_only = st.selectbox(
        "Show Transactions",
        ["All", "Fraud Only (actual)", "Legit Only (actual)",
         "Critical Risk", "Suspicious", "Clear"],
    )

    cols_to_show = ["TransactionID", "TransactionAmt", "FraudProb", "RiskTier"]
    for c in ["HourOfDay", "DeviceRisk", "ProductCD", "ActualFraud"]:
        if c in df_filtered.columns:
            cols_to_show.append(c)

    df_view = df_filtered[cols_to_show].copy()

    # Apply search
    if search_id.strip():
        try:
            sid = int(search_id.strip())
            df_view = df_view[df_view["TransactionID"] == sid]
        except ValueError:
            st.error("Please enter a valid numeric TransactionID.")

    # Apply show filter
    if show_only == "Fraud Only (actual)" and "ActualFraud" in df_view.columns:
        df_view = df_view[df_view["ActualFraud"] == 1]
    elif show_only == "Legit Only (actual)" and "ActualFraud" in df_view.columns:
        df_view = df_view[df_view["ActualFraud"] == 0]
    elif show_only in ["Critical Risk", "Suspicious", "Clear"]:
        tier_map = {"Critical Risk": "Critical"}
        df_view = df_view[df_view["RiskTier"] == tier_map.get(show_only, show_only)]

    st.markdown(f"**Showing {len(df_view):,} transactions**")

    # ── Colour-coded table ────────────────────────────────────────────────
    def highlight_row(row):
        tier = row.get("RiskTier", "Clear")
        if tier == "Critical":
            return ["background-color: #fee2e2"] * len(row)
        elif tier == "Suspicious":
            return ["background-color: #fef9c3"] * len(row)
        return ["background-color: #f0fdf4"] * len(row)

    styled_table = (
        df_view.head(500)
        .style
        .apply(highlight_row, axis=1)
        .format({
            "TransactionAmt": "${:,.2f}",
            "FraudProb"     : "{:.4f}",
        })
    )
    st.dataframe(styled_table, use_container_width=True, height=420)

    # ── Individual transaction drill-down ─────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Live Risk Score — Single Transaction</div>',
                unsafe_allow_html=True)

    tid_input = st.number_input(
        "Enter TransactionID for risk score",
        min_value=int(df_pred["TransactionID"].min()),
        max_value=int(df_pred["TransactionID"].max()),
        step=1,
    )

    if st.button("🚀 Get Risk Score", type="primary"):
        row = df_pred[df_pred["TransactionID"] == tid_input]
        if row.empty:
            st.error(f"TransactionID {tid_input} not found in the current dataset.")
        else:
            row = row.iloc[0]
            prob  = row["FraudProb"]
            tier  = row["RiskTier"]
            amt   = row["TransactionAmt"]

            badge_cls = {
                "Critical"  : "badge-critical",
                "Suspicious": "badge-suspicious",
                "Clear"     : "badge-clear",
            }.get(tier, "badge-clear")

            decision = (
                "🚫 BLOCK — Immediate review required" if tier == "Critical"
                else "⚠️ FLAG — Step-up authentication recommended" if tier == "Suspicious"
                else "✅ ALLOW — Normal transaction"
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Fraud Probability", f"{prob:.4f}", f"Threshold: {THRESHOLD:.3f}")
            col2.metric("Transaction Amount", f"${amt:,.2f}")
            col3.metric("Risk Tier", tier)

            st.markdown(f"**Decision:** {decision}")

            if PLOTLY_OK:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    delta={"reference": THRESHOLD * 100},
                    gauge={
                        "axis"     : {"range": [0, 100], "ticksuffix": "%"},
                        "bar"      : {"color": "#ef4444" if prob >= THRESHOLD else "#22c55e"},
                        "steps"    : [
                            {"range": [0,  40], "color": "#dcfce7"},
                            {"range": [40, 75], "color": "#fef9c3"},
                            {"range": [75, 100],"color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line" : {"color": "#1e293b", "width": 3},
                            "thickness": 0.8,
                            "value": THRESHOLD * 100,
                        },
                    },
                    title={"text": f"Fraud Risk Gauge — TxID {tid_input}"},
                    number={"suffix": "%", "valueformat": ".2f"},
                ))
                gauge.update_layout(height=300,
                                    margin=dict(t=50, b=10, l=30, r=30),
                                    paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(gauge, use_container_width=True)

    # ── Amount Scatter ─────────────────────────────────────────────────────
    if PLOTLY_OK and "HourOfDay" in df_filtered.columns:
        st.markdown("---")
        st.markdown('<div class="section-header">Transaction Amount vs Hour of Day</div>',
                    unsafe_allow_html=True)
        sample = df_filtered.sample(min(3000, len(df_filtered)), random_state=42)
        fig_sc = px.scatter(
            sample, x="HourOfDay", y="TransactionAmt",
            color="FraudProb", color_continuous_scale="RdYlGn_r",
            opacity=0.55, log_y=True,
            labels={"HourOfDay":"Hour of Day", "TransactionAmt":"Amount ($)",
                    "FraudProb":"Fraud Prob"},
            title="Amount vs Hour — coloured by Fraud Probability",
        )
        fig_sc.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="#f8fafc")
        st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ███  PAGE 3 — SHAP EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainer":
    st.markdown("# 🧠 SHAP Explainer — Why Did the Model Flag This Transaction?")
    st.caption(
        "Enter a TransactionID to see which features pushed the fraud probability "
        "up or down, with a plain-English explanation."
    )

    tid_shap = st.number_input(
        "TransactionID",
        min_value=int(df_pred["TransactionID"].min()),
        max_value=int(df_pred["TransactionID"].max()),
        step=1,
        key="shap_tid",
    )

    run_shap = st.button("🔬 Explain This Transaction", type="primary")

    if run_shap:
        row = df_pred[df_pred["TransactionID"] == tid_shap]
        if row.empty:
            st.error("TransactionID not found.")
        elif model is None or scaler is None:
            st.warning(
                "Model or scaler artefacts not found. "
                "Please run **analysis.ipynb** first, then restart the app."
            )
        else:
            row_data = row.iloc[0]
            prob     = row_data["FraudProb"]
            tier     = row_data["RiskTier"]

            # ── Header ────────────────────────────────────────────────────
            col1, col2, col3 = st.columns(3)
            col1.metric("Fraud Probability",  f"{prob:.4f}")
            col2.metric("Risk Tier",          tier)
            col3.metric("Transaction Amount", f"${row_data['TransactionAmt']:,.2f}")

            st.markdown("---")

            # ── SHAP waterfall ─────────────────────────────────────────────
            feat_cols_available = [c for c in FEAT_COLS if c in row_data.index]
            if not feat_cols_available:
                st.warning("Feature columns not available in stored predictions.")
            else:
                with st.spinner("Computing SHAP values …"):
                    try:
                        X_row    = pd.DataFrame([row_data[feat_cols_available]],
                                                columns=feat_cols_available)
                        X_row_sc = pd.DataFrame(
                            scaler.transform(X_row), columns=feat_cols_available
                        )
                        explainer    = shap.TreeExplainer(model)
                        shap_vals    = explainer(X_row_sc)

                        fig_wf, ax_wf = plt.subplots(figsize=(12, 6))
                        shap.plots.waterfall(shap_vals[0], max_display=15, show=False)
                        ax_wf = plt.gca()
                        ax_wf.set_title(
                            f"SHAP Waterfall — TxID {tid_shap}  |  "
                            f"P(fraud)={prob:.4f}  |  {tier} Risk",
                            fontsize=12, fontweight="bold"
                        )
                        plt.tight_layout()
                        st.pyplot(fig_wf, use_container_width=True)
                        plt.close()

                        # ── Plain-English Explanation ──────────────────────
                        st.markdown("---")
                        st.markdown("### 📝 Plain-English Explanation")

                        shap_arr  = shap_vals.values[0]
                        feat_names = feat_cols_available
                        pairs     = sorted(zip(feat_names, shap_arr),
                                           key=lambda x: abs(x[1]), reverse=True)[:8]

                        verdict = (
                            "🚨 **FLAGGED AS FRAUD**" if prob >= THRESHOLD
                            else "✅ **CLEARED AS LEGITIMATE**"
                        )
                        st.markdown(
                            f"**Model verdict:** {verdict} "
                            f"(fraud probability = `{prob:.4f}`, "
                            f"threshold = `{THRESHOLD:.3f}`)"
                        )

                        st.markdown("**Top factors driving this prediction:**")
                        for i, (feat, val) in enumerate(pairs, 1):
                            direction = "⬆️ increased" if val > 0 else "⬇️ decreased"
                            raw_val   = row_data.get(feat, "N/A")
                            st.markdown(
                                f"**{i}.** `{feat}` = `{raw_val}` → "
                                f"{direction} fraud risk by `{val:+.4f}` SHAP units"
                            )

                        st.markdown("---")
                        st.markdown("### 💡 What Should the Analyst Do?")
                        if tier == "Critical":
                            st.error(
                                "**Block immediately.** This transaction exceeds the critical "
                                "threshold. Flag the card, notify the cardholder, and escalate "
                                "to the fraud investigation team."
                            )
                        elif tier == "Suspicious":
                            st.warning(
                                "**Step-up authentication.** Send a one-time password to the "
                                "registered mobile number. Allow the transaction only after "
                                "successful verification."
                            )
                        else:
                            st.success(
                                "**Allow.** Low fraud probability. Log for passive monitoring "
                                "and continue standard velocity checks."
                            )

                    except Exception as e:
                        st.error(f"SHAP computation failed: {e}")
                        st.info(
                            "This usually means the model artefact and stored predictions "
                            "are from different runs. Re-run the notebook end-to-end."
                        )

    # ── Global SHAP summary image ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌐 Global SHAP Feature Importance")
    shap_img = os.path.join(BASE, "..", "shap_summary.png")
    if os.path.exists(shap_img):
        st.image(shap_img, caption="Top 20 features by mean |SHAP value|",
                 use_column_width=True)
    else:
        st.info("Run the notebook to generate `shap_summary.png`, then refresh.")

    # ── Feature importance chart (static fallback) ─────────────────────────
    if model is not None and FEAT_COLS and PLOTLY_OK:
        try:
            fi = pd.Series(model.feature_importances_, index=FEAT_COLS)
            top_fi = fi.nlargest(20).sort_values()
            fig_fi = px.bar(
                top_fi, orientation="h",
                labels={"value": "Importance", "index": "Feature"},
                title="LightGBM Native Feature Importance (Top 20)",
                color=top_fi.values,
                color_continuous_scale="Blues",
            )
            fig_fi.update_layout(height=500, showlegend=False,
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="#f8fafc",
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception:
            pass
