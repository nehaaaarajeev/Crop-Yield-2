# ============================================================
# CROP YIELD DASHBOARD — app.py
# Streamlit + Plotly + AI-Powered Insights (Anthropic API)
# Run: streamlit run app.py
# ============================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix
)

# ─────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 Crop Yield Intelligence Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }
    section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E0E0; }

    .kpi-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px 18px 16px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 4px solid var(--accent);
        text-align: center;
    }
    .kpi-label { font-size: 13px; color: #6B7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 32px; font-weight: 800; color: #111827; margin: 6px 0 4px; }
    .kpi-sub   { font-size: 12px; color: #9CA3AF; }

    .section-header {
        font-size: 19px; font-weight: 700; color: #1F2937;
        border-left: 4px solid #16A34A; padding-left: 10px;
        margin-bottom: 6px;
    }

    .insight-box {
        background: linear-gradient(135deg, #F0FDF4, #ECFDF5);
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0 18px 0;
        font-size: 14px;
        color: #1F2937;
        line-height: 1.7;
    }
    .insight-label {
        font-size: 11px;
        font-weight: 700;
        color: #16A34A;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }

    button[data-baseweb="tab"] { font-size: 14px !important; font-weight: 600 !important; }
    details summary { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# COLOUR PALETTES
# ─────────────────────────────────────────────────
SEASON_COLORS   = {"Kharif": "#16A34A", "Rabi": "#2563EB", "Zaid": "#F97316"}
SOIL_COLORS     = {"Loamy": "#92400E", "Clay": "#7F1D1D", "Sandy": "#FBBF24", "Silty": "#6B7280", "Black": "#111827"}
SUCCESS_COLORS  = {1: "#16A34A", 0: "#DC2626"}
SUCCESS_LABELS  = {1: "Success", 0: "Failure"}
PLOTLY_TEMPLATE = "plotly_white"

# ─────────────────────────────────────────────────
# AI INSIGHT ENGINE
# ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_ai_insight(chart_title: str, data_summary: str) -> str:
    """
    Calls Anthropic API to generate a 2-3 sentence insight.
    Cached so it doesn't re-fire on every Streamlit rerun
    as long as the chart title + data summary are unchanged.
    API key must be set in Streamlit Cloud secrets as ANTHROPIC_API_KEY.
    """
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return (
                "⚠️ AI insights are disabled. "
                "Add ANTHROPIC_API_KEY to your Streamlit Cloud secrets "
                "(App settings → Secrets) to enable them."
            )
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            f'You are an expert agricultural data analyst.\n'
            f'Analyse the following data summary from the chart "{chart_title}" '
            f'and provide exactly 2-3 concise, actionable sentences of insight.\n'
            f'Focus on patterns, anomalies, or recommendations valuable to a farmer or agricultural officer.\n'
            f'Do NOT use bullet points. Write in plain flowing sentences only.\n\n'
            f'Data summary:\n{data_summary}'
        )
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=180,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"AI insight unavailable: {str(e)}"


def show_insight(chart_title: str, data_summary: str):
    """Renders the styled AI insight box below a chart."""
    with st.spinner("🤖 Generating AI insight..."):
        insight = get_ai_insight(chart_title, data_summary)
    st.markdown(
        f'<div class="insight-box">'
        f'<div class="insight-label">🤖 AI Insight</div>'
        f'{insight}'
        f'</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING  (cached)
# ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "Crop_Yield.csv"),
        encoding="latin1"
    )
    return df


@st.cache_data
def preprocess_and_train(df_raw):
    df = df_raw.copy()

    # Null handling
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Label encoding
    df_enc = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df_enc[col] = le.fit_transform(df_enc[col])

    # Split
    X = df_enc.drop(columns=["Yield Success"])
    y = df_enc["Yield Success"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Train
    dt  = DecisionTreeClassifier(random_state=42)
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    gbt = GradientBoostingClassifier(n_estimators=100, random_state=42)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gbt.fit(X_train, y_train)
    trained = {"Decision Tree": dt, "Random Forest": rf, "Gradient Boosted Trees": gbt}

    # Evaluate
    eval_rows = []
    for name, mdl in trained.items():
        y_pred = mdl.predict(X_test)
        eval_rows.append({
            "Model":     name,
            "Train Acc": round(accuracy_score(y_train, mdl.predict(X_train)), 4),
            "Test Acc":  round(accuracy_score(y_test,  y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test,    y_pred, zero_division=0), 4),
            "CM":        confusion_matrix(y_test, y_pred),
            "FI":        dict(zip(X.columns, mdl.feature_importances_))
        })
    return trained, eval_rows, X_test, y_test, X.columns.tolist()


# ─────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────
df_raw = load_data()
trained_models, eval_rows, X_test, y_test, feature_names = preprocess_and_train(df_raw)
df = df_raw.copy()
df["Yield Label"] = df["Yield Success"].map(SUCCESS_LABELS)

# ─────────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/wheat.png", width=60)
    st.title("🌾 Crop Yield\nDashboard")
    st.caption("Powered by ML + AI Insights")
    st.divider()

    st.markdown("### 🔍 Filters")
    sel_crop   = st.multiselect("Crop Type",        sorted(df["Crop Type"].unique()),       default=sorted(df["Crop Type"].unique()))
    sel_season = st.multiselect("Season",            sorted(df["Season"].unique()),           default=sorted(df["Season"].unique()))
    sel_soil   = st.multiselect("Soil Type",         sorted(df["Soil Type"].unique()),        default=sorted(df["Soil Type"].unique()))
    sel_irr    = st.multiselect("Irrigation Type",   sorted(df["Irrigation Type"].unique()),  default=sorted(df["Irrigation Type"].unique()))
    sel_seed   = st.multiselect("Seed Quality",      sorted(df["Seed Quality"].unique()),     default=sorted(df["Seed Quality"].unique()))
    sel_farm   = st.multiselect("Farming Practice",  sorted(df["Farming Practice"].unique()), default=sorted(df["Farming Practice"].unique()))

    st.markdown("#### 📊 Numeric Ranges")
    sel_rain = st.slider("Rainfall (mm)",
        float(df["Rainfall (mm)"].min()), float(df["Rainfall (mm)"].max()),
        (float(df["Rainfall (mm)"].min()), float(df["Rainfall (mm)"].max())), step=1.0)
    sel_ph = st.slider("Soil pH",
        float(df["Soil Ph"].min()), float(df["Soil Ph"].max()),
        (float(df["Soil Ph"].min()), float(df["Soil Ph"].max())), step=0.1)
    sel_fert = st.slider("Fertilizer Used (kg)",
        float(df["Fertilizer Used (kg)"].min()), float(df["Fertilizer Used (kg)"].max()),
        (float(df["Fertilizer Used (kg)"].min()), float(df["Fertilizer Used (kg)"].max())), step=1.0)

    st.divider()
    st.caption("© 2025 Crop Yield Intelligence")

# Apply filters
mask = (
    df["Crop Type"].isin(sel_crop) &
    df["Season"].isin(sel_season) &
    df["Soil Type"].isin(sel_soil) &
    df["Irrigation Type"].isin(sel_irr) &
    df["Seed Quality"].isin(sel_seed) &
    df["Farming Practice"].isin(sel_farm) &
    df["Rainfall (mm)"].between(*sel_rain) &
    df["Soil Ph"].between(*sel_ph) &
    df["Fertilizer Used (kg)"].between(*sel_fert)
)
dff = df[mask].copy()

# ─────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#16A34A,#15803D);
            padding:28px 32px;border-radius:14px;margin-bottom:20px;'>
    <h1 style='color:white;margin:0;font-size:30px;'>
        🌾 Crop Yield Intelligence Dashboard
    </h1>
    <p style='color:#BBF7D0;margin:6px 0 0;font-size:15px;'>
        End-to-end crop performance analytics · ML model insights · 🤖 AI-powered chart analysis
    </p>
</div>
""", unsafe_allow_html=True)
st.caption(f"📦 Showing **{len(dff):,}** records after filters  |  Total dataset: **{len(df):,}** records")

# ─────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────
tab_overview, tab_eda, tab_model, tab_fi = st.tabs([
    "🏠 Overview", "📊 EDA", "🤖 Model Performance", "🌟 Feature Importance"
])

# ════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════
with tab_overview:

    # ── KPI Cards ──────────────────────────────
    st.markdown('<p class="section-header">📌 Key Performance Indicators</p>', unsafe_allow_html=True)

    success_rate  = round(dff["Yield Success"].mean() * 100, 1) if len(dff) else 0
    avg_exp_yield = round(dff["Expected Yield (kg per acre)"].mean(), 1) if len(dff) else 0
    avg_act_yield = round(dff["Actual Yield (kg per acre)"].mean(), 1) if len(dff) else 0
    avg_rainfall  = round(dff["Rainfall (mm)"].mean(), 1) if len(dff) else 0
    avg_ph        = round(dff["Soil Ph"].mean(), 2) if len(dff) else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "#16A34A", "✅ Yield Success Rate",  f"{success_rate}%",      "of filtered crops"),
        (c2, "#2563EB", "📦 Avg Expected Yield",   f"{avg_exp_yield:,} kg", "per acre"),
        (c3, "#7C3AED", "📦 Avg Actual Yield",     f"{avg_act_yield:,} kg", "per acre"),
        (c4, "#0891B2", "🌧️ Avg Rainfall",         f"{avg_rainfall} mm",   "per season"),
        (c5, "#D97706", "🧪 Avg Soil pH",          f"{avg_ph}",            "pH units"),
    ]
    for col, accent, label, value, sub in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:{accent}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    kpi_summary = (
        f"Overall yield success rate: {success_rate}%. "
        f"Average expected yield: {avg_exp_yield} kg/acre vs actual yield: {avg_act_yield} kg/acre "
        f"(gap: {round(avg_exp_yield - avg_act_yield, 1)} kg/acre). "
        f"Average rainfall: {avg_rainfall} mm. Average soil pH: {avg_ph}. "
        f"Filtered dataset: {len(dff)} records."
    )
    show_insight("Key Performance Indicators Overview", kpi_summary)

    st.divider()

    # ── Comparative Bar Charts ──────────────────
    st.markdown('<p class="section-header">📊 Yield Success by Category</p>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        grp = dff.groupby(["Crop Type", "Yield Label"]).size().reset_index(name="Count")
        fig = px.bar(grp, x="Crop Type", y="Count", color="Yield Label",
                     barmode="group",
                     color_discrete_map={"Success": "#16A34A", "Failure": "#DC2626"},
                     title="Yield Success by Crop Type",
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(legend_title_text="", title_font_size=14, height=360)
        st.plotly_chart(fig, use_container_width=True)
        show_insight("Yield Success by Crop Type", grp.to_string(index=False))

    with col_b:
        grp2 = dff.groupby(["Season", "Yield Label"]).size().reset_index(name="Count")
        fig2 = px.bar(grp2, x="Season", y="Count", color="Season",
                      facet_col="Yield Label",
                      color_discrete_map=SEASON_COLORS,
                      title="Yield Success by Season",
                      template=PLOTLY_TEMPLATE)
        fig2.update_layout(title_font_size=14, height=360, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        show_insight("Yield Success by Season (Kharif / Rabi / Zaid)", grp2.to_string(index=False))

    with col_c:
        grp3 = dff.groupby(["Soil Type", "Yield Label"]).size().reset_index(name="Count")
        fig3 = px.bar(grp3, x="Soil Type", y="Count", color="Soil Type",
                      facet_col="Yield Label",
                      color_discrete_map=SOIL_COLORS,
                      title="Yield Success by Soil Type",
                      template=PLOTLY_TEMPLATE)
        fig3.update_layout(title_font_size=14, height=360, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        show_insight("Yield Success by Soil Type", grp3.to_string(index=False))

    st.divider()

    # ── Socio-economic Charts ───────────────────
    st.markdown('<p class="section-header">💳 Socio-Economic Factors</p>', unsafe_allow_html=True)
    pie_c1, pie_c2 = st.columns(2)

    with pie_c1:
        credit_df = dff.groupby(["Access to Credit", "Yield Label"]).size().reset_index(name="Count")
        credit_df["Credit Label"] = credit_df["Access to Credit"].map({1: "With Credit", 0: "No Credit"})
        fig_p1 = px.sunburst(credit_df, path=["Credit Label", "Yield Label"], values="Count",
                              color="Yield Label",
                              color_discrete_map={"Success": "#16A34A", "Failure": "#DC2626"},
                              title="Yield Success by Access to Credit",
                              template=PLOTLY_TEMPLATE)
        fig_p1.update_layout(title_font_size=14, height=380)
        st.plotly_chart(fig_p1, use_container_width=True)
        show_insight(
            "Yield Success by Access to Credit",
            credit_df[["Credit Label","Yield Label","Count"]].to_string(index=False)
        )

    with pie_c2:
        subsidy_df = dff.groupby(["Govt. Subsidy Received", "Yield Label"]).size().reset_index(name="Count")
        subsidy_df["Subsidy Label"] = subsidy_df["Govt. Subsidy Received"].map({1: "Got Subsidy", 0: "No Subsidy"})
        fig_p2 = px.sunburst(subsidy_df, path=["Subsidy Label", "Yield Label"], values="Count",
                              color="Yield Label",
                              color_discrete_map={"Success": "#16A34A", "Failure": "#DC2626"},
                              title="Yield Success by Govt. Subsidy",
                              template=PLOTLY_TEMPLATE)
        fig_p2.update_layout(title_font_size=14, height=380)
        st.plotly_chart(fig_p2, use_container_width=True)
        show_insight(
            "Yield Success by Government Subsidy",
            subsidy_df[["Subsidy Label","Yield Label","Count"]].to_string(index=False)
        )

    st.divider()

    # ── Farmer Experience Trend ─────────────────
    st.markdown('<p class="section-header">📈 Yield Success vs Farmer Experience</p>', unsafe_allow_html=True)
    exp_grp = dff.groupby("Farmer Experience (years)")["Yield Success"].mean().reset_index()
    exp_grp.columns = ["Experience (years)", "Success Rate"]
    exp_grp["Success Rate (%)"] = (exp_grp["Success Rate"] * 100).round(1)

    fig_trend = px.line(exp_grp, x="Experience (years)", y="Success Rate (%)",
                        markers=True, line_shape="spline",
                        color_discrete_sequence=["#16A34A"],
                        title="Yield Success Rate (%) by Farmer Experience",
                        template=PLOTLY_TEMPLATE)
    fig_trend.update_traces(line_width=2.5, marker_size=7)
    fig_trend.add_hline(y=50, line_dash="dash", line_color="#DC2626",
                        annotation_text="50% Baseline", annotation_position="top left")
    fig_trend.update_layout(title_font_size=15, height=380,
                             xaxis_title="Farmer Experience (years)",
                             yaxis_title="Success Rate (%)")
    st.plotly_chart(fig_trend, use_container_width=True)

    exp_summary = (
        f"Experience range: {int(exp_grp['Experience (years)'].min())}–"
        f"{int(exp_grp['Experience (years)'].max())} years. "
        f"Highest success rate: {exp_grp['Success Rate (%)'].max()}% at "
        f"{int(exp_grp.loc[exp_grp['Success Rate (%)'].idxmax(), 'Experience (years)'])} years experience. "
        f"Lowest: {exp_grp['Success Rate (%)'].min()}%."
    )
    show_insight("Yield Success Rate vs Farmer Experience", exp_summary)

    with st.expander("📋 View Raw Data Sample"):
        st.dataframe(dff.head(50), use_container_width=True, height=280)


# ════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════
with tab_eda:

    st.markdown('<p class="section-header">📦 Distribution Analysis</p>', unsafe_allow_html=True)
    dist_c1, dist_c2 = st.columns(2)

    with dist_c1:
        fig_box = px.box(dff, x="Yield Label", y="Rainfall (mm)", color="Yield Label",
                         color_discrete_map={"Success": "#16A34A", "Failure": "#DC2626"},
                         title="Rainfall Distribution vs Yield Success",
                         template=PLOTLY_TEMPLATE, points="outliers")
        fig_box.update_layout(title_font_size=14, height=400, showlegend=False,
                               xaxis_title="Yield Outcome", yaxis_title="Rainfall (mm)")
        st.plotly_chart(fig_box, use_container_width=True)

        r_s = dff[dff["Yield Success"]==1]["Rainfall (mm)"]
        r_f = dff[dff["Yield Success"]==0]["Rainfall (mm)"]
        show_insight("Rainfall Distribution vs Yield Success",
            f"Success — mean: {r_s.mean():.1f} mm, median: {r_s.median():.1f} mm, std: {r_s.std():.1f} mm. "
            f"Failure — mean: {r_f.mean():.1f} mm, median: {r_f.median():.1f} mm, std: {r_f.std():.1f} mm."
        )

    with dist_c2:
        fig_hist = px.histogram(dff, x="Soil Ph", color="Yield Label",
                                nbins=30, barmode="overlay", opacity=0.7,
                                color_discrete_map={"Success": "#16A34A", "Failure": "#DC2626"},
                                title="Soil pH Distribution: Success vs Failure",
                                template=PLOTLY_TEMPLATE)
        fig_hist.update_layout(title_font_size=14, height=400,
                                xaxis_title="Soil pH", yaxis_title="Count",
                                legend_title_text="Yield Outcome")
        st.plotly_chart(fig_hist, use_container_width=True)

        p_s = dff[dff["Yield Success"]==1]["Soil Ph"]
        p_f = dff[dff["Yield Success"]==0]["Soil Ph"]
        show_insight("Soil pH Distribution: Success vs Failure",
            f"Success — mean pH: {p_s.mean():.2f}, range: {p_s.min():.1f}–{p_s.max():.1f}. "
            f"Failure — mean pH: {p_f.mean():.2f}, range: {p_f.min():.1f}–{p_f.max():.1f}."
        )

    st.divider()

    # ── Correlation Heatmap ─────────────────────
    st.markdown('<p class="section-header">🔥 Correlation Heatmap (Numeric Features)</p>', unsafe_allow_html=True)
    num_cols = [
        "Rainfall (mm)", "Soil Ph", "Fertilizer Used (kg)", "Pesticide Used (kg)",
        "Expected Yield (kg per acre)", "Actual Yield (kg per acre)",
        "Farm Size", "Soil Moisture (%)", "Avg Temperature (°C)",
        "Farmer Experience (years)", "Yield Success"
    ]
    corr = dff[num_cols].corr().round(2)

    fig_hm = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale="RdYlGn", zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text}", textfont_size=9,
        hoverongaps=False
    ))
    fig_hm.update_layout(
        title="Correlation Matrix — Numeric Variables",
        title_font_size=15, height=560,
        template=PLOTLY_TEMPLATE, xaxis_tickangle=-40,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    target_corr = corr["Yield Success"].drop("Yield Success").abs().sort_values(ascending=False)
    top3 = target_corr.head(3)
    show_insight("Correlation Heatmap — Numeric Variables",
        f"Top 3 features correlated with Yield Success: "
        f"{top3.index[0]} ({corr.loc[top3.index[0],'Yield Success']:+.2f}), "
        f"{top3.index[1]} ({corr.loc[top3.index[1],'Yield Success']:+.2f}), "
        f"{top3.index[2]} ({corr.loc[top3.index[2],'Yield Success']:+.2f}). "
        f"Matrix covers {len(num_cols)} numeric variables across {len(dff)} records."
    )

    st.divider()

    # ── Scatter ─────────────────────────────────
    st.markdown('<p class="section-header">🌦️ Scatter: Rainfall vs Actual Yield</p>', unsafe_allow_html=True)
    fig_sc = px.scatter(dff, x="Rainfall (mm)", y="Actual Yield (kg per acre)",
                        color="Yield Label", symbol="Season",
                        color_discrete_map={"Success": "#16A34A", "Failure": "#DC2626"},
                        opacity=0.6,
                        title="Rainfall vs Actual Yield — coloured by outcome, shaped by season",
                        template=PLOTLY_TEMPLATE)
    fig_sc.update_layout(title_font_size=14, height=420, legend_title_text="Outcome / Season")
    st.plotly_chart(fig_sc, use_container_width=True)

    sc_corr = dff["Rainfall (mm)"].corr(dff["Actual Yield (kg per acre)"])
    show_insight("Rainfall vs Actual Yield Scatter",
        f"Pearson correlation between rainfall and actual yield: {sc_corr:.3f}. "
        f"Rainfall range: {dff['Rainfall (mm)'].min():.0f}–{dff['Rainfall (mm)'].max():.0f} mm. "
        f"Yield range: {dff['Actual Yield (kg per acre)'].min()}–{dff['Actual Yield (kg per acre)'].max()} kg/acre. "
        f"Records: {len(dff[dff['Yield Success']==1])} success, {len(dff[dff['Yield Success']==0])} failure."
    )

    with st.expander("📊 Descriptive Statistics"):
        st.dataframe(
            dff[num_cols].describe().T.style.background_gradient(cmap="Greens"),
            use_container_width=True
        )


# ════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════
with tab_model:

    st.markdown('<p class="section-header">📐 Model Evaluation Summary</p>', unsafe_allow_html=True)

    m_cols = st.columns(len(eval_rows))
    for col, row in zip(m_cols, eval_rows):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:#2563EB;margin-bottom:8px;">
                <div class="kpi-label">{row['Model']}</div>
                <div class="kpi-value" style="font-size:22px;">{row['Test Acc']*100:.1f}%</div>
                <div class="kpi-sub">Test Accuracy</div>
            </div>""", unsafe_allow_html=True)

    # Grouped bar chart
    fig_eval = go.Figure()
    metrics = ["Train Acc", "Test Acc", "Precision", "Recall"]
    colors  = ["#2563EB", "#16A34A", "#7C3AED", "#D97706"]
    for metric, color in zip(metrics, colors):
        fig_eval.add_trace(go.Bar(
            name=metric,
            x=[r["Model"] for r in eval_rows],
            y=[r[metric] for r in eval_rows],
            marker_color=color,
            text=[f"{r[metric]:.3f}" for r in eval_rows],
            textposition="outside"
        ))
    fig_eval.update_layout(
        barmode="group", title="Model Comparison — All Metrics",
        template=PLOTLY_TEMPLATE,
        yaxis=dict(range=[0, 1.1], title="Score"),
        xaxis_title="Model", height=420, title_font_size=15,
        legend_title_text="Metric"
    )
    st.plotly_chart(fig_eval, use_container_width=True)

    show_insight("Model Comparison — All Metrics",
        " | ".join([
            f"{r['Model']}: Train={r['Train Acc']:.3f}, Test={r['Test Acc']:.3f}, "
            f"Precision={r['Precision']:.3f}, Recall={r['Recall']:.3f}"
            for r in eval_rows
        ])
    )

    with st.expander("📋 Full Evaluation Table"):
        metric_rows = [{k: v for k, v in r.items() if k not in ("CM","FI")} for r in eval_rows]
        eval_disp = pd.DataFrame(metric_rows)
        eval_disp.columns = ["Model","Train Accuracy","Test Accuracy","Precision","Recall"]
        st.dataframe(
            eval_disp.set_index("Model").style
            .background_gradient(cmap="Greens", subset=["Test Accuracy","Precision","Recall"])
            .format("{:.4f}", subset=["Train Accuracy","Test Accuracy","Precision","Recall"]),
            use_container_width=True
        )

    st.divider()

    # ── Confusion Matrices ──────────────────────
    st.markdown('<p class="section-header">🔲 Confusion Matrices</p>', unsafe_allow_html=True)
    class_labels = ["Failure", "Success"]
    cm_cols = st.columns(3)

    for col, row in zip(cm_cols, eval_rows):
        cm = row["CM"]
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        with col:
            annotations = []
            quad = [["TN","FP"],["FN","TP"]]
            for i in range(2):
                for j in range(2):
                    annotations.append(dict(
                        x=class_labels[j], y=class_labels[i],
                        text=f"<b>{cm[i,j]}</b><br>"
                             f"<span style='font-size:11px;color:#888'>{quad[i][j]}</span>",
                        showarrow=False, font=dict(size=16, color="#111827"),
                        xref="x", yref="y"
                    ))
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=class_labels, y=class_labels,
                colorscale=[[0,"#FEF2F2"],[0.5,"#BFDBFE"],[1,"#1D4ED8"]],
                showscale=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            ))
            fig_cm.update_layout(
                title=dict(text=row["Model"], font_size=13, x=0.5),
                annotations=annotations,
                xaxis_title="Predicted", yaxis_title="Actual",
                height=320, template=PLOTLY_TEMPLATE,
                margin=dict(l=10, r=10, t=40, b=30)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            show_insight(
                f"Confusion Matrix — {row['Model']}",
                f"TP={tp}, TN={tn}, FP={fp}, FN={fn}. "
                f"Test Accuracy={row['Test Acc']:.3f}, Precision={row['Precision']:.3f}, "
                f"Recall={row['Recall']:.3f}. "
                f"False positives (crop predicted as success but failed): {fp}. "
                f"False negatives (crop predicted as failure but succeeded): {fn}."
            )


# ════════════════════════════════════════════════
# TAB 4 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════
with tab_fi:

    st.markdown('<p class="section-header">🌟 Feature Importance — All Models</p>', unsafe_allow_html=True)

    for row in eval_rows:
        name      = row["Model"]
        fi_series = pd.Series(row["FI"]).sort_values(ascending=True)
        top_feat  = fi_series.idxmax()
        top_score = fi_series.max()

        with st.container():
            fig_fi = go.Figure(go.Bar(
                x=fi_series.values,
                y=fi_series.index.tolist(),
                orientation="h",
                marker=dict(color=fi_series.values, colorscale="Greens", showscale=False),
                text=[f"{v:.4f}" for v in fi_series.values],
                textposition="outside",
                hovertemplate="%{y}: %{x:.4f}<extra></extra>"
            ))
            fig_fi.update_layout(
                title=dict(text=f"Feature Importance — {name}", font_size=15, x=0),
                xaxis_title="Importance Score", yaxis_title="Feature",
                height=430, template=PLOTLY_TEMPLATE,
                margin=dict(l=20, r=80, t=50, b=40)
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            top5 = fi_series.sort_values(ascending=False).head(5)
            show_insight(
                f"Feature Importance — {name}",
                f"Top 5 features: "
                + ", ".join([f"{k} ({v:.4f})" for k,v in top5.items()])
                + f". Most important: '{top_feat}' (score: {top_score:.4f}). "
                f"Total features evaluated: {len(fi_series)}."
            )
            st.divider()

    # Side-by-side comparison
    with st.expander("📊 Side-by-side Feature Importance Comparison"):
        fi_data = {row["Model"]: row["FI"] for row in eval_rows}
        fi_df   = pd.DataFrame(fi_data).fillna(0)
        fi_df   = fi_df.loc[fi_df.mean(axis=1).sort_values(ascending=False).index]

        fi_colors = {
            "Decision Tree": "#2563EB",
            "Random Forest": "#16A34A",
            "Gradient Boosted Trees": "#F97316"
        }
        fig_compare = go.Figure()
        for model_name, color in fi_colors.items():
            fig_compare.add_trace(go.Bar(
                name=model_name,
                x=fi_df.index.tolist(),
                y=fi_df[model_name].tolist(),
                marker_color=color
            ))
        fig_compare.update_layout(
            barmode="group",
            title="Feature Importance Comparison Across All Models",
            template=PLOTLY_TEMPLATE, height=460,
            xaxis_title="Feature", yaxis_title="Importance Score",
            xaxis_tickangle=-35, legend_title_text="Model",
            title_font_size=15
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        show_insight(
            "Feature Importance Comparison Across All Models",
            f"Decision Tree top feature: {pd.Series(eval_rows[0]['FI']).idxmax()}. "
            f"Random Forest top feature: {pd.Series(eval_rows[1]['FI']).idxmax()}. "
            f"Gradient Boosted Trees top feature: {pd.Series(eval_rows[2]['FI']).idxmax()}. "
            f"Comparing {len(fi_df)} features across 3 models."
        )
