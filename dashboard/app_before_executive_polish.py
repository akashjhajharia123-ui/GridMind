from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import r2_score


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "final_test_predictions.csv"
)

ANOMALIES_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "anomalies"
    / "final_test_anomalies.csv"
)

FINAL_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "final_test_metrics.json"
)

ANOMALY_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "anomaly_summary.json"
)

MODEL_COMPARISON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "model_comparison.csv"
)

BACKTEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "backtest_results.csv"
)

FINAL_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_final.joblib"
)


FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "lag_24",
    "lag_48",
    "lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GridMind · Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL STYLE
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --gm-bg: #06101d;
        --gm-panel: #0b1b2d;
        --gm-panel-soft: rgba(11, 27, 45, 0.82);
        --gm-border: rgba(255, 255, 255, 0.085);
        --gm-text: #edf7ff;
        --gm-muted: #8da5ba;
        --gm-cyan: #46d7ff;
        --gm-green: #35dfa5;
        --gm-purple: #8b7cff;
        --gm-red: #ff6685;
        --gm-amber: #ffbf69;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 14% 4%,
                rgba(0, 210, 180, 0.075),
                transparent 25%
            ),
            radial-gradient(
                circle at 88% 2%,
                rgba(70, 120, 255, 0.08),
                transparent 24%
            ),
            #06101d;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #081728 0%,
                #071321 100%
            );
        border-right:
            1px solid rgba(255, 255, 255, 0.08);
    }

    .block-container {
        max-width: 1540px;
        padding-top: 1.6rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    .gm-hero {
        padding: 1.65rem 1.8rem;
        border-radius: 22px;
        border:
            1px solid rgba(255, 255, 255, 0.10);
        background:
            linear-gradient(
                135deg,
                rgba(17, 45, 72, 0.96),
                rgba(7, 20, 35, 0.96)
            );
        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.18);
        margin-bottom: 1.25rem;
    }

    .gm-eyebrow {
        color: #55e3bd;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .gm-title {
        color: #f4fbff;
        font-size: 2.65rem;
        line-height: 1.02;
        font-weight: 850;
        letter-spacing: -0.055em;
        margin: 0;
    }

    .gm-subtitle {
        color: #9cb2c6;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 900px;
        margin-top: 0.75rem;
        margin-bottom: 0;
    }

    .gm-pill {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.42rem 0.75rem;
        border-radius: 999px;
        color: #67e8c0;
        background: rgba(32, 201, 151, 0.12);
        border:
            1px solid rgba(32, 201, 151, 0.32);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.025em;
    }

    div[data-testid="stMetric"] {
        min-height: 132px;
        padding: 1rem 1.05rem;
        border-radius: 17px;
        background:
            linear-gradient(
                145deg,
                rgba(14, 35, 57, 0.96),
                rgba(9, 25, 42, 0.96)
            );
        border:
            1px solid rgba(255, 255, 255, 0.085);
        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.12);
    }

    div[data-testid="stMetricLabel"] {
        color: #9cb1c4;
    }

    div[data-testid="stMetricValue"] {
        color: #f4faff;
    }

    div[data-testid="stMetricDelta"] {
        white-space: normal;
        overflow: visible;
    }

    .gm-section {
        margin-top: 1.3rem;
        margin-bottom: 0.8rem;
    }

    .gm-section-title {
        color: #eff8ff;
        font-size: 1.28rem;
        font-weight: 780;
        letter-spacing: -0.025em;
    }

    .gm-section-copy {
        color: #8299ae;
        font-size: 0.9rem;
        line-height: 1.55;
        margin-top: 0.2rem;
    }

    .gm-card {
        padding: 1.1rem 1.2rem;
        border-radius: 16px;
        border:
            1px solid rgba(255, 255, 255, 0.085);
        background: rgba(11, 27, 45, 0.78);
        color: #a9bfd2;
        line-height: 1.65;
    }

    .gm-card strong {
        color: #edf7ff;
    }

    .gm-protocol {
        padding: 1rem 1.1rem;
        border-left: 3px solid #35dfa5;
        border-radius: 12px;
        background: rgba(11, 27, 45, 0.72);
        color: #9eb5c8;
        line-height: 1.65;
        margin-bottom: 0.8rem;
    }

    .gm-protocol-title {
        color: #edf7ff;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    [data-testid="stDataFrame"] {
        border:
            1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: rgba(8, 23, 39, 0.72);
        border-radius: 14px;
        padding: 0.35rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Softer, theme-consistent multiselect chips */
    [data-baseweb="tag"] {
        background-color: rgba(53, 223, 165, 0.14) !important;
        border: 1px solid rgba(53, 223, 165, 0.30) !important;
        color: #8ff0cf !important;
    }

    [data-baseweb="tag"] span {
        color: #8ff0cf !important;
    }

    /* Prevent metric delta clipping */
    div[data-testid="stMetricDelta"] {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        max-width: 100% !important;
    }

    div[data-testid="stMetricDelta"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }

    /* Cleaner tab emphasis */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(53, 223, 165, 0.08);
    }

    /* Final public-dashboard chart polish */
    .modebar-container,
    .modebar {
        display: none !important;
    }

    /* Subtle Plotly container treatment */
    [data-testid="stPlotlyChart"] {
        border-radius: 16px;
        overflow: hidden;
    }

    hr {
        border-color:
            rgba(255, 255, 255, 0.08);
    }

    /* ===== GRIDMIND ULTIMATE FINAL POLISH ===== */
    html { scroll-behavior:smooth; }

    .block-container {
        max-width:1480px;
        padding:1.25rem clamp(1rem,3vw,2.4rem) 4rem;
    }

    [data-testid="stSidebar"] {
        box-shadow:18px 0 55px rgba(0,0,0,.12);
    }

    .gm-hero {
        position:relative;
        overflow:hidden;
        padding:clamp(1.55rem,3vw,2.35rem);
        border-radius:24px;
        background:
            radial-gradient(circle at 92% 10%,rgba(139,124,255,.16),transparent 27%),
            radial-gradient(circle at 10% 0%,rgba(53,223,165,.13),transparent 28%),
            linear-gradient(135deg,rgba(16,43,69,.98),rgba(7,20,35,.98));
        border:1px solid rgba(153,204,255,.12);
        box-shadow:0 28px 75px rgba(0,0,0,.22),inset 0 1px 0 rgba(255,255,255,.035);
    }

    .gm-hero::after {
        content:"";
        position:absolute;
        width:280px;height:280px;
        right:-110px;top:-160px;
        border-radius:50%;
        border:1px solid rgba(70,215,255,.14);
        box-shadow:0 0 0 42px rgba(70,215,255,.025),0 0 0 88px rgba(139,124,255,.018);
        pointer-events:none;
    }

    .gm-title {
        font-size:clamp(2.35rem,5vw,4rem);
        background:linear-gradient(90deg,#f7fcff 0%,#dff8ff 55%,#8ff0cf 100%);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        background-clip:text;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap:.3rem;padding:.38rem;
        border:1px solid rgba(255,255,255,.065);
        background:rgba(7,20,35,.72);
        backdrop-filter:blur(14px);
    }

    .stTabs [data-baseweb="tab"] {
        min-height:44px;color:#9db2c5;font-weight:700;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color:#f4fbff !important;
        background:linear-gradient(135deg,rgba(53,223,165,.12),rgba(70,215,255,.07));
    }

    div[data-testid="stMetric"] {
        min-height:138px;
        padding:1.08rem 1.12rem;
        border-radius:18px;
        background:
            radial-gradient(circle at 100% 0%,rgba(70,215,255,.055),transparent 36%),
            linear-gradient(145deg,rgba(14,35,57,.98),rgba(8,24,41,.98));
        border:1px solid rgba(142,190,225,.13);
        box-shadow:0 14px 38px rgba(0,0,0,.13),inset 0 1px 0 rgba(255,255,255,.025);
        transition:transform .18s ease,border-color .18s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform:translateY(-2px);
        border-color:rgba(70,215,255,.22);
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div {
        overflow:visible !important;
        text-overflow:clip !important;
        white-space:nowrap !important;
        max-width:none !important;
    }

    div[data-testid="stMetricValue"] {
        font-size:clamp(1.72rem,2.55vw,2.55rem) !important;
        letter-spacing:-.045em;
    }

    .gm-section { margin-top:2rem;margin-bottom:.85rem; }
    .gm-section-title { font-size:clamp(1.22rem,2vw,1.48rem);font-weight:820; }

    [data-testid="stPlotlyChart"] {
        border:1px solid rgba(255,255,255,.055);
        border-radius:18px;
        background:rgba(7,20,35,.24);
        box-shadow:0 12px 34px rgba(0,0,0,.08);
    }

    [data-testid="stDataFrame"] {
        border:1px solid rgba(142,190,225,.13);
        border-radius:16px;
        box-shadow:0 14px 36px rgba(0,0,0,.10);
    }

    [data-testid="stExpander"] {
        border:1px solid rgba(142,190,225,.13) !important;
        border-radius:14px !important;
        background:rgba(8,24,41,.52);
        overflow:hidden;
    }

    .gm-protocol {
        min-height:146px;
        padding:1.18rem 1.22rem;
        border-top:1px solid rgba(255,255,255,.055);
        border-right:1px solid rgba(255,255,255,.055);
        border-bottom:1px solid rgba(255,255,255,.055);
        background:linear-gradient(145deg,rgba(12,31,51,.94),rgba(8,24,41,.86));
        box-shadow:0 12px 30px rgba(0,0,0,.08);
    }

    .gm-integrity-head {
        display:flex;align-items:center;justify-content:space-between;gap:16px;
        padding:20px 22px;margin:20px 0 16px;border-radius:18px;
        background:
            radial-gradient(circle at 92% 0%,rgba(53,223,165,.10),transparent 28%),
            linear-gradient(135deg,#0b2135,#091a2b);
        border:1px solid rgba(112,183,222,.18);
        box-shadow:0 16px 38px rgba(0,0,0,.10);
    }

    .gm-integrity-title { color:#f1f9ff;font-size:1.12rem;font-weight:850; }
    .gm-integrity-copy { color:#8fa4b8;font-size:.84rem;margin-top:4px; }

    .gm-integrity-pill {
        flex:0 0 auto;padding:8px 13px;border-radius:999px;
        background:rgba(53,223,165,.10);
        border:1px solid rgba(53,223,165,.36);
        color:#54e6b7;font-size:.76rem;font-weight:900;
    }

    .gm-integrity-card {
        min-height:178px;padding:20px;border-radius:17px;
        background:
            radial-gradient(circle at 100% 0%,rgba(70,215,255,.05),transparent 34%),
            linear-gradient(145deg,#0b2135,#091a2b);
        border:1px solid rgba(112,183,222,.18);
        box-shadow:0 14px 34px rgba(0,0,0,.10),inset 0 1px 0 rgba(255,255,255,.02);
    }

    .gm-integrity-label {
        color:#8fa4b8;font-size:.70rem;font-weight:800;
        letter-spacing:.10em;text-transform:uppercase;
    }

    .gm-integrity-value {
        margin-top:18px;color:#eef6ff;font-size:.96rem;
        font-weight:900;line-height:1.35;overflow-wrap:anywhere;
    }

    .gm-integrity-value.ok { color:#43d9aa; }
    .gm-integrity-detail { margin-top:9px;color:#8fa4b8;font-size:.76rem;line-height:1.45; }

    .gm-status-dot {
        display:inline-block;width:7px;height:7px;margin-right:8px;
        border-radius:50%;background:#43d9aa;
        box-shadow:0 0 0 4px rgba(67,217,170,.10);vertical-align:1px;
    }

    @media(max-width:900px) {
        .gm-integrity-head { align-items:flex-start;flex-direction:column; }
        div[data-testid="stMetric"] { min-height:124px; }
    }

    @media(max-width:640px) {
        .block-container { padding-left:.85rem;padding-right:.85rem; }
        .gm-hero { border-radius:19px; }
        .gm-integrity-card { min-height:auto; }
        .stTabs [data-baseweb="tab"] { padding-left:.65rem;padding-right:.65rem;font-size:.84rem; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOADERS
# ============================================================

@st.cache_data
def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


@st.cache_data
def load_csv(path, parse_dates=None):
    return pd.read_csv(
        path,
        parse_dates=parse_dates,
    )


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def require_artifacts():
    required = [
        PREDICTIONS_PATH,
        FINAL_METRICS_PATH,
        ANOMALY_SUMMARY_PATH,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        st.error(
            "Required GridMind artifacts are missing."
        )

        for path in missing:
            st.code(str(path))

        st.stop()


# ============================================================
# CHART HELPERS
# ============================================================

def apply_chart_theme(
    fig,
    height=420,
    hovermode="x unified",
):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#a9bfd2",
        ),
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=25,
        ),
        hovermode=hovermode,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.05)",
        zeroline=False,
    )

    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.05)",
        zeroline=False,
    )

    return fig


def section(title, copy):
    st.markdown(
        f"""
        <div class="gm-section">
            <div class="gm-section-title">
                {title}
            </div>
            <div class="gm-section-copy">
                {copy}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def calculate_window_metrics(df):
    if df.empty:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
        }

    actual = df["AEP_MW"].to_numpy()
    predicted = df["prediction"].to_numpy()

    residual = actual - predicted

    mae = np.mean(
        np.abs(residual)
    )

    rmse = np.sqrt(
        np.mean(residual ** 2)
    )

    nonzero = actual != 0

    mape = np.mean(
        np.abs(
            residual[nonzero]
            / actual[nonzero]
        )
    ) * 100

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
    }


# ============================================================
# DATA
# ============================================================

require_artifacts()

predictions = load_csv(
    PREDICTIONS_PATH,
    parse_dates=["Datetime"],
)

predictions = predictions.sort_values(
    "Datetime"
).reset_index(drop=True)

if "residual" not in predictions.columns:
    predictions["residual"] = (
        predictions["AEP_MW"]
        - predictions["prediction"]
    )

predictions["absolute_error"] = (
    predictions["residual"].abs()
)

anomalies = pd.DataFrame()

if ANOMALIES_PATH.exists():
    anomalies = load_csv(
        ANOMALIES_PATH,
        parse_dates=["Datetime"],
    )

final_metrics = load_json(
    FINAL_METRICS_PATH
)

anomaly_summary = load_json(
    ANOMALY_SUMMARY_PATH
)

model_comparison = pd.DataFrame()

if MODEL_COMPARISON_PATH.exists():
    model_comparison = load_csv(
        MODEL_COMPARISON_PATH
    )

backtest = pd.DataFrame()

if BACKTEST_PATH.exists():
    backtest = load_csv(
        BACKTEST_PATH
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## ⚡ GridMind"
)

st.sidebar.caption(
    "Energy Intelligence Control Center"
)

st.sidebar.divider()

min_date = (
    predictions["Datetime"]
    .min()
    .date()
)

max_date = (
    predictions["Datetime"]
    .max()
    .date()
)

default_start = max(
    min_date,
    max_date - pd.Timedelta(days=29),
)

date_range = st.sidebar.date_input(
    "Analysis window",
    value=(
        default_start,
        max_date,
    ),
    min_value=min_date,
    max_value=max_date,
)

show_anomalies = st.sidebar.toggle(
    "Show anomaly markers",
    value=True,
)

severity_options = [
    "moderate",
    "high",
    "critical",
]

selected_severities = (
    st.sidebar.multiselect(
        "Anomaly severity",
        options=severity_options,
        default=severity_options,
    )
)

direction_options = [
    "spike",
    "drop",
]

selected_directions = (
    st.sidebar.multiselect(
        "Anomaly direction",
        options=direction_options,
        default=direction_options,
    )
)

st.sidebar.divider()

st.sidebar.markdown(
    "**Operational status**"
)

st.sidebar.success(
    "Random Forest · Frozen Candidate"
)

st.sidebar.caption(
    "Final evaluation protocol committed "
    "before reserved-test scoring."
)

st.sidebar.divider()

st.sidebar.caption(
    "GridMind v2 · Portfolio Build"
)


# ============================================================
# FILTERING
# ============================================================

if (
    isinstance(date_range, tuple)
    and len(date_range) == 2
):
    start_date, end_date = date_range
else:
    start_date = default_start
    end_date = max_date


start_timestamp = pd.Timestamp(
    start_date
)

end_timestamp = (
    pd.Timestamp(end_date)
    + pd.Timedelta(days=1)
    - pd.Timedelta(microseconds=1)
)

filtered_predictions = predictions[
    predictions["Datetime"].between(
        start_timestamp,
        end_timestamp,
    )
].copy()

filtered_anomalies = anomalies.copy()

if not filtered_anomalies.empty:
    filtered_anomalies = (
        filtered_anomalies[
            filtered_anomalies[
                "Datetime"
            ].between(
                start_timestamp,
                end_timestamp,
            )
        ]
        .copy()
    )

    if selected_severities:
        filtered_anomalies = (
            filtered_anomalies[
                filtered_anomalies[
                    "severity"
                ].isin(
                    selected_severities
                )
            ]
        )
    else:
        filtered_anomalies = (
            filtered_anomalies.iloc[0:0]
        )

    if selected_directions:
        filtered_anomalies = (
            filtered_anomalies[
                filtered_anomalies[
                    "direction"
                ].isin(
                    selected_directions
                )
            ]
        )
    else:
        filtered_anomalies = (
            filtered_anomalies.iloc[0:0]
        )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="gm-hero">
<div class="gm-eyebrow">Energy Intelligence Platform</div>
<div class="gm-title">GridMind</div>
<div class="gm-subtitle">
Leakage-aware hourly demand forecasting, residual intelligence,
model diagnostics, and operational anomaly surveillance in one
decision workspace.
</div>
<span class="gm-pill">● FINAL TEST EVALUATED · MODEL FROZEN</span>
</div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GLOBAL KPIs
# ============================================================

rf_metrics = final_metrics[
    "random_forest"
]

baseline_metrics = final_metrics[
    "seasonal_naive"
]

improvement = final_metrics[
    "relative_mape_improvement_percent"
]

window_metrics = calculate_window_metrics(
    filtered_predictions
)

visible_anomaly_count = len(
    filtered_anomalies
)

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Final Test MAPE",
    f"{rf_metrics['MAPE']:.2f}%",
    delta=f"{improvement:.2f}% vs baseline",
)

k2.metric(
    "Final Test MAE",
    f"{rf_metrics['MAE']:,.0f} MW",
)

k3.metric(
    "Final Test RMSE",
    f"{rf_metrics['RMSE']:,.0f} MW",
)

k4.metric(
    "Visible Anomalies",
    f"{visible_anomaly_count:,}",
    delta=(
        f"{anomaly_summary['anomaly_rate_percent']:.3f}% "
        "full-test rate"
    ),
    delta_color="off",
)


# ============================================================
# TABS
# ============================================================

(
    overview_tab,
    anomaly_tab,
    diagnostics_tab,
    methodology_tab,
) = st.tabs(
    [
        "Overview",
        "Anomaly Intelligence",
        "Model Diagnostics",
        "Methodology",
    ]
)


# ============================================================
# OVERVIEW TAB
# ============================================================

with overview_tab:
    section(
        "Demand Forecast Timeline",
        (
            "Observed demand versus frozen final-model "
            "predictions for the selected analysis window."
        ),
    )

    timeline = go.Figure()

    timeline.add_trace(
        go.Scatter(
            x=filtered_predictions[
                "Datetime"
            ],
            y=filtered_predictions[
                "AEP_MW"
            ],
            mode="lines",
            name="Actual Demand",
            line=dict(
                width=1.8,
                color="#46d7ff",
            ),
        )
    )

    timeline.add_trace(
        go.Scatter(
            x=filtered_predictions[
                "Datetime"
            ],
            y=filtered_predictions[
                "prediction"
            ],
            mode="lines",
            name="Forecast",
            line=dict(
                width=1.5,
                color="#8b7cff",
            ),
        )
    )

    if (
        show_anomalies
        and not filtered_anomalies.empty
    ):
        spike_df = filtered_anomalies[
            filtered_anomalies[
                "direction"
            ] == "spike"
        ]

        drop_df = filtered_anomalies[
            filtered_anomalies[
                "direction"
            ] == "drop"
        ]

        if not spike_df.empty:
            timeline.add_trace(
                go.Scatter(
                    x=spike_df["Datetime"],
                    y=spike_df["AEP_MW"],
                    mode="markers",
                    name="Spike anomaly",
                    marker=dict(
                        size=9,
                        color="#ffbf69",
                        symbol="triangle-up",
                        line=dict(
                            width=1,
                            color="#fff0d8",
                        ),
                    ),
                    customdata=spike_df[
                        [
                            "severity",
                            "severity_score",
                            "residual",
                        ]
                    ],
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Actual: %{y:,.0f} MW<br>"
                        "Severity: %{customdata[0]}<br>"
                        "Score: %{customdata[1]:.2f}<br>"
                        "Residual: %{customdata[2]:,.0f} MW"
                        "<extra></extra>"
                    ),
                )
            )

        if not drop_df.empty:
            timeline.add_trace(
                go.Scatter(
                    x=drop_df["Datetime"],
                    y=drop_df["AEP_MW"],
                    mode="markers",
                    name="Drop anomaly",
                    marker=dict(
                        size=9,
                        color="#ff6685",
                        symbol="triangle-down",
                        line=dict(
                            width=1,
                            color="#ffd5de",
                        ),
                    ),
                    customdata=drop_df[
                        [
                            "severity",
                            "severity_score",
                            "residual",
                        ]
                    ],
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Actual: %{y:,.0f} MW<br>"
                        "Severity: %{customdata[0]}<br>"
                        "Score: %{customdata[1]:.2f}<br>"
                        "Residual: %{customdata[2]:,.0f} MW"
                        "<extra></extra>"
                    ),
                )
            )

    timeline.update_layout(
        yaxis_title="Demand (MW)",
        xaxis_title=None,
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.06,
            )
        ),
    )

    apply_chart_theme(
        timeline,
        height=560,
    )

    st.plotly_chart(
        timeline,
        width="stretch",
        config={
            "displaylogo": False,
            "responsive": True,
        },
    )

    section(
        "Selected-Window Performance",
        (
            "Error metrics recomputed dynamically "
            "for the active sidebar date range."
        ),
    )

    w1, w2, w3, w4 = st.columns(4)

    w1.metric(
        "Window MAPE",
        f"{window_metrics['mape']:.2f}%",
    )

    w2.metric(
        "Window MAE",
        f"{window_metrics['mae']:,.0f} MW",
    )

    w3.metric(
        "Window RMSE",
        f"{window_metrics['rmse']:,.0f} MW",
    )

    w4.metric(
        "Hours Displayed",
        f"{len(filtered_predictions):,}",
    )

    left, right = st.columns(
        [1.1, 0.9]
    )

    with left:
        section(
            "Residual Distribution",
            (
                "Signed forecast error; values near "
                "zero indicate closer predictions."
            ),
        )

        hist = px.histogram(
            filtered_predictions,
            x="residual",
            nbins=55,
            labels={
                "residual":
                    "Residual (MW)",
            },
        )

        hist.update_traces(
            marker_color="#46d7ff",
            opacity=0.78,
        )

        hist.add_vline(
            x=0,
            line_dash="dash",
            line_color="#35dfa5",
        )

        hist.update_layout(
            showlegend=False,
            yaxis_title="Hours",
        )

        apply_chart_theme(
            hist,
            height=390,
        )

        st.plotly_chart(
            hist,
            width="stretch",
        )

    with right:
        section(
            "Baseline Benchmark",
            (
                "Final reserved-test MAPE comparison "
                "against the seasonal-naive baseline."
            ),
        )

        benchmark_df = pd.DataFrame({
            "Model": [
                "Seasonal Naive",
                "Random Forest",
            ],
            "MAPE": [
                baseline_metrics["MAPE"],
                rf_metrics["MAPE"],
            ],
        })

        benchmark = px.bar(
            benchmark_df,
            x="Model",
            y="MAPE",
            text_auto=".2f",
        )

        benchmark.update_traces(
            marker_color=[
                "#5f7185",
                "#35dfa5",
            ],
            textposition="outside",
        )

        benchmark.update_layout(
            showlegend=False,
            yaxis_title="MAPE (%)",
        )

        apply_chart_theme(
            benchmark,
            height=390,
        )

        st.plotly_chart(
            benchmark,
            width="stretch",
        )


# ============================================================
# ANOMALY TAB
# ============================================================

with anomaly_tab:
    section(
        "Residual Anomaly Intelligence",
        (
            "Events exceeding the frozen q=0.99 "
            "threshold calibrated on unseen "
            "validation residuals."
        ),
    )

    a1, a2, a3, a4 = st.columns(4)

    spike_count = (
        int(
            (
                filtered_anomalies[
                    "direction"
                ] == "spike"
            ).sum()
        )
        if not filtered_anomalies.empty
        else 0
    )

    drop_count = (
        int(
            (
                filtered_anomalies[
                    "direction"
                ] == "drop"
            ).sum()
        )
        if not filtered_anomalies.empty
        else 0
    )

    critical_count = (
        int(
            (
                filtered_anomalies[
                    "severity"
                ] == "critical"
            ).sum()
        )
        if not filtered_anomalies.empty
        else 0
    )

    a1.metric(
        "Filtered Events",
        f"{len(filtered_anomalies):,}",
    )

    a2.metric(
        "Demand Spikes",
        f"{spike_count:,}",
    )

    a3.metric(
        "Demand Drops",
        f"{drop_count:,}",
    )

    a4.metric(
        "Critical Events",
        f"{critical_count:,}",
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        section(
            "Severity Mix",
            (
                "Distribution of currently visible "
                "anomaly events by severity band."
            ),
        )

        if filtered_anomalies.empty:
            st.info(
                "No anomalies match the current filters."
            )
        else:
            severity_order = [
                "moderate",
                "high",
                "critical",
            ]

            severity_counts = (
                filtered_anomalies[
                    "severity"
                ]
                .value_counts()
                .reindex(
                    severity_order,
                    fill_value=0,
                )
                .reset_index()
            )

            severity_counts.columns = [
                "Severity",
                "Count",
            ]

            donut = px.pie(
                severity_counts,
                names="Severity",
                values="Count",
                hole=0.62,
                color="Severity",
                color_discrete_map={
                    "moderate": "#46d7ff",
                    "high": "#ffbf69",
                    "critical": "#ff6685",
                },
            )

            donut.update_traces(
                textinfo="percent+label",
                textposition="inside",
                insidetextorientation="horizontal",
                sort=False,
                marker=dict(
                    line=dict(
                        color="rgba(6,16,29,.85)",
                        width=2,
                    )
                ),
            )

            donut.update_layout(
                margin=dict(l=35, r=35, t=65, b=35),
                uniformtext_minsize=11,
                uniformtext_mode="hide",
            )

            apply_chart_theme(
                donut,
                height=390,
                hovermode="closest",
            )

            st.plotly_chart(
                donut,
                width="stretch",
            )

    with chart_right:
        section(
            "Direction Mix",
            (
                "Positive residuals indicate demand "
                "spikes; negative residuals indicate drops."
            ),
        )

        if filtered_anomalies.empty:
            st.info(
                "No anomalies match the current filters."
            )
        else:
            direction_counts = (
                filtered_anomalies[
                    "direction"
                ]
                .value_counts()
                .reset_index()
            )

            direction_counts.columns = [
                "Direction",
                "Count",
            ]

            direction_chart = px.bar(
                direction_counts,
                x="Direction",
                y="Count",
                text_auto=True,
                color="Direction",
                color_discrete_map={
                    "spike": "#ffbf69",
                    "drop": "#ff6685",
                },
            )

            direction_chart.update_layout(
                showlegend=False,
            )

            apply_chart_theme(
                direction_chart,
                height=390,
            )

            st.plotly_chart(
                direction_chart,
                width="stretch",
            )

    section(
        "Highest-Severity Events",
        (
            "Ranked operational events after applying "
            "the active date, severity, and direction filters."
        ),
    )

    if filtered_anomalies.empty:
        st.info(
            "No anomaly events match the current filters."
        )
    else:
        top_events = (
            filtered_anomalies
            .sort_values(
                "severity_score",
                ascending=False,
            )
            .head(20)
            [
                [
                    "Datetime",
                    "AEP_MW",
                    "prediction",
                    "residual",
                    "severity_score",
                    "direction",
                    "severity",
                ]
            ]
            .copy()
        )

        top_events = top_events.rename(
            columns={
                "Datetime": "Timestamp",
                "AEP_MW": "Actual MW",
                "prediction": "Forecast MW",
                "residual": "Residual MW",
                "severity_score":
                    "Severity Score",
                "direction": "Direction",
                "severity": "Severity",
            }
        )

        top_events[
            "Actual MW"
        ] = top_events[
            "Actual MW"
        ].round(1)

        top_events[
            "Forecast MW"
        ] = top_events[
            "Forecast MW"
        ].round(1)

        top_events[
            "Residual MW"
        ] = top_events[
            "Residual MW"
        ].round(1)

        top_events[
            "Severity Score"
        ] = top_events[
            "Severity Score"
        ].round(2)

        st.dataframe(
            top_events,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# DIAGNOSTICS TAB
# ============================================================

with diagnostics_tab:
    section(
        "Forecast Diagnostics",
        (
            "Residual behavior, rolling error drift, "
            "calibration quality, and model explainability."
        ),
    )

    if not filtered_predictions.empty:
        diagnostic_r2 = r2_score(
            filtered_predictions["AEP_MW"],
            filtered_predictions["prediction"],
        )

        diagnostic_bias = float(
            filtered_predictions["residual"].mean()
        )

        diagnostic_median_abs_error = float(
            filtered_predictions["absolute_error"].median()
        )

        diagnostic_p95_abs_error = float(
            filtered_predictions["absolute_error"].quantile(0.95)
        )
    else:
        diagnostic_r2 = np.nan
        diagnostic_bias = np.nan
        diagnostic_median_abs_error = np.nan
        diagnostic_p95_abs_error = np.nan

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Window R²",
        f"{diagnostic_r2:.3f}",
    )

    d2.metric(
        "Mean Bias",
        f"{diagnostic_bias:,.0f} MW",
        help=(
            "Actual minus predicted demand. "
            "Negative values indicate average overprediction."
        ),
    )

    d3.metric(
        "Median Absolute Error",
        f"{diagnostic_median_abs_error:,.0f} MW",
    )

    d4.metric(
        "95th Percentile Error",
        f"{diagnostic_p95_abs_error / 1000:.2f}k MW",
    )

    diag_left, diag_right = st.columns(2)

    with diag_left:
        section(
            "Residuals Over Time",
            (
                "Signed errors reveal systematic "
                "under- and over-prediction periods."
            ),
        )

        residual_time = go.Figure()

        residual_time.add_trace(
            go.Scatter(
                x=filtered_predictions[
                    "Datetime"
                ],
                y=filtered_predictions[
                    "residual"
                ],
                mode="lines",
                name="Residual",
                line=dict(
                    width=1.2,
                    color="#46d7ff",
                ),
            )
        )

        residual_time.add_hline(
            y=0,
            line_dash="dash",
            line_color="#35dfa5",
        )

        residual_time.update_layout(
            yaxis_title="Residual (MW)",
        )

        apply_chart_theme(
            residual_time,
            height=410,
        )

        st.plotly_chart(
            residual_time,
            width="stretch",
        )

    with diag_right:
        section(
            "Actual vs Predicted",
            (
                "Closer alignment to the diagonal "
                "indicates stronger forecast calibration."
            ),
        )

        scatter_source = (
            filtered_predictions
            .sample(
                n=min(
                    5000,
                    len(filtered_predictions),
                ),
                random_state=42,
            )
            if len(filtered_predictions) > 0
            else filtered_predictions
        )

        scatter = px.scatter(
            scatter_source,
            x="AEP_MW",
            y="prediction",
            opacity=0.35,
            labels={
                "AEP_MW": "Actual Demand (MW)",
                "prediction":
                    "Predicted Demand (MW)",
            },
        )

        scatter.update_traces(
            marker=dict(
                color="#8b7cff",
                size=5,
            )
        )

        if not scatter_source.empty:
            min_value = min(
                scatter_source[
                    "AEP_MW"
                ].min(),
                scatter_source[
                    "prediction"
                ].min(),
            )

            max_value = max(
                scatter_source[
                    "AEP_MW"
                ].max(),
                scatter_source[
                    "prediction"
                ].max(),
            )

            scatter.add_trace(
                go.Scatter(
                    x=[
                        min_value,
                        max_value,
                    ],
                    y=[
                        min_value,
                        max_value,
                    ],
                    mode="lines",
                    name="Perfect forecast",
                    line=dict(
                        dash="dash",
                        color="#35dfa5",
                    ),
                )
            )

        apply_chart_theme(
            scatter,
            height=410,
            hovermode="closest",
        )

        st.plotly_chart(
            scatter,
            width="stretch",
        )

    section(
        "Rolling Error Trend",
        (
            "Seven-day rolling MAE highlights periods "
            "where forecast difficulty changes over time."
        ),
    )

    rolling_df = (
        filtered_predictions[
            [
                "Datetime",
                "absolute_error",
            ]
        ]
        .copy()
    )

    rolling_df["rolling_mae_7d"] = (
        rolling_df[
            "absolute_error"
        ]
        .rolling(
            window=24 * 7,
            min_periods=24,
        )
        .mean()
    )

    rolling_chart = go.Figure()

    rolling_chart.add_trace(
        go.Scatter(
            x=rolling_df["Datetime"],
            y=rolling_df[
                "rolling_mae_7d"
            ],
            mode="lines",
            name="7-day rolling MAE",
            line=dict(
                width=2,
                color="#ffbf69",
            ),
            fill="tozeroy",
            fillcolor=(
                "rgba(255,191,105,0.08)"
            ),
        )
    )

    rolling_chart.update_layout(
        yaxis_title="MAE (MW)",
    )

    apply_chart_theme(
        rolling_chart,
        height=390,
    )

    st.plotly_chart(
        rolling_chart,
        width="stretch",
        config={
            "displaylogo": False,
            "responsive": True,
        },
    )

    diag_bottom_left, diag_bottom_right = (
        st.columns(2)
    )

    with diag_bottom_left:
        section(
            "Model Comparison",
            (
                "Validation-stage challenger comparison "
                "when the saved artifact is available."
            ),
        )

        if (
            not model_comparison.empty
            and "model"
            in model_comparison.columns
            and "MAPE"
            in model_comparison.columns
        ):
            comparison_chart = px.bar(
                model_comparison,
                x="model",
                y="MAPE",
                text_auto=".2f",
            )

            comparison_chart.update_traces(
                marker_color="#46d7ff",
                textposition="outside",
            )

            comparison_chart.update_layout(
                showlegend=False,
                xaxis_title=None,
                yaxis_title="MAPE (%)",
            )

            apply_chart_theme(
                comparison_chart,
                height=390,
            )

            st.plotly_chart(
                comparison_chart,
                width="stretch",
            )
        else:
            st.info(
                "Model comparison artifact is "
                "not available in the expected schema."
            )

    with diag_bottom_right:
        section(
            "Feature Importance",
            (
                "Random Forest impurity-based importance "
                "for the frozen final model."
            ),
        )

        if FINAL_MODEL_PATH.exists():
            try:
                final_model = load_model(
                    FINAL_MODEL_PATH
                )

                importances = getattr(
                    final_model,
                    "feature_importances_",
                    None,
                )

                if (
                    importances is not None
                    and len(importances)
                    == len(FEATURE_COLUMNS)
                ):
                    importance_df = pd.DataFrame({
                        "Feature": FEATURE_COLUMNS,
                        "Importance": importances,
                    }).sort_values(
                        "Importance",
                        ascending=True,
                    )

                    importance_chart = px.bar(
                        importance_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        text="Importance",
                    )

                    importance_chart.update_traces(
                        marker_color="#35dfa5",
                        texttemplate="%{text:.3f}",
                        textposition="outside",
                        cliponaxis=False,
                    )

                    importance_chart.update_layout(
                        showlegend=False,
                    )

                    apply_chart_theme(
                        importance_chart,
                        height=390,
                    )

                    st.plotly_chart(
                        importance_chart,
                        width="stretch",
                    )
                else:
                    st.info(
                        "Feature importance metadata "
                        "does not match dashboard features."
                    )

            except Exception as error:
                st.warning(
                    "Could not load feature importance: "
                    f"{error}"
                )
        else:
            st.info(
                "Final model artifact is unavailable."
            )

    if not backtest.empty:
        section(
            "Backtest Stability",
            (
                "Expanding-window MAPE across temporal folds, "
                "followed by the preserved raw evaluation record."
            ),
        )

        required_backtest_columns = {
            "fold",
            "model",
            "MAPE",
        }

        if required_backtest_columns.issubset(
            backtest.columns
        ):
            backtest_chart = px.line(
                backtest,
                x="fold",
                y="MAPE",
                color="model",
                markers=True,
                labels={
                    "fold": "Temporal Fold",
                    "MAPE": "MAPE (%)",
                    "model": "Model",
                },
                color_discrete_map={
                    "Random Forest": "#35dfa5",
                    "HistGradientBoosting": "#8b7cff",
                    "Seasonal Naive": "#71869a",
                },
            )

            backtest_chart.update_traces(
                line=dict(width=2.4),
                marker=dict(size=8),
            )

            backtest_chart.update_xaxes(
                dtick=1,
            )

            apply_chart_theme(
                backtest_chart,
                height=430,
                hovermode="x unified",
            )

            st.plotly_chart(
                backtest_chart,
                width="stretch",
                config={
                    "displaylogo": False,
                    "responsive": True,
                },
            )

            rf_rows = backtest[
                backtest["model"]
                == "Random Forest"
            ]

            hgb_rows = backtest[
                backtest["model"]
                == "HistGradientBoosting"
            ]

            if (
                not rf_rows.empty
                and not hgb_rows.empty
            ):
                rf_mean = rf_rows["MAPE"].mean()
                rf_std = rf_rows["MAPE"].std(ddof=0)

                hgb_mean = hgb_rows["MAPE"].mean()
                hgb_std = hgb_rows["MAPE"].std(ddof=0)

                st.markdown(
                    f'''
<div class="gm-card">
<strong>Selection rationale.</strong>
Random Forest achieved mean backtest MAPE of
<strong>{rf_mean:.3f}%</strong>
with fold variability of
<strong>{rf_std:.3f}</strong>.
HistGradientBoosting achieved
<strong>{hgb_mean:.3f}%</strong>
with variability of
<strong>{hgb_std:.3f}</strong>.
The final selection followed the committed rule:
models within the practical competitiveness margin
were resolved using holdout validation performance,
without reopening the reserved final test set.
</div>
                    ''',
                    unsafe_allow_html=True,
                )

        with st.expander(
            "View raw backtest evaluation record"
        ):
            st.dataframe(
                backtest,
                width="stretch",
                hide_index=True,
            )


# ============================================================
# METHODOLOGY TAB
# ============================================================

with methodology_tab:
    section(
        "Evaluation Integrity",
        (
            "GridMind separates model development, "
            "selection, final evaluation, and anomaly "
            "calibration to reduce temporal leakage."
        ),
    )

    m1, m2 = st.columns(2)

    with m1:
        st.markdown(
            """
            <div class="gm-protocol">
                <div class="gm-protocol-title">
                    01 · Chronological Splitting
                </div>
                Train, validation, and reserved-test
                periods preserve temporal order.
                Random shuffling is not used.
            </div>

            <div class="gm-protocol">
                <div class="gm-protocol-title">
                    02 · Leakage-Safe Features
                </div>
                Lag and rolling features use historical
                demand information rather than future
                target values.
            </div>

            <div class="gm-protocol">
                <div class="gm-protocol-title">
                    03 · Challenger Evaluation
                </div>
                Random Forest and
                HistGradientBoosting were compared
                against a seasonal-naive benchmark.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            """
            <div class="gm-protocol">
                <div class="gm-protocol-title">
                    04 · Expanding-Window Backtesting
                </div>
                Temporal folds evaluate stability across
                multiple historical forecast origins.
            </div>

            <div class="gm-protocol">
                <div class="gm-protocol-title">
                    05 · Frozen Final Protocol
                </div>
                The final evaluation script was committed
                before reserved-test evaluation.
            </div>

            <div class="gm-protocol">
                <div class="gm-protocol-title">
                    06 · Separate Anomaly Calibration
                </div>
                A train-only calibration model predicts
                unseen validation data; q=0.99 residual
                threshold is then frozen for test scoring.
            </div>
            """,
            unsafe_allow_html=True,
        )

    section(
        "Final Reserved-Test Results",
        (
            "The reserved test set is considered consumed "
            "and is not used for post-hoc model tuning."
        ),
    )

    result_table = pd.DataFrame({
        "Model": [
            "Seasonal Naive",
            "Random Forest",
        ],
        "MAE (MW)": [
            baseline_metrics["MAE"],
            rf_metrics["MAE"],
        ],
        "RMSE (MW)": [
            baseline_metrics["RMSE"],
            rf_metrics["RMSE"],
        ],
        "MAPE (%)": [
            baseline_metrics["MAPE"],
            rf_metrics["MAPE"],
        ],
    })

    result_table[
        "MAE (MW)"
    ] = result_table[
        "MAE (MW)"
    ].round(2)

    result_table[
        "RMSE (MW)"
    ] = result_table[
        "RMSE (MW)"
    ].round(2)

    result_table[
        "MAPE (%)"
    ] = result_table[
        "MAPE (%)"
    ].round(3)

    st.dataframe(
        result_table,
        width="stretch",
        hide_index=True,
    )

    section(
        "Anomaly Calibration Record",
        (
            "Saved metadata documenting threshold "
            "provenance and final-test usage."
        ),
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Threshold Quantile",
        (
            f"q={anomaly_summary['threshold_quantile']:.2f}"
        ),
    )

    c2.metric(
        "Frozen Threshold",
        (
            f"{anomaly_summary['threshold_mw']:,.0f} MW"
        ),
    )

    c3.metric(
        "Full-Test Events",
        (
            f"{anomaly_summary['anomaly_count']:,}"
        ),
    )

    c4.metric(
        "Full-Test Rate",
        (
            f"{anomaly_summary['anomaly_rate_percent']:.3f}%"
        ),
    )

    calibration_scope = anomaly_summary.get(
        "calibration_model_training_scope", "train_only"
    )
    threshold_source = anomaly_summary.get(
        "threshold_source", "unseen_validation_residuals"
    )
    validation_seen = anomaly_summary.get(
        "validation_seen_by_calibration_model", False
    )
    threshold_fit_test = anomaly_summary.get(
        "threshold_fit_on_final_test", False
    )
    final_test_usage = anomaly_summary.get(
        "final_test_usage", "scoring_only"
    )

    integrity_tone = (
        "REVIEW REQUIRED"
        if (validation_seen or threshold_fit_test)
        else "FROZEN · LEAKAGE SAFE"
    )

    st.markdown(
        f"""<div class="gm-integrity-head">
<div>
<div class="gm-integrity-title">Calibration Integrity</div>
<div class="gm-integrity-copy">Threshold provenance and reserved-test isolation</div>
</div>
<div class="gm-integrity-pill">{calibration_scope.upper()} · {integrity_tone}</div>
</div>""",
        unsafe_allow_html=True,
    )

    meta1, meta2, meta3, meta4 = st.columns(4)

    with meta1:
        threshold_source_label = (
            threshold_source
            .replace("_", " ")
            .upper()
        )

        st.markdown(
            f"""<div class="gm-integrity-card">
<div class="gm-integrity-label">Threshold Source</div>
<div class="gm-integrity-value">{threshold_source_label}</div>
<div class="gm-integrity-detail">Frozen from held-out residual evidence</div>
</div>""",
            unsafe_allow_html=True,
        )

    with meta2:
        validation_status = (
            "REVIEW REQUIRED" if validation_seen else "LEAKAGE SAFE"
        )
        validation_detail = (
            "Calibration model saw validation"
            if validation_seen
            else "Validation exposure: none"
        )
        validation_class = "" if validation_seen else " ok"
        validation_icon = (
            "!" if validation_seen
            else '<span class="gm-status-dot"></span>'
        )
        st.markdown(
            f"""<div class="gm-integrity-card">
<div class="gm-integrity-label">Validation Exposure</div>
<div class="gm-integrity-value{validation_class}">{validation_icon}{validation_status}</div>
<div class="gm-integrity-detail">{validation_detail}</div>
</div>""",
            unsafe_allow_html=True,
        )

    with meta3:
        threshold_status = (
            "REVIEW REQUIRED" if threshold_fit_test else "TEST UNTOUCHED"
        )
        threshold_detail = (
            "Threshold used final-test data"
            if threshold_fit_test
            else "Fit on final test: no"
        )
        threshold_class = "" if threshold_fit_test else " ok"
        threshold_icon = (
            "!" if threshold_fit_test
            else '<span class="gm-status-dot"></span>'
        )
        st.markdown(
            f"""<div class="gm-integrity-card">
<div class="gm-integrity-label">Threshold Fitting</div>
<div class="gm-integrity-value{threshold_class}">{threshold_icon}{threshold_status}</div>
<div class="gm-integrity-detail">{threshold_detail}</div>
</div>""",
            unsafe_allow_html=True,
        )

    with meta4:
        usage_label = final_test_usage.replace("_", " ").upper()
        st.markdown(
            f"""<div class="gm-integrity-card">
<div class="gm-integrity-label">Final-Test Usage</div>
<div class="gm-integrity-value">{usage_label}</div>
<div class="gm-integrity-detail">Evaluation boundary preserved</div>
</div>""",
            unsafe_allow_html=True,
        )
