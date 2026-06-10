# -*- coding: utf-8 -*-
"""Streamlit UI for compost germination-index prediction."""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "model" / "model_bundle.joblib"
TEMPLATE_PATH = APP_DIR / "data" / "single_prediction_template.csv"
EXAMPLE_PATH = APP_DIR / "data" / "example_input.csv"


st.set_page_config(
    page_title="Compost GI Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
:root {
  --ink: #24323a;
  --muted: #667680;
  --teal: #55b8ae;
  --blue: #6ec6dc;
  --coral: #f2a07b;
  --navy: #5874a6;
  --soft: #f6faf9;
  --line: #d9e5e2;
}
.main .block-container {
  padding-top: 1rem;
  padding-bottom: 2.2rem;
  max-width: 1280px;
}
h1, h2, h3 {
  color: var(--ink);
  letter-spacing: 0;
  font-weight: 850;
}
.hero {
  padding: 1.25rem 1.35rem 1.1rem 1.35rem;
  border: 1px solid var(--line);
  background: linear-gradient(135deg, #f7fbfb 0%, #eef8f5 55%, #fff8f4 100%);
  border-radius: 10px;
}
.hero h1 {
  font-size: 2.45rem;
  margin: 0 0 0.35rem 0;
  line-height: 1.12;
}
.hero p {
  color: var(--muted);
  font-size: 1.14rem;
  line-height: 1.45;
  margin: 0;
}
.metric-card {
  min-height: 116px;
  padding: 1rem 1.05rem;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
  box-shadow: 0 1px 12px rgba(31, 49, 51, 0.04);
}
.metric-card .label {
  color: var(--muted);
  font-size: 0.96rem;
  font-weight: 700;
  text-transform: uppercase;
}
.metric-card .value {
  color: var(--ink);
  font-size: 2rem;
  font-weight: 800;
  line-height: 1.15;
}
.result-card {
  padding: 1.1rem 1.25rem;
  border: 1px solid #bfe3dd;
  border-radius: 10px;
  background: #f4fbf9;
}
.result-card .value {
  color: #1e6f68;
  font-size: 3rem;
  font-weight: 850;
  line-height: 1.05;
}
.small-note {
  color: var(--muted);
  font-size: 1rem;
  line-height: 1.4;
}
.warn-box {
  padding: 0.75rem 0.9rem;
  border-left: 4px solid var(--coral);
  background: #fff7f1;
  color: #6b4637;
  border-radius: 6px;
}
div.stButton > button {
  width: 100%;
  border-radius: 7px;
  font-weight: 800;
  font-size: 1.08rem;
  border: 1px solid #2f8f86;
  background: #2f8f86;
  color: white;
  min-height: 3rem;
}
div.stDownloadButton > button {
  border-radius: 7px;
  font-weight: 750;
  font-size: 1.02rem;
}
div[data-testid="stMetricValue"], div[data-testid="stMarkdownContainer"] p,
div[data-testid="stWidgetLabel"] p {
  font-size: 1.06rem;
}
div[data-testid="stWidgetLabel"] p {
  color: var(--ink);
  font-weight: 760;
  line-height: 1.25;
}
div[data-baseweb="input"] input, div[data-baseweb="select"] {
  font-size: 1.06rem;
}
section[data-testid="stSidebar"] {
  font-size: 1.04rem;
}
section[data-testid="stSidebar"] h2 {
  font-size: 1.45rem;
}
.stTabs [data-baseweb="tab"] {
  font-size: 1.08rem;
  font-weight: 800;
}
.section-title {
  font-size: 1.18rem;
  font-weight: 850;
  color: var(--ink);
  margin: 0.25rem 0 0.55rem 0;
}
.feature-help {
  font-size: 0.96rem;
  color: var(--muted);
  margin-top: -0.25rem;
}
</style>
"""


UI_LABELS = {
    "Waste type": "Waste type",
    "Initial pH": "Initial pH",
    "Initial carbon to nitrogen": "Initial C/N",
    "Initial moisture content (%)": "Initial moisture (%)",
    "Compost time (day)": "Compost time (d)",
    "Temperature_corr": "Temperature (°C)",
    "Moisture_corr": "Moisture (%)",
    "pH_corr": "Process pH",
    "EC（ms/cm-1）": "EC (mS cm⁻¹)",
    "Ammonia氨mg/kg": "NH₄⁺-N (mg kg⁻¹)",
    "Nitrate硝酸盐mg/kg": "NO₃⁻-N (mg kg⁻¹)",
    "总氮(%)": "TN (%)",
    "总有机碳变化(%)": "TOC change (%)",
    "有机质含量变化(%)": "OM change (%)",
}

GROUP_LABELS = {
    "Feedstock and initial conditions": "Feedstock and initial conditions",
    "Composting process": "Composting process",
    "Laboratory chemistry": "Laboratory chemistry",
}


@st.cache_resource(show_spinner=False)
def load_bundle() -> dict:
    return joblib.load(BUNDLE_PATH)


def as_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    return buffer.getvalue()


def ensure_features(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy()
    missing = [feature for feature in features if feature not in work.columns]
    for feature in missing:
        work[feature] = np.nan
    return work[features], missing


def label_for(feature: str, metadata: dict | None = None) -> str:
    if feature in UI_LABELS:
        return UI_LABELS[feature]
    if metadata is not None:
        return metadata["display_names"].get(feature, feature)
    return feature


def range_warnings(row: pd.Series, metadata: dict) -> list[str]:
    warnings = []
    profile = metadata["numeric_profile"]
    for feature, stats in profile.items():
        value = row.get(feature)
        if pd.isna(value):
            warnings.append(f"{label_for(feature, metadata)} is missing and will be imputed.")
            continue
        try:
            value = float(value)
        except Exception:
            warnings.append(f"{label_for(feature, metadata)} is not numeric and will be imputed.")
            continue
        if value < stats["q01"] or value > stats["q99"]:
            warnings.append(f"{label_for(feature, metadata)} is outside the 1st-99th percentile training range.")
    waste = row.get("Waste type")
    if pd.isna(waste) or str(waste).strip() == "":
        warnings.append("Waste type is missing and will be imputed.")
    elif str(waste) not in metadata["waste_types"]:
        warnings.append("Waste type was not observed during training; prediction is extrapolative.")
    return warnings


def prediction_interval_label(value: float, metadata: dict) -> str:
    lo = metadata["training_target_min"]
    hi = metadata["training_target_max"]
    if value < lo or value > hi:
        return "Outside training target range"
    if value < 50:
        return "Low maturity signal"
    if value < 80:
        return "Intermediate maturity signal"
    return "High maturity signal"


def render_metric(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="small-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


bundle = load_bundle()
model = bundle["model"]
metadata = bundle["metadata"]
features = bundle["feature_columns"]
profile = metadata["numeric_profile"]

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
      <h1>Compost Germination Index Predictor</h1>
      <p>XGBoost model for estimating compost maturity as GI (%). Designed for rapid screening with transparent validation and applicability checks.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
random_metrics = metadata["metrics_mean"].get("random", {})
group_metrics = metadata["metrics_mean"].get("group", {})
with metric_cols[0]:
    render_metric("Model", metadata["model_name"], metadata["feature_set"])
with metric_cols[1]:
    render_metric("Random test R²", f"{random_metrics.get('test_r2', np.nan):.3f}", "within-distribution")
with metric_cols[2]:
    render_metric("Group test R²", f"{group_metrics.get('test_r2', np.nan):.3f}", "batch-aware robustness")
with metric_cols[3]:
    render_metric("Training rows", f"{metadata['training_rows']:,}", "GI-labelled records")

st.sidebar.header("Model scope")
st.sidebar.write("Target: **GI (%)**")
st.sidebar.write("Main model: **XGBoost / P1 feature protocol**")
st.sidebar.caption("Use group-test metrics as the conservative robustness estimate. Random-test metrics are within-distribution controls.")
with st.sidebar.expander("Caveats", expanded=False):
    for caveat in metadata["caveats"]:
        st.write(f"- {caveat}")

tab_single, tab_batch, tab_about = st.tabs(["Single prediction", "Batch prediction", "Model details"])

with tab_single:
    st.subheader("Single-sample GI prediction")
    row = {}

    group_cols = st.columns(3)
    for idx, (group_name, group_features) in enumerate(metadata["feature_groups"].items()):
        with group_cols[idx]:
            st.markdown(f"<div class='section-title'>{GROUP_LABELS.get(group_name, group_name)}</div>", unsafe_allow_html=True)
            for feature in group_features:
                if feature == "Waste type":
                    default = metadata["waste_types"].index(metadata["waste_types"][0]) if metadata["waste_types"] else 0
                    row[feature] = st.selectbox(label_for(feature, metadata), metadata["waste_types"], index=default)
                    continue
                stats = profile[feature]
                row[feature] = st.number_input(
                    label_for(feature, metadata),
                    value=float(stats["median"]),
                    step=max((stats["q95"] - stats["q05"]) / 100, 0.01),
                    format="%.4f",
                    help=f"Training range: {stats['min']:.3g} to {stats['max']:.3g}",
                )

    predict = st.button("Predict GI", type="primary")
    if predict:
        input_df = pd.DataFrame([row])[features]
        prediction = float(model.predict(input_df)[0])
        warnings = range_warnings(input_df.iloc[0], metadata)
        label = prediction_interval_label(prediction, metadata)
        st.markdown(
            f"""
            <div class="result-card">
              <div class="small-note">Predicted germination index</div>
              <div class="value">{prediction:.2f}%</div>
              <div class="small-note">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if warnings:
            st.markdown("<div class='warn-box'><b>Applicability warnings</b><br>" + "<br>".join(warnings) + "</div>", unsafe_allow_html=True)
        else:
            st.success("Input values are within the model's recorded training-domain checks.")

with tab_batch:
    st.subheader("Batch prediction")
    st.write("Upload a CSV or Excel file with the required model columns. Extra columns are preserved in the output.")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.download_button(
            "Download template CSV",
            data=TEMPLATE_PATH.read_bytes(),
            file_name="compost_gi_prediction_template.csv",
            mime="text/csv",
        )
    with col_b:
        st.download_button(
            "Download example CSV",
            data=EXAMPLE_PATH.read_bytes(),
            file_name="compost_gi_example_input.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            raw = pd.read_csv(uploaded)
        else:
            raw = pd.read_excel(uploaded)
        st.write("Preview")
        st.dataframe(raw.head(20), use_container_width=True)
        model_input, missing_cols = ensure_features(raw, features)
        pred = model.predict(model_input)
        result = raw.copy()
        result["Predicted GI (%)"] = pred
        result["Maturity signal"] = [prediction_interval_label(float(v), metadata) for v in pred]

        warning_counts = []
        for _, row_i in model_input.iterrows():
            warning_counts.append(len(range_warnings(row_i, metadata)))
        result["Applicability warning count"] = warning_counts

        if missing_cols:
            st.warning("Missing feature columns were added as blank and will be imputed: " + ", ".join(missing_cols))
        if any(warning_counts):
            st.warning("Some rows are outside training-domain checks. See the warning-count column.")

        st.write("Predictions")
        st.dataframe(result.head(100), use_container_width=True)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download predictions CSV",
                data=result.to_csv(index=False).encode("utf-8-sig"),
                file_name="compost_gi_predictions.csv",
                mime="text/csv",
            )
        with dl2:
            st.download_button(
                "Download predictions Excel",
                data=as_excel_bytes(result),
                file_name="compost_gi_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

with tab_about:
    st.subheader("Model details")
    st.write(
        "This app uses the same feature protocol as the paper figures: feedstock descriptors, process conditions, and laboratory chemistry variables."
    )
    st.dataframe(pd.DataFrame(metadata["metrics_raw"]), use_container_width=True)
    st.markdown("**Required input columns**")
    label_table = pd.DataFrame(
        {
            "Model column": features,
            "UI label": [label_for(feature, metadata) for feature in features],
        }
    )
    st.dataframe(label_table, use_container_width=True, hide_index=True)
