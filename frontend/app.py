from __future__ import annotations

# Standard library imports
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

# Third-party imports
import pandas as pd
import requests
import streamlit as st

# ============================================================
# BACKEND CONFIGURATION
# ============================================================

# Base URL of the FastAPI backend used by the Streamlit frontend.
# In deployment, this can be overridden using an environment variable.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Maximum wait time for backend requests before showing an error.
REQUEST_TIMEOUT = 12

# Explanations shown for each predicted risk level.
RISK_HELP = {
    "LOW": "Low likelihood of severe service impact. Continue standard monitoring.",
    "MEDIUM": "Moderate risk. Validate signals, review recent changes, and monitor closely.",
    "HIGH": "High likelihood of major service impact. Immediate investigation is recommended.",
}

# Descriptions for each severity input option.
# This helps users understand the historical severity context.
SEVERITY_HELP = {
    "severity_type 1": "Lower historical severity context.",
    "severity_type 2": "Moderate historical severity context.",
    "severity_type 3": "Elevated historical severity context.",
    "severity_type 4": "High historical severity context.",
    "severity_type 5": "Very high historical severity context.",
}

# Default selectable event features shown in the UI.
DEFAULT_EVENT_OPTIONS = [f"event_type_{i}" for i in [1, 5, 11, 15, 21, 35, 49, 54]]

# Default selectable resource features shown in the UI.
DEFAULT_RESOURCE_OPTIONS = [f"resource_type_{i}" for i in [2, 4, 6, 8, 10]]

# Default selectable log features shown in the UI.
DEFAULT_LOG_OPTIONS = [f"log_feature_{i}" for i in [40, 68, 82, 170, 172, 203, 225, 312]]

# Main project description shown in the sidebar.
PROJECT_DESCRIPTION = (
    "An Explainable Machine Learning-Based Framework for Fault Prediction and "
    "Isolation in Telecommunication Networks"
)

# Improve pandas display behavior for longer text content.
pd.set_option("display.max_colwidth", None)

# ============================================================
# PAGE SETUP
# ============================================================

# Configure the Streamlit page.
st.set_page_config(
    page_title="NetGuard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM STYLING
# ============================================================

# Custom CSS for layout, spacing, cards, colors, and sidebar appearance.
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37,99,235,0.11), transparent 26%),
                radial-gradient(circle at top right, rgba(16,185,129,0.07), transparent 22%),
                linear-gradient(180deg, #050816 0%, #07101f 56%, #0a1429 80%);
            color: #E5E7EB;
        }

        section[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top left, rgba(37,99,235,0.18), transparent 28%),
                linear-gradient(180deg, #07101f 0%, #0a1429 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        .ng-sidebar-title {
            font-size: 3rem;
            font-weight: 850;
            color: #F8FAFC;
            line-height: 1.08;
            margin-bottom: 0.35rem;
        }

        .ng-sidebar-desc {
            font-size: 1rem;
            line-height: 1.55;
            color: #C7D7F6;
            margin-bottom: 1rem;
        }

        .ng-page-title {
            font-size: 2rem;
            font-weight: 750;
            color: #F8FAFC;
            line-height: 1.12;
            margin-top: 2rem;
            margin-bottom: 0.3rem;
        }

        .ng-page-note {
            color: #A7B0C3;
            font-size: 0.96rem;
            margin-bottom: 1.0rem;
        }

        .ng-card {
            background: rgba(10, 18, 40, 0.92);
            border: 1px solid rgba(91, 116, 190, 0.22);
            border-radius: 18px;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.18);
            margin-bottom: 0.95rem;
        }

        .ng-kpi {
            background: linear-gradient(180deg, rgba(9,17,40,0.98), rgba(10,18,40,0.86));
            border: 1px solid rgba(91, 116, 190, 0.22);
            border-radius: 18px;
            padding: 16px;
            min-height: 124px;
            margin-bottom: 0.8rem;
        }

        .ng-kpi-label {
            font-size: 0.95rem;
            color: #A7B0C3;
            margin-bottom: 8px;
        }

        .ng-kpi-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #F8FAFC;
            line-height: 1.05;
            word-break: break-word;
        }

        .ng-kpi-sub {
            font-size: 0.92rem;
            color: #9FB0D3;
            margin-top: 8px;
        }

        .ng-banner {
            border-radius: 16px;
            padding: 14px 18px;
            font-weight: 500;
            margin-bottom: 18px;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .ng-low {
            background: rgba(34,197,94,0.16);
            color: #C7F9D4;
        }

        .ng-medium {
            background: rgba(234,179,8,0.18);
            color: #FFF1B3;
        }

        .ng-high {
            background: rgba(239,68,68,0.17);
            color: #FFD1D1;
        }

        .ng-section-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
            color: #F8FAFC;
        }

        .ng-muted {
            color: #9CA3AF;
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(96,165,250,0.28);
            background: linear-gradient(180deg, rgba(15,23,42,1), rgba(12,20,40,1));
            color: white;
            font-weight: 600;
            min-height: 2.8rem;
        }

        .stDownloadButton > button {
            border-radius: 12px;
        }

        .stSelectbox label,
        .stMultiSelect label,
        .stTextInput label,
        .stNumberInput label,
        .stSlider label {
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] .stSuccess,
        [data-testid="stSidebar"] .stError {
            margin-top: 1rem;
            margin-bottom: 0.85rem;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            font-size: 0.95rem !important;
            font-weight: 600 !important;
        }

        section[data-testid="stSidebar"] .stRadio > label {
            font-size: 0.9rem !important;
            font-weight: 700 !important;
            color: #B8C4DD !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# API HELPER FUNCTIONS
# ============================================================

def api_get(path: str, params: Dict[str, Any] | None = None) -> Tuple[bool, Any]:
    """
    Send a GET request to the backend API.

    Returns:
        (True, json_data) if successful
        (False, error_message) if failed
    """
    try:
        response = requests.get(
            f"{API_URL}{path}",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        if response.ok:
            return True, response.json()
        return False, f"{response.status_code}: {response.text}"
    except Exception as exc:
        return False, str(exc)


def api_post(path: str, payload: Dict[str, Any]) -> Tuple[bool, Any]:
    """
    Send a POST request to the backend API.

    Returns:
        (True, json_data) if successful
        (False, error_message) if failed
    """
    try:
        response = requests.post(
            f"{API_URL}{path}",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        if response.ok:
            return True, response.json()
        return False, f"{response.status_code}: {response.text}"
    except Exception as exc:
        return False, str(exc)

# ============================================================
# GENERAL HELPER FUNCTIONS
# ============================================================

def risk_class_name(risk_level: str) -> str:
    """
    Map a risk level to a CSS class used for banner styling.
    """
    rl = str(risk_level).upper().strip()
    if rl == "LOW":
        return "ng-low"
    if rl == "HIGH":
        return "ng-high"
    return "ng-medium"


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    If conversion fails, return the default.
    """
    try:
        return float(value)
    except Exception:
        return default


def format_pct(value: Any) -> str:
    """
    Convert a probability/confidence value to percentage string.
    Example: 0.889 -> 88.9%
    """
    return f"{safe_float(value) * 100:.1f}%"


def parse_feature_labels(items: List[str]) -> Dict[str, float]:
    """
    Convert selected event/resource feature names into sparse dictionary format.
    Example:
        ["event_type_1", "event_type_5"]
        -> {"event_type_1": 1.0, "event_type_5": 1.0}
    """
    return {item: 1.0 for item in items}


def parse_log_inputs(selected_logs: List[str], log_values: Dict[str, float]) -> Dict[str, float]:
    """
    Convert selected log features and their entered strengths
    into the sparse format expected by the backend.
    """
    out: Dict[str, float] = {}
    for key in selected_logs:
        out[key] = safe_float(log_values.get(key, 1.0), 1.0)
    return out


def build_payload(
    location_num: int,
    severity_num: int,
    event_types: List[str],
    resource_types: List[str],
    log_types: List[str],
    log_values: Dict[str, float],
) -> Dict[str, Any]:
    """
    Build the final payload sent to the backend prediction endpoint.
    """
    return {
        "location": f"location {location_num}",
        "severity_type": f"severity_type {severity_num}",
        "event_features": parse_feature_labels(event_types),
        "resource_features": parse_feature_labels(resource_types),
        "log_features": parse_log_inputs(log_types, log_values),
    }


def clean_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a dataframe safer for Streamlit rendering in local apps
    and for general dataframe cleaning.

    This is still useful for regular history/report tables.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    out.columns = [str(col) for col in out.columns]
    out = out.where(pd.notnull(out), None)

    for col in out.columns:
        out[col] = out[col].map(
            lambda x: x if isinstance(x, (int, float, str, bool, type(None), pd.Timestamp)) else str(x)
        )
        out[col] = out[col].astype("object")

    return out


def dataframe_from_top_features(top_features: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert backend SHAP top-feature output into a readable dataframe.
    """
    rows = []
    for item in top_features or []:
        shap_val = safe_float(item.get("shap_value"))
        rows.append(
            {
                "Feature": str(item.get("feature", "")),
                "Influence": float(round(shap_val, 5)),
                "Abs Influence": float(abs(round(shap_val, 5))),
            }
        )

    df = pd.DataFrame(rows, columns=["Feature", "Influence", "Abs Influence"])

    if not df.empty:
        df = df.sort_values("Abs Influence", ascending=False).reset_index(drop=True)

    return clean_dataframe_for_streamlit(df)


def dataframe_from_probabilities(probabilities: Dict[str, float]) -> pd.DataFrame:
    """
    Convert backend class probabilities into a readable dataframe.
    """
    label_map = {"0": "LOW", "1": "MEDIUM", "2": "HIGH"}
    rows = []

    for cls, prob in (probabilities or {}).items():
        rows.append(
            {
                "Class": f"{cls} ({label_map.get(str(cls), str(cls))})",
                "Probability": float(safe_float(prob)),
            }
        )

    df = pd.DataFrame(rows, columns=["Class", "Probability"])

    if not df.empty:
        df = df.sort_values("Probability", ascending=False).reset_index(drop=True)

    return clean_dataframe_for_streamlit(df)

# ============================================================
# UI RENDER HELPER FUNCTIONS
# ============================================================

def render_page_header(page_title: str, page_note: str) -> None:
    """
    Render the main page title and supporting note.
    """
    st.markdown(f'<div class="ng-page-title">{page_title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ng-page-note">{page_note}</div>', unsafe_allow_html=True)


def render_kpi(title: str, value: str, subtitle: str = "") -> None:
    """
    Render a KPI card used throughout the dashboard and result pages.
    """
    st.markdown(
        f"""
        <div class="ng-kpi">
            <div class="ng-kpi-label">{title}</div>
            <div class="ng-kpi-value">{value}</div>
            <div class="ng-kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_banner(risk_level: str) -> None:
    """
    Render the colored risk banner after prediction.
    """
    rl = str(risk_level).upper().strip()
    message = RISK_HELP.get(rl, "Model output ready for analyst review.")
    css = risk_class_name(rl)

    st.markdown(
        f'<div class="ng-banner {css}">{rl} RISK — {message}</div>',
        unsafe_allow_html=True,
    )


def render_card(title: str, body: str) -> None:
    """
    Render a reusable text card.
    """
    st.markdown(
        f"""
        <div class="ng-card">
            <div class="ng-section-title">{title}</div>
            <div class="ng-muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_api_status() -> None:
    """
    Check backend health and show online/offline status in the sidebar.
    """
    ok, _ = api_get("/health")
    if ok:
        st.success("API Online")
    else:
        st.error("API Offline")


def explain_inputs_block() -> None:
    """
    Show an expandable explanation section for the input and output fields.
    """
    with st.expander("What do these inputs and outputs mean?", expanded=False):
        st.markdown(
            """
**Location**  
Represents the telecom network site or node associated with the incident.

**Severity Type**  
Historical severity context associated with the reported fault pattern.

**Event Signals**  
Alarm or event indicators detected by the monitoring environment.

**Resource Signals**  
Related network resources or components linked to the incident.

**Log Signals**  
Important log signatures associated with the case. Higher values indicate stronger log presence or volume.

**Predicted Severity / Risk Level**  
The class predicted by the model from the provided incident signals.

**Confidence**  
The model's probability for the predicted class. Higher confidence means the model is more certain, not that it is guaranteed correct.

**Fault Category**  
A rule-based interpretation of the dominant signal pattern to support fault isolation.

**Recommended Checks**  
Investigation guidance for the engineer after the prediction is produced.
            """
        )

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    # Sidebar project title.
    st.markdown('<div class="ng-sidebar-title">NetGuard</div>', unsafe_allow_html=True)

    # Sidebar project description.
    st.markdown(
        f'<div class="ng-sidebar-desc">{PROJECT_DESCRIPTION}</div>',
        unsafe_allow_html=True,
    )

    # Backend status indicator.
    show_api_status()
    st.markdown("---")

    # Main app navigation.
    page = st.radio(
        "Navigation",
        options=["Dashboard", "Analyze Incident", "Incident History", "Reports"],
    )

# ============================================================
# SESSION STATE
# ============================================================

# Store latest prediction result so it remains visible after reruns.
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

# Store latest prediction ID from backend.
if "prediction_id" not in st.session_state:
    st.session_state["prediction_id"] = None

# Store latest request payload used for prediction.
if "last_payload" not in st.session_state:
    st.session_state["last_payload"] = None

# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "Dashboard":
    render_page_header(
        "Dashboard",
        "Operational overview of recent telecom fault assessments and system outputs.",
    )

    # Load stored predictions from backend.
    ok, history = api_get("/predictions", params={"limit": 100})
    rows = history if ok and isinstance(history, list) else []

    # Calculate summary values.
    total_cases = len(rows)
    high_count = sum(1 for row in rows if str(row.get("risk_level", "")).upper() == "HIGH")
    medium_count = sum(1 for row in rows if str(row.get("risk_level", "")).upper() == "MEDIUM")
    avg_conf = (
        sum(safe_float(row.get("confidence")) for row in rows) / total_cases
        if total_cases else 0.0
    )

    # KPI summary cards.
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        render_kpi("Assessments Logged", str(total_cases), "Stored prediction records")
    with col_b:
        render_kpi("High Risk Cases", str(high_count), "Require immediate review")
    with col_c:
        render_kpi("Medium Risk Cases", str(medium_count), "Need validation and monitoring")
    with col_d:
        render_kpi("Average Confidence", format_pct(avg_conf), "Across recent outputs")

    st.markdown("### Recent Activity")

    if rows:
        hist_df = pd.DataFrame(rows)
        hist_df["created_at"] = pd.to_datetime(hist_df["created_at"], errors="coerce")
        hist_df["confidence_pct"] = hist_df["confidence"].apply(lambda x: f"{safe_float(x) * 100:.1f}%")

        # Table shown in dashboard recent activity.
        show_df = hist_df[
            ["id", "created_at", "location", "severity_type", "risk_level", "confidence_pct", "fault_category"]
        ].copy()

        show_df.columns = [
            "ID",
            "Created At",
            "Location",
            "Severity Type",
            "Risk Level",
            "Confidence",
            "Fault Category",
        ]

        show_df = clean_dataframe_for_streamlit(show_df)
        st.dataframe(show_df, use_container_width=True)

        # Risk distribution bar chart.
        risk_counts = hist_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]

        st.markdown("### Risk Distribution")
        st.bar_chart(risk_counts.set_index("Risk Level"), height=320)
    else:
        st.info("No predictions found yet. Run an incident assessment to populate the dashboard.")

# ============================================================
# ANALYZE INCIDENT PAGE
# ============================================================

elif page == "Analyze Incident":
    render_page_header(
        "Analyze Incident",
        "Enter incident details manually, run the model, and review severity, explanation, and likely isolation direction.",
    )

    explain_inputs_block()

    # Two-column layout:
    # left side = inputs
    # right side = summary cards
    left, right = st.columns([1.22, 1.0], gap="large")

    with left:
        st.subheader("Incident Inputs")

        # Telecom site/location input.
        location_num = st.number_input(
            "Location ID",
            min_value=1,
            max_value=2000,
            value=1,
            step=1,
            help="Represents the telecom site or network location linked to the incident.",
        )

        # Severity input.
        severity_options = [1, 2, 3, 4, 5]
        severity_num = st.selectbox(
            "Severity Type",
            options=severity_options,
            index=0,
            help="Historical severity context associated with the selected incident.",
        )
        st.caption(SEVERITY_HELP.get(f"severity_type {severity_num}", ""))

        # Event feature inputs.
        event_types = st.multiselect(
            "Event Signals",
            options=DEFAULT_EVENT_OPTIONS,
            default=[],
            help="Alarm or event signals triggered for this incident.",
        )

        # Resource feature inputs.
        resource_types = st.multiselect(
            "Resource Signals",
            options=DEFAULT_RESOURCE_OPTIONS,
            default=[],
            help="Related network resources or components involved in the incident.",
        )

        # Log feature inputs.
        log_types = st.multiselect(
            "Log Signals",
            options=DEFAULT_LOG_OPTIONS,
            default=[],
            help="Important log patterns related to the incident.",
        )

        # Numeric strengths for each selected log feature.
        st.markdown("##### Log Signal Strength")
        st.caption("Set a value for each selected log signal. Higher values represent stronger signal volume or presence.")

        log_values: Dict[str, float] = {}
        if log_types:
            log_cols = st.columns(2)
            for idx, log_key in enumerate(log_types):
                with log_cols[idx % 2]:
                    log_values[log_key] = st.number_input(
                        f"{log_key}",
                        min_value=0.0,
                        max_value=1000.0,
                        value=1.0,
                        step=1.0,
                    )

        # Build request payload for backend.
        payload = build_payload(
            location_num=location_num,
            severity_num=severity_num,
            event_types=event_types,
            resource_types=resource_types,
            log_types=log_types,
            log_values=log_values,
        )

        # Save latest payload in session state.
        st.session_state["last_payload"] = payload

        # Main action button.
        run = st.button("Run NetGuard Assessment", type="primary")

    with right:
        st.subheader("Input Summary")

        # Small summary cards showing current selected inputs.
        c1, c2 = st.columns(2)
        with c1:
            render_kpi("Selected Location", f"location {location_num}", "Telecom site identifier")
        with c2:
            render_kpi("Severity Context", f"severity_type {severity_num}", "Historical severity input")

        c3, c4 = st.columns(2)
        with c3:
            render_kpi("Event Signals", str(len(event_types)), "Selected event indicators")
        with c4:
            render_kpi("Resource Signals", str(len(resource_types)), "Selected resource indicators")

        c5, c6 = st.columns(2)
        with c5:
            render_kpi("Log Signals", str(len(log_types)), "Selected log signatures")
        with c6:
            render_kpi("Log Volume Sum", f"{sum(log_values.values()):.1f}", "Sum of selected log strengths")

        # Short system workflow card.
        st.markdown(
            """
            <div class="ng-card">
                <div class="ng-section-title">Assessment Flow</div>
                <div class="ng-muted">
                    1. Predict fault severity from incident signals<br>
                    2. Explain the prediction using feature influence<br>
                    3. Suggest fault isolation direction and investigation checks
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Run prediction when user presses the button.
    if run:
        with st.spinner("Running NetGuard assessment..."):
            ok, response = api_post("/predict", payload)

        if ok and isinstance(response, dict):
            st.session_state["prediction_result"] = response.get("result")
            st.session_state["prediction_id"] = response.get("prediction_id")
            st.success("Assessment completed successfully.")
        else:
            st.error(f"Prediction failed: {response}")

    # Read stored result from session state.
    result = st.session_state.get("prediction_result")
    prediction_id = st.session_state.get("prediction_id")

    # Only show prediction section if a result exists.
    if result:
        st.markdown("---")
        render_banner(str(result.get("risk_level", "MEDIUM")))

        st.subheader("Prediction")

        # Main output KPI cards.
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            render_kpi(
                "Risk Level",
                str(result.get("risk_level", "N/A")),
                RISK_HELP.get(str(result.get("risk_level", "")).upper(), ""),
            )
        with p2:
            render_kpi(
                "Severity Class",
                str(result.get("predicted_severity", "N/A")),
                "0 = Low, 1 = Medium, 2 = High",
            )
        with p3:
            render_kpi(
                "Confidence",
                format_pct(result.get("confidence", 0.0)),
                "Model certainty for the selected class",
            )
        with p4:
            render_kpi(
                "Fault Category",
                str(result.get("fault_category") or "N/A"),
                "Isolation-oriented interpretation",
            )

        st.subheader("Explanation")

        # Natural-language explanation returned by backend.
        st.markdown(
            f"""
            <div class="ng-card" style="border-left: 4px solid #60A5FA;">
                <div class="ng-section-title">Prediction Summary</div>
                <div class="ng-muted">{result.get("reason", "No explanation available.")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Two-column section:
        # left = top contributing features
        # right = class probability breakdown
        exp_left, exp_right = st.columns([1.04, 1.0], gap="large")

        with exp_left:
            st.markdown("#### Top Contributing Features")

            # Convert backend explanation output into a display dataframe.
            top_df = dataframe_from_top_features(result.get("top_features", []))

            st.markdown('<div class="ng-card">', unsafe_allow_html=True)

            # Render as markdown text instead of st.table/st.dataframe
            # to avoid deployment Arrow LargeUtf8 frontend errors.
            if not top_df.empty:
                st.markdown("**Top Features**")
                for _, row in top_df.iterrows():
                    feature = str(row["Feature"])
                    influence = safe_float(row["Influence"])
                    abs_influence = safe_float(row["Abs Influence"])
                    st.markdown(
                        f"- **{feature}** — Influence: `{influence:.5f}`, Abs Influence: `{abs_influence:.5f}`"
                    )
            else:
                st.info("No top features returned by the backend.")

            st.markdown("</div>", unsafe_allow_html=True)

        with exp_right:
            st.markdown("#### Class Probability Breakdown")

            # Convert backend class probabilities into display dataframe.
            prob_df = dataframe_from_probabilities(result.get("class_probabilities", {}))

            st.markdown('<div class="ng-card">', unsafe_allow_html=True)

            if not prob_df.empty:
                # Create numeric copy for charting.
                chart_df = prob_df.copy()
                chart_df["Probability"] = pd.to_numeric(
                    chart_df["Probability"], errors="coerce"
                ).fillna(0.0)

                # Show class probability chart.
                st.bar_chart(chart_df.set_index("Class"), height=300)

                # Render probability values as markdown instead of dataframe/table.
                st.markdown("**Probability Values**")
                for _, row in chart_df.iterrows():
                    label = str(row["Class"])
                    prob = safe_float(row["Probability"])
                    st.markdown(f"- **{label}**: {prob:.4f} ({prob * 100:.1f}%)")
            else:
                st.info("No class probabilities available.")

            st.markdown("</div>", unsafe_allow_html=True)

        # Isolation summary and recommended checks section.
        iso_left, iso_right = st.columns(2, gap="large")

        with iso_left:
            st.markdown("#### Isolation Summary")
            st.markdown(
                f"""
                <div class="ng-card">
                    <div class="ng-section-title">{str(result.get("fault_category") or "Mixed Operational Signals")}</div>
                    <div class="ng-muted">{str(result.get("isolation_summary") or "No isolation summary available.")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with iso_right:
            st.markdown("#### Recommended Checks")
            checks = result.get("recommended_checks") or []

            if checks:
                checks_html = "".join([f"<li>{str(check)}</li>" for check in checks])
                st.markdown(
                    f"""
                    <div class="ng-card">
                        <ul style="margin-top:0.4rem; padding-left:1.2rem;">
                            {checks_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("No recommended checks returned.")

        # Build exportable JSON payload containing both input and output.
        export_payload = {
            "prediction_id": prediction_id,
            "payload": st.session_state.get("last_payload"),
            "result": result,
            "generated_at": datetime.now().isoformat(),
        }

        # Download button for exporting latest assessment.
        st.download_button(
            "Download Assessment JSON",
            data=json.dumps(export_payload, indent=2),
            file_name=f"netguard_assessment_{prediction_id or 'latest'}.json",
            mime="application/json",
        )

# ============================================================
# INCIDENT HISTORY PAGE
# ============================================================

elif page == "Incident History":
    render_page_header(
        "Incident History",
        "Browse past assessments, apply filters, and inspect detailed explanations stored by the backend.",
    )

    # Filter controls.
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        hist_limit = st.slider("Records", min_value=10, max_value=200, value=50, step=10)
    with filter_col2:
        hist_risk = st.selectbox("Risk Filter", options=["ALL", "LOW", "MEDIUM", "HIGH"])
    with filter_col3:
        min_conf_pct = st.slider("Minimum Confidence (%)", min_value=0, max_value=100, value=0, step=5)

    # Build API query parameters from filters.
    params: Dict[str, Any] = {"limit": hist_limit}
    if hist_risk != "ALL":
        params["risk_level"] = hist_risk
    if min_conf_pct > 0:
        params["min_confidence"] = min_conf_pct / 100.0

    # Load filtered history records.
    ok, history = api_get("/predictions", params=params)

    if not ok:
        st.error(f"Could not load history: {history}")
    elif not history:
        st.info("No history records match the current filters.")
    else:
        hist_df = pd.DataFrame(history)
        hist_df["created_at"] = pd.to_datetime(hist_df["created_at"], errors="coerce")
        hist_df["confidence_pct"] = hist_df["confidence"].apply(lambda x: f"{safe_float(x) * 100:.1f}%")

        # Main incident history display table.
        display_df = hist_df[
            [
                "id",
                "created_at",
                "location",
                "severity_type",
                "risk_level",
                "confidence_pct",
                "fault_category",
                "event_count",
                "resource_count",
                "log_count",
                "log_volume_sum",
            ]
        ].copy()

        display_df.columns = [
            "ID",
            "Created At",
            "Location",
            "Severity Type",
            "Risk Level",
            "Confidence",
            "Fault Category",
            "Event Count",
            "Resource Count",
            "Log Count",
            "Log Volume Sum",
        ]

        display_df = clean_dataframe_for_streamlit(display_df)
        st.dataframe(display_df, use_container_width=True)

        # Let user select a record and inspect its detail.
        prediction_ids = display_df["ID"].tolist()
        selected_id = st.selectbox("Open assessment detail", options=prediction_ids)

        ok_detail, detail = api_get(
            f"/predictions/{selected_id}",
            params={"include_explanations": True},
        )

        if ok_detail and isinstance(detail, dict):
            st.markdown("### Assessment Detail")

            # Detail summary KPI cards.
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                render_kpi("Risk Level", str(detail.get("risk_level", "N/A")), "Stored prediction output")
            with d2:
                render_kpi("Confidence", format_pct(detail.get("confidence", 0.0)), "Stored model confidence")
            with d3:
                render_kpi("Location", str(detail.get("location", "N/A")), "Incident location input")
            with d4:
                render_kpi("Severity Type", str(detail.get("severity_type", "N/A")), "Historical severity input")

            # Natural-language reason and isolation summary.
            render_card("Reason", str(detail.get("reason", "")))

            if detail.get("isolation_summary"):
                render_card("Isolation Summary", str(detail.get("isolation_summary")))

            # Stored feature explanation shown safely as markdown list.
            exp_df = dataframe_from_top_features(detail.get("explanations", []))
            if not exp_df.empty:
                st.markdown("#### Stored Top Features")
                for _, row in exp_df.iterrows():
                    feature = str(row["Feature"])
                    influence = safe_float(row["Influence"])
                    abs_influence = safe_float(row["Abs Influence"])
                    st.markdown(
                        f"- **{feature}** — Influence: `{influence:.5f}`, Abs Influence: `{abs_influence:.5f}`"
                    )
        else:
            st.warning(f"Could not load detail for ID {selected_id}.")

# ============================================================
# REPORTS PAGE
# ============================================================

elif page == "Reports":
    render_page_header(
        "Reports",
        "View high-level summaries of risk levels, confidence, event activity, and fault categories.",
    )

    # Load larger set of stored records for reporting.
    ok, history = api_get("/predictions", params={"limit": 200})
    rows = history if ok and isinstance(history, list) else []

    if not rows:
        st.info("No stored assessments available for reporting.")
    else:
        df = pd.DataFrame(rows)
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["confidence_pct"] = df["confidence"].apply(lambda x: round(safe_float(x) * 100, 1))

        # Risk summary table.
        st.markdown("### Risk Summary")
        risk_summary = (
            df.groupby("risk_level", dropna=False)
            .agg(
                assessments=("id", "count"),
                avg_confidence=("confidence_pct", "mean"),
                avg_event_count=("event_count", "mean"),
                avg_log_volume=("log_volume_sum", "mean"),
            )
            .reset_index()
        )
        risk_summary = clean_dataframe_for_streamlit(risk_summary)
        st.dataframe(risk_summary, use_container_width=True)

        # Fault category summary table and chart.
        st.markdown("### Fault Category Summary")
        if "fault_category" in df.columns:
            cat_summary = (
                df.groupby("fault_category", dropna=False)
                .agg(count=("id", "count"))
                .sort_values("count", ascending=False)
                .reset_index()
            )
            cat_summary = clean_dataframe_for_streamlit(cat_summary)
            st.dataframe(cat_summary, use_container_width=True)

            if not cat_summary.empty:
                chart_df = cat_summary.copy()
                chart_df["count"] = pd.to_numeric(chart_df["count"], errors="coerce").fillna(0)
                st.bar_chart(chart_df.set_index("fault_category"), height=320)

        # CSV export for full stored history.
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download History CSV",
            data=csv_data,
            file_name="netguard_history_report.csv",
            mime="text/csv",
        )