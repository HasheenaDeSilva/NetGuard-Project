from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st
import os

# _BACKEND CONFIGURATION_

# Base URL of the FastAPI backend used by the Streamlit frontend.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Maximum wait time for backend requests before showing an error.
REQUEST_TIMEOUT = 12

# Explanations for each predicted risk level.
# These are shown in the UI so users can quickly understand the result.
RISK_HELP = {
    "LOW": "Low likelihood of severe service impact. Continue standard monitoring.",
    "MEDIUM": "Moderate risk. Validate signals, review recent changes, and monitor closely.",
    "HIGH": "High likelihood of major service impact. Immediate investigation is recommended.",
}

# Descriptions for the severity input selected by the user.
# This helps explain what the historical severity context means.
SEVERITY_HELP = {
    "severity_type 1": "Lower historical severity context.",
    "severity_type 2": "Moderate historical severity context.",
    "severity_type 3": "Elevated historical severity context.",
    "severity_type 4": "High historical severity context.",
    "severity_type 5": "Very high historical severity context.",
}

# Default feature options shown in the incident analysis form.
# These act as selectable sample features for event, resource, and log signals.
DEFAULT_EVENT_OPTIONS = [f"event_type_{i}" for i in [1, 5, 11, 15, 21, 35, 49, 54]]
DEFAULT_RESOURCE_OPTIONS = [f"resource_type_{i}" for i in [2, 4, 6, 8, 10]]
DEFAULT_LOG_OPTIONS = [f"log_feature_{i}" for i in [40, 68, 82, 170, 172, 203, 225, 312]]

# Main research/project description used in the sidebar.
PROJECT_DESCRIPTION = (
    "An Explainable Machine Learning-Based Framework for Fault Prediction and "
    "Isolation in Telecommunication Networks"
)


# _PAGE SETUP_

# Streamlit page configuration for title, icon, width, and sidebar state.
st.set_page_config(
    page_title="NetGuard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# _CUSTOM STYLING_

# Custom CSS used to create a professional dark theme layout.
# This controls colors, spacing, cards, KPI boxes, sidebar look, and buttons.
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

        .stDataFrame, .stTable {
            border-radius: 14px;
            overflow: hidden;
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


# _API HELPERS_

# These helper functions centralize communication with the backend API.
# They return:
#   - True + JSON data if successful
#   - False + error message if failed
def api_get(path: str, params: Dict[str, Any] | None = None) -> Tuple[bool, Any]:
    """Send a GET request to the backend and return success flag plus response body."""
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
    """Send a POST request to the backend and return success flag plus response body."""
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


# _GENERAL HELPERS_

# Utility functions used throughout the UI for formatting,
# safe conversions, feature preparation, and table creation.
def risk_class_name(risk_level: str) -> str:
    """Return CSS class name used to style the risk banner."""
    rl = str(risk_level).upper().strip()
    if rl == "LOW":
        return "ng-low"
    if rl == "HIGH":
        return "ng-high"
    return "ng-medium"


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert any value safely to float."""
    try:
        return float(value)
    except Exception:
        return default


def format_pct(value: Any) -> str:
    """Format numeric probability or confidence as percentage."""
    return f"{safe_float(value) * 100:.1f}%"


def parse_feature_labels(items: List[str]) -> Dict[str, float]:
    """Convert selected event/resource feature names into sparse 1.0 dictionary."""
    return {item: 1.0 for item in items}


def parse_log_inputs(selected_logs: List[str], log_values: Dict[str, float]) -> Dict[str, float]:
    """Convert selected log features and their strengths into sparse backend format."""
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
    """Build API prediction payload from current UI input values."""
    return {
        "location": f"location {location_num}",
        "severity_type": f"severity_type {severity_num}",
        "event_features": parse_feature_labels(event_types),
        "resource_features": parse_feature_labels(resource_types),
        "log_features": parse_log_inputs(log_types, log_values),
    }


def dataframe_from_top_features(top_features: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert top SHAP features into a readable table."""
    rows = []
    for item in top_features or []:
        shap_val = safe_float(item.get("shap_value"))
        rows.append(
            {
                "Feature": str(item.get("feature", "")),
                "Influence": round(shap_val, 5),
                "Abs Influence": abs(round(shap_val, 5)),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Abs Influence", ascending=False).reset_index(drop=True)
    return df


def dataframe_from_probabilities(probabilities: Dict[str, float]) -> pd.DataFrame:
    """Convert class probabilities into a display table with human-readable labels."""
    label_map = {"0": "LOW", "1": "MEDIUM", "2": "HIGH"}
    rows = []
    for cls, prob in probabilities.items():
        rows.append(
            {
                "Class": f"{cls} ({label_map.get(str(cls), str(cls))})",
                "Probability": safe_float(prob),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Probability", ascending=False).reset_index(drop=True)
    return df


# _UI RENDER HELPERS_

# These functions build reusable interface components
# like page headers, KPI cards, banners, information cards, and helper sections.
def render_page_header(page_title: str, page_note: str) -> None:
    """Render the main page title and supporting note for the selected page."""
    st.markdown(
        f'<div class="ng-page-title">{page_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="ng-page-note">{page_note}</div>',
        unsafe_allow_html=True,
    )


def render_kpi(title: str, value: str, subtitle: str = "") -> None:
    """Render a custom KPI card."""
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
    """Render a colored banner describing the predicted risk level."""
    rl = str(risk_level).upper().strip()
    message = RISK_HELP.get(rl, "Model output ready for analyst review.")
    css = risk_class_name(rl)
    st.markdown(
        f'<div class="ng-banner {css}">{rl} RISK — {message}</div>',
        unsafe_allow_html=True,
    )


def render_card(title: str, body: str) -> None:
    """Render a reusable text content card."""
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
    """Check backend availability and show current API status in the sidebar."""
    ok, _ = api_get("/health")
    if ok:
        st.success("API Online")
    else:
        st.error("API Offline")


def explain_inputs_block() -> None:
    """Expandable helper that explains inputs and outputs."""
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


# _SIDEBAR_

# Sidebar provides project identity, backend status,
# and navigation across the main sections of the system.
with st.sidebar:
    # Project title shown at the top of the sidebar.
    st.markdown('<div class="ng-sidebar-title">NetGuard</div>', unsafe_allow_html=True)

    # Research/project description shown below the title.
    st.markdown(
        f'<div class="ng-sidebar-desc">{PROJECT_DESCRIPTION}</div>',
        unsafe_allow_html=True,
    )

    # Current backend health state.
    show_api_status()
    st.markdown("---")

    # Main page navigation.
    page = st.radio(
        "Navigation",
        options=["Dashboard", "Analyze Incident", "Incident History", "Reports"],
    )


# _SESSION STATE_

# Session state keeps important values available across reruns.
# This allows prediction results and payloads to remain visible
# even after Streamlit refreshes the page interaction.
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

if "prediction_id" not in st.session_state:
    st.session_state["prediction_id"] = None

if "last_payload" not in st.session_state:
    st.session_state["last_payload"] = None


# _DASHBOARD PAGE_

# Displays a high-level summary of recent stored assessments
# including counts, average confidence, and recent activity.
if page == "Dashboard":
    # Page title and short explanation for the dashboard.
    render_page_header(
        "Dashboard",
        "Operational overview of recent telecom fault assessments and system outputs.",
    )

    # Load recent predictions from the backend.
    ok, history = api_get("/predictions", params={"limit": 100})
    rows = history if ok and isinstance(history, list) else []

    # Calculate quick dashboard metrics.
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
        # Convert backend history to a DataFrame for easier display.
        hist_df = pd.DataFrame(rows)
        hist_df["created_at"] = pd.to_datetime(hist_df["created_at"], errors="coerce")
        hist_df["confidence_pct"] = hist_df["confidence"].apply(lambda x: f"{safe_float(x) * 100:.1f}%")

        # Keep only the most important columns for the dashboard table.
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

        st.dataframe(show_df, use_container_width=True, hide_index=True)

        # Create a small risk distribution chart.
        risk_counts = hist_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]

        st.markdown("### Risk Distribution")
        st.bar_chart(risk_counts.set_index("Risk Level"), height=320)
    else:
        st.info("No predictions found yet. Run an incident assessment to populate the dashboard.")


# _ANALYZE INCIDENT PAGE_

# Main page where the user enters incident data,
# runs the model, and reviews predictions + explanations.
elif page == "Analyze Incident":
    # Page title and short explanation for the analysis page.
    render_page_header(
        "Analyze Incident",
        "Enter incident details manually, run the model, and review severity, explanation, and likely isolation direction.",
    )

    # Expandable help content for users and reviewers.
    explain_inputs_block()

    # Two-column layout:
    # left = form inputs
    # right = summary of the selected input values
    left, right = st.columns([1.22, 1.0], gap="large")

    with left:
        st.subheader("Incident Inputs")

        # Numeric location identifier for the telecom site.
        location_num = st.number_input(
            "Location ID",
            min_value=1,
            max_value=2000,
            value=1,
            step=1,
            help="Represents the telecom site or network location linked to the incident.",
        )

        # Historical severity context selected by the user.
        severity_options = [1, 2, 3, 4, 5]
        severity_num = st.selectbox(
            "Severity Type",
            options=severity_options,
            index=0,
            help="Historical severity context associated with the selected incident.",
        )
        st.caption(SEVERITY_HELP.get(f"severity_type {severity_num}", ""))

        # Event-related signals selected by the user.
        event_types = st.multiselect(
            "Event Signals",
            options=DEFAULT_EVENT_OPTIONS,
            default=[],
            help="Alarm or event signals triggered for this incident.",
        )

        # Resource-related signals selected by the user.
        resource_types = st.multiselect(
            "Resource Signals",
            options=DEFAULT_RESOURCE_OPTIONS,
            default=[],
            help="Related network resources or components involved in the incident.",
        )

        # Log-related signals selected by the user.
        log_types = st.multiselect(
            "Log Signals",
            options=DEFAULT_LOG_OPTIONS,
            default=[],
            help="Important log patterns related to the incident.",
        )

        # Numeric strength values for each selected log signal.
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

        # Build the request payload exactly as expected by the backend API.
        payload = build_payload(
            location_num=location_num,
            severity_num=severity_num,
            event_types=event_types,
            resource_types=resource_types,
            log_types=log_types,
            log_values=log_values,
        )

        # Store the most recent payload so it can be exported later.
        st.session_state["last_payload"] = payload

        # Button to trigger the model assessment.
        run = st.button("Run NetGuard Assessment", type="primary")

    with right:
        st.subheader("Input Summary")

        # Summary cards showing the selected incident inputs.
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

        # Short process overview for the user.
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

    # Run prediction only when the button is pressed.
    if run:
        with st.spinner("Running NetGuard assessment..."):
            ok, response = api_post("/predict", payload)

        if ok and isinstance(response, dict):
            # Store the latest result in session state so it persists across reruns.
            st.session_state["prediction_result"] = response.get("result")
            st.session_state["prediction_id"] = response.get("prediction_id")
            st.success("Assessment completed successfully.")
        else:
            st.error(f"Prediction failed: {response}")

    # Read the stored prediction result.
    result = st.session_state.get("prediction_result")
    prediction_id = st.session_state.get("prediction_id")

    # Show prediction output only when a result is available.
    if result:
        st.markdown("---")
        render_banner(str(result.get("risk_level", "MEDIUM")))

        st.subheader("Prediction")

        # Main prediction result cards.
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

        # Natural-language explanation returned by the backend.
        st.markdown(
            f"""
            <div class="ng-card" style="border-left: 4px solid #60A5FA;">
                <div class="ng-section-title">Prediction Summary</div>
                <div class="ng-muted">{result.get("reason", "No explanation available.")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Two-column explanation section:
        # left = feature influence
        # right = class probabilities
        exp_left, exp_right = st.columns([1.04, 1.0], gap="large")

        with exp_left:
            st.markdown("#### Top Contributing Features")
            top_df = dataframe_from_top_features(result.get("top_features", []))
            st.markdown('<div class="ng-card">', unsafe_allow_html=True)
            if not top_df.empty:
                st.dataframe(top_df, use_container_width=True, hide_index=True)
            else:
                st.info("No top features returned by the backend.")
            st.markdown("</div>", unsafe_allow_html=True)

        with exp_right:
            st.markdown("#### Class Probability Breakdown")
            prob_df = dataframe_from_probabilities(result.get("class_probabilities", {}))
            st.markdown('<div class="ng-card">', unsafe_allow_html=True)
            if not prob_df.empty:
                chart_df = prob_df.copy().set_index("Class")
                st.bar_chart(chart_df, height=300)
                st.dataframe(show_df, use_container_width=True, hide_index=True)
            else:
                st.info("No class probabilities available.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Fault isolation summary and recommended checks.
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

        # Build export content using both input payload and output result.
        export_payload = {
            "prediction_id": prediction_id,
            "payload": st.session_state.get("last_payload"),
            "result": result,
            "generated_at": datetime.now().isoformat(),
        }

        # Downloadable JSON export for demo or reporting use.
        st.download_button(
            "Download Assessment JSON",
            data=json.dumps(export_payload, indent=2),
            file_name=f"netguard_assessment_{prediction_id or 'latest'}.json",
            mime="application/json",
            width="content",
        )


# _INCIDENT HISTORY PAGE_

# Shows stored prediction records with filters
# and allows the user to inspect details of one record.
elif page == "Incident History":
    # Page title and short explanation for the history page.
    render_page_header(
        "Incident History",
        "Browse past assessments, apply filters, and inspect detailed explanations stored by the backend.",
    )

    # History filter controls.
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        hist_limit = st.slider("Records", min_value=10, max_value=200, value=50, step=10)
    with filter_col2:
        hist_risk = st.selectbox("Risk Filter", options=["ALL", "LOW", "MEDIUM", "HIGH"])
    with filter_col3:
        min_conf_pct = st.slider("Minimum Confidence (%)", min_value=0, max_value=100, value=0, step=5)

    # Build backend query parameters from the selected filters.
    params: Dict[str, Any] = {"limit": hist_limit}
    if hist_risk != "ALL":
        params["risk_level"] = hist_risk
    if min_conf_pct > 0:
        params["min_confidence"] = min_conf_pct / 100.0

    # Load filtered prediction history from the backend.
    ok, history = api_get("/predictions", params=params)

    if not ok:
        st.error(f"Could not load history: {history}")
    elif not history:
        st.info("No history records match the current filters.")
    else:
        # Convert history records into a table.
        hist_df = pd.DataFrame(history)
        hist_df["created_at"] = pd.to_datetime(hist_df["created_at"], errors="coerce")
        hist_df["confidence_pct"] = hist_df["confidence"].apply(lambda x: f"{safe_float(x) * 100:.1f}%")

        # Clean display table for the user.
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

        st.dataframe(top_df, use_container_width=True, hide_index=True)

        # Let the user choose one assessment to inspect in detail.
        prediction_ids = display_df["ID"].tolist()
        selected_id = st.selectbox("Open assessment detail", options=prediction_ids)

        # Load the selected record with explanation details included.
        ok_detail, detail = api_get(
            f"/predictions/{selected_id}",
            params={"include_explanations": True},
        )

        if ok_detail and isinstance(detail, dict):
            st.markdown("### Assessment Detail")

            # Summary KPI cards for the selected record.
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                render_kpi("Risk Level", str(detail.get("risk_level", "N/A")), "Stored prediction output")
            with d2:
                render_kpi("Confidence", format_pct(detail.get("confidence", 0.0)), "Stored model confidence")
            with d3:
                render_kpi("Location", str(detail.get("location", "N/A")), "Incident location input")
            with d4:
                render_kpi("Severity Type", str(detail.get("severity_type", "N/A")), "Historical severity input")

            # Explanation cards for the selected record.
            render_card("Reason", str(detail.get("reason", "")))

            if detail.get("isolation_summary"):
                render_card("Isolation Summary", str(detail.get("isolation_summary")))

            # Stored feature explanation table.
            exp_df = dataframe_from_top_features(detail.get("explanations", []))
            if not exp_df.empty:
                st.markdown("#### Stored Top Features")
                st.dataframe(show_df, use_container_width=True, hide_index=True)
        else:
            st.warning(f"Could not load detail for ID {selected_id}.")


# _REPORTS PAGE_

# Provides summary analytics and downloadable reporting output
# based on stored assessments in the backend database.
elif page == "Reports":
    # Page title and short explanation for the reports page.
    render_page_header(
        "Reports",
        "View high-level summaries of risk levels, confidence, event activity, and fault categories.",
    )

    # Load a larger prediction history window for reporting.
    ok, history = api_get("/predictions", params={"limit": 200})
    rows = history if ok and isinstance(history, list) else []

    if not rows:
        st.info("No stored assessments available for reporting.")
    else:
        # Convert records to DataFrame for grouped summaries.
        df = pd.DataFrame(rows)
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["confidence_pct"] = df["confidence"].apply(lambda x: round(safe_float(x) * 100, 1))

        # Group by risk level for a high-level summary.
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
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        # Group by fault category to see most common isolation patterns.
        st.markdown("### Fault Category Summary")
        if "fault_category" in df.columns:
            cat_summary = (
                df.groupby("fault_category", dropna=False)
                .agg(count=("id", "count"))
                .sort_values("count", ascending=False)
                .reset_index()
            )
            st.dataframe(show_df, use_container_width=True, hide_index=True)

            if not cat_summary.empty:
                st.bar_chart(cat_summary.set_index("fault_category"), height=320)

        # Export the full report dataset as CSV.
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download History CSV",
            data=csv_data,
            file_name="netguard_history_report.csv",
            mime="text/csv",
            width="content",
        )