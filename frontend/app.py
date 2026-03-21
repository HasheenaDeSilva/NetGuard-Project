from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st

# Read backend connection details from environment variables.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))

# Sidebar branding shown across the app.
PROJECT_TITLE = "NetGuard"
PROJECT_DESCRIPTION = (
    "An Explainable Machine Learning-Based Framework for Fault Prediction and "
    "Isolation in Telecommunication Networks"
)

# These are used in banners and KPI cards so outputs are easier to understand.
RISK_HELP = {
    "LOW": "Low likelihood of severe service impact. Continue standard monitoring.",
    "MEDIUM": "Moderate risk. Validate signals, review recent changes, and monitor closely.",
    "HIGH": "High likelihood of major service impact. Immediate investigation is recommended.",
}

# Helper text for the severity input chosen by the analyst.
SEVERITY_HELP = {
    "severity_type 1": "Lower historical severity context.",
    "severity_type 2": "Moderate historical severity context.",
    "severity_type 3": "Elevated historical severity context.",
    "severity_type 4": "High historical severity context.",
    "severity_type 5": "Very high historical severity context.",
}

# These ranges only control the selectable IDs shown in the frontend.
# They do not define the model itself; the backend still performs the actual prediction.
EVENT_SIGNAL_IDS = list(range(1, 51))
RESOURCE_SIGNAL_IDS = list(range(1, 51))
LOG_SIGNAL_IDS = list(range(1, 101))

# Prevent long strings in dataframes from being visually cut off.
pd.set_option("display.max_colwidth", None)

# Must be called before rendering the page.
st.set_page_config(
    page_title="NetGuard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS.
# This controls page colors, cards, KPI boxes, sidebar appearance, and spacing.
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59,130,246,0.12), transparent 26%),
                radial-gradient(circle at top right, rgba(16,185,129,0.08), transparent 24%),
                linear-gradient(180deg, #050816 0%, #081122 58%, #0a1429 100%);
            color: #E5E7EB;
        }
        section[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top left, rgba(37,99,235,0.18), transparent 28%),
                linear-gradient(180deg, #07101f 0%, #0a1429 100%);
            border-right: 1px solid rgba(255,255,255,0.07);
        }
        .block-container {
            max-width: 1480px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        .ng-sidebar-title {
            font-size: 3rem;
            font-weight: 850;
            color: #F8FAFC;
            line-height: 1.02;
            margin-bottom: 0.35rem;
        }
        .ng-sidebar-desc {
            font-size: 0.98rem;
            line-height: 1.55;
            color: #C7D7F6;
            margin-bottom: 1rem;
        }
        .ng-page-title {
            font-size: 2.15rem;
            font-weight: 800;
            color: #F8FAFC;
            line-height: 1.08;
            margin-top: 2rem;
            margin-bottom: 0.2rem;
        }
        .ng-page-note {
            color: #A7B0C3;
            font-size: 0.98rem;
            margin-bottom: 1.15rem;
        }
        .ng-card {
            background: rgba(10, 18, 40, 0.92);
            border: 1px solid rgba(91, 116, 190, 0.22);
            border-radius: 18px;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.18);
            margin-bottom: 1rem;
        }
        .ng-kpi {
            background: linear-gradient(180deg, rgba(9,17,40,0.98), rgba(10,18,40,0.88));
            border: 1px solid rgba(91, 116, 190, 0.22);
            border-radius: 18px;
            padding: 16px;
            min-height: 128px;
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
            font-weight: 600;
            margin-bottom: 18px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .ng-low { background: rgba(34,197,94,0.16); color: #C7F9D4; }
        .ng-medium { background: rgba(234,179,8,0.18); color: #FFF1B3; }
        .ng-high { background: rgba(239,68,68,0.17); color: #FFD1D1; }
        .ng-section-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
            color: #F8FAFC;
        }
        .ng-muted {
            color: #A7B0C3;
            font-size: 0.93rem;
            line-height: 1.65;
        }
        .ng-chip {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            border: 1px solid rgba(96,165,250,0.25);
            background: rgba(59,130,246,0.12);
            color: #D6E7FF;
            font-size: 0.8rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid rgba(96,165,250,0.28);
            background: linear-gradient(180deg, rgba(15,23,42,1), rgba(12,20,40,1));
            color: white;
            font-weight: 600;
            min-height: 2.8rem;
        }
        .stSelectbox label,
        .stMultiSelect label,
        .stNumberInput label,
        .stTextInput label,
        .stSlider label {
            font-weight: 600 !important;
        }
        div[data-testid="stProgressBar"] > div > div {
            border-radius: 999px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Wrapper for backend GET requests.
# Returning (success, data_or_error) keeps error handling simple across the whole app.
def api_get(path: str, params: Dict[str, Any] | None = None) -> Tuple[bool, Any]:
    """Send a GET request to the backend."""
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=REQUEST_TIMEOUT)
        if response.ok:
            return True, response.json()
        return False, f"{response.status_code}: {response.text}"
    except Exception as exc:
        return False, str(exc)


# Wrapper for backend POST requests.
# Used when sending incident data to the prediction endpoint.
def api_post(path: str, payload: Dict[str, Any]) -> Tuple[bool, Any]:
    """Send a POST request to the backend."""
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT)
        if response.ok:
            return True, response.json()
        return False, f"{response.status_code}: {response.text}"
    except Exception as exc:
        return False, str(exc)


def risk_class_name(risk_level: str) -> str:
    """Map risk label to CSS class for banner colors."""
    rl = str(risk_level).upper().strip()
    if rl == "LOW":
        return "ng-low"
    if rl == "HIGH":
        return "ng-high"
    return "ng-medium"


# Safe conversion helper so invalid values never crash the UI.
# Very useful when backend data may contain None, strings, or missing values.
def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert values to float."""
    try:
        return float(value)
    except Exception:
        return default


def format_pct(value: Any) -> str:
    """Convert probability 0.889 -> 88.9%."""
    return f"{safe_float(value) * 100:.1f}%"


# Convert a selected feature list into sparse binary input format.
# Example: if the user selected event_type_1 and event_type_5,
# only those chosen features are sent with value 1.0.
def parse_sparse_binary(feature_names: List[str]) -> Dict[str, float]:
    """
    Convert a feature list into sparse binary format.
    Example: ['event_type_1', 'event_type_5'] becomes:
    {'event_type_1': 1.0, 'event_type_5': 1.0}
    """
    return {name: 1.0 for name in feature_names if str(name).strip()}


# Log features are different from binary inputs because they carry numeric intensity values.
# This function keeps each selected log feature linked to its entered strength.
def parse_log_inputs(log_feature_names: List[str], log_values: Dict[str, float]) -> Dict[str, float]:
    """
    Convert selected log features into sparse numeric format.
    Each selected log feature gets the entered numeric strength.
    """
    return {name: safe_float(log_values.get(name, 1.0), 1.0) for name in log_feature_names}


# Build the exact JSON payload expected by the backend.
# The frontend converts simple UI choices like event ID 3 into model-style feature names
# such as event_type_3 before sending them to /predict.
def build_payload(
    location_num: int,
    severity_num: int,
    event_ids: List[int],
    resource_ids: List[int],
    log_ids: List[int],
    log_values: Dict[str, float],
) -> Dict[str, Any]:
    """
    Build the backend request payload.

    The frontend converts selected numeric IDs into backend feature keys.
    The backend still decides how to preprocess them and which final
    selected features are used by the model.
    """
    event_features = [f"event_type_{i}" for i in event_ids]
    resource_features = [f"resource_type_{i}" for i in resource_ids]
    log_features = [f"log_feature_{i}" for i in log_ids]

    return {
        "location": f"location {location_num}",
        "severity_type": f"severity_type {severity_num}",
        "event_features": parse_sparse_binary(event_features),
        "resource_features": parse_sparse_binary(resource_features),
        "log_features": parse_log_inputs(log_features, log_values),
    }


# Streamlit tables can fail or look messy if values contain unsupported types.
# This function normalizes columns, replaces NaN, and converts unusual objects to strings.
def clean_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """Make dataframe values safer for Streamlit display."""
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


# Convert SHAP-style explanation records from the backend into a compact table.
# "Abs Influence" is used only for sorting so the strongest contributions appear first.
def dataframe_from_top_features(top_features: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert backend explanation data into a compact dataframe.
    Only Feature and Influence are shown to keep the UI simple.
    """
    rows = []
    for item in top_features or []:
        rows.append(
            {
                "Feature": str(item.get("feature", "")),
                "Influence": round(safe_float(item.get("shap_value")), 5),
                "Abs Influence": abs(round(safe_float(item.get("shap_value")), 5)),
            }
        )

    df = pd.DataFrame(rows, columns=["Feature", "Influence", "Abs Influence"])

    if not df.empty:
        df = (
            df.sort_values("Abs Influence", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )

    return clean_dataframe_for_streamlit(df)


# Convert raw probability output into a readable table.
# The backend uses numeric classes, so this maps them to LOW / MEDIUM / HIGH labels.
def dataframe_from_probabilities(probabilities: Dict[str, float]) -> pd.DataFrame:
    """Convert class probabilities from backend into a dataframe."""
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


def render_page_header(page_title: str, page_note: str) -> None:
    """Show page title and short page note."""
    st.markdown(f'<div class="ng-page-title">{page_title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ng-page-note">{page_note}</div>', unsafe_allow_html=True)


def render_kpi(title: str, value: str, subtitle: str = "") -> None:
    """Show a KPI card."""
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


def render_card(title: str, body: str) -> None:
    """Show a general content card."""
    st.markdown(
        f"""
        <div class="ng-card">
            <div class="ng-section-title">{title}</div>
            <div class="ng-muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Shows a color-coded banner after prediction using the CSS class derived from the risk label.
def render_banner(risk_level: str) -> None:
    """Show colored risk banner after prediction."""
    rl = str(risk_level).upper().strip()
    css = risk_class_name(rl)
    message = RISK_HELP.get(rl, "Model output ready for analyst review.")
    st.markdown(f'<div class="ng-banner {css}">{rl} RISK — {message}</div>', unsafe_allow_html=True)


def render_feature_table(top_df: pd.DataFrame) -> None:
    """Show top contributing features in a compact table."""
    if top_df.empty:
        st.info("No top features returned by the backend.")
        return

    display_df = top_df.copy()
    display_df["Influence"] = display_df["Influence"].apply(lambda x: round(float(x), 5))

    st.dataframe(
        display_df[["Feature", "Influence"]],
        use_container_width=True,
        hide_index=True,
    )


# Render probabilities visually using progress bars so users can quickly compare class confidence.
def render_probability_progress(prob_df: pd.DataFrame) -> None:
    """Show class probabilities using progress bars."""
    if prob_df.empty:
        st.info("No class probabilities available.")
        return

    for _, row in prob_df.iterrows():
        label = str(row["Class"])
        prob = min(max(safe_float(row["Probability"]), 0.0), 1.0)

        st.markdown(
            f"""
            <div class="ng-card" style="padding: 14px 16px 10px 16px; margin-bottom: 0.8rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                    <div style="font-weight:700; color:#F8FAFC;">{label}</div>
                    <div style="color:#C7D2FE; font-weight:600;">{prob * 100:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(prob)


# Small health check so the sidebar can immediately show whether the backend is reachable.
def show_api_status() -> None:
    """Check backend health and show Online / Offline message."""
    ok, _ = api_get("/health")
    if ok:
        st.success("API Online")
    else:
        st.error("API Offline")


# Helps non-technical users understand what each input and output means.
# This is a UX explanation block, not part of the ML pipeline itself.
def explain_inputs_block() -> None:
    """
    Simple explanation block for non-domain experts.
    This avoids backend and pipeline technical wording.
    """
    with st.expander("What do these inputs and outputs mean?", expanded=False):
        st.markdown(
            """
**Location ID**  
The telecom site or node related to the incident.

**Severity Type**  
The historical severity context used by the model.

**Event Signal IDs**  
IDs of active event indicators.

**Resource Signal IDs**  
IDs of active resource indicators.

**Log Signals**  
IDs and values for active log indicators.

**Predicted Risk Level**  
The model's risk label: LOW, MEDIUM, or HIGH.

**Confidence**  
The model probability assigned to the predicted class.

**Fault Category / Isolation Summary**  
Operational guidance returned by the backend to help troubleshooting.
            """
        )

# Sidebar is the main navigation control for the whole app.
with st.sidebar:
    st.markdown(f'<div class="ng-sidebar-title">{PROJECT_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ng-sidebar-desc">{PROJECT_DESCRIPTION}</div>', unsafe_allow_html=True)

    show_api_status()

    st.markdown("---")
    page = st.radio("Navigation", options=["Dashboard", "Analyze Incident", "Incident History", "Reports"])


# Session state keeps results alive across reruns.
# Without this, Streamlit would clear the current prediction every time a widget changes.
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "prediction_id" not in st.session_state:
    st.session_state["prediction_id"] = None
if "last_payload" not in st.session_state:
    st.session_state["last_payload"] = None


if page == "Dashboard":
    render_page_header(
        "Dashboard",
        "Operational overview of recent telecom fault assessments and model outputs.",
    )

    ok, history = api_get("/predictions", params={"limit": 100})
    rows = history if ok and isinstance(history, list) else []

    total_cases = len(rows)
    high_count = sum(1 for row in rows if str(row.get("risk_level", "")).upper() == "HIGH")
    medium_count = sum(1 for row in rows if str(row.get("risk_level", "")).upper() == "MEDIUM")
    avg_conf = sum(safe_float(row.get("confidence")) for row in rows) / total_cases if total_cases else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("Assessments Logged", str(total_cases), "Stored prediction records")
    with c2:
        render_kpi("High Risk Cases", str(high_count), "Require immediate review")
    with c3:
        render_kpi("Medium Risk Cases", str(medium_count), "Need validation and monitoring")
    with c4:
        render_kpi("Average Confidence", format_pct(avg_conf), "Across recent outputs")

    st.markdown("### Recent Activity")
    if rows:
        hist_df = pd.DataFrame(rows)
        hist_df["created_at"] = pd.to_datetime(hist_df.get("created_at"), errors="coerce")
        hist_df["confidence_pct"] = hist_df["confidence"].apply(format_pct)

        display_cols = [
            "id", "created_at", "location", "severity_type", "risk_level", "confidence_pct", "fault_category"
        ]
        show_df = clean_dataframe_for_streamlit(hist_df[display_cols].copy())
        show_df.columns = [
            "ID", "Created At", "Location", "Severity Type", "Risk Level", "Confidence", "Fault Category"
        ]
        st.dataframe(show_df, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions found yet. Run an incident assessment to populate the dashboard.")


elif page == "Analyze Incident":
    render_page_header(
        "Analyze Incident",
        "Enter incident details, run the deployed model, and review the prediction results.",
    )

    explain_inputs_block()

    left, right = st.columns([1.18, 1.0], gap="large")

    with left:
        st.subheader("Incident Inputs")

        location_num = st.number_input(
            "Location ID",
            min_value=1,
            max_value=2000,
            value=1,
            step=1,
            help="Represents the telecom site or network location linked to the incident.",
        )

        severity_num = st.selectbox(
            "Severity Type",
            options=[1, 2, 3, 4, 5],
            index=0,
            help="Historical severity context associated with the selected incident.",
        )
        st.caption(SEVERITY_HELP.get(f"severity_type {severity_num}", ""))

        event_ids = st.multiselect(
            "Event Signal IDs",
            options=EVENT_SIGNAL_IDS,
            default=[],
            help="Select active event indicators related to this incident.",
        )

        resource_ids = st.multiselect(
            "Resource Signal IDs",
            options=RESOURCE_SIGNAL_IDS,
            default=[],
            help="Select active resource indicators related to this incident.",
        )

        log_ids = st.multiselect(
            "Log Signal IDs",
            options=LOG_SIGNAL_IDS,
            default=[],
            help="Select active log indicators related to this incident.",
        )

        # Each selected log feature can have its own numeric strength.
        # This is important because log signals are not just present/absent like binary features.
        st.markdown("##### Log Signal Strength")
        st.caption("Set a value for each selected log signal.")

        log_values: Dict[str, float] = {}
        if log_ids:
            log_feature_names = [f"log_feature_{i}" for i in log_ids]
            cols = st.columns(2)

            for idx, log_key in enumerate(log_feature_names):
                with cols[idx % 2]:
                    log_values[log_key] = st.number_input(
                        f"{log_key}",
                        min_value=0.0,
                        max_value=10000.0,
                        value=1.0,
                        step=1.0,
                        key=f"log_strength_{log_key}",
                    )

        # Build the exact backend request object from the current UI selections.
        payload = build_payload(
            location_num=location_num,
            severity_num=int(severity_num),
            event_ids=event_ids,
            resource_ids=resource_ids,
            log_ids=log_ids,
            log_values=log_values,
        )

        st.session_state["last_payload"] = payload
        run = st.button("Run NetGuard Assessment", type="primary")

    with right:
        st.subheader("Input Summary")

        r1, r2 = st.columns(2)
        with r1:
            render_kpi("Selected Location", f"location {location_num}", "Telecom site identifier")
        with r2:
            render_kpi("Severity Context", f"severity_type {severity_num}", "Historical severity input")

        r3, r4 = st.columns(2)
        with r3:
            render_kpi("Event Signals", str(len(event_ids)), "Selected event indicators")
        with r4:
            render_kpi("Resource Signals", str(len(resource_ids)), "Selected resource indicators")

        r5, r6 = st.columns(2)
        with r5:
            render_kpi("Log Signals", str(len(log_ids)), "Selected log indicators")
        with r6:
            render_kpi("Log Volume Sum", f"{sum(log_values.values()):.1f}", "Sum of selected log strengths")

        st.markdown(
            """
            <div class="ng-card">
                <div class="ng-section-title">Assessment Flow</div>
                <div class="ng-muted">
                    1. Enter incident details<br>
                    2. Run the assessment<br>
                    3. Review the predicted risk level<br>
                    4. Check the explanation and recommended actions
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Prediction is only triggered when the user clicks the button.
    # The backend handles model inference, explanation generation, and stored output.
    if run:
        with st.spinner("Running NetGuard assessment..."):
            ok, response = api_post("/predict", payload)

        if ok and isinstance(response, dict):
            st.session_state["prediction_result"] = response.get("result")
            st.session_state["prediction_id"] = response.get("prediction_id")
            st.success("Assessment completed successfully.")
        else:
            st.error(f"Prediction failed: {response}")

    result = st.session_state.get("prediction_result")
    prediction_id = st.session_state.get("prediction_id")

    # If a result exists in session state, keep showing it even after Streamlit reruns.
    if result:
        st.markdown("---")
        render_banner(str(result.get("risk_level", "MEDIUM")))

        st.subheader("Prediction")

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            render_kpi(
                "Risk Level",
                str(result.get("risk_level", "N/A")),
                RISK_HELP.get(str(result.get("risk_level", "")).upper(), "")
            )
        with p2:
            render_kpi(
                "Severity Class",
                str(result.get("predicted_severity", "N/A")),
                "0 = Low, 1 = Medium, 2 = High"
            )
        with p3:
            render_kpi(
                "Confidence",
                format_pct(result.get("confidence", 0.0)),
                "Model certainty for the selected class"
            )
        with p4:
            render_kpi(
                "Fault Category",
                str(result.get("fault_category") or "N/A"),
                "Isolation-oriented interpretation"
            )

        st.subheader("Explanation")
        render_card("Prediction Summary", str(result.get("reason", "No explanation available.")))

        exp_left, exp_right = st.columns([1.05, 1.0], gap="large")
        with exp_left:
            st.markdown("#### Top Contributing Features")
            top_df = dataframe_from_top_features(result.get("top_features", []))
            render_feature_table(top_df)

        with exp_right:
            st.markdown("#### Class Probability Breakdown")
            prob_df = dataframe_from_probabilities(result.get("class_probabilities", {}))
            render_probability_progress(prob_df)

        iso_left, iso_right = st.columns(2, gap="large")
        with iso_left:
            render_card(
                str(result.get("fault_category") or "Isolation Summary"),
                str(result.get("isolation_summary") or "No isolation summary available."),
            )

        with iso_right:
            checks = result.get("recommended_checks") or []
            if checks:
                render_card("Recommended Checks", "<br>".join([f"• {str(check)}" for check in checks]))
            else:
                st.info("No recommended checks returned.")

        # Export the full assessment for later review, reporting, or debugging.
        st.download_button(
            "Download Assessment JSON",
            data=json.dumps(
                {
                    "prediction_id": prediction_id,
                    "payload": st.session_state.get("last_payload"),
                    "result": result,
                    "generated_at": datetime.now().isoformat(),
                },
                indent=2,
            ),
            file_name=f"netguard_assessment_{prediction_id or 'latest'}.json",
            mime="application/json",
        )


elif page == "Incident History":
    render_page_header(
        "Incident History",
        "Browse past assessments and inspect stored prediction details.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        hist_limit = st.slider("Records", 10, 200, 50, 10)
    with c2:
        hist_risk = st.selectbox("Risk Filter", ["ALL", "LOW", "MEDIUM", "HIGH"])
    with c3:
        min_conf_pct = st.slider("Minimum Confidence (%)", 0, 100, 0, 5)

    # Build optional filters dynamically before calling the backend.
    params: Dict[str, Any] = {"limit": hist_limit}
    if hist_risk != "ALL":
        params["risk_level"] = hist_risk
    if min_conf_pct > 0:
        params["min_confidence"] = min_conf_pct / 100.0

    ok, history = api_get("/predictions", params=params)

    if not ok:
        st.error(f"Could not load history: {history}")
    elif not history:
        st.info("No history records match the current filters.")
    else:
        hist_df = pd.DataFrame(history)
        hist_df["created_at"] = pd.to_datetime(hist_df.get("created_at"), errors="coerce")
        hist_df["confidence_pct"] = hist_df["confidence"].apply(format_pct)

        display_df = hist_df[
            [
                "id", "created_at", "location", "severity_type", "risk_level", "confidence_pct",
                "fault_category", "event_count", "resource_count", "log_count", "log_volume_sum"
            ]
        ].copy()

        display_df.columns = [
            "ID", "Created At", "Location", "Severity Type", "Risk Level", "Confidence",
            "Fault Category", "Event Count", "Resource Count", "Log Count", "Log Volume Sum"
        ]

        st.dataframe(clean_dataframe_for_streamlit(display_df), use_container_width=True, hide_index=True)

        selected_id = st.selectbox("Open assessment detail", options=display_df["ID"].tolist())
        ok_detail, detail = api_get(f"/predictions/{selected_id}", params={"include_explanations": True})

        if ok_detail and isinstance(detail, dict):
            st.markdown("### Assessment Detail")

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                render_kpi("Risk Level", str(detail.get("risk_level", "N/A")), "Stored prediction output")
            with d2:
                render_kpi("Confidence", format_pct(detail.get("confidence", 0.0)), "Stored model confidence")
            with d3:
                render_kpi("Location", str(detail.get("location", "N/A")), "Incident location input")
            with d4:
                render_kpi("Severity Type", str(detail.get("severity_type", "N/A")), "Historical severity input")

            render_card("Reason", str(detail.get("reason", "")))

            if detail.get("isolation_summary"):
                render_card("Isolation Summary", str(detail.get("isolation_summary")))

            exp_df = dataframe_from_top_features(detail.get("explanations", []))
            if not exp_df.empty:
                st.markdown("#### Stored Top Features")
                render_feature_table(exp_df)
        else:
            st.warning(f"Could not load detail for ID {selected_id}.")


elif page == "Reports":
    render_page_header(
        "Reports",
        "View high-level summaries of risk levels, confidence, and fault categories.",
    )

    ok, history = api_get("/predictions", params={"limit": 200})
    rows = history if ok and isinstance(history, list) else []

    if not rows:
        st.info("No stored assessments available for reporting.")
    else:
        df = pd.DataFrame(rows)
        df["created_at"] = pd.to_datetime(df.get("created_at"), errors="coerce")
        df["confidence_pct"] = df["confidence"].apply(lambda x: round(safe_float(x) * 100, 1))

        st.markdown("### Risk Summary")

        # Group stored predictions by risk level to create report aggregates.
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
        st.dataframe(clean_dataframe_for_streamlit(risk_summary), use_container_width=True, hide_index=True)

        if "fault_category" in df.columns:
            st.markdown("### Fault Category Summary")

            # Summarize which fault categories appear most often in stored assessments.
            cat_summary = (
                df.groupby("fault_category", dropna=False)
                .agg(count=("id", "count"))
                .sort_values("count", ascending=False)
                .reset_index()
            )
            st.dataframe(clean_dataframe_for_streamlit(cat_summary), use_container_width=True, hide_index=True)

        st.markdown("### Report Highlights")

        total_records = len(df)
        avg_conf_report = df["confidence_pct"].mean() if total_records else 0.0
        most_common_fault = (
            str(df["fault_category"].mode().iloc[0])
            if "fault_category" in df.columns and not df["fault_category"].dropna().empty
            else "N/A"
        )

        h1, h2, h3 = st.columns(3)
        with h1:
            render_kpi("Total Records", str(total_records), "Assessments in report window")
        with h2:
            render_kpi("Average Confidence", f"{avg_conf_report:.1f}%", "Across stored assessments")
        with h3:
            render_kpi("Most Common Fault Category", most_common_fault, "Frequent isolation pattern")

        # Export all stored report data to CSV for offline analysis or inclusion in reports.
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download History CSV",
            data=csv_data,
            file_name="netguard_history_report.csv",
            mime="text/csv",
        )