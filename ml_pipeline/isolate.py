from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _safe_float(x: Any) -> float:
    # Convert any value safely to float
    # If conversion fails, return 0.0
    try:
        return float(x)
    except Exception:
        return 0.0


def _top_feature_names(top_features: List[Dict[str, Any]]) -> List[str]:
    # Extract only feature names from SHAP top feature output
    return [
        str(item.get("feature"))
        for item in (top_features or [])
        if item.get("feature") is not None
    ]


def isolate_fault(
    x: Dict[str, Any],
    top_features: List[Dict[str, Any]],
    risk_level: str,
) -> Tuple[str, str, List[str]]:
    # Get top feature names only
    top = _top_feature_names(top_features)

    # Check what kind of features dominate the prediction
    has_log = any(
        f.startswith("log_feature_") or "log_volume" in f or "logvol_" in f
        for f in top
    )
    has_event = any(
        f.startswith("event_type_") or "event_count" in f
        for f in top
    )
    has_resource = any(
        f.startswith("resource_type_") or "resource_count" in f
        for f in top
    )
    has_severity = any(f.startswith("severity_type_") for f in top)
    has_location = any(f == "location_num" for f in top)

    # Extract location number if available
    location_num = int(_safe_float(x.get("location_num", 0)))

    # Decide fault category using simple rule-based logic
    if has_log and not has_event:
        category = "Software/Log Anomaly"
        summary = (
            f"The {risk_level} risk prediction is mainly influenced by log-related signals, "
            "suggesting a software, configuration, or repeated exception pattern rather than a pure event burst."
        )
        checks = [
            "Check recent abnormal log spikes and repeated signatures.",
            "Review recent configuration or deployment changes.",
            "Inspect affected software services and restart history.",
        ]

    elif has_event and has_log:
        category = "Alarm Burst / Incident Cascade"
        summary = (
            f"The {risk_level} risk prediction is influenced by both event and log signals, "
            "suggesting an alarm burst or cascading incident."
        )
        checks = [
            "Identify the first alarm in the sequence.",
            "Check upstream dependency failures and correlated sites.",
            "Review whether one event triggered multiple secondary alarms.",
        ]

    elif has_severity and not has_log and not has_event:
        category = "Historical Severity Pattern"
        summary = (
            f"The {risk_level} risk prediction is strongly influenced by severity-pattern signals, "
            "which suggests the case resembles historically similar severity behavior."
        )
        checks = [
            "Compare with similar historical fault cases.",
            "Review prior incidents with the same severity context.",
            "Validate whether this case follows an already known recovery pattern.",
        ]

    elif has_location or has_resource or has_event or has_log:
        category = "Mixed Operational Signals"
        summary = (
            f"The {risk_level} risk prediction is driven by a combination of location, resource, event, "
            "and/or log signals. This suggests a mixed operational pattern rather than a single obvious root cause."
        )
        checks = [
            "Review top contributing features together.",
            "Inspect location-specific incident history and nearby sites.",
            "Correlate events, resources, and logs before final diagnosis.",
        ]

    else:
        category = "Mixed Operational Signals"
        summary = (
            f"The {risk_level} risk prediction does not show one dominant signal group. "
            "Manual investigation is recommended."
        )
        checks = [
            "Review the input row and top features manually.",
            "Compare against known incident patterns.",
            "Escalate if similar predictions repeat frequently.",
        ]

    # Add location information to the summary if available
    if location_num > 0:
        summary += f" Related site indicator: location {location_num}."

    return category, summary, checks