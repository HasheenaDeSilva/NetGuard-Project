from __future__ import annotations

import numpy as np
import pandas as pd


def _non_volume_log_cols(columns: list[str]) -> list[str]:
    # Get normal log feature columns like:
    # log_feature_40, log_feature_82
    # but exclude volume columns like log_feature_40_vol
    return [
        c for c in columns
        if c.startswith("log_feature_") and not c.endswith("_vol")
    ]


def _volume_log_cols(columns: list[str]) -> list[str]:
    # Get log volume columns like:
    # log_feature_40_vol, log_feature_82_vol
    return [
        c for c in columns
        if c.startswith("log_feature_") and c.endswith("_vol")
    ]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # Check whether the input is valid
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame.")

    # Work on a copy so the original DataFrame will not be changed
    out = df.copy()

    # Detect column groups from their prefixes
    event_cols = [c for c in out.columns if c.startswith("event_type_")]
    resource_cols = [c for c in out.columns if c.startswith("resource_type_")]
    severity_cols = [c for c in out.columns if c.startswith("severity_type_")]
    log_presence_cols = _non_volume_log_cols(list(out.columns))
    log_volume_cols = _volume_log_cols(list(out.columns))

    # Convert all detected grouped columns into numeric values
    # Non-numeric values become NaN, then replaced with 0
    for col in event_cols + resource_cols + severity_cols + log_presence_cols + log_volume_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    
    # _Event-based engineered features_

    # Total sum of event values across all event_type_* columns
    out["event_count"] = out[event_cols].sum(axis=1) if event_cols else 0.0

    # Number of different event columns that are active (> 0)
    out["unique_event_types"] = (
        (out[event_cols] > 0).sum(axis=1) if event_cols else 0.0
    )

    # _Resource-based engineered features_

    # Total sum of resource values
    out["resource_count"] = out[resource_cols].sum(axis=1) if resource_cols else 0.0

    # Number of unique resource types active
    out["unique_resource_types"] = (
        (out[resource_cols] > 0).sum(axis=1) if resource_cols else 0.0
    )
    
    # _Severity-based engineered features_

    # Sum of severity_type_* values
    out["severity_type_count"] = (
        out[severity_cols].sum(axis=1) if severity_cols else 0.0
    )

    # Number of active severity type columns
    out["unique_severity_types"] = (
        (out[severity_cols] > 0).sum(axis=1) if severity_cols else 0.0
    )

    # _Log-based engineered features_
  

    if log_volume_cols:
        # If special volume columns exist, use them directly
        log_vol_df = out[log_volume_cols]

        # Count how many log volume features are active
        out["log_count"] = (log_vol_df > 0).sum(axis=1).astype(float)

        # Count unique active log features
        out["unique_log_features"] = (log_vol_df > 0).sum(axis=1).astype(float)

        # Total log volume
        out["log_volume_sum"] = log_vol_df.sum(axis=1)

        # Average non-zero log volume
        out["log_volume_mean"] = log_vol_df.replace(0, np.nan).mean(axis=1).fillna(0.0)

        # Maximum log volume seen in the row
        out["log_volume_max"] = log_vol_df.max(axis=1)

        # Standard deviation of non-zero log volume values
        out["log_volume_std"] = log_vol_df.replace(0, np.nan).std(axis=1).fillna(0.0)

    else:
        # If *_vol columns do not exist,
        # use normal log_feature_* columns as fallback
        log_df = out[log_presence_cols] if log_presence_cols else pd.DataFrame(index=out.index)

        if not log_df.empty:
            # Count active log features
            out["log_count"] = (log_df > 0).sum(axis=1).astype(float)

            # Count unique active log features
            out["unique_log_features"] = (log_df > 0).sum(axis=1).astype(float)

            # Sum of all log feature values
            out["log_volume_sum"] = log_df.sum(axis=1)

            # Average non-zero log values
            out["log_volume_mean"] = log_df.replace(0, np.nan).mean(axis=1).fillna(0.0)

            # Highest log feature value
            out["log_volume_max"] = log_df.max(axis=1)

            # Spread/variation in log values
            out["log_volume_std"] = log_df.replace(0, np.nan).std(axis=1).fillna(0.0)
        else:
            # If no log columns exist at all, create the features with zeros
            out["log_count"] = 0.0
            out["unique_log_features"] = 0.0
            out["log_volume_sum"] = 0.0
            out["log_volume_mean"] = 0.0
            out["log_volume_max"] = 0.0
            out["log_volume_std"] = 0.0

    # -----------------------------
    # Ratio features
    # These help the model understand relationships
    # between events, resources, and logs
    # -----------------------------

    # Events per resource
    out["events_per_resource"] = np.where(
        out["resource_count"] > 0,
        out["event_count"] / out["resource_count"],
        0.0,
    )

    # Logs per event
    out["logs_per_event"] = np.where(
        out["event_count"] > 0,
        out["log_count"] / out["event_count"],
        0.0,
    )

    # Average log volume per active log feature
    out["logvol_per_log"] = np.where(
        out["log_count"] > 0,
        out["log_volume_sum"] / out["log_count"],
        0.0,
    )

    # Average log volume per resource
    out["logvol_per_resource"] = np.where(
        out["resource_count"] > 0,
        out["log_volume_sum"] / out["resource_count"],
        0.0,
    )

    # -----------------------------
    # Log transform for skewed features
    # This reduces the effect of very large values
    # -----------------------------
    
    out["log1p_log_volume_sum"] = np.log1p(np.clip(out["log_volume_sum"], a_min=0, a_max=None))
    out["log1p_log_volume_mean"] = np.log1p(np.clip(out["log_volume_mean"], a_min=0, a_max=None))
    out["log1p_log_volume_max"] = np.log1p(np.clip(out["log_volume_max"], a_min=0, a_max=None))

    # Replace infinity values and missing values with 0
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)

    # IMPORTANT: return the engineered DataFrame
    return out