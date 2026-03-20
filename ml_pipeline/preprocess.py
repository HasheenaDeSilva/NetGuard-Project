from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _extract_numeric_value(value: Any) -> int:
    # Convert strings like:
    # "location 344" -> 344
    # "severity_type 4" -> 4
    # "203" -> 203
    # 203 -> 203

    # If missing value, return 0
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0

    # If already integer, return directly
    if isinstance(value, (int, np.integer)):
        return int(value)

    # If float, convert to integer
    if isinstance(value, (float, np.floating)):
        return int(value)

    # Try to extract the number from the end of the string
    s = str(value).strip()
    match = pd.Series([s]).str.extract(r"(\d+)$", expand=False).iloc[0]

    # If no number found, return 0
    if pd.isna(match):
        return 0

    return int(match)


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Convert all columns except raw location string into numeric
    out = df.copy()

    for col in out.columns:
        if col == "location":
            # Keep original location text unchanged
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _add_location_num(out: pd.DataFrame) -> pd.DataFrame:
    # Ensure numeric location_num exists for model input
    if "location_num" not in out.columns:
        if "location" in out.columns:
            out["location_num"] = out["location"].apply(_extract_numeric_value)
        else:
            out["location_num"] = 0
    else:
        out["location_num"] = out["location_num"].apply(_extract_numeric_value)

    return out


def _expand_severity_type(out: pd.DataFrame) -> pd.DataFrame:
    # Model expects one-hot severity_type_* columns
    # Example:
    # severity_type = "severity_type 4"
    # becomes severity_type_4 = 1

    sev_cols = [c for c in out.columns if c.startswith("severity_type_")]

    # If already expanded, just make sure numeric
    if sev_cols:
        for col in sev_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
        return out

    # Otherwise extract the severity value from severity_type
    if "severity_type" in out.columns:
        sev_num_series = out["severity_type"].apply(_extract_numeric_value)
    else:
        sev_num_series = pd.Series([0] * len(out), index=out.index)

    # Create common severity columns 1 to 5
    for k in [1, 2, 3, 4, 5]:
        out[f"severity_type_{k}"] = (sev_num_series == k).astype(np.uint8)

    return out


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    # Validate input first
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame.")

    # Make a copy so original input is not modified
    out = df.copy()

    # Add numeric location column
    out = _add_location_num(out)

    # Expand severity_type into one-hot columns
    out = _expand_severity_type(out)

    # Convert remaining usable columns into numeric values
    out = _ensure_numeric(out)

    # Clean invalid values
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)

    return out