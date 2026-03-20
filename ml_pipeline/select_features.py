from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd


# Go from:
# ml_pipeline/select_features.py
# up to project root: NetGuard/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Main artifacts folder in the project
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Possible files that may contain the selected feature list
# We try these first because they are the most suitable for model inference
CANDIDATE_SELECTED_FEATURE_FILES = [
    ARTIFACTS_DIR / "selected_features_pivot.json",
    ARTIFACTS_DIR / "selected_features.json",
    ARTIFACTS_DIR / "final_model" / "selected_features_pivot.json",
    ARTIFACTS_DIR / "final_model" / "selected_features.json",
]

# Fallback files that may contain the full feature list
# These are used only if selected feature files are not found
CANDIDATE_FULL_FEATURE_FILES = [
    ARTIFACTS_DIR / "feature_names_pivot.json",
    ARTIFACTS_DIR / "final_model" / "feature_names_pivot.json",
]


def _load_json_list(path: Path) -> List[str]:
    # Open the JSON file and load its contents
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Make sure the file contains a non-empty list
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path.name} must contain a non-empty list of feature names.")

    # Make sure every item in the list is a string
    if not all(isinstance(x, str) for x in data):
        raise ValueError(f"All entries in {path.name} must be strings.")

    return data


def load_selected_features() -> List[str]:
    # First try the selected-feature files
    # These are preferred because they match the trained model input
    for path in CANDIDATE_SELECTED_FEATURE_FILES:
        if path.exists():
            return _load_json_list(path)

    # If not found, try fallback full-feature files
    for path in CANDIDATE_FULL_FEATURE_FILES:
        if path.exists():
            return _load_json_list(path)

    # If nothing exists, raise a clear error showing every path checked
    tried = [str(p) for p in CANDIDATE_SELECTED_FEATURE_FILES + CANDIDATE_FULL_FEATURE_FILES]
    raise FileNotFoundError(f"No feature list file found. Tried: {tried}")


def select_features_for_model(
    df: pd.DataFrame,
    selected_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    # Validate the input DataFrame
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame.")

    # Work on a copy so the original DataFrame will not be changed
    out = df.copy()

    # If the feature list is not given manually, load it from JSON file
    if selected_features is None:
        selected_features = load_selected_features()

    # Find features expected by the model but missing in the input DataFrame
    missing_cols = [col for col in selected_features if col not in out.columns]

    # Add any missing columns with default value 0
    # This keeps the model input shape correct
    if missing_cols:
        missing_df = pd.DataFrame(0, index=out.index, columns=missing_cols)
        out = pd.concat([out, missing_df], axis=1)

    # Reorder columns exactly as the model expects
    # Any extra columns not needed by the model are dropped here
    out = out.reindex(columns=selected_features, fill_value=0)

    # Convert everything to numeric for safe model input
    # Non-numeric values become NaN, then replaced with 0
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Return a clean copy aligned to model feature order
    return out.copy()