from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

# Move to the project root first, then go into artifacts folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Saved scaler path
SCALER_PATH = ARTIFACTS_DIR / "scaler_pivot.pkl"

# Keep False unless the final trained model requires scaling
APPLY_SCALING = False


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    # Check input
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame.")

    # Work on a copy
    out = df.copy()

    # If scaling is not required, return unchanged data
    if not APPLY_SCALING:
        return out

    # If scaling is enabled, scaler file must exist
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found at: {SCALER_PATH}")

    # Load trained scaler
    scaler = joblib.load(SCALER_PATH)

    # Align DataFrame columns with scaler expected columns
    if hasattr(scaler, "feature_names_in_"):
        needed = list(scaler.feature_names_in_)

        # Add any missing columns with 0
        for col in needed:
            if col not in out.columns:
                out[col] = 0

        # Reorder columns to match training order
        out = out[needed]

    # Transform using scaler
    scaled = scaler.transform(out)

    # Return scaled data as DataFrame
    return pd.DataFrame(scaled, columns=out.columns, index=out.index)