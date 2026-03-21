from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap

from ml_pipeline.feature_engineering import add_engineered_features
from ml_pipeline.isolate import isolate_fault
from ml_pipeline.preprocess import preprocess_input
from ml_pipeline.scale import scale_features
from ml_pipeline.select_features import select_features_for_model

# Project root = NetGuard/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# Try these model locations in order
CANDIDATE_MODEL_FILES = [
    ARTIFACTS_DIR / "final_model" / "lightgbm_final_model.joblib",
]

# Map numeric model output to label
RISK_LABELS = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
}

def _resolve_model_path() -> Path:
    # Find the first model file that actually exists
    for model_path in CANDIDATE_MODEL_FILES:
        if model_path.exists():
            return model_path

    # If none found, raise clear error
    tried = ", ".join(str(p) for p in CANDIDATE_MODEL_FILES)
    raise FileNotFoundError(f"Model file not found. Tried: {tried}")


def load_model():
    # Load trained model from disk
    return joblib.load(_resolve_model_path())


def _prepare_features(input_df: pd.DataFrame) -> pd.DataFrame:
    # Full feature pipeline before prediction:
    # 1. preprocess raw input
    # 2. add engineered features
    # 3. keep only selected model features
    # 4. scale if needed
    df = preprocess_input(input_df)
    df = add_engineered_features(df)
    df = select_features_for_model(df)
    df = scale_features(df)
    return df.copy()


def _extract_row_shap_values(
    shap_values: Any,
    row_index: int,
    class_index: int,
) -> np.ndarray:
    # SHAP output can come in different shapes depending on model/version

    if isinstance(shap_values, list):
        # Old multiclass SHAP format:
        # list[class] -> array(rows, features)
        return np.asarray(shap_values[class_index])[row_index]

    shap_arr = np.asarray(shap_values)

    if shap_arr.ndim == 3:
        # New format:
        # (rows, features, classes)
        return shap_arr[row_index, :, class_index]

    if shap_arr.ndim == 2:
        # Binary or already selected format:
        # (rows, features)
        return shap_arr[row_index]

    raise ValueError(f"Unexpected SHAP output shape: {shap_arr.shape}")


def explain_prediction(
    input_df: pd.DataFrame,
    class_index: Optional[int] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    # Validate inputs
    if input_df is None or not isinstance(input_df, pd.DataFrame) or input_df.empty:
        raise ValueError("input_df must be a non-empty pandas DataFrame.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    # Load model and prepare features
    model = load_model()
    df_model = _prepare_features(input_df)

    # Predict class probabilities if available
    if hasattr(model, "predict_proba"):
        pred_proba = model.predict_proba(df_model)
        pred_classes = np.argmax(pred_proba, axis=1)
    else:
        pred_proba = None
        pred_classes = model.predict(df_model)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Get SHAP values
    shap_values = explainer.shap_values(df_model)

    explanations: List[Dict[str, Any]] = []

    # Build explanation for each row
    for i in range(len(df_model)):
        # Use provided class_index, otherwise use predicted class
        current_class = int(pred_classes[i]) if class_index is None else int(class_index)

        # Extract SHAP values for this row and class
        row_shap = _extract_row_shap_values(shap_values, i, current_class)

        # Build DataFrame for sorting top features
        shap_df = pd.DataFrame(
            {
                "feature": df_model.columns,
                "shap_value": row_shap,
            }
        )

        # Sort by absolute importance
        shap_df["abs_shap"] = shap_df["shap_value"].abs()
        shap_df = shap_df.sort_values("abs_shap", ascending=False)

        # Store explanation result for this row
        row_result: Dict[str, Any] = {
            "row_index": int(i),
            "predicted_class": current_class,
            "top_features": [
                {
                    "feature": str(row["feature"]),
                    "shap_value": float(row["shap_value"]),
                }
                for _, row in shap_df.head(top_k).iterrows()
            ],
        }

        # Add probability breakdown if model supports it
        if pred_proba is not None:
            row_result["class_probabilities"] = {
                str(j): float(pred_proba[i][j]) for j in range(pred_proba.shape[1])
            }

        explanations.append(row_result)

    return {"explanations": explanations}


def _build_reason(risk_level: str, top_features: List[Dict[str, Any]]) -> str:
    # Build a simple readable reason using top features
    if not top_features:
        return f"The model predicted {risk_level} risk."

    feature_names = [str(x.get("feature", "unknown")) for x in top_features[:5]]
    return f"The model predicted {risk_level} risk mainly due to: {', '.join(feature_names)}."


def predict_fault_severity(input_df: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
    # Main end-to-end prediction function
    # Returns only the first row result

    if input_df is None or not isinstance(input_df, pd.DataFrame) or input_df.empty:
        raise ValueError("input_df must be a non-empty pandas DataFrame.")

    # Keep raw copy
    raw_df = input_df.copy()

    # Preprocess raw input for rule-based isolation use
    prepared_df = preprocess_input(raw_df)

    # Prepare final model features
    model_df = _prepare_features(raw_df)

    # Load trained model
    model = load_model()

    # Predict class
    pred = model.predict(model_df)
    predicted_class = int(np.asarray(pred)[0])

    class_probabilities: Dict[str, float] = {}
    confidence = 0.0

    # Get class probabilities and confidence if supported
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(model_df)
        class_probabilities = {
            str(i): float(proba[0][i]) for i in range(proba.shape[1])
        }
        confidence = float(np.max(proba[0]))
    else:
        # If model has no probability output, assume full certainty
        confidence = 1.0

    # Convert class to label
    risk_level = RISK_LABELS.get(predicted_class, str(predicted_class))

    # Generate explanation for predicted class
    exp = explain_prediction(raw_df, class_index=predicted_class, top_k=top_k)
    exp_row = (exp.get("explanations") or [{}])[0]
    top_features = exp_row.get("top_features") or []

    # Use preprocessed first row for rule-based isolation summary
    raw_row = prepared_df.iloc[0].to_dict()
    fault_category, isolation_summary, recommended_checks = isolate_fault(
        raw_row,
        top_features,
        risk_level,
    )

    # Return all outputs together
    return {
        "predicted_severity": predicted_class,
        "risk_level": risk_level,
        "confidence": confidence,
        "reason": _build_reason(risk_level, top_features),
        "top_features": top_features,
        "fault_category": fault_category,
        "isolation_summary": isolation_summary,
        "recommended_checks": recommended_checks,
        "class_probabilities": class_probabilities,
    }