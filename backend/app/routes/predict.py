from __future__ import annotations

from typing import Dict

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Explanation, FaultPrediction
from ..schemas import FaultInput, PredictResponse
from ml_pipeline.predict import predict_fault_severity

# Create router group for prediction endpoints
router = APIRouter(tags=["Prediction"])


def _safe_sum(values: Dict[str, float]) -> float:
    # Safely sum numeric values from a dictionary
    # Non-numeric values are skipped
    total = 0.0

    for value in values.values():
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue

    return total


def _count_positive_features(values: Dict[str, float]) -> float:
    # Count how many feature values are greater than 0
    # Used for summary counts shown in dashboard/history
    count = 0.0

    for value in values.values():
        try:
            if float(value) > 0:
                count += 1.0
        except (TypeError, ValueError):
            continue

    return count


def _build_model_row(payload: FaultInput) -> dict:
    # Convert API request payload into a single row dictionary
    # This row is later converted into a pandas DataFrame for ML pipeline
    row = {
        "location": payload.location,
        "severity_type": payload.severity_type,
    }

    # Add sparse feature dictionaries into the same row
    row.update(payload.event_features)
    row.update(payload.resource_features)
    row.update(payload.log_features)

    return row


@router.post("/predict", response_model=PredictResponse)
def predict(payload: FaultInput, db: Session = Depends(get_db)):
    # Main prediction endpoint
    # 1. build one-row input DataFrame
    # 2. call ML pipeline
    # 3. save result to database
    # 4. return structured response to frontend
    try:
        # Convert request payload into one-row DataFrame
        row = _build_model_row(payload)
        input_df = pd.DataFrame([row])

        # Run the ML prediction pipeline
        pred_out = predict_fault_severity(input_df)

        # Normalize returned values
        predicted_severity = int(pred_out.get("predicted_severity", 0))
        confidence = max(0.0, min(1.0, float(pred_out.get("confidence", 0.0))))
        risk_level = str(pred_out.get("risk_level", "")).strip().upper()
        reason = str(pred_out.get("reason", "")).strip()

        # Optional fault isolation outputs
        fault_category = pred_out.get("fault_category")
        isolation_summary = pred_out.get("isolation_summary")

        # Keep top 10 explanation features at most
        top_features = list(pred_out.get("top_features") or [])[:10]

        # Create database row for prediction history
        prediction = FaultPrediction(
            location=payload.location,
            severity_type=payload.severity_type,
            event_count=_count_positive_features(payload.event_features),
            resource_count=_count_positive_features(payload.resource_features),
            log_count=_count_positive_features(payload.log_features),
            log_volume_sum=_safe_sum(payload.log_features),
            predicted_severity=predicted_severity,
            confidence=confidence,
            risk_level=risk_level,
            reason=reason,
            fault_category=str(fault_category) if fault_category is not None else None,
            isolation_summary=str(isolation_summary) if isolation_summary is not None else None,
        )

        # Save explanation rows linked to this prediction
        for feat in top_features:
            feature_name = str(feat.get("feature", "")).strip()

            # Skip empty feature names
            if not feature_name:
                continue

            prediction.explanations.append(
                Explanation(
                    feature=feature_name,
                    shap_value=float(feat.get("shap_value", 0.0)),
                )
            )

        # Save prediction and explanations to database
        db.add(prediction)
        db.commit()
        db.refresh(prediction)

        # Return structured response expected by frontend
        return {
            "prediction_id": prediction.id,
            "result": {
                "predicted_severity": predicted_severity,
                "risk_level": risk_level,
                "confidence": confidence,
                "reason": reason,
                "top_features": top_features,
                "fault_category": prediction.fault_category,
                "isolation_summary": prediction.isolation_summary,
                "recommended_checks": list(pred_out.get("recommended_checks") or []),
                "class_probabilities": dict(pred_out.get("class_probabilities") or {}),
            },
        }

    except ValueError as exc:
        # Input/validation related issues
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except FileNotFoundError as exc:
        # Missing model/scaler/feature-list artifact issues
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    except Exception as exc:
        # Any unexpected prediction error
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc