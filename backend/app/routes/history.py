from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload

from ..database import get_db
from ..models import FaultPrediction
from ..schemas import PredictionDetail, PredictionHistoryItem, TopFeature

# Create router group for prediction history endpoints
router = APIRouter(tags=["History"])


@router.get("/predictions", response_model=list[PredictionHistoryItem])
def get_predictions(
    limit: int = Query(50, ge=1, le=500),
    location: str | None = None,
    risk_level: str | None = None,
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    db: Session = Depends(get_db),
):
    # Build base query
    q = db.query(FaultPrediction)

    # Optional filter by exact location
    if location:
        q = q.filter(FaultPrediction.location == location.strip())

    # Optional filter by risk level
    if risk_level:
        q = q.filter(FaultPrediction.risk_level == risk_level.strip().upper())

    # Optional filter by minimum confidence
    if min_confidence is not None:
        q = q.filter(FaultPrediction.confidence >= float(min_confidence))

    # Optional filter for created_at start datetime
    if start_time is not None:
        q = q.filter(FaultPrediction.created_at >= start_time)

    # Optional filter for created_at end datetime
    if end_time is not None:
        q = q.filter(FaultPrediction.created_at <= end_time)

    # Order latest first and limit returned rows
    results = (
        q.order_by(FaultPrediction.created_at.desc())
        .limit(int(limit))
        .all()
    )

    return results


@router.get("/predictions/{prediction_id}", response_model=PredictionDetail)
def get_prediction_detail(
    prediction_id: int,
    include_explanations: bool = True,
    db: Session = Depends(get_db),
):
    # Start base query
    q = db.query(FaultPrediction)

    # Optionally eager-load explanation rows in the same query
    if include_explanations:
        q = q.options(joinedload(FaultPrediction.explanations))

    # Find the requested prediction row
    prediction = q.filter(FaultPrediction.id == prediction_id).first()

    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Build explanation output list
    explanations = []
    if include_explanations:
        explanations = [
            TopFeature(
                feature=item.feature,
                shap_value=float(item.shap_value),
            )
            for item in (prediction.explanations or [])
        ]

    # Return a structured detailed response
    return PredictionDetail(
        id=prediction.id,
        created_at=prediction.created_at,
        location=prediction.location,
        severity_type=prediction.severity_type,
        event_count=float(prediction.event_count),
        resource_count=float(prediction.resource_count),
        log_count=float(prediction.log_count),
        log_volume_sum=float(prediction.log_volume_sum),
        predicted_severity=int(prediction.predicted_severity),
        confidence=float(prediction.confidence),
        risk_level=prediction.risk_level,
        reason=prediction.reason,
        fault_category=prediction.fault_category,
        isolation_summary=prediction.isolation_summary,
        explanations=explanations,
    )