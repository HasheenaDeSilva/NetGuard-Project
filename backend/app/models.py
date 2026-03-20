from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


class FaultPrediction(Base):
    # Main table storing one prediction record per assessment
    __tablename__ = "fault_predictions"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Timestamp when the prediction row was created
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # _Original input summary_

    # Location entered by user, for example: "location 344"
    location = Column(String, default="", nullable=False, index=True)

    # Severity context entered by user, for example: "severity_type 4"
    severity_type = Column(String, default="", nullable=False)

    # _Dashboard/history summary values_

    # Number of active event features in the request
    event_count = Column(Float, default=0.0, nullable=False)

    # Number of active resource features in the request
    resource_count = Column(Float, default=0.0, nullable=False)

    # Number of active log features in the request
    log_count = Column(Float, default=0.0, nullable=False)

    # Sum of log values/strengths in the request
    log_volume_sum = Column(Float, default=0.0, nullable=False)

    # _Prediction outputs_

    # Numeric model prediction, for example: 0, 1, 2
    predicted_severity = Column(Integer, nullable=False)

    # Model confidence score between 0 and 1
    confidence = Column(Float, nullable=False, default=0.0)

    # Risk label, for example: LOW, MEDIUM, HIGH
    risk_level = Column(String, default="", nullable=False, index=True)

    # Human-readable explanation summary
    reason = Column(Text, default="", nullable=False)

    # _Fault isolation outputs_

    # Simple rule-based category for investigation
    fault_category = Column(String, nullable=True)

    # Text summary for likely isolation direction
    isolation_summary = Column(Text, nullable=True)

    # One-to-many relationship:
    # one prediction can have many explanation rows (top SHAP features)
    explanations = relationship(
        "Explanation",
        back_populates="prediction",
        cascade="all, delete-orphan",
        order_by="Explanation.id.asc()",
    )


class Explanation(Base):
    # Table storing feature-level explanation rows for each prediction
    __tablename__ = "explanations"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key linking to the fault_predictions table
    prediction_id = Column(
        Integer,
        ForeignKey("fault_predictions.id"),
        nullable=False,
        index=True,
    )

    # Feature name returned by SHAP or explanation logic
    feature = Column(String, nullable=False)

    # SHAP value / influence score for that feature
    shap_value = Column(Float, nullable=False)

    # Relationship back to parent prediction
    prediction = relationship("FaultPrediction", back_populates="explanations")