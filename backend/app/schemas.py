from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class TopFeature(BaseModel):
    # One explanation feature item
    feature: str
    shap_value: float


class PredictionResult(BaseModel):
    # Main prediction outputs returned to frontend
    predicted_severity: int
    risk_level: str
    confidence: float
    reason: str

    # Top explanation features
    top_features: List[TopFeature] = Field(default_factory=list)

    # Fault isolation outputs
    fault_category: Optional[str] = None
    isolation_summary: Optional[str] = None

    # Rule-based engineer guidance
    recommended_checks: List[str] = Field(default_factory=list)

    # Probability for each class, for example {"0": 0.1, "1": 0.7, "2": 0.2}
    class_probabilities: Dict[str, float] = Field(default_factory=dict)


class FaultInput(BaseModel):
    # Main user inputs from frontend
    location: str = Field(..., examples=["location 344"])
    severity_type: str = Field(..., examples=["severity_type 4"])

    # Sparse dictionaries of selected feature signals
    event_features: Dict[str, float] = Field(default_factory=dict)
    resource_features: Dict[str, float] = Field(default_factory=dict)
    log_features: Dict[str, float] = Field(default_factory=dict)

    @field_validator("location")
    @classmethod
    def validate_location(cls, value: str) -> str:
        # Remove extra spaces
        value = str(value).strip()

        # Location must not be empty
        if not value:
            raise ValueError("location is required.")

        return value

    @field_validator("severity_type")
    @classmethod
    def validate_severity_type(cls, value: str) -> str:
        # Remove extra spaces
        value = str(value).strip()

        # Severity type must not be empty
        if not value:
            raise ValueError("severity_type is required.")

        return value

    @field_validator("event_features")
    @classmethod
    def validate_event_features(cls, value: Dict[str, float]) -> Dict[str, float]:
        # Validate event feature keys and convert values to float
        cleaned: Dict[str, float] = {}

        for key, val in value.items():
            key = str(key).strip()

            # Event feature keys must follow notebook/training naming pattern
            if not key.startswith("event_type_"):
                raise ValueError("All event_features keys must start with 'event_type_'.")

            cleaned[key] = float(val)

        return cleaned

    @field_validator("resource_features")
    @classmethod
    def validate_resource_features(cls, value: Dict[str, float]) -> Dict[str, float]:
        # Validate resource feature keys and convert values to float
        cleaned: Dict[str, float] = {}

        for key, val in value.items():
            key = str(key).strip()

            # Resource feature keys must follow training naming pattern
            if not key.startswith("resource_type_"):
                raise ValueError("All resource_features keys must start with 'resource_type_'.")

            cleaned[key] = float(val)

        return cleaned

    @field_validator("log_features")
    @classmethod
    def validate_log_features(cls, value: Dict[str, float]) -> Dict[str, float]:
        # Validate log feature keys and convert values to float
        cleaned: Dict[str, float] = {}

        for key, val in value.items():
            key = str(key).strip()

            # Log feature keys must follow training naming pattern
            if not key.startswith("log_feature_"):
                raise ValueError("All log_features keys must start with 'log_feature_'.")

            cleaned[key] = float(val)

        return cleaned


class PredictResponse(BaseModel):
    # Response returned by POST /predict
    prediction_id: int
    result: PredictionResult


class PredictionHistoryItem(BaseModel):
    # Short history item returned in /predictions list
    id: int
    created_at: datetime
    location: str
    severity_type: str

    event_count: float
    resource_count: float
    log_count: float
    log_volume_sum: float

    predicted_severity: int
    confidence: float
    risk_level: str
    reason: str
    fault_category: Optional[str] = None
    isolation_summary: Optional[str] = None

    # Allow Pydantic to read directly from SQLAlchemy model objects
    model_config = {"from_attributes": True}


class PredictionDetail(PredictionHistoryItem):
    # Detailed history item including explanation rows
    explanations: List[TopFeature] = Field(default_factory=list)

    model_config = {"from_attributes": True}