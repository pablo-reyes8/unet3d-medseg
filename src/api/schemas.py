from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    device: str = Field(..., example="cuda")
    model_loaded: bool = Field(..., example=True)
    model_path: str
    detail: Optional[str] = None


class ModelMetadata(BaseModel):
    model_path: str
    num_classes: int
    class_names: List[str]
    in_channels: int
    base_channels: int
    norm: str
    dropout: float
    pad_multiple: int
    clip_percentiles: Tuple[int, int]
    default_threshold: float
    device: str


class PredictionSummary(BaseModel):
    filename: str
    input_shape: Tuple[int, int, int]
    padded_shape: Tuple[int, int, int]
    runtime_ms: float
    device: str
    threshold_used: float
    class_histogram: Dict[int, int]


class ErrorResponse(BaseModel):
    detail: str

