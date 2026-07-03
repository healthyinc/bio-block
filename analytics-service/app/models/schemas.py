

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "bio-block-analytics"
    version: str
    port: int = 3003


class DescriptiveResponse(BaseModel):
    analysis_type: str = "descriptive"
    source_dataset_cid: str
    row_count: int
    columns_analyzed: list
    results: dict
    result_cid: Optional[str] = None
    tx_hash: Optional[str] = None


class VisualizationResponse(BaseModel):
    analysis_type: str = "graphical"
    source_dataset_cid: str
    chart_type: str
    chart_config: Dict[str, Any]
    image: str = Field(..., description="Base64-encoded PNG image")
    row_count: int
    result_cid: Optional[str] = None
    tx_hash: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str


class RegistryResultResponse(BaseModel):
    result_cid: str
    data: dict


class RegistryDatasetResponse(BaseModel):
    dataset_cid: str
    result_cids: List[str]
