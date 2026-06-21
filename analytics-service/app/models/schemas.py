

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


class VisualizationResponse(BaseModel):
    analysis_type: str = "graphical"
    source_dataset_cid: str
    chart_type: str
    chart_config: Dict[str, Any]
    image: str = Field(..., description="Base64-encoded PNG image")
    row_count: int


class ErrorResponse(BaseModel):
    detail: str


class RegistryResultResponse(BaseModel):
    result_cid: str
    data: dict
    message: str = "Placeholder until IPFS integration is complete"


class RegistryDatasetResponse(BaseModel):
    dataset_cid: str
    result_cids: List[str]
    message: str = "Placeholder until IPFS/Contract integration is complete"
