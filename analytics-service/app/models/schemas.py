

from __future__ import annotations

from typing import Any

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


class ErrorResponse(BaseModel):
    detail: str
