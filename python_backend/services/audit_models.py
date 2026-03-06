from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class AuditLogEntry(BaseModel):
    id: str
    timestamp: str
    operation: str
    wallet_address: Optional[str] = None
    document_id: Optional[str] = None
    status: str
    details: Optional[str] = None
    integrity_hash: str


class AuditQueryResponse(BaseModel):
    logs: List[AuditLogEntry]
    total: int
