import chromadb
import hashlib
import uuid
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any


class AuditLogger:
    def __init__(self, chroma_client: chromadb.ClientAPI):
        self.collection = chroma_client.get_or_create_collection(
            name="audit_logs"
        )

    @staticmethod
    def _generate_integrity_hash(
        entry_id: str,
        timestamp: str,
        operation: str,
        wallet_address: str,
        document_id: str,
        status: str,
        details: str,
    ) -> str:
        payload = f"{entry_id}|{timestamp}|{operation}|{wallet_address}|{document_id}|{status}|{details}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def log_operation(
        self,
        operation: str,
        status: str = "SUCCESS",
        wallet_address: Optional[str] = None,
        document_id: Optional[str] = None,
        details: Optional[str] = None,
    ) -> str:
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        integrity_hash = self._generate_integrity_hash(
            entry_id,
            timestamp,
            operation,
            wallet_address or "",
            document_id or "",
            status,
            details or "",
        )

        document_text = (
            f"{operation} by {wallet_address or 'anonymous'} "
            f"on {document_id or 'N/A'} at {timestamp} — {status}"
        )

        metadata = {
            "timestamp": timestamp,
            "operation": operation,
            "wallet_address": wallet_address or "",
            "document_id": document_id or "",
            "status": status,
            "details": details or "",
            "integrity_hash": integrity_hash,
        }

        self.collection.add(
            ids=[entry_id],
            documents=[document_text],
            metadatas=[metadata],
        )

        return entry_id

    def verify_integrity(self, entry_id: str) -> bool:
        result = self.collection.get(
            ids=[entry_id], include=["metadatas"]
        )

        if not result["ids"]:
            return False

        meta = result["metadatas"][0]
        expected = self._generate_integrity_hash(
            entry_id,
            meta["timestamp"],
            meta["operation"],
            meta["wallet_address"],
            meta["document_id"],
            meta["status"],
            meta["details"],
        )
        return expected == meta["integrity_hash"]

    def query_logs(
        self,
        wallet_address: Optional[str] = None,
        operation: Optional[str] = None,
        document_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        filters = []
        if wallet_address:
            filters.append({"wallet_address": wallet_address})
        if operation:
            filters.append({"operation": operation})
        if document_id:
            filters.append({"document_id": document_id})

        where_clause = None
        if len(filters) > 1:
            where_clause = {"$and": filters}
        elif len(filters) == 1:
            where_clause = filters[0]

        if where_clause:
            result = self.collection.get(
                where=where_clause,
                include=["metadatas", "documents"],
            )
        else:
            result = self.collection.get(
                include=["metadatas", "documents"],
            )

        logs = []
        if result["ids"]:
            for i, entry_id in enumerate(result["ids"]):
                meta = result["metadatas"][i]
                logs.append({
                    "id": entry_id,
                    "timestamp": meta["timestamp"],
                    "operation": meta["operation"],
                    "wallet_address": meta["wallet_address"],
                    "document_id": meta["document_id"],
                    "status": meta["status"],
                    "details": meta["details"],
                    "integrity_hash": meta["integrity_hash"],
                })

        logs.sort(key=lambda x: x["timestamp"], reverse=True)
        return logs[:limit]

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        result = self.collection.get(
            ids=[entry_id], include=["metadatas", "documents"]
        )

        if not result["ids"]:
            return None

        meta = result["metadatas"][0]
        return {
            "id": entry_id,
            "timestamp": meta["timestamp"],
            "operation": meta["operation"],
            "wallet_address": meta["wallet_address"],
            "document_id": meta["document_id"],
            "status": meta["status"],
            "details": meta["details"],
            "integrity_hash": meta["integrity_hash"],
        }
