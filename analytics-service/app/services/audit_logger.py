"""Append-only audit logger with SHA-256 hash-chain integrity."""

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AuditLogger:
    """JSON-file-backed audit log with hash-chain tamper detection.

    Each entry's integrity_hash = SHA256(entry_id | timestamp | operation |
    wallet | cid | status | details | prev_hash). Changing any field
    or reordering entries breaks the chain.
    """

    def __init__(self, log_path: str = "audit_log.json"):
        self._path = log_path
        self._lock = threading.Lock()
        self._entries: List[Dict[str, Any]] = []
        self._load()

    # -- persistence --

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    self._entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._entries = []
        else:
            self._entries = []

    def _flush(self):
        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._entries, f, indent=2)
        os.replace(tmp, self._path)

    # -- hashing --

    @staticmethod
    def _hash_entry(
        entry_id: str,
        timestamp: str,
        operation: str,
        wallet_address: str,
        dataset_cid: str,
        status: str,
        details: str,
        prev_hash: str,
    ) -> str:
        payload = (
            f"{entry_id}|{timestamp}|{operation}|{wallet_address}"
            f"|{dataset_cid}|{status}|{details}|{prev_hash}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- public API --

    def log_operation(
        self,
        operation: str,
        wallet_address: str = "",
        dataset_cid: str = "",
        status: str = "SUCCESS",
        details: str = "",
    ) -> str:
        """Append a new audit entry. Returns the entry_id."""
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        prev_hash = self._entries[-1]["integrity_hash"] if self._entries else "genesis"

        integrity_hash = self._hash_entry(
            entry_id, timestamp, operation,
            wallet_address, dataset_cid, status, details, prev_hash,
        )

        entry = {
            "id": entry_id,
            "timestamp": timestamp,
            "operation": operation,
            "wallet_address": wallet_address,
            "dataset_cid": dataset_cid,
            "status": status,
            "details": details,
            "prev_hash": prev_hash,
            "integrity_hash": integrity_hash,
        }

        with self._lock:
            self._entries.append(entry)
            self._flush()

        return entry_id

    def verify_entry(self, entry_id: str) -> Dict[str, Any]:
        """Verify a single entry's hash and its chain link.

        Returns dict with valid (bool), entry data, and reason on failure.
        """
        entry = self.get_entry(entry_id)
        if entry is None:
            return {"valid": False, "reason": "entry not found"}

        idx = next(
            (i for i, e in enumerate(self._entries) if e["id"] == entry_id), -1
        )

        # check prev_hash linkage
        expected_prev = self._entries[idx - 1]["integrity_hash"] if idx > 0 else "genesis"
        if entry["prev_hash"] != expected_prev:
            return {
                "valid": False,
                "entry": entry,
                "reason": "prev_hash chain broken",
            }

        # recompute this entry's hash
        expected_hash = self._hash_entry(
            entry["id"],
            entry["timestamp"],
            entry["operation"],
            entry["wallet_address"],
            entry["dataset_cid"],
            entry["status"],
            entry["details"],
            entry["prev_hash"],
        )
        if expected_hash != entry["integrity_hash"]:
            return {
                "valid": False,
                "entry": entry,
                "reason": "integrity_hash mismatch — data tampered",
            }

        return {"valid": True, "entry": entry}

    def verify_chain(self) -> Dict[str, Any]:
        """Walk the full chain and report first broken link."""
        for i, entry in enumerate(self._entries):
            expected_prev = self._entries[i - 1]["integrity_hash"] if i > 0 else "genesis"
            if entry["prev_hash"] != expected_prev:
                return {
                    "valid": False,
                    "broken_at": entry["id"],
                    "index": i,
                    "reason": "prev_hash mismatch",
                }

            expected_hash = self._hash_entry(
                entry["id"],
                entry["timestamp"],
                entry["operation"],
                entry["wallet_address"],
                entry["dataset_cid"],
                entry["status"],
                entry["details"],
                entry["prev_hash"],
            )
            if expected_hash != entry["integrity_hash"]:
                return {
                    "valid": False,
                    "broken_at": entry["id"],
                    "index": i,
                    "reason": "integrity_hash mismatch",
                }

        return {"valid": True, "total_entries": len(self._entries)}

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        return next((e for e in self._entries if e["id"] == entry_id), None)

    def query_logs(
        self,
        wallet_address: Optional[str] = None,
        operation: Optional[str] = None,
        dataset_cid: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        results = self._entries
        if wallet_address:
            results = [e for e in results if e["wallet_address"] == wallet_address]
        if operation:
            results = [e for e in results if e["operation"] == operation]
        if dataset_cid:
            results = [e for e in results if e["dataset_cid"] == dataset_cid]
        # newest first
        return list(reversed(results))[:limit]
