import pytest
import uuid
import chromadb
from services.audit_logger import AuditLogger


@pytest.fixture
def audit_logger():
    client = chromadb.Client()
    logger = AuditLogger(client)
    logger.collection = client.get_or_create_collection(
        name=f"audit_test_{uuid.uuid4().hex[:8]}"
    )
    return logger


class TestLogOperation:
    def test_creates_entry_and_returns_id(self, audit_logger):
        entry_id = audit_logger.log_operation(
            operation="STORE",
            wallet_address="0xabc123",
            document_id="doc_001",
            details="test store",
        )
        assert entry_id is not None
        assert len(entry_id) > 0

    def test_entry_is_retrievable(self, audit_logger):
        entry_id = audit_logger.log_operation(
            operation="DELETE",
            wallet_address="0xdef456",
            document_id="doc_002",
        )
        entry = audit_logger.get_entry(entry_id)
        assert entry is not None
        assert entry["operation"] == "DELETE"
        assert entry["wallet_address"] == "0xdef456"
        assert entry["document_id"] == "doc_002"
        assert entry["status"] == "SUCCESS"

    def test_anonymous_operation(self, audit_logger):
        entry_id = audit_logger.log_operation(
            operation="SEARCH",
            details="query: cancer dataset",
        )
        entry = audit_logger.get_entry(entry_id)
        assert entry["wallet_address"] == ""
        assert entry["operation"] == "SEARCH"

    def test_custom_status(self, audit_logger):
        entry_id = audit_logger.log_operation(
            operation="STORE",
            status="FAILED",
            wallet_address="0x111",
            details="disk full",
        )
        entry = audit_logger.get_entry(entry_id)
        assert entry["status"] == "FAILED"


class TestIntegrityVerification:
    def test_valid_entry_passes_verification(self, audit_logger):
        entry_id = audit_logger.log_operation(
            operation="STORE",
            wallet_address="0xabc",
            document_id="doc_100",
        )
        assert audit_logger.verify_integrity(entry_id) is True

    def test_nonexistent_entry_fails_verification(self, audit_logger):
        assert audit_logger.verify_integrity("nonexistent-id") is False

    def test_tampered_entry_fails_verification(self, audit_logger):
        entry_id = audit_logger.log_operation(
            operation="STORE",
            wallet_address="0xabc",
            document_id="doc_tamper",
        )

        audit_logger.collection.update(
            ids=[entry_id],
            metadatas=[{
                "timestamp": "tampered",
                "operation": "STORE",
                "wallet_address": "0xabc",
                "document_id": "doc_tamper",
                "status": "SUCCESS",
                "details": "",
                "integrity_hash": "fake_hash",
            }],
        )

        assert audit_logger.verify_integrity(entry_id) is False


class TestQueryLogs:
    def test_query_all_logs(self, audit_logger):
        audit_logger.log_operation(operation="STORE", wallet_address="0xa")
        audit_logger.log_operation(operation="DELETE", wallet_address="0xb")

        logs = audit_logger.query_logs()
        assert len(logs) == 2

    def test_filter_by_wallet(self, audit_logger):
        audit_logger.log_operation(operation="STORE", wallet_address="0xAAA")
        audit_logger.log_operation(operation="DELETE", wallet_address="0xBBB")

        logs = audit_logger.query_logs(wallet_address="0xAAA")
        assert len(logs) == 1
        assert logs[0]["wallet_address"] == "0xAAA"

    def test_filter_by_operation(self, audit_logger):
        audit_logger.log_operation(operation="STORE", wallet_address="0x1")
        audit_logger.log_operation(operation="DELETE", wallet_address="0x2")
        audit_logger.log_operation(operation="STORE", wallet_address="0x3")

        logs = audit_logger.query_logs(operation="STORE")
        assert len(logs) == 2
        assert all(log["operation"] == "STORE" for log in logs)

    def test_filter_by_document_id(self, audit_logger):
        audit_logger.log_operation(operation="STORE", document_id="doc_X")
        audit_logger.log_operation(operation="UPDATE", document_id="doc_Y")

        logs = audit_logger.query_logs(document_id="doc_X")
        assert len(logs) == 1
        assert logs[0]["document_id"] == "doc_X"

    def test_limit_results(self, audit_logger):
        for i in range(10):
            audit_logger.log_operation(operation="SEARCH", details=f"q{i}")

        logs = audit_logger.query_logs(limit=3)
        assert len(logs) == 3

    def test_empty_collection_returns_empty(self, audit_logger):
        logs = audit_logger.query_logs()
        assert logs == []


class TestGetEntry:
    def test_existing_entry(self, audit_logger):
        entry_id = audit_logger.log_operation(
            operation="UPDATE",
            wallet_address="0xget",
            document_id="doc_get",
        )
        entry = audit_logger.get_entry(entry_id)
        assert entry is not None
        assert entry["id"] == entry_id

    def test_nonexistent_entry_returns_none(self, audit_logger):
        result = audit_logger.get_entry("does-not-exist")
        assert result is None


class TestAppendOnly:
    def test_entries_accumulate(self, audit_logger):
        audit_logger.log_operation(operation="STORE")
        assert len(audit_logger.query_logs()) == 1

        audit_logger.log_operation(operation="DELETE")
        assert len(audit_logger.query_logs()) == 2

        audit_logger.log_operation(operation="SEARCH")
        assert len(audit_logger.query_logs()) == 3
