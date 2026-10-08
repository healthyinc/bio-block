import logging
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.bm25_retrieval import BM25QueryError  # noqa: E402
from services.bm25_retrieval import BM25RetrievalService  # noqa: E402
from services.bm25_retrieval import SearchableDocument  # noqa: E402
from services.bm25_retrieval import UnsafeDocumentError  # noqa: E402
from services.bm25_retrieval import tokenize  # noqa: E402


def _document(
    document_id,
    text,
    *,
    modality="text",
    metadata=None,
    anonymization_status="completed",
    privacy_validation_status="passed",
):
    return SearchableDocument(
        document_id=document_id,
        modality=modality,
        anonymized_text=text,
        anonymization_status=anonymization_status,
        safe_metadata=metadata or {},
        privacy_validation_status=privacy_validation_status,
    )


def test_tokenizer_is_case_insensitive_and_preserves_clinical_tokens():
    assert tokenize("  HbA1c, E11.9; COVID-19 and type 2 diabetes  ") == [
        "hba1c",
        "e11.9",
        "covid-19",
        "and",
        "type",
        "2",
        "diabetes",
    ]


def test_relevant_document_ranks_first():
    service = BM25RetrievalService()
    service.index_documents(
        [
            _document("doc-unrelated", "Routine respiratory follow-up."),
            _document("doc-diabetes", "Type 2 diabetes care with HbA1c review."),
        ]
    )

    results = service.search("type 2 diabetes")

    assert results[0]["document_id"] == "doc-diabetes"
    assert results[0]["rank"] == 1
    assert results[0]["score"] > 0


def test_rare_term_has_more_weight_than_common_term():
    service = BM25RetrievalService()
    service.index_documents(
        [
            _document("doc-rare", "common rareterm"),
            _document("doc-common-a", "common routine"),
            _document("doc-common-b", "common standard"),
        ]
    )

    rare_score = service.search("rareterm")[0]["score"]
    common_score = next(
        result["score"]
        for result in service.search("common")
        if result["document_id"] == "doc-rare"
    )

    assert rare_score > common_score


def test_search_is_case_insensitive():
    service = BM25RetrievalService()
    service.upsert_document(_document("doc-1", "Hypertension medication review."))

    assert service.search("HYPERTENSION")[0]["document_id"] == "doc-1"


@pytest.mark.parametrize("query", ["E11.9", "e11.9"])
def test_clinical_code_search_handles_punctuation(query):
    service = BM25RetrievalService()
    service.upsert_document(
        _document(
            "doc-code",
            "Diabetes assessment.",
            metadata={"clinical_codes": ["E11.9"]},
        )
    )

    assert service.search(query)[0]["document_id"] == "doc-code"


def test_hyphenated_term_search():
    service = BM25RetrievalService()
    service.upsert_document(_document("doc-covid", "COVID-19 follow-up completed."))

    assert service.search("covid-19")[0]["document_id"] == "doc-covid"


def test_repeated_query_terms_contribute_repeatedly():
    service = BM25RetrievalService()
    service.upsert_document(_document("doc-1", "asthma management"))

    single_score = service.search("asthma")[0]["score"]
    repeated_score = service.search("asthma asthma")[0]["score"]

    assert repeated_score == pytest.approx(single_score * 2)


def test_empty_query_is_rejected():
    service = BM25RetrievalService()

    with pytest.raises(BM25QueryError):
        service.search("   ...   ")


@pytest.mark.parametrize("top_k", [0, 101, True])
def test_top_k_is_validated(top_k):
    service = BM25RetrievalService()

    with pytest.raises(BM25QueryError):
        service.search("asthma", top_k=top_k)


def test_empty_index_returns_no_results():
    assert BM25RetrievalService().search("diabetes") == []


def test_no_match_returns_empty_results():
    service = BM25RetrievalService()
    service.upsert_document(_document("doc-1", "asthma review"))

    assert service.search("nonexistentterm") == []


def test_ties_are_deterministic_by_document_id():
    service = BM25RetrievalService()
    service.index_documents(
        [
            _document("doc-b", "same clinical text"),
            _document("doc-a", "same clinical text"),
        ]
    )

    assert [result["document_id"] for result in service.search("clinical")] == [
        "doc-a",
        "doc-b",
    ]


def test_modality_filter_is_applied_before_top_k():
    service = BM25RetrievalService()
    service.index_documents(
        [
            _document("doc-text", "MRI brain finding", modality="text"),
            _document("doc-mri", "MRI brain finding", modality="MRI"),
        ]
    )

    results = service.search("MRI brain", top_k=1, filters={"modality": "mri"})

    assert [result["document_id"] for result in results] == ["doc-mri"]


def test_unsupported_filter_is_rejected():
    with pytest.raises(BM25QueryError):
        BM25RetrievalService().search("asthma", filters={"owner": "value"})


def test_duplicate_id_upsert_replaces_document():
    service = BM25RetrievalService()
    service.upsert_document(_document("doc-1", "asthma"))
    service.upsert_document(_document("doc-1", "hypertension"))

    assert service.document_count == 1
    assert service.search("asthma") == []
    assert service.search("hypertension")[0]["document_id"] == "doc-1"


def test_rebuild_replaces_the_existing_index():
    service = BM25RetrievalService()
    service.upsert_document(_document("old", "asthma"))

    count = service.rebuild(
        [
            _document("new-a", "diabetes"),
            _document("new-b", "hypertension"),
        ]
    )

    assert count == 2
    assert service.search("asthma") == []
    assert service.search("diabetes")[0]["document_id"] == "new-a"


def test_rebuild_is_atomic_when_a_document_is_unsafe():
    service = BM25RetrievalService()
    service.upsert_document(_document("existing", "asthma"))

    with pytest.raises(UnsafeDocumentError):
        service.rebuild(
            [
                _document("safe", "diabetes"),
                _document("unsafe", "content", anonymization_status="failed"),
            ]
        )

    assert service.document_count == 1
    assert service.search("asthma")[0]["document_id"] == "existing"


def test_term_frequency_contributes_to_ranking():
    service = BM25RetrievalService()
    service.index_documents(
        [
            _document("frequent", "asthma asthma asthma"),
            _document("single", "asthma"),
        ]
    )

    assert service.search("asthma")[0]["document_id"] == "frequent"


def test_document_length_normalization_favors_concise_match():
    service = BM25RetrievalService()
    service.index_documents(
        [
            _document("short", "asthma plan"),
            _document("long", "asthma " + "unrelated " * 30),
        ]
    )

    assert service.search("asthma")[0]["document_id"] == "short"


def test_zero_average_document_length_is_safe():
    service = BM25RetrievalService()
    service.upsert_document(_document("empty", "", modality="---"))

    assert service.average_document_length == 0
    assert service.search("asthma") == []


def test_snippet_comes_only_from_anonymized_text():
    service = BM25RetrievalService(snippet_length=60)
    service.upsert_document(
        _document(
            "doc-1",
            "Anonymized diabetes summary " + "safe " * 30,
            metadata={"clinical_terms": ["metadata-only-term"]},
        )
    )

    result = service.search("metadata-only-term")[0]

    assert "Anonymized diabetes summary" in result["snippet"]
    assert "metadata-only-term" not in result["snippet"]
    assert len(result["snippet"]) <= 62


def test_document_id_is_returned_but_not_searchable():
    service = BM25RetrievalService()
    service.upsert_document(_document("secret-id-token", "asthma summary"))

    assert service.search("secret-id-token") == []
    assert service.search("asthma")[0]["document_id"] == "secret-id-token"


def test_approved_safe_metadata_is_searchable_and_canonicalized():
    service = BM25RetrievalService()
    service.upsert_document(
        _document(
            "doc-1",
            "Clinical review.",
            metadata={"Document-Type": "radiology", "disease tags": ["oncology"]},
        )
    )

    result = service.search("oncology")[0]

    assert result["safe_metadata"] == {
        "document_type": "radiology",
        "disease_tags": ["oncology"],
    }


@pytest.mark.parametrize(
    "status",
    ["failed", "pending", "completed_with_warnings", ""],
)
def test_non_completed_anonymization_is_rejected(status):
    service = BM25RetrievalService()

    with pytest.raises(UnsafeDocumentError):
        service.upsert_document(
            _document("doc-1", "asthma", anonymization_status=status)
        )


def test_missing_anonymization_status_is_rejected():
    service = BM25RetrievalService()

    with pytest.raises(UnsafeDocumentError):
        service.upsert_document(
            {
                "document_id": "doc-1",
                "modality": "text",
                "anonymized_text": "asthma",
            }
        )


@pytest.mark.parametrize("status", ["failed", "failed_privacy_validation", "pending"])
def test_failed_privacy_validation_is_rejected(status):
    service = BM25RetrievalService()

    with pytest.raises(UnsafeDocumentError):
        service.upsert_document(
            _document("doc-1", "asthma", privacy_validation_status=status)
        )


@pytest.mark.parametrize(
    "unsafe_key",
    [
        "Patient_Name",
        "patient-id",
        "SSN",
        "MRN",
        "original.filename",
        "raw DICOM metadata",
        "temporary-path",
        "exact_coordinates",
        "email_address",
    ],
)
def test_unsafe_metadata_keys_are_rejected_separator_insensitively(unsafe_key):
    service = BM25RetrievalService()

    with pytest.raises(UnsafeDocumentError):
        service.upsert_document(
            _document("doc-1", "asthma", metadata={unsafe_key: "unsafe-value"})
        )


def test_unapproved_metadata_is_rejected():
    with pytest.raises(UnsafeDocumentError):
        BM25RetrievalService().upsert_document(
            _document("doc-1", "asthma", metadata={"arbitrary_field": "value"})
        )


def test_raw_bytes_are_rejected_at_any_document_boundary():
    service = BM25RetrievalService()

    with pytest.raises(UnsafeDocumentError):
        service.upsert_document(
            {
                "document_id": "doc-1",
                "modality": "text",
                "anonymized_text": "asthma",
                "anonymization_status": "completed",
                "file_bytes": b"synthetic-bytes",
            }
        )
    with pytest.raises(UnsafeDocumentError):
        service.upsert_document(
            _document(
                "doc-2",
                "asthma",
                metadata={"clinical_terms": [b"synthetic-bytes"]},
            )
        )


def test_raw_content_fields_are_rejected():
    with pytest.raises(UnsafeDocumentError):
        BM25RetrievalService().upsert_document(
            {
                "document_id": "doc-1",
                "modality": "text",
                "anonymized_text": "safe summary",
                "anonymization_status": "completed",
                "raw_content": "not eligible",
            }
        )


def test_delete_document_and_clear():
    service = BM25RetrievalService()
    service.upsert_document(_document("doc-1", "asthma"))

    assert service.delete_document("doc-1") is True
    assert service.delete_document("doc-1") is False
    assert service.document_count == 0

    service.upsert_document(_document("doc-2", "diabetes"))
    service.clear()
    assert service.document_count == 0


@pytest.fixture
def api_context():
    import main

    main.bm25_retrieval_service.clear()
    yield main, TestClient(main.app)
    main.bm25_retrieval_service.clear()


def test_bm25_endpoint_returns_ranked_safe_results(api_context):
    main, client = api_context
    main.bm25_retrieval_service.index_documents(
        [
            _document("doc-other", "Routine asthma review.", modality="text"),
            _document(
                "doc-diabetes",
                "Type 2 diabetes assessment with E11.9.",
                modality="report",
                metadata={"document_type": "clinical-summary"},
            ),
        ]
    )

    response = client.post(
        "/api/v1/search/bm25",
        json={"query": "type 2 diabetes", "top_k": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["search_type"] == "bm25"
    assert body["query_metadata"] == {"token_count": 3}
    assert body["result_count"] == 1
    assert body["index_document_count"] == 2
    assert body["retrieval_status"] == "completed"
    result = body["results"][0]
    assert result["document_id"] == "doc-diabetes"
    assert set(result) == {
        "document_id",
        "rank",
        "score",
        "modality",
        "safe_metadata",
        "snippet",
    }
    serialized = response.text.lower()
    for forbidden in (
        "patientname",
        "original_filename",
        "raw_content",
        "file_bytes",
        "temporary_path",
    ):
        assert forbidden not in serialized


def test_bm25_endpoint_modality_filter(api_context):
    main, client = api_context
    main.bm25_retrieval_service.index_documents(
        [
            _document("text-doc", "MRI brain", modality="text"),
            _document("mri-doc", "MRI brain", modality="MRI"),
        ]
    )

    response = client.post(
        "/api/v1/search/bm25",
        json={"query": "MRI brain", "modality": "mri"},
    )

    assert response.status_code == 200
    assert [item["document_id"] for item in response.json()["results"]] == [
        "mri-doc"
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "asthma", "top_k": 0},
        {"query": "asthma", "top_k": 101},
        {"query": "asthma", "owner": "unsupported"},
    ],
)
def test_bm25_endpoint_rejects_invalid_requests(api_context, payload):
    _, client = api_context

    response = client.post("/api/v1/search/bm25", json=payload)

    assert response.status_code == 422
    assert "traceback" not in response.text.lower()


def test_bm25_endpoint_no_match_is_empty(api_context):
    main, client = api_context
    main.bm25_retrieval_service.upsert_document(_document("doc-1", "asthma"))

    response = client.post(
        "/api/v1/search/bm25",
        json={"query": "nonexistentterm"},
    )

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["result_count"] == 0
    assert response.json()["retrieval_status"] == "no_matches"


def test_bm25_endpoint_empty_index_is_clean(api_context):
    _, client = api_context

    response = client.post("/api/v1/search/bm25", json={"query": "asthma"})

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["index_document_count"] == 0
    assert response.json()["retrieval_status"] == "empty_index"


def test_bm25_endpoint_does_not_log_raw_query(api_context, caplog):
    main, client = api_context
    main.bm25_retrieval_service.upsert_document(_document("doc-1", "asthma"))
    sensitive_query = "synthetic-sensitive-query-value"

    with caplog.at_level(logging.INFO, logger="main"):
        response = client.post(
            "/api/v1/search/bm25",
            json={"query": sensitive_query},
        )

    assert response.status_code == 200
    assert sensitive_query not in caplog.text
    assert "query_length=" in caplog.text


def test_bm25_endpoint_hides_internal_errors(api_context, monkeypatch):
    main, client = api_context

    def fail_search(*args, **kwargs):
        raise RuntimeError("synthetic internal path must stay private")

    monkeypatch.setattr(main.bm25_retrieval_service, "search", fail_search)
    response = client.post("/api/v1/search/bm25", json={"query": "asthma"})

    assert response.status_code == 500
    assert response.json() == {"detail": "BM25 search failed"}
    assert "internal path" not in response.text.lower()
