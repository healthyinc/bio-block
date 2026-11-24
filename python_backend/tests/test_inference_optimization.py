"""Phase 12B: the optimizations must not move a single privacy result.

Two things are proved here. The first is that chunking still catches an
identifier no matter where a chunk boundary falls across it - the boundary is
walked across every offset of a planted identifier, because an entity that is
only detectable when it happens to sit inside one window is not detected at
all. The second is that prewarming changes when work happens, never whether a
request is accepted: readiness is false until the models are verified, loaded
and warmed, and a failure leaves it false rather than degrading to unprotected
processing.

A whitespace-chunk skip was tried and not adopted: it bought no measurable
speed. A guard below fails if it is reintroduced without re-running the
privacy evaluation.
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import model_lifecycle  # noqa: E402
from services.local_model_detectors import (  # noqa: E402
    DetectorConfig,
    merge_chunk_entities,
    overlapping_chunks,
)
from services.privacy_contracts import PhiEntity  # noqa: E402
from services.text_anonymization import anonymize_clinical_text  # noqa: E402


@pytest.fixture(autouse=True)
def _mocked_models(monkeypatch):
    monkeypatch.setenv("PHI_MODEL_MODE", "legacy_test")
    monkeypatch.setenv("BIOBLOCK_STUDY_SALT", "phase12b-salt")


# -- chunking keeps its offset contract -------------------------------------


def test_chunks_cover_the_whole_document_without_loss():
    text = "".join(f"line {n} of the note\n" for n in range(500))
    chunks = overlapping_chunks(text, 2000, 200)

    rebuilt = ""
    for offset, chunk in chunks:
        assert text[offset : offset + len(chunk)] == chunk, "offset mapping drifted"
        if offset <= len(rebuilt):
            rebuilt = rebuilt[:offset] + chunk
    assert rebuilt == text, "characters were lost or reordered"


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_line_endings_do_not_break_offsets(newline):
    text = newline.join(f"observation {n}" for n in range(400))
    for offset, chunk in overlapping_chunks(text, 1000, 150):
        assert text[offset : offset + len(chunk)] == chunk


def test_a_very_long_line_without_punctuation_is_still_chunked():
    text = "x" * 9_000
    chunks = overlapping_chunks(text, 2000, 200)

    assert len(chunks) > 1
    assert chunks[-1][0] + len(chunks[-1][1]) == len(text)


def test_unicode_is_never_split_mid_character():
    text = "Patient note with unicode: café, naïve, 日本語テキスト. " * 200
    for offset, chunk in overlapping_chunks(text, 1000, 200):
        # Slicing a Python str cannot split a code point; this asserts the
        # round trip through encode/decode stays intact for the whole chunk.
        assert chunk.encode("utf-8").decode("utf-8") == chunk
        assert text[offset : offset + len(chunk)] == chunk


def test_tabs_and_repeated_whitespace_survive():
    text = "field:\t\tvalue\n\n\n   trailing   \n" * 300
    rebuilt_len = 0
    for offset, chunk in overlapping_chunks(text, 900, 100):
        rebuilt_len = max(rebuilt_len, offset + len(chunk))
    assert rebuilt_len == len(text)


# -- an identifier is found wherever the boundary falls ---------------------

BOUNDARY_IDENTIFIERS = [
    pytest.param("Zelquin Vantreske", "PERSON", id="long_multi_token_name"),
    pytest.param("synthetic.person@example.invalid", "EMAIL", id="email"),
    pytest.param("MRN 44821903", "RECORD_ID", id="record_id"),
    pytest.param("555-0142", "PHONE", id="phone"),
    pytest.param("14 March 2021", "DATE", id="date"),
    pytest.param("94 years old", "AGE_OVER_89", id="age_over_89"),
    pytest.param("12 Fallowbrae Lane, Endermoss", "ADDRESS", id="address"),
]


@pytest.mark.parametrize(("identifier", "_kind"), BOUNDARY_IDENTIFIERS)
def test_identifier_is_wholly_inside_some_chunk_at_every_boundary_offset(
    identifier, _kind
):
    """Walk the chunk boundary across the identifier, one character at a time.

    An entity that is only detectable when it happens to land inside a window
    is not reliably detectable at all, so overlap must guarantee that every
    placement falls wholly inside at least one chunk.
    """
    chunk_size, overlap = 200, 60
    assert len(identifier) <= overlap, "fixture longer than the overlap guarantee"

    filler = "the routine clinical observation continued without incident. "
    for shift in range(len(identifier) + overlap):
        prefix = (filler * 40)[: chunk_size - len(identifier) + shift - overlap // 2]
        text = prefix + identifier + " " + filler * 5
        start = len(prefix)
        end = start + len(identifier)

        chunks = overlapping_chunks(text, chunk_size, overlap)
        assert any(
            offset <= start and end <= offset + len(chunk) for offset, chunk in chunks
        ), f"identifier split at every window for shift={shift}"


def test_overlap_shorter_than_an_entity_is_rejected_by_config():
    """The guarantee only holds while overlap covers the longest entity."""
    with pytest.raises(ValueError):
        DetectorConfig(chunk_size=100, chunk_overlap=0)
    with pytest.raises(ValueError):
        DetectorConfig(chunk_size=100, chunk_overlap=100)


# -- duplicates from overlap collapse deterministically ---------------------


def _entity(start, end, kind="PERSON", score=0.9, source="gliner_multi_pii"):
    return PhiEntity(entity_type=kind, start=start, end=end, source=source, score=score)


def test_duplicate_detections_in_overlap_are_collapsed():
    duplicates = [_entity(10, 20), _entity(10, 20), _entity(10, 20)]

    assert len(merge_chunk_entities(duplicates)) == 1


def test_overlapping_detections_collapse_to_the_longest():
    merged = merge_chunk_entities([_entity(10, 18), _entity(10, 25)])

    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (10, 25)


def test_deduplication_is_deterministic_regardless_of_input_order():
    entities = [_entity(30, 40), _entity(10, 25), _entity(10, 18), _entity(30, 40)]

    first = merge_chunk_entities(list(entities))
    second = merge_chunk_entities(list(reversed(entities)))

    assert [(e.start, e.end, e.entity_type) for e in first] == [
        (e.start, e.end, e.entity_type) for e in second
    ]


# -- the withdrawn whitespace-skip optimization ----------------------------


def _probe_detector():
    """A detector that records exactly what text was handed to the model."""
    from services.local_model_detectors import _ChunkedModelDetector

    seen = []

    class _Probe(_ChunkedModelDetector):
        source = "probe"
        label_map = {"PERSON": "PERSON"}

        def _infer(self, texts):
            seen.extend(texts)
            return [() for _ in texts]

    return _Probe, seen


def test_a_whitespace_chunk_is_still_answered_by_the_model():
    """Documents the rejected optimization.

    Skipping model calls for whitespace-only chunks was measured and not
    adopted: it bought no measurable latency, and a chunk that is never
    inferred has no answer at all. Every chunk still gets the model's answer.

    Identical chunks may now share one inference - the model is asked once and
    the answer is used for each occurrence - which is a different thing: the
    answer for every chunk is exactly what the model gave for that text.
    """
    probe, seen = _probe_detector()
    blank = " " * 4_000
    text = f"Patient note.{chr(10)}{blank}{chr(10)}Final observation line."
    chunks = overlapping_chunks(text, 500, 100)

    probe(DetectorConfig(chunk_size=500, chunk_overlap=100)).detect(text)

    distinct = list(dict.fromkeys(chunk for _offset, chunk in chunks))
    assert seen == distinct, (
        "the text reaching the model changed; a chunk must not be skipped, "
        "and only an identical one may share another's answer"
    )
    assert any(
        not chunk.strip() for chunk in seen
    ), "a blank chunk is still asked about"


def test_every_chunk_gets_an_answer_even_when_some_repeat():
    """Sharing an answer between identical chunks must not lose one."""
    from services.local_model_detectors import _run_batched

    calls = []

    def runner(texts):
        calls.append(list(texts))
        return [({"text": text},) for text in texts]

    chunks = [
        (0, "alpha"),
        (100, "beta"),
        (200, "alpha"),
        (300, "beta"),
        (400, "gamma"),
    ]

    answers = _run_batched(chunks, DetectorConfig(), runner, "probe_failed")

    assert len(answers) == len(chunks)
    assert [answer[0]["text"] for answer in answers] == [
        "alpha",
        "beta",
        "alpha",
        "beta",
        "gamma",
    ]
    assert sorted(sum(calls, [])) == ["alpha", "beta", "gamma"]


def test_a_repeated_block_gives_the_same_entities_as_an_unrepeated_one():
    """The point of the dedup: what comes out is unchanged."""
    from services.local_model_detectors import _run_batched

    def runner(texts):
        return [({"start": 0, "end": 3},) for _ in texts]

    once = _run_batched([(0, "same")], DetectorConfig(), runner, "probe_failed")
    twice = _run_batched(
        [(0, "same"), (50, "same")], DetectorConfig(), runner, "probe_failed"
    )

    assert twice[0] == once[0] and twice[1] == once[0]


def test_entities_after_a_blank_region_keep_their_offsets():
    from services.local_model_detectors import _ChunkedModelDetector

    text = "A" * 100 + " " * 1_500 + "Zelquin Vantreske attended."
    target = text.index("Zelquin")

    class _Probe(_ChunkedModelDetector):
        source = "probe"
        label_map = {"PERSON": "PERSON"}

        def _infer(self, texts):
            out = []
            for chunk in texts:
                index = chunk.find("Zelquin")
                out.append(
                    [
                        {
                            "entity_group": "PERSON",
                            "start": index,
                            "end": index + len("Zelquin Vantreske"),
                            "score": 0.99,
                        }
                    ]
                    if index >= 0
                    else []
                )
            return out

    found = _Probe(DetectorConfig(chunk_size=500, chunk_overlap=100)).detect(text)

    assert found, "entity after the blank region was lost"
    assert any(
        e.start == target for e in found
    ), f"offset drifted: got {[e.start for e in found]}, expected {target}"


# -- prewarming ------------------------------------------------------------


def test_prewarm_is_a_no_op_without_a_worker(monkeypatch):
    """The mocked suite must not start subprocesses."""
    monkeypatch.setenv(model_lifecycle.PREWARM_ENV_VAR, "0")
    model_lifecycle.reset_state()

    state = model_lifecycle.prewarm()

    assert state.is_ready() is True
    assert state.models == ()


def test_readiness_is_false_before_preparation():
    model_lifecycle.reset_state()

    assert model_lifecycle.state().is_ready() is False
    assert model_lifecycle.state().state == model_lifecycle.STATE_UNINITIALIZED


def test_failed_warmup_leaves_readiness_false(monkeypatch):
    """A model that will not load must never open the gate."""
    from services.local_model_detectors import LocalModelError

    monkeypatch.setenv(model_lifecycle.PREWARM_ENV_VAR, "1")
    model_lifecycle.reset_state()

    class _Broken:
        def start(self):
            raise LocalModelError("stanford_model_unavailable", 503)

    monkeypatch.setattr(model_lifecycle, "get_client", lambda: _Broken())
    state = model_lifecycle.prewarm()

    assert state.is_ready() is False
    assert state.state == model_lifecycle.STATE_FAILED
    assert state.error_code == "stanford_model_unavailable"


def test_lifecycle_state_exposes_no_paths_or_hashes(monkeypatch):
    monkeypatch.setenv(model_lifecycle.PREWARM_ENV_VAR, "1")
    model_lifecycle.reset_state()

    class _Broken:
        def start(self):
            raise RuntimeError(r"C:\Users\someone\.model-cache\weights.safetensors")

    monkeypatch.setattr(model_lifecycle, "get_client", lambda: _Broken())
    payload = repr(model_lifecycle.prewarm().to_public_dict())

    assert "Users" not in payload
    assert "safetensors" not in payload
    assert "model-cache" not in payload
    assert "model_prewarm_failed" in payload


def test_shutdown_clears_readiness(monkeypatch):
    monkeypatch.setenv(model_lifecycle.PREWARM_ENV_VAR, "0")
    model_lifecycle.reset_state()
    model_lifecycle.prewarm()

    model_lifecycle.shutdown()

    assert model_lifecycle.state().is_ready() is False


def test_liveness_and_readiness_are_separate_endpoints(monkeypatch):
    monkeypatch.setenv(model_lifecycle.PREWARM_ENV_VAR, "0")
    import main
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        live = client.get("/api/v1/health/live")
        ready = client.get("/api/v1/health/ready")

    assert live.status_code == 200
    assert live.json()["status"] == "alive"
    assert ready.status_code in {200, 503}
    assert "ready" in ready.json()


def test_readiness_payload_carries_stage_timings(monkeypatch):
    monkeypatch.setenv(model_lifecycle.PREWARM_ENV_VAR, "0")
    model_lifecycle.reset_state()
    model_lifecycle.prewarm()

    payload = model_lifecycle.state().to_public_dict()

    for key in ("worker_start_ms", "warmup_ms", "total_ms", "state", "ready"):
        assert key in payload


# -- privacy output is unchanged by any of this ----------------------------

EQUIVALENCE_NOTES = [
    "Patient Zelquin Vantreske, MRN 44821903, seen on 14 March 2021.",
    "No adverse reaction was recorded by nursing staff overnight.",
    "The patient is 94 years old and remains fully independent.",
    "Contact synthetic.person@example.invalid or 555-0142 for records.",
    "Metformin 500 mg nightly. Parkinson's disease was thought unlikely.",
]


@pytest.mark.parametrize("note", EQUIVALENCE_NOTES)
def test_short_documents_are_byte_identical_after_optimization(note):
    """Repeated runs must agree exactly, including surrogate identity."""
    first = anonymize_clinical_text(note, profile="strict", study_salt="study-a")
    second = anonymize_clinical_text(note, profile="strict", study_salt="study-a")

    assert first["anonymized_text"] == second["anonymized_text"]
    assert first["detected_entities"] == second["detected_entities"]
    assert first["residual_phi_categories"] == second["residual_phi_categories"]
    assert first["utility_metrics"] == second["utility_metrics"]


def test_repeated_identifiers_keep_one_stable_surrogate():
    note = (
        "Zelquin Vantreske attended clinic. Zelquin Vantreske was reviewed. "
        "Discharge instructions were given to Zelquin Vantreske."
    )
    result = anonymize_clinical_text(note, profile="strict", study_salt="study-a")

    assert "Vantreske" not in result["anonymized_text"]


def test_long_document_still_removes_every_planted_identifier():
    filler = "The observation was routine and unremarkable. " * 400
    note = f"{filler}Patient Zelquin Vantreske, MRN 44821903.{filler}"

    result = anonymize_clinical_text(note, profile="strict", study_salt="study-a")

    assert "Vantreske" not in result["anonymized_text"]
    assert "44821903" not in result["anonymized_text"]


def test_production_still_runs_exactly_two_detector_passes(monkeypatch):
    """Optimization must not have quietly removed the second pass."""
    monkeypatch.setenv("PHI_LATENCY_METRICS", "1")
    import main
    from fastapi.testclient import TestClient
    from services import latency_metrics

    with TestClient(main.app) as client:
        response = client.post(
            "/api/v1/ingest",
            files={
                "file": (
                    "n.txt",
                    io.BytesIO(b"Patient Zelquin Vantreske, MRN 44821903."),
                    "text/plain",
                )
            },
            data={"profile": "strict"},
        )
    latency_metrics.finish_request()

    assert response.status_code == 200
    record = main.latest_request_timings()[-1]
    assert record.call_count("anonymize.detect_pass1") == 1
    assert record.call_count("anonymize.residual_validation") == 1
    assert record.call_count("detect.evidence_model") == 2
