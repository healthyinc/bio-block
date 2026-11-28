import hashlib
import os
import re
import string
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from services import numeric_context
from services.clinical_preservation import unattributed_losses
from services.clinical_vocabulary import is_identifier_label_text
from services.detection_evidence import (
    EVIDENCE_DETERMINISTIC,
    EVIDENCE_MODEL,
    NON_IDENTIFIER_CATEGORIES,
    EvidenceAssessment,
    assess_finding,
    contains_proper_noun_token,
    count_agreeing,
    evidence_type,
    has_clinical_support,
    identifier_token_spans,
    is_confirmed_clinical_marker,
    occurrence_support,
    trim_clinical_edges,
)
from services.edit_justification import AMBIGUOUS_MODEL_EDIT, AMBIGUOUS_NUMERIC_VALUE
from services.latency_metrics import stage as _timed_stage
from services.local_model_detectors import (
    MODE_OFFLINE,
    SOURCE_GLINER,
    SOURCE_STANFORD,
    GlinerPiiDetector,
    LocalModelError,
    StanfordClinicalDetector,
    calibrated_config_for,
    resolve_model_mode,
)
from services.model_client import RemoteModelDetector, worker_enabled
from services.ner_phi_detector import (
    NerPhiDetectionError,
    SpacyNerPhiDetector,
    configured_model_name,
    noun_evidence,
    proper_noun_offsets,
)
from services.occurrence import DecisionLog, OccurrenceDecision
from services.phi_detection import (
    SOURCE_CONTEXT_RULE,
    SOURCE_REPEAT_PROPAGATION,
    SOURCE_STRICT_PROPER_NOUN,
    AgeOverThresholdDetector,
    DetectedEntity,
    PhiDetector,
    StructuredPatternDetector,
    expand_partial_identifiers,
    fold_dashes,
    propagate_repeated_identifiers,
    resolve_overlaps,
)
from services.privacy_profiles import (
    PrivacyProfileError,
    get_privacy_profile,
    validate_privacy_profile,
)
from services.request_deadline import (
    RequestDeadlineExceeded,
)
from services.request_deadline import check as _deadline_check
from services.residual_identifier_scan import date_fragments, foreign_token_entities
from services.residual_identifier_scan import scan as scan_structured_identifiers
from services.residual_profile import (
    PASS_FIRST,
    PASS_RESIDUAL,
    ResidualProfileError,
    build_evidence,
)
from services.residual_profile import (
    configured_profile as _configured_validator_profile,
)
from services.residual_profile import required_sources as _required_detector_sources
from services.surrogates import (
    LEGACY_SURROGATE_PATTERN,
    SURROGATE_FAMILIES,
    SURROGATE_PATTERN,
    SurrogateAllocator,
    looks_like_provider,
)
from services.text_utility import measure_text_utility
from services.transformation_provenance import (
    KIND_PLACEHOLDER,
    KIND_SURROGATE,
    ProvenanceBuilder,
    TransformationProvenance,
    kind_for_replacement,
)

SUPPORTED_PROFILES = {"strict", "research"}
STUDY_SALT_ENV_VAR = "BIOBLOCK_STUDY_SALT"
HASH_LENGTH = 8
MAX_TEXT_BYTES = 256 * 1024
#: Safe Harbor aggregation for an age above 89.
AGE_AGGREGATE_REPLACEMENT = "90+"

_ID_ENTITY_TYPES = {
    "MEDICAL_RECORD_NUMBER",
    "PATIENT_ID",
    "HEALTH_PLAN_ID",
    "INSURANCE_ID",
    "ACCESSION_NUMBER",
    "DEVICE_ID",
}
_IDENTIFIER_AT_END = re.compile(r"([A-Z0-9][A-Z0-9-]{3,30})\b", re.IGNORECASE)
DIRECT_IDENTIFIER_REDACTIONS = {
    "PERSON": "<REDACTED_NAME>",
    "MEDICAL_RECORD_NUMBER": "<REDACTED_MRN>",
    "PATIENT_ID": "<REDACTED_PATIENT_ID>",
    "HEALTH_PLAN_ID": "<REDACTED_HEALTH_PLAN>",
    "INSURANCE_ID": "<REDACTED_HEALTH_PLAN>",
    "ACCESSION_NUMBER": "<REDACTED_ACCESSION>",
    "DEVICE_ID": "<REDACTED_DEVICE_ID>",
}


class TextAnonymizationError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def stable_hash(
    value: str,
    salt: str,
    length: int = HASH_LENGTH,
    entity_type: str = "VALUE",
) -> str:
    """Build a research-profile compatibility surrogate, not a Safe Harbor result."""
    normalized = value.strip().lower()
    digest = hashlib.sha256(f"{salt}:{entity_type}:{normalized}".encode("utf-8"))
    return digest.hexdigest().upper()[:length]


def pseudonymize_person(value: str, salt: str) -> str:
    """Research-only compatibility helper; strict processing never calls this."""
    return f"PERSON_{stable_hash(value, salt, entity_type='PERSON')}"


def pseudonymize_mrn(value: str, salt: str) -> str:
    return f"MRN_{stable_hash(value, salt, entity_type='MEDICAL_RECORD_NUMBER')}"


def pseudonymize_patient_id(value: str, salt: str) -> str:
    return f"PATIENT_ID_{stable_hash(value, salt, entity_type='PATIENT_ID')}"


def pseudonymize_health_plan(value: str, salt: str) -> str:
    return f"HEALTH_PLAN_{stable_hash(value, salt, entity_type='HEALTH_PLAN_ID')}"


def pseudonymize_accession(value: str, salt: str) -> str:
    return f"ACCESSION_{stable_hash(value, salt, entity_type='ACCESSION_NUMBER')}"


def pseudonymize_device(value: str, salt: str) -> str:
    return f"DEVICE_{stable_hash(value, salt, entity_type='DEVICE_ID')}"


def _profile_settings(profile: str) -> tuple[str, Dict[str, Any]]:
    try:
        normalized = validate_privacy_profile(profile)
        return normalized, get_privacy_profile(normalized)
    except PrivacyProfileError as exc:
        raise TextAnonymizationError(exc.detail, status_code=exc.status_code) from exc


def _normalize_profile(profile: str) -> str:
    return _profile_settings(profile)[0]


def _detectors(
    model_name: str,
    profile: str,
    pass_kind: str = PASS_FIRST,
    validator_profile: Optional[str] = None,
) -> Tuple[PhiDetector, ...]:
    """Build the detector chain for the currently configured model mode.

    An unrecognized mode raises ``LocalModelError`` rather than silently
    dropping the model adapters.

    ``pass_kind`` selects which of the two production passes this is. Only the
    residual pass consults the validator profile; the first pass always runs
    the complete chain, so no configuration can remove GLiNER from detection.
    """
    chain = _build_detectors(model_name, profile, resolve_model_mode())
    if pass_kind == PASS_FIRST:
        return chain

    wanted = set(
        _required_detector_sources(
            pass_kind, validator_profile or _configured_validator_profile()
        )
    )
    filtered = tuple(
        detector for detector in chain if _detector_source(detector) in wanted
    )
    if not filtered:
        # Every profile names at least one model; an empty chain here means
        # the configuration and the built chain disagree, which must not
        # silently become "nothing found".
        raise NerPhiDetectionError("residual_validator_empty", status_code=500)
    return filtered


def _detector_source(detector: PhiDetector) -> str:
    """The source name a detector reports, or its class as a fallback."""
    return getattr(detector, "source", None) or {
        "StructuredPatternDetector": "structured_pattern",
        "AgeOverThresholdDetector": "age_rule",
        "SpacyNerPhiDetector": "ner",
    }.get(type(detector).__name__, type(detector).__name__)


@lru_cache(maxsize=8)
def _build_detectors(
    model_name: str,
    profile: str,
    model_mode: str,
) -> Tuple[PhiDetector, ...]:
    """Cached detector chain. Model adapters only propose spans."""
    if model_mode == MODE_OFFLINE:
        # Each model carries its own calibrated threshold. They were selected
        # jointly against the combined chain's recall, not in isolation, so
        # they are not interchangeable.
        #
        # The main API environment cannot hold the model stack without
        # downgrading ChromaDB's pins, so when the worker is enabled inference
        # runs out of process. Both paths apply the same calibration; only the
        # execution boundary differs.
        if worker_enabled():
            stanford: PhiDetector = RemoteModelDetector("stanford", SOURCE_STANFORD)
            gliner: PhiDetector = RemoteModelDetector("gliner", SOURCE_GLINER)
        else:
            stanford = StanfordClinicalDetector(calibrated_config_for(SOURCE_STANFORD))
            gliner = GlinerPiiDetector(calibrated_config_for(SOURCE_GLINER))
        return (
            StructuredPatternDetector(),
            AgeOverThresholdDetector(),
            stanford,
            gliner,
            SpacyNerPhiDetector(
                model_name,
                high_recall_proper_nouns=profile == "strict",
            ),
        )
    return (
        StructuredPatternDetector(),
        AgeOverThresholdDetector(),
        SpacyNerPhiDetector(
            model_name,
            high_recall_proper_nouns=profile == "strict",
        ),
    )


#: Evidence from the most recent pass of each kind, for the caller to attach
#: to its result. Request-scoped in practice because a request runs both
#: passes before the next one starts on this thread.
_LAST_VALIDATION_EVIDENCE: Dict[str, Any] = {}
#: The occurrence decisions of the pass in progress, per context (thread and job).
_DECISIONS: ContextVar[Optional[DecisionLog]] = ContextVar(
    "_occurrence_decisions", default=None
)
#: Each call's pass summaries, keyed by pass. A holder per call, not a module
#: dict: two jobs anonymizing at once must never report each other's counts.
#: Mutable, so a detection step run in a copied context still writes to it.
_DECISION_SUMMARIES: ContextVar[Optional[Dict[str, Dict[str, Any]]]] = ContextVar(
    "_occurrence_decision_summaries", default=None
)


def occurrence_summary(pass_kind: str) -> Dict[str, Any]:
    """Value-free counts of this call's occurrence decisions for one pass."""
    holder = _DECISION_SUMMARIES.get()
    return dict((holder or {}).get(pass_kind) or {})


def _decide(
    entity,
    text: str,
    action: str,
    reason: str,
    pass_kind: str,
    agreeing: int = 1,
    clinical: str = "",
    conflict: str = "",
) -> None:
    log = _DECISIONS.get()
    if log is None:
        return
    log.add(
        OccurrenceDecision(
            surface="text",
            location=f"{pass_kind}:offset",
            start=entity.start,
            end=entity.end,
            entity_type=entity.entity_type,
            detector=entity.source,
            deterministic=evidence_type(entity.source) == EVIDENCE_DETERMINISTIC,
            neural_score=getattr(entity, "score", None),
            clinical_evidence=clinical,
            context=numeric_context.segment(text, entity.start, entity.end),
            conflict=conflict,
            action=action,
            reason=reason,
            replacement_family=SURROGATE_FAMILIES.get(entity.entity_type, ""),
            provenance=pass_kind,
            agreeing_detectors=agreeing,
        )
    )


def validation_evidence(pass_kind: str):
    """Evidence recorded by the most recent pass of this kind, if any."""
    return _LAST_VALIDATION_EVIDENCE.get(pass_kind)


#: Off unless a caller opens :func:`decision_trace`. Evaluation harnesses need
#: to know which rule decided each candidate; production callers must not pay
#: for a record they never read, and the request path never opens one.
_DECISION_TRACE: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "phi_decision_trace", default=None
)


@contextmanager
def decision_trace() -> Iterator[List[Dict[str, Any]]]:
    """Record the evidence verdict on every candidate, for evaluation only.

    Each record carries the pass, the candidate's offsets in the text that
    pass examined, and the assessment: category, detector, evidence type,
    reading, action and rationale. It never carries the span's text, so a
    trace of a document full of identifiers is still safe to write down.
    """
    records: List[Dict[str, Any]] = []
    token = _DECISION_TRACE.set(records)
    try:
        yield records
    finally:
        _DECISION_TRACE.reset(token)


def _trace_assessment(pass_kind: str, entity: DetectedEntity, assessment) -> None:
    records = _DECISION_TRACE.get()
    if records is None:
        return
    records.append(
        {
            "pass": pass_kind,
            "start": int(entity.start),
            "end": int(entity.end),
            "proposed_category": entity.entity_type,
            **assessment.to_public_dict(),
        }
    )


def _trace_span(stage: str, entity: DetectedEntity, text: str, **fields) -> None:
    """Record one value-free span during an opted-in diagnostic trace."""
    records = _DECISION_TRACE.get()
    if records is None:
        return
    records.append(
        {
            "pass": stage,
            "start": int(entity.start),
            "end": int(entity.end),
            "length": int(entity.end - entity.start),
            "shape": normalized_shape(text[entity.start : entity.end]),
            "proposed_category": entity.entity_type,
            "detector": entity.source,
            "original_label": entity.original_label,
            **fields,
        }
    )


#: Runs of one character class, so a shape says how a span is written without
#: saying what it says. "900-12-3456" becomes "d{3}-d{2}-d{4}" and "Dr Smith"
#: becomes "A a{1} A a{4}" - enough to recognise a rule's mistake, never
#: enough to recover the value.
_SHAPE_CLASSES = (
    (str.isdigit, "d"),
    (str.isupper, "A"),
    (str.islower, "a"),
)


def normalized_shape(span: str, limit: int = 64) -> str:
    """A character-class skeleton of a span, carrying no value."""
    pieces: List[str] = []
    for character in span[:limit]:
        symbol = next(
            (mark for test, mark in _SHAPE_CLASSES if test(character)),
            " " if character.isspace() else character,
        )
        if pieces and pieces[-1][0] == symbol and symbol in "daA":
            head, _, count = pieces[-1].partition("*")
            pieces[-1] = f"{head}*{int(count or 1) + 1}"
        else:
            pieces.append(symbol)
    shaped = "".join(
        piece if "*" not in piece else f"{piece[0]}{{{piece.split('*')[1]}}}"
        for piece in pieces
    )
    return shaped + ("..." if len(span) > limit else "")


def _trace_residual(
    finding: Dict[str, Any],
    start: int,
    end: int,
    shape: str,
    text: str = "",
    context: str = "",
) -> None:
    """Record value-free residual coordinates only in an explicitly opened trace."""
    records = _DECISION_TRACE.get()
    if records is None:
        return
    records.append(
        {
            "pass": "residual_finding",
            "start": int(start),
            "end": int(end),
            "shape": shape,
            **finding,
        }
    )


def _detect_entities(
    text: str,
    model_name: str,
    profile: str,
    pass_kind: str = PASS_FIRST,
    validator_profile: Optional[str] = None,
) -> List[DetectedEntity]:
    detected: List[DetectedEntity] = []
    try:
        validator_profile = validator_profile or _configured_validator_profile()
    except ResidualProfileError as exc:
        # A misconfigured profile blocks rather than selecting a default.
        raise NerPhiDetectionError(exc.error_code, status_code=500) from exc

    completed: List[str] = []
    try:
        # Built inside the guard: an unknown model mode raises LocalModelError
        # here, and that must convert to the fail-closed detection error
        # rather than escaping as a different exception type.
        chain = _detectors(model_name, profile, pass_kind, validator_profile)
        # What this chain is expected to report. Derived from the chain
        # actually built, because the mocked test mode legitimately has no
        # pinned models; the profile's own requirement is checked below.
        expected = tuple(_detector_source(detector) for detector in chain)
        for detector in chain:
            # One stage per detector, named after the source it reports, so a
            # slow model is attributable rather than hidden inside a chain
            # total. These are children of the surrounding pass.
            # Named by the detector's own source where it declares one, and
            # by its class otherwise, so every stage is attributable.
            _detector_name = (
                getattr(detector, "source", None) or type(detector).__name__
            )
            # Between detectors is the one place a stuck chain can be
            # stopped cheaply. It cannot interrupt a detector already inside
            # a model call; the worker client's read timeout does that.
            _deadline_check(f"detect.{_detector_name}")
            with _timed_stage(f"detect.{_detector_name}"):
                proposed = detector.detect(text)
                detected.extend(proposed)
                for entity in proposed:
                    _trace_span(f"{pass_kind}.detector_candidate", entity, text)
            # Recorded only after the detector returns, so a detector that
            # raised cannot be counted as having covered the text.
            completed.append(_detector_source(detector))
    except LocalModelError as exc:
        # Model load, checksum, inference, and timeout failures block the
        # artifact; they never degrade to returning unredacted content.
        raise NerPhiDetectionError(exc.error_code, exc.status_code) from exc
    except NerPhiDetectionError:
        raise
    except RequestDeadlineExceeded as exc:
        # Out of time blocks exactly as a model failure does. Keeping its own
        # code makes an over-long request distinguishable from a broken model
        # in metrics; both produce no artifact.
        raise NerPhiDetectionError(exc.error_code, status_code=exc.status_code) from exc
    except Exception as exc:
        raise NerPhiDetectionError("phi_detection_failed", status_code=500) from exc

    # Two separate checks. First: every detector in the built chain reported.
    # A detector that silently produced nothing is indistinguishable from a
    # clean document, so partial coverage must never be reported as one.
    if set(expected) - set(completed):
        raise NerPhiDetectionError("incomplete_detector_coverage", status_code=500)

    # Second: when the pinned models are in play, the chain must actually
    # contain what the profile requires. This is what stops a configuration
    # or wiring mistake from quietly running a thinner validator.
    if resolve_model_mode() == MODE_OFFLINE:
        required = set(_required_detector_sources(pass_kind, validator_profile))
        if required - set(completed):
            raise NerPhiDetectionError("incomplete_detector_coverage", status_code=500)
    _LAST_VALIDATION_EVIDENCE[pass_kind] = build_evidence(
        pass_kind, validator_profile, tuple(completed)
    )

    if pass_kind == PASS_FIRST:
        # The residual pass's structured-identifier scan, run here as well:
        # an identifier-shaped value ("MG-3.4-77120" after "Account") must be
        # replaced by the first pass, not preserved by a clinical-token rule
        # and caught only by the second. The second pass still runs it on its
        # own, so the two stay independent.
        deterministic = [
            (entity.start, entity.end)
            for entity in detected
            if evidence_type(entity.source) == EVIDENCE_DETERMINISTIC
        ]
        detected.extend(
            entity
            for entity in scan_structured_identifiers(text)
            if not any(
                low < entity.end and entity.start < high for low, high in deterministic
            )
        )
        # A token shaped like the sanitizer's own, in text it has not written yet,
        # is the document's: it is replaced, not left for a model to claim.
    log = DecisionLog()
    token = _DECISIONS.set(log)
    try:
        with _timed_stage("detect.evidence_model"):
            kept, review = _apply_evidence_model(
                detected,
                text,
                pass_kind,
                pinned_ran={SOURCE_STANFORD, SOURCE_GLINER} <= set(completed),
            )
    finally:
        _DECISIONS.reset(token)
    kept = _merge_overlapping_name_candidates(kept, text)
    for entity in kept:
        _trace_span(f"{pass_kind}.evidence_kept", entity, text)
    with _timed_stage("detect.normalize_merge"):
        if pass_kind == PASS_FIRST:
            # A model can stop inside an identifier, and a confirmed value can
            # recur where no detector looked. Both only widen what is removed;
            # neither can hand a span back to the output. The residual pass
            # runs its own independent scan instead, so a mistake here cannot
            # hide an identifier from both passes.
            kept = expand_partial_identifiers(kept, text)
            kept = kept + propagate_repeated_identifiers(kept, text)
            # A short number nothing explains - no contact label, no clinical
            # structure - is replaced in the candidate and the file is held.
            propagated = [
                entity for entity in kept if entity.source == SOURCE_REPEAT_PROPAGATION
            ]
            ambiguous = _unresolved_numbers(text, kept, detected)
            if ambiguous:
                kept = kept + ambiguous
                review = sorted(set(review) | {AMBIGUOUS_NUMERIC_VALUE})
            token = _DECISIONS.set(log)
            try:
                for entity in propagated:
                    _decide(
                        entity,
                        text,
                        "replace",
                        "same_identifier_own_context_not_clinical",
                        pass_kind,
                    )
                for entity in ambiguous:
                    _decide(entity, text, "hold", "unresolved_short_number", pass_kind)
            finally:
                _DECISIONS.reset(token)
        holder = _DECISION_SUMMARIES.get()
        if holder is not None:
            holder[pass_kind] = log.summary()
        resolved = _merge_adjacent_same_type(resolve_overlaps(kept, len(text)), text)
        for entity in resolved:
            _trace_span(f"{pass_kind}.conflict_resolved", entity, text)
    return resolved, review


#: Labels that a clinical term can be wrongly assigned. A detection of one of
#: these over recorded clinical vocabulary is a misclassification, whichever
#: detector produced it.
_NAME_SHAPED_TYPES = frozenset(
    {"PERSON", "ORGANIZATION", "FACILITY", "LOCATION", "ADDRESS", "MEDICAL_CONDITION"}
)


def _without_edge_punctuation(entity: DetectedEntity, text: str) -> DetectedEntity:
    """A model span without the full stop or comma it swallowed at either edge.

    "U/L." read as a span escaped the measurement it belongs to, because the
    full stop lies outside the result-with-unit; judged without it, the unit
    is inside its measurement and is kept. Deterministic matches are exact.
    """
    if evidence_type(entity.source) == EVIDENCE_DETERMINISTIC:
        return entity
    if not 0 <= entity.start < entity.end <= len(text):
        return entity
    start, end = entity.start, entity.end
    while end > start and text[end - 1] in ".,;:":
        end -= 1
    while start < end and text[start] in ".,;:":
        start += 1
    if start >= end or (start, end) == (entity.start, entity.end):
        return entity
    return DetectedEntity(
        entity_type=entity.entity_type,
        start=start,
        end=end,
        source=entity.source,
        score=entity.score,
        original_label=entity.original_label,
    )


def _unresolved_numbers(text: str, kept, detected) -> List[DetectedEntity]:
    """Standalone short numbers with neither a contact label nor clinical structure.

    Judged one occurrence at a time. Typed table cells are resolved by their
    format handler before they reach this function; an untyped value remains
    ambiguous even when it is the whole input.
    """
    if not text:
        return []
    # A model proposal is not evidence that a number is benign.  Only exact
    # deterministic evidence may suppress the independent ambiguity rule.
    covered = [
        (entity.start, entity.end)
        for entity in list(kept) + list(detected)
        if evidence_type(entity.source) == EVIDENCE_DETERMINISTIC
    ]
    # A number wholly inside a span the evidence model is already replacing
    # ("2026" inside the date "March 14, 2026") is not ambiguous: it goes with
    # its enclosing identifier. Left as its own candidate it would outrank that
    # wider model span in overlap resolution and strand "March 14,".
    enclosing = [
        (entity.start, entity.end)
        for entity in kept
        if entity.entity_type not in NON_IDENTIFIER_CATEGORIES
        and entity.source != SOURCE_NUMERIC_CONTEXT
    ]
    found = []
    for start, end in numeric_context.unresolved_numbers(text):
        if any(low < end and start < high for low, high in covered):
            continue
        if any(
            low <= start and end <= high and (low, high) != (start, end)
            for low, high in enclosing
        ):
            continue
        found.append(
            DetectedEntity(
                entity_type="AMBIGUOUS_NUMBER",
                start=start,
                end=end,
                source=SOURCE_NUMERIC_CONTEXT,
                original_label="unresolved_short_number",
            )
        )
    return found


def _without_clinical_edges(entity: DetectedEntity, text: str) -> DetectedEntity:
    """The entity with readings, scores and bed numbers cut off its edges.

    Only a model's span is trimmed; a deterministic match is exactly the
    identifier it matched. See detection_evidence.trim_clinical_edges.
    """
    if evidence_type(entity.source) == EVIDENCE_DETERMINISTIC:
        return entity
    start, end = trim_clinical_edges(text, entity.start, entity.end, entity.entity_type)
    if start >= end or (start, end) == (entity.start, entity.end):
        return entity
    return DetectedEntity(
        entity_type=entity.entity_type,
        start=start,
        end=end,
        source=entity.source,
        score=entity.score,
        original_label=entity.original_label,
    )


def _codes_inside_clinical_span(
    entity: DetectedEntity,
    text: str,
    pass_kind: str,
) -> List[DetectedEntity]:
    """Identifier-shaped tokens a model handed over inside a clinical span.

    A model calling a span a medical condition takes it out of the eighteen
    categories entirely, and the span is then preserved whole. That is right
    for the words and wrong for a code written among them: GLiNER read
    "palm-vein template PVT123456" as a condition and the biometric
    identifier survived into a released note, twice.

    Each such token is removed on its own, as the unique identifier its syntax
    says it is, and the clinical words keep their place. Nothing else here
    changes: the span itself still goes through the evidence model, which
    preserves it.
    """
    if entity.entity_type not in NON_IDENTIFIER_CATEGORIES:
        return []
    codes = []
    for start, end in identifier_token_spans(text, entity.start, entity.end):
        code = DetectedEntity(
            entity_type="OTHER_UNIQUE_ID",
            start=start,
            end=end,
            source=entity.source,
            score=entity.score,
            original_label=entity.original_label,
        )
        _trace_assessment(
            pass_kind,
            code,
            EvidenceAssessment(
                "OTHER_UNIQUE_ID",
                entity.source,
                EVIDENCE_MODEL,
                "identifier",
                "redact",
                "identifier_syntax_inside_a_non_identifier_span",
            ),
        )
        codes.append(code)
    return codes


def _pinned_models_confirm(
    entities: List[DetectedEntity], start: int, end: int
) -> bool:
    """Whether a pinned model proposed a name-shaped reading overlapping this span."""
    return any(
        other.source in (SOURCE_STANFORD, SOURCE_GLINER)
        and other.entity_type in _NAME_SHAPED_TYPES
        and other.start < end
        and start < other.end
        for other in entities
    )


def _apply_evidence_model(
    entities: List[DetectedEntity],
    text: str,
    pass_kind: str = PASS_FIRST,
    pinned_ran: bool = False,
) -> Tuple[List[DetectedEntity], List[str]]:
    """Decide each candidate by evidence strength rather than word membership.

    Applied to every source. The pinned models label "Parkinson" a PERSON as
    readily as spaCy does, so filtering only the detectors we control leaves
    the same clinical term destroyed by a different route.

    Returns the spans to redact and the review reasons raised by candidates
    that could not be resolved either way. An ambiguous span is neither
    deleted on suspicion nor kept on hope: the document goes to a human.
    """
    kept: List[DetectedEntity] = []
    review: List[str] = []
    #: Uncertain readings, decided once every candidate has been assessed.
    uncertain: List[DetectedEntity] = []
    # Parsed once per document, not once per candidate. None means the parse
    # failed, which the evidence model treats as no information rather than
    # as "no proper nouns here".
    parsed = noun_evidence(text, configured_model_name())
    proper_nouns, opening_modifiers = parsed if parsed is not None else (None, ())
    unjustified: List[DetectedEntity] = []
    justified: List[DetectedEntity] = []
    # What an exact identifier match already replaces. A model span that takes an
    # exact match along with the words around it ("AHPRA MED0001234567" over the
    # registration number) is judged on those words alone: the match is replaced
    # on its own evidence, and the label beside it is not part of what it replaces.
    exact_spans = [
        (entity.start, entity.end)
        for entity in entities
        if evidence_type(entity.source) == EVIDENCE_DETERMINISTIC
        and entity.entity_type not in NON_IDENTIFIER_CATEGORIES
    ]

    def normalised_parts(candidate: DetectedEntity) -> List[DetectedEntity]:
        pieces: List[DetectedEntity] = []
        # Cut once on the span as the model drew it, and again once it has been
        # normalised. The second cut matters: a model that stops inside a
        # registration number ("AHPRA MED") is widened to the whole word, which is
        # what takes the number into a span that began at its label. The first
        # matters too: it judges a span that began at a label before its edges are
        # trimmed against readings and clinical words.
        for drawn in _outside_exact_matches(
            _line_parts(candidate, text), exact_spans, text
        ):
            part = _whole_words(drawn, text)
            part = _without_edge_punctuation(part, text)
            part = _without_clinical_edges(part, text)
            part = _with_whole_tokens(part, text)
            pieces.extend(_outside_exact_matches([part], exact_spans, text))
        return pieces

    support_cache: Dict[str, Any] = {}

    def support_of(entity) -> Any:
        surface = text[entity.start : entity.end].strip()
        if surface not in support_cache:
            support_cache[surface] = occurrence_support(
                entities, text, entity.start, entity.end
            )
        return support_cache[surface]

    for candidate in entities:
        for entity in normalised_parts(candidate):
            kept.extend(_codes_inside_clinical_span(entity, text, pass_kind))
            agreeing = count_agreeing(
                entities,
                entity.start,
                entity.end,
                entity.entity_type,
                exclude_source=entity.source,
            )
            assessment = assess_finding(
                text,
                entity.start,
                entity.end,
                entity.entity_type,
                entity.source,
                agreeing_detectors=agreeing,
                clinical_support=has_clinical_support(
                    entities, entity.start, entity.end
                ),
                contains_proper_noun=contains_proper_noun_token(
                    proper_nouns, entity.start, entity.end
                ),
                opening_modifier=any(
                    low <= entity.start and entity.end <= high
                    for low, high in opening_modifiers
                ),
                # Both pinned models read this text and neither proposed a name
                # for this span: a capitalisation-only claim stands alone.
                opening_unconfirmed=pinned_ran
                and not _pinned_models_confirm(entities, entity.start, entity.end),
                occurrence_support=lambda entity=entity: support_of(entity),
            )
            _trace_assessment(pass_kind, entity, assessment)
            rationale = str(getattr(assessment, "rationale", "") or "")
            _decide(
                entity,
                text,
                {"redact": "replace", "review": "hold"}.get(
                    assessment.action, "retain"
                ),
                rationale,
                pass_kind,
                agreeing,
                clinical=rationale if assessment.action == "preserve" else "",
                conflict=(
                    "unjustified" if not getattr(assessment, "justified", True) else ""
                ),
            )
            if assessment.action == "redact":
                if assessment.category != entity.entity_type:
                    # The evidence named what the span is: a code a model
                    # called a phone number is replaced as a unique
                    # identifier, not given a phone placeholder.
                    entity = DetectedEntity(
                        entity_type=assessment.category,
                        start=entity.start,
                        end=entity.end,
                        source=entity.source,
                        score=entity.score,
                        original_label=entity.original_label,
                    )
                kept.append(entity)
                (
                    justified if getattr(assessment, "justified", True) else unjustified
                ).append(entity)
            elif assessment.action == "review":
                # An uncertain model reading wholly inside an exact
                # deterministic identifier cannot survive the transformation:
                # the enclosing span is already replaced.  Stanford can, for
                # example, call the numeric part of an email a phone number.
                # Holding that document adds no safety because the complete
                # email match removes the uncertain inner span either way.
                enclosed_by_exact_identifier = any(
                    other.start <= entity.start
                    and other.end >= entity.end
                    and evidence_type(other.source) == EVIDENCE_DETERMINISTIC
                    and other.entity_type not in NON_IDENTIFIER_CATEGORIES
                    for other in entities
                )
                if not enclosed_by_exact_identifier:
                    uncertain.append(entity)
            # "preserve" keeps the clinical content and adds no review reason.
    for entity in uncertain:
        # The same reasoning for any span that is being replaced: "01103" in
        # an address the model redacts whole, read by another model as "03" a
        # phone number, is gone from the output either way. Only an uncertain
        # span wholly inside a replaced identifier is excused; one that
        # reaches past it, or that nothing replaces, still goes to a person.
        enclosed_by_replacement = any(
            other.start <= entity.start
            and other.end >= entity.end
            and other.entity_type not in NON_IDENTIFIER_CATEGORIES
            for other in kept
        )
        if not enclosed_by_replacement:
            review.append(f"ambiguous_{entity.entity_type.lower()}_requires_review")
    for entity in unjustified:
        # Applied, privacy first. Excused only where a justified replacement
        # of the same text already removes it; otherwise the edit rests on a
        # model's label alone and a person decides before anything releases.
        covered = any(
            other.start <= entity.start and other.end >= entity.end
            for other in justified
        ) or any(
            # An exact identifier match over the same text replaces it
            # whatever the model called it: "CGA-112-4471" read as an
            # organization is inside the accession number the regex found.
            other.start <= entity.start
            and other.end >= entity.end
            and evidence_type(other.source) == EVIDENCE_DETERMINISTIC
            and other.entity_type not in NON_IDENTIFIER_CATEGORIES
            for other in entities
        )
        if not covered and pass_kind == PASS_FIRST:
            review.append(AMBIGUOUS_MODEL_EDIT)
    return kept, sorted(set(review))


#: Provenance of an unresolved-number replacement.
SOURCE_NUMERIC_CONTEXT = "numeric_context"

_WORD_CHARACTER = re.compile(r"[A-Za-z0-9'’]")


_BRACKETED_TOKEN = re.compile(
    r"<(?:[A-Z][A-Z]*(?:_[A-Z]+)*_\d{1,6}|REDACTED(?:_[A-Z]+)+)>"
)


def _whole_words(entity: DetectedEntity, text: str) -> DetectedEntity:
    """A model span widened to whole words: "Knee M" is assessed as "Knee MRI".

    A span that stops inside a word leaves a fragment ("RI") behind for the
    next reader to misjudge, and is judged on a word nobody wrote.
    Deterministic matches are exactly what their expression matched.
    """
    if evidence_type(entity.source) == EVIDENCE_DETERMINISTIC:
        return entity
    if not 0 <= entity.start < entity.end <= len(text):
        return entity
    start, end = entity.start, entity.end

    def joined(left: str, right: str) -> bool:
        # One word continuing, not two run together: "SmithSarah" is a
        # lower-to-upper join between two names and is never widened across.
        return bool(
            _WORD_CHARACTER.match(left) and _WORD_CHARACTER.match(right)
        ) and not (left.islower() and right.isupper())

    while start > 0 and joined(text[start - 1], text[start]):
        start -= 1
    while end < len(text) and joined(text[end - 1], text[end]):
        end += 1
    if entity.entity_type == "PERSON":
        # A hyphenated surname is one name: "Ashgrove-Nwankwo" read only to
        # the hyphen leaves "-Nwankwo" for another model to call a facility.
        # Only a hyphen between two capitalised name parts joins them.
        while (
            end + 1 < len(text)
            and text[end] == "-"
            and text[end + 1].isupper()
            and text[end - 1].isalpha()
        ):
            end += 2  # the hyphen and the capital after it
            while end < len(text) and joined(text[end - 1], text[end]):
                end += 1
        while start > 1 and text[start - 1] == "-" and text[start].isupper():
            hyphen = start - 1
            word_start = hyphen
            while (
                word_start > 0
                and _WORD_CHARACTER.match(text[word_start - 1])
                and (
                    word_start == hyphen
                    or joined(text[word_start - 1], text[word_start])
                )
            ):
                word_start -= 1
            if word_start == hyphen or not text[word_start].isupper():
                break
            start = word_start
    # Inside a token-shaped run ("<PERSON_2>"), take the brackets too, so no
    # stray "<" or ">" is left beside the replacement.
    start, end = _span_with_whole_tokens(text, start, end)
    if (start, end) == (entity.start, entity.end):
        return entity
    return DetectedEntity(
        entity_type=entity.entity_type,
        start=start,
        end=end,
        source=entity.source,
        score=entity.score,
        original_label=entity.original_label,
    )


def _span_with_whole_tokens(text: str, start: int, end: int) -> Tuple[int, int]:
    """The span widened over every placeholder token it only partly covers.

    "URL_2>" or the text up to a token's first character would leave a stray
    bracket beside the replacement ("<<ORGANIZATION_1>"), so any overlap takes
    the whole token.
    """
    window_start, window_end = max(0, start - 64), min(len(text), end + 64)
    for token in _BRACKETED_TOKEN.finditer(text, window_start, window_end):
        if token.start() < end and start < token.end():
            start, end = min(start, token.start()), max(end, token.end())
    return start, end


def _with_whole_tokens(entity: DetectedEntity, text: str) -> DetectedEntity:
    """A model span, after any trimming, never ends inside a placeholder token."""
    if evidence_type(entity.source) == EVIDENCE_DETERMINISTIC:
        return entity
    if not 0 <= entity.start < entity.end <= len(text):
        return entity
    start, end = _span_with_whole_tokens(text, entity.start, entity.end)
    if (start, end) == (entity.start, entity.end):
        return entity
    return DetectedEntity(
        entity_type=entity.entity_type,
        start=start,
        end=end,
        source=entity.source,
        score=entity.score,
        original_label=entity.original_label,
    )


def _line_parts(entity: DetectedEntity, text: str) -> List[DetectedEntity]:
    """A model span cut at line breaks and table cells, one entity per part.

    No identifier runs across a line break or a table cell boundary - a
    single "|" with space on both sides - but models draw spans that do:
    Stanford read the end of a reference on one line and the clinical score
    opening the next as a single identifier, and replacing it removed the
    score. Each line is assessed on its own, so a part that is an identifier
    is still replaced and a part that is not is kept. A deterministic match
    is exactly what its expression matched and is left whole, as is any span
    that does not lie inside this text.
    """
    if evidence_type(entity.source) == EVIDENCE_DETERMINISTIC:
        return [entity]
    if not 0 <= entity.start < entity.end <= len(text):
        return [entity]
    segment = text[entity.start : entity.end]
    cuts = [
        index
        for index, character in enumerate(segment)
        if character == "\n" or _is_cell_separator(segment, index)
    ]
    if not cuts:
        return [entity]
    parts: List[DetectedEntity] = []
    bounds = [-1] + cuts + [len(segment)]
    for left, right in zip(bounds, bounds[1:]):
        line = segment[left + 1 : right]
        offset = entity.start + left + 1
        stripped = line.strip()
        if stripped:
            start = offset + line.index(stripped)
            parts.append(
                DetectedEntity(
                    entity_type=entity.entity_type,
                    start=start,
                    end=start + len(stripped),
                    source=entity.source,
                    score=entity.score,
                    original_label=entity.original_label,
                )
            )
    return parts


def _outside_exact_matches(
    parts: List[DetectedEntity],
    exact_spans: List[Tuple[int, int]],
    text: str,
) -> List[DetectedEntity]:
    """A model part that is a few words and an exact match, judged on the words alone.

    A deterministic part is exactly what its expression matched and is returned
    whole. A model part that begins or ends with an exact span - "AHPRA
    MED0001234567", "Dr Smith 4417" - and has at most three words besides it is
    cut at that span, and the remainder is judged on its own: the match is
    replaced on its own evidence, and the label or title beside it is not part of
    what it replaces. A span that holds the match in its middle, or has more
    words around it (an address that ends in a code), is judged whole as before.
    """
    result: List[DetectedEntity] = []
    for part in parts:
        if evidence_type(part.source) == EVIDENCE_DETERMINISTIC or not exact_spans:
            result.append(part)
            continue
        cut = None
        for low, high in exact_spans:
            if not (part.start <= low < high <= part.end) or (low, high) == (
                part.start,
                part.end,
            ):
                continue
            if low == part.start:
                remainder = (high, part.end)
            elif high == part.end:
                remainder = (part.start, low)
            else:
                continue
            if len(text[remainder[0] : remainder[1]].split()) <= 3:
                cut = remainder
                break
        if cut is None:
            result.append(part)
            continue
        piece = text[cut[0] : cut[1]]
        stripped = piece.strip()
        if not stripped:
            continue
        start = cut[0] + len(piece) - len(piece.lstrip())
        result.append(
            DetectedEntity(
                entity_type=part.entity_type,
                start=start,
                end=start + len(stripped),
                source=part.source,
                score=part.score,
                original_label=part.original_label,
            )
        )
    return result


def _is_cell_separator(segment: str, index: int) -> bool:
    """A lone "|" with whitespace or the segment edge on both sides.

    A doubled "||" and a tight "a|b" can be part of an identifier and are
    never cuts.
    """
    if segment[index] != "|":
        return False
    before = segment[index - 1] if index > 0 else " "
    after = segment[index + 1] if index + 1 < len(segment) else " "
    return before in " \t" and after in " \t"


#: Only horizontal whitespace, a hyphen or a possessive may sit between two
#: fragments of one name. Anything else is two different entities - and a line
#: break in particular is not a gap inside a name. Allowing one merged
#: "Template FP-TEMPLATE-52123" and the "Fax" on the next line into a single
#: span, and the replacement swallowed a whole field.
#: The hyphen is escaped: written bare between a tab and an apostrophe it made the
#: class the range TAB..APOSTROPHE, which contains the line feed and the carriage
#: return, so two spans on adjacent lines were merged and the line break between
#: them was replaced along with them. The end is ``\Z``, not ``$``: ``$`` also
#: matches before a final line feed, so a gap that was only a line break passed.
_NAME_GAP = re.compile(r"^[ \t\-']*\Z")
_NAME_CONTEXT_PREFIX = re.compile(
    r"^\s*(?:(?:dr|mr|mrs|ms|professor)\.?|patient(?:\s+name)?|name|"
    r"attending\s+physician)\s*:?\s*$",
    re.IGNORECASE,
)


def _merge_overlapping_name_candidates(
    entities: List[DetectedEntity], text: str
) -> List[DetectedEntity]:
    """Union overlapping identifier-name readings before priority resolution.

    A short context-rule PERSON span must not erase a complete PERSON span from
    a pinned model merely because rules rank above models. Components are
    merged only when every member is name-shaped and at least one says PERSON;
    clinical candidates never enter ``entities`` and cannot widen the result.
    """
    ordered = sorted(
        (entity for entity in entities if entity.entity_type in _NAME_SHAPED_TYPES),
        key=lambda e: (e.start, e.end, e.entity_type),
    )
    output: List[DetectedEntity] = [
        entity for entity in entities if entity.entity_type not in _NAME_SHAPED_TYPES
    ]
    index = 0
    while index < len(ordered):
        component = [ordered[index]]
        stop = ordered[index].end
        index += 1
        while index < len(ordered) and ordered[index].start < stop:
            component.append(ordered[index])
            stop = max(stop, ordered[index].end)
            index += 1
        categories = {entity.entity_type for entity in component}
        # A capitalisation rule's PERSON reading is the weakest evidence in a
        # component. Beside a model's or a rule's reading of the same text as
        # something else - an address, a place, an organisation - it must not
        # retype the whole span as a person: "42 Example Street" becomes a
        # location, not a patient. A PERSON that any other source also holds
        # still merges as before.
        heuristic_person_only = (
            "PERSON" in categories
            and all(
                entity.source == SOURCE_STRICT_PROPER_NOUN
                for entity in component
                if entity.entity_type == "PERSON"
            )
            and any(
                entity.entity_type != "PERSON"
                and entity.source != SOURCE_STRICT_PROPER_NOUN
                for entity in component
            )
        )
        if len(component) > 1 and "PERSON" in categories and not heuristic_person_only:
            people = [entity for entity in component if entity.entity_type == "PERSON"]
            representative = max(
                people,
                key=lambda entity: (
                    entity.source == SOURCE_CONTEXT_RULE,
                    entity.end - entity.start,
                    -entity.start,
                    entity.source,
                ),
            )
            merged_start = min(entity.start for entity in component)
            context_starts = [
                entity.start
                for entity in people
                if entity.source == SOURCE_CONTEXT_RULE
            ]
            if context_starts:
                contextual_start = min(context_starts)
                # Models sometimes include the title or field label in their
                # PERSON span.  Keep that prefix outside the replacement so
                # provider classification can still see "Dr." and explicit
                # fields keep their readable label.  A genuine earlier name
                # token is never trimmed: only a recognized context prefix is.
                if merged_start < contextual_start and _NAME_CONTEXT_PREFIX.fullmatch(
                    text[merged_start:contextual_start]
                ):
                    merged_start = contextual_start
            # A naming-context rule says person outright. Without one, the
            # reading that covers the most text names what the span is: a
            # model span for "14 Marlowe Terrace, Ashbourne, DE6 1QX" is an
            # address even if a shorter reading of "Marlowe" says person.
            merged_type = "PERSON"
            if not context_starts:
                longest = max(
                    component,
                    key=lambda e: (e.end - e.start, e.entity_type == "PERSON"),
                )
                if (
                    longest.entity_type != "PERSON"
                    and longest.end - longest.start
                    > max(p.end - p.start for p in people)
                ):
                    merged_type, representative = longest.entity_type, longest
            merged = DetectedEntity(
                entity_type=merged_type,
                start=merged_start,
                end=max(e.end for e in component),
                source=representative.source,
                score=representative.score,
                original_label=representative.original_label,
            )
            _trace_span(
                "name_overlap_merged",
                merged,
                text,
                merged_candidates=len(component),
                merged_categories=sorted(categories),
                merged_sources=sorted({entity.source for entity in component}),
            )
            output.append(merged)
        else:
            output.extend(component)
    return sorted(
        output, key=lambda entity: (entity.start, entity.end, entity.entity_type)
    )


def _merge_adjacent_same_type(
    entities: List[DetectedEntity],
    text: str,
) -> List[DetectedEntity]:
    """Join same-type spans split across a name by different detectors.

    Two detectors proposing "Padmavathi" and "Venkataraghavan" separately
    resolve to two adjacent spans, which would then receive two different
    surrogates for one person. Merging them keeps one person as one entity.
    """
    if not entities:
        return entities
    ordered = sorted(entities, key=lambda e: (e.start, e.end))
    merged: List[DetectedEntity] = [ordered[0]]
    for entity in ordered[1:]:
        previous = merged[-1]
        gap = text[previous.end : entity.start]
        if (
            entity.entity_type == previous.entity_type
            and entity.entity_type in _NAME_SHAPED_TYPES
            # A real separator must be present. Two spans that touch with no
            # gap at all are two entities written without a space, not one
            # name that detectors split.
            and 1 <= len(gap) <= 2
            and _NAME_GAP.match(gap)
        ):
            merged[-1] = DetectedEntity(
                entity_type=previous.entity_type,
                start=previous.start,
                end=entity.end,
                source=previous.source,
                score=previous.score,
                original_label=previous.original_label,
            )
            continue
        merged.append(entity)
    return merged


def _hash_value(entity_type: str, value: str) -> str:
    if entity_type not in _ID_ENTITY_TYPES:
        return value

    match = _IDENTIFIER_AT_END.search(value.strip())
    if match:
        return match.group(1)
    return value


def _date_shift_days(salt: str) -> int:
    value = int(stable_hash("date-shift", salt, length=6), 16)
    days = value % 731 - 365
    return days or 17


def _shift_date_text(value: str, salt: str) -> str:
    cleaned = value.strip()
    formats = [
        ("%Y-%m-%d", "%Y-%m-%d"),
        ("%m/%d/%Y", "%m/%d/%Y"),
        ("%m/%d/%y", "%m/%d/%Y"),
    ]
    for input_format, output_format in formats:
        try:
            shifted = datetime.strptime(cleaned, input_format) + timedelta(
                days=_date_shift_days(salt)
            )
            return shifted.strftime(output_format)
        except ValueError:
            continue
    return "<REDACTED_DATE>"


def _replacement_for(
    entity_type: str,
    value: str,
    salt: str,
    settings: Dict[str, Any],
    allocator: "SurrogateAllocator | None" = None,
    is_provider: bool = False,
) -> str:
    # Safe Harbor requires ages over 89 to be aggregated rather than removed.
    # "94 years old" becomes "90+ years old", which keeps the clinical fact.
    if entity_type == "AGE_OVER_89":
        return AGE_AGGREGATE_REPLACEMENT
    if entity_type == "AGE_UNCERTAIN":
        # Cannot be resolved to a number, so it is removed and the document is
        # routed to review rather than guessed at in either direction.
        return "<REDACTED_AGE>"

    if settings["text_identifier_strategy"] == "redact":
        # A consistent study-local surrogate preserves coreference where a
        # fixed placeholder would collapse every mention into one token. Any
        # type with a surrogate form uses one, not only the direct-identifier
        # set, so a facility or organisation stays distinguishable too.
        if allocator is not None:
            surrogate_type = (
                "PERSON_PROVIDER"
                if entity_type == "PERSON" and is_provider
                else entity_type
            )
            surrogate = allocator.surrogate_for(surrogate_type, value)
            if surrogate is not None:
                return surrogate
        if entity_type in DIRECT_IDENTIFIER_REDACTIONS:
            return DIRECT_IDENTIFIER_REDACTIONS[entity_type]

    hash_value = _hash_value(entity_type, value)
    if entity_type == "PERSON":
        return pseudonymize_person(hash_value, salt)
    if entity_type == "MEDICAL_RECORD_NUMBER":
        return pseudonymize_mrn(hash_value, salt)
    if entity_type == "PATIENT_ID":
        return pseudonymize_patient_id(hash_value, salt)
    if entity_type in {"HEALTH_PLAN_ID", "INSURANCE_ID"}:
        return pseudonymize_health_plan(hash_value, salt)
    if entity_type == "ACCESSION_NUMBER":
        return pseudonymize_accession(hash_value, salt)
    if entity_type == "DEVICE_ID":
        return pseudonymize_device(hash_value, salt)
    if entity_type == "EMAIL_ADDRESS":
        return "<REDACTED_EMAIL>"
    if entity_type == "PHONE_NUMBER":
        return "<REDACTED_PHONE>"
    if entity_type in {"US_SSN", "SSN"}:
        return "<REDACTED_SSN>"
    if entity_type in {"DATE", "DATE_TIME"}:
        if settings["date_strategy"] == "shift":
            return _shift_date_text(value, salt)
        return "<REDACTED_DATE>"
    if entity_type == "TIME":
        return "<REDACTED_TIME>"
    if entity_type == "URL":
        return "<REDACTED_URL>"
    if entity_type == "IP_ADDRESS":
        return "<REDACTED_IP_ADDRESS>"
    return f"<REDACTED_{entity_type}>"


def _replace_entities(
    text: str,
    entities: Iterable[DetectedEntity],
    salt: str,
    settings: Dict[str, Any],
    allocator: "SurrogateAllocator | None" = None,
    applied: "List[Tuple[int, int, str, str]] | None" = None,
):
    """Rewrite the document and record where each replacement landed.

    Built left to right so the provenance offsets are correct by
    construction. Replacing back to front keeps *input* offsets valid but
    makes every recorded output offset wrong the moment an earlier entity is
    replaced, because that replacement shifts everything after it. Surrogate
    numbering also follows reading order this way, which is what a reader
    expects.
    """
    ordered = sorted(entities, key=lambda item: (item.start, item.end))
    provenance = ProvenanceBuilder()
    pieces: List[str] = []
    output_length = 0
    cursor = 0

    for entity in ordered:
        if entity.start < cursor:
            # Overlap resolution should have removed these, but a stray
            # overlapping span must never corrupt the output.
            continue
        untouched = text[cursor : entity.start]
        pieces.append(untouched)
        output_length += len(untouched)

        original_value = text[entity.start : entity.end]
        is_provider = entity.entity_type == "PERSON" and looks_like_provider(
            text, entity.start
        )
        replacement = _replacement_for(
            entity.entity_type,
            original_value,
            salt,
            settings,
            allocator=allocator,
            is_provider=is_provider,
        )
        pieces.append(replacement)
        if applied is not None:
            # In memory only, for a caller that must place each replacement
            # back into its own cell. Never serialized.
            applied.append((entity.start, entity.end, entity.entity_type, replacement))
        provenance.record(
            output_length,
            len(replacement),
            kind_for_replacement(entity.entity_type, replacement),
            entity.entity_type,
            token=replacement,
        )
        output_length += len(replacement)
        cursor = entity.end

    pieces.append(text[cursor:])
    return "".join(pieces), provenance.build()


def _entity_summary(entities: Iterable[DetectedEntity]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for entity in entities:
        summary[entity.entity_type] = summary.get(entity.entity_type, 0) + 1
    return summary


def _source_summary(entities: Iterable[DetectedEntity]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for entity in entities:
        summary[entity.source] = summary.get(entity.source, 0) + 1
    return summary


def _read_local_env_salt() -> str | None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#") or "=" not in cleaned:
            continue
        key, value = cleaned.split("=", 1)
        if key.strip() == STUDY_SALT_ENV_VAR:
            return value.strip().strip("\"'")

    return None


def _resolve_study_salt(study_salt: str | None) -> str:
    configured_salt = (
        study_salt or os.getenv(STUDY_SALT_ENV_VAR) or _read_local_env_salt()
    )
    if configured_salt and configured_salt.strip():
        return configured_salt.strip()

    raise TextAnonymizationError(
        f"Text anonymization salt is not configured. Set {STUDY_SALT_ENV_VAR}.",
        status_code=500,
    )


def detect_clinical_phi(text: str) -> Dict[str, int]:
    """Return safe entity counts without returning source text or offsets."""
    if not isinstance(text, str):
        raise TextAnonymizationError("Text input must be a string")
    if not text.strip():
        return {}
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise TextAnonymizationError(
            f"Text input exceeds the {MAX_TEXT_BYTES} byte limit",
            status_code=413,
        )

    try:
        entities, _review = _detect_entities(text, configured_model_name(), "strict")
    except NerPhiDetectionError as exc:
        raise TextAnonymizationError(
            exc.error_code,
            status_code=exc.status_code,
        ) from exc
    return _entity_summary(entities)


def anonymize_clinical_text(
    text: str,
    profile: str = "strict",
    study_salt: str | None = None,
) -> Dict[str, Any]:
    result, _entities = _anonymize_clinical_text(text, profile, study_salt)
    return result


def anonymize_clinical_text_with_spans(
    text: str,
    profile: str = "strict",
    study_salt: str | None = None,
    allocator: "SurrogateAllocator | None" = None,
) -> Tuple[Dict[str, Any], List[Tuple[int, int, str]]]:
    """The same result, plus the input offsets of every replaced span.

    For format handlers that must place a redaction on a page or an image.
    The offsets refer only to the transient input text: they are never
    serialized, logged or returned to a caller outside the process.
    """
    result, entities = _anonymize_clinical_text(
        text, profile, study_salt, allocator=allocator
    )
    return result, [(int(e.start), int(e.end), e.entity_type) for e in entities]


def anonymize_clinical_text_with_replacements(
    text: str,
    profile: str = "strict",
    study_salt: str | None = None,
    allocator: "SurrogateAllocator | None" = None,
) -> Tuple[Dict[str, Any], List[Tuple[int, int, str, str]]]:
    """The same full_v1 result, plus each applied replacement and its input offsets.

    For a caller that anonymizes many separate values - the cells of a table -
    in a few calls and must put each replacement back into the value it came
    from, rather than splitting the joined output. ``allocator`` lets those
    calls share one study-local surrogate allocator, so one person keeps one
    surrogate across calls. Detection, transformation and residual validation
    are exactly those of ``anonymize_clinical_text``. Nothing returned here is
    serialized.
    """
    applied: List[Tuple[int, int, str, str]] = []
    result, _entities = _anonymize_clinical_text(
        text, profile, study_salt, allocator=allocator, applied=applied
    )
    return result, applied


def replacement_for_value(
    entity_type: str,
    value: str,
    profile: str = "strict",
    allocator: "SurrogateAllocator | None" = None,
    study_salt: str | None = None,
    is_provider: bool = False,
) -> str:
    """The profile's replacement for one confirmed value, as full_v1 would write it."""
    _privacy_profile, settings = _profile_settings(profile)
    if settings["text_identifier_strategy"] == "redact":
        salt = ""
    else:
        salt = _resolve_study_salt(study_salt)
    return _replacement_for(
        entity_type, value, salt, settings, allocator=allocator, is_provider=is_provider
    )


def _anonymize_clinical_text(
    text: str,
    profile: str = "strict",
    study_salt: str | None = None,
    allocator: "SurrogateAllocator | None" = None,
    applied: "List[Tuple[int, int, str, str]] | None" = None,
) -> Tuple[Dict[str, Any], List[DetectedEntity]]:
    token = _DECISION_SUMMARIES.set({})
    try:
        return _anonymize_clinical_text_in_scope(
            text, profile, study_salt, allocator, applied
        )
    finally:
        _DECISION_SUMMARIES.reset(token)


def _anonymize_clinical_text_in_scope(
    text: str,
    profile: str,
    study_salt: str | None,
    allocator: "SurrogateAllocator | None",
    applied: "List[Tuple[int, int, str, str]] | None",
) -> Tuple[Dict[str, Any], List[DetectedEntity]]:
    if not isinstance(text, str):
        raise TextAnonymizationError("Text input must be a string")
    if not text.strip():
        raise TextAnonymizationError("Text input is empty")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise TextAnonymizationError(
            f"Text input exceeds the {MAX_TEXT_BYTES} byte limit",
            status_code=413,
        )

    privacy_profile, settings = _profile_settings(profile)
    if settings["text_identifier_strategy"] == "redact":
        salt = ""
    else:
        salt = _resolve_study_salt(study_salt)

    try:
        model_name = configured_model_name()
        with _timed_stage("anonymize.detect_pass1"):
            # Detectors and the evidence model read the text with every dash
            # variant written as a hyphen (same offsets): a soft hyphen in a
            # PDF's text layer must not split a surname or a phone number.
            entities, detection_review = _detect_entities(
                fold_dashes(text), model_name, privacy_profile
            )
    except NerPhiDetectionError as exc:
        raise TextAnonymizationError(
            exc.error_code,
            status_code=exc.status_code,
        ) from exc

    # One allocator per call. It dies with the request, so two studies
    # containing the same person get unrelated surrogates and no cross-study
    # linkage is created. The mapping is never returned or persisted.
    # A caller may pass one allocator for a set of calls that together make up
    # one upload, such as the cells of one table.
    if allocator is None and settings["text_identifier_strategy"] == "redact":
        allocator = SurrogateAllocator()
    if allocator is not None:
        # A token already in the source keeps its number; nobody else gets it.
        allocator.reserve_existing(text)
    with _timed_stage("anonymize.transform"):
        anonymized_text, provenance = _replace_entities(
            text, entities, salt, settings, allocator, applied
        )

    # An age reference that cannot be resolved to a number cannot be judged
    # against the Safe Harbor threshold, so the document needs a human.
    review_reasons = sorted(
        {
            "uncertain_age_reference"
            for entity in entities
            if entity.entity_type == "AGE_UNCERTAIN"
        }
        | set(detection_review)
        # The clinical-utility gate: a protected invariant (a unit after a
        # result, a stain or sequence in context) removed by anything but a
        # deterministic identifier rule is a utility loss nobody approved.
        | set(
            unattributed_losses(
                text,
                [
                    (
                        entity.start,
                        entity.end,
                        evidence_type(entity.source) == EVIDENCE_DETERMINISTIC,
                    )
                    for entity in entities
                ],
            )
        )
    )

    # Measured here because this is the only place that holds both the
    # original spans and the output. The removed values are needed to score
    # utility honestly - deleting PHI is the goal, not a utility loss - and
    # they never leave this function: only ratios and counts are returned.
    with _timed_stage("anonymize.utility_measure"):
        utility_metrics = measure_text_utility(
            text,
            anonymized_text,
            redacted_spans=[(e.start, e.end) for e in entities],
        )

    # Scanned once, here, because this is the only place holding the
    # provenance map for the text it just produced.
    # A full detector pass of its own, so its per-detector stages nest
    # beneath this one. Its cost must not be added to anonymize.detect_pass1:
    # the two passes run over different text.
    with _timed_stage("anonymize.residual_validation"):
        second_pass = residual_findings(anonymized_text, provenance)

    result = {
        "anonymization_status": "completed",
        "privacy_profile": privacy_profile,
        "utility_metrics": utility_metrics,
        "date_strategy": settings["date_strategy"],
        "text_identifier_strategy": settings["text_identifier_strategy"],
        "anonymized_text": anonymized_text,
        "entity_count": len(entities),
        "detected_entities": _entity_summary(entities),
        "detection_sources": _source_summary(entities),
        # Counts of distinct entities replaced, per surrogate kind. Never the
        # mapping and never an original value.
        "surrogate_counts": allocator.counts() if allocator else {},
        # Computed here, not by the caller, because this is the only place
        # holding the provenance map. Handing the map out instead would put
        # generated-region offsets into every serialized response, and every
        # caller that forgot to pass it back would silently re-scan surrogates
        # as if they were surviving text - the exact defect Phase 11 removes.
        "residual_phi_categories": _residual_summary(second_pass),
        # The classified second-pass record for each finding: detector,
        # category, evidence type, location type, whether it overlaps a
        # generated region, and the classification. No value, no offsets, no
        # sentence - a diagnostic report on a privacy pipeline is exactly the
        # document that ends up pasted into a ticket.
        "residual_findings": second_pass,
        # How many regions the sanitizer wrote, by kind. Counts only: the
        # offsets stay inside this function, where the validator uses them.
        "provenance_counts": provenance.counts(),
        "occurrence_decisions": occurrence_summary(PASS_FIRST),
        "review_required_reasons": review_reasons,
        "ner_model": model_name,
        "trained_ner_active": True,
    }
    return result, entities


# ---------------------------------------------------------------------------
# Post-redaction validation (Phase 4)
# ---------------------------------------------------------------------------

# Placeholders and research-profile surrogates are our own output, not PHI.
# They are masked out before the residual scan so the validator measures what
# survived redaction rather than re-flagging the redaction itself.
_PLACEHOLDER_PATTERN = re.compile(r"<REDACTED_[A-Z0-9_]+>")
_SURROGATE_PATTERN = re.compile(
    r"\b(?:PERSON|MRN|PATIENT_ID|HEALTH_PLAN|ACCESSION|DEVICE)"
    rf"_[0-9A-F]{{{HASH_LENGTH}}}\b"
)


def mask_release_placeholders(text: str) -> str:
    """Blank out our own replacement tokens, preserving offsets.

    Study-local surrogates and the Safe Harbor age aggregate are our own
    output, so the residual scan must not re-detect them as PHI.
    """
    masked = _PLACEHOLDER_PATTERN.sub(lambda match: " " * len(match.group(0)), text)
    masked = _SURROGATE_PATTERN.sub(lambda match: " " * len(match.group(0)), masked)
    masked = SURROGATE_PATTERN.sub(lambda match: " " * len(match.group(0)), masked)
    masked = LEGACY_SURROGATE_PATTERN.sub(
        lambda match: " " * len(match.group(0)), masked
    )
    return masked.replace(
        AGE_AGGREGATE_REPLACEMENT, " " * len(AGE_AGGREGATE_REPLACEMENT)
    )


# -- classification vocabulary for second-pass findings ---------------------
#
# Every residual finding lands in exactly one of these. Four are benign and
# three block release; nothing is dropped silently, because a finding with no
# recorded classification is a finding nobody reviewed.

CLASSIFICATION_GENUINE_PHI = "genuine_surviving_phi"
CLASSIFICATION_PLAUSIBLE_PHI = "additional_plausible_phi"
CLASSIFICATION_EXACT_SURROGATE = "exact_generated_surrogate"
CLASSIFICATION_PLACEHOLDER = "anonymizer_placeholder"
CLASSIFICATION_DETECTOR_ARTEFACT = "detector_artefact_modified_context"
CLASSIFICATION_USEFUL_CLINICAL = "useful_clinical_content"
CLASSIFICATION_MALFORMED = "malformed_output"

RESIDUAL_CLASSIFICATIONS = (
    CLASSIFICATION_GENUINE_PHI,
    CLASSIFICATION_PLAUSIBLE_PHI,
    CLASSIFICATION_EXACT_SURROGATE,
    CLASSIFICATION_PLACEHOLDER,
    CLASSIFICATION_DETECTOR_ARTEFACT,
    CLASSIFICATION_USEFUL_CLINICAL,
    CLASSIFICATION_MALFORMED,
)

#: A generated token that got truncated, nested or otherwise mangled leaves a
#: fragment like "<REDACTED_PERSON" or a bare "PATIENT_" behind. A fragment is
#: not PHI, but it is proof the sanitizer's own writing is unsound, so the
#: document must not be released on the strength of a scan of it.
_PLACEHOLDER_FRAGMENT = re.compile(
    r"<\s*(?:REDACTED_?|(?:"
    + "|".join(sorted(set(SURROGATE_FAMILIES.values()), key=len, reverse=True))
    + r")_)[A-Z0-9_]*(?![A-Z0-9_]*>)"
)
_SURROGATE_STEM = re.compile(
    r"\b(?:PATIENT|PROVIDER|FACILITY|ORG|PLACE|ADDRESS|RECORD|PATIENTID|PLAN"
    r"|ACCESSION|DEVICE|IDENTIFIER|USER)_(?![0-9A-Z])"
)


def _region_text_is_well_formed(kind: str, span: str) -> bool:
    """Whether a recorded region still holds the token we meant to write."""
    if not span:
        return False
    if kind == KIND_PLACEHOLDER:
        return bool(_PLACEHOLDER_PATTERN.fullmatch(span))
    if kind == KIND_SURROGATE:
        return bool(
            SURROGATE_PATTERN.fullmatch(span)
            or LEGACY_SURROGATE_PATTERN.fullmatch(span)
            or _SURROGATE_PATTERN.fullmatch(span)
        )
    # Generalized ages and shifted dates are ordinary text by design; they are
    # malformed only if a replacement marker leaked into them.
    return "<" not in span and ">" not in span


def _malformed_output_findings(
    anonymized_text: str,
    provenance: "TransformationProvenance",
) -> List[Dict[str, Any]]:
    """Structural check on the sanitizer's own output.

    This is not a PHI scan. It asks whether the text we are about to judge is
    the text we intended to write: a half-written placeholder means the
    replacement loop produced something we cannot reason about, and reasoning
    about it anyway is how a partially-redacted value gets released. Findings
    here block, and they carry a shape, never a value.
    """
    findings: List[Dict[str, Any]] = []

    for region in provenance.regions:
        span = anonymized_text[region.start : region.end]
        if not _region_text_is_well_formed(region.kind, span):
            findings.append(
                {
                    "detector": "output_structure_validator",
                    "category": "MALFORMED_REPLACEMENT",
                    "evidence_type": EVIDENCE_DETERMINISTIC,
                    "location_type": "inside_generated_region",
                    "overlaps_generated_region": True,
                    "classification": CLASSIFICATION_MALFORMED,
                    "blocking": True,
                }
            )

    for pattern in (_PLACEHOLDER_FRAGMENT, _SURROGATE_STEM):
        for match in pattern.finditer(anonymized_text):
            if provenance.covering(match.start(), match.end()):
                continue
            findings.append(
                {
                    "detector": "output_structure_validator",
                    "category": "MALFORMED_REPLACEMENT",
                    "evidence_type": EVIDENCE_DETERMINISTIC,
                    "location_type": provenance.location_type(
                        match.start(), match.end()
                    ),
                    "overlaps_generated_region": bool(
                        provenance.touching(match.start(), match.end())
                    ),
                    "classification": CLASSIFICATION_MALFORMED,
                    "blocking": True,
                }
            )

    return findings


#: How an extracted fragment reads. Used by the pixel pipelines, where an OCR
#: box carries a few words with no sentence around them.
EXTRACTED_PHI = "phi"
EXTRACTED_NON_PHI = "non_phi"
EXTRACTED_UNCERTAIN = "uncertain"


def classify_extracted_text(text: str) -> str:
    """Classify a short extracted string as PHI, useful non-PHI, or unknown.

    Burned-in image text is the case this exists for. Blacking out every box
    an OCR engine reports makes the residual-text count zero and destroys the
    laterality marker, the scale bar and the burned-in measurement along with
    the patient name - a clean privacy number bought with the diagnostic
    content the image was kept for.

    Three answers, and the middle one is the point: only text that is
    *recognisably* clinical is preserved. Everything the pipeline cannot place
    comes back uncertain, and an uncertain fragment is treated as PHI by the
    caller, not waved through.
    """
    if not text or not text.strip():
        return EXTRACTED_NON_PHI
    # This is a deliberately closed whole-fragment grammar.  Check it before
    # general NER because short pathology markers such as ER and PR are often
    # mislabeled as initials.  A marker beside any other text no longer
    # matches the grammar and still goes through the complete detector chain.
    if is_confirmed_clinical_marker(text):
        return EXTRACTED_NON_PHI

    try:
        entities, review = _detect_entities(text, configured_model_name(), "strict")
    except NerPhiDetectionError as exc:
        raise TextAnonymizationError(
            exc.error_code, status_code=exc.status_code
        ) from exc

    if entities:
        return EXTRACTED_PHI
    if review:
        return EXTRACTED_UNCERTAIN
    return EXTRACTED_UNCERTAIN


#: Characters that carry no identity of their own. Stripped from the
#: uncovered remainder of a straddling prediction before it is assessed,
#: so a trailing comma is not mistaken for surviving content.
_BOUNDARY_CHARACTERS = string.whitespace + ".,;:()[]{}<>'\"-/\\"


#: Honorifics only: words that announce a person and are never one.
_HONORIFICS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "miss",
        "mx",
        "prof",
        "professor",
        "doctor",
        "sir",
        "dame",
        "nurse",
        "sister",
        "rev",
        "fr",
    }
)
_PERSON_SURROGATE = re.compile(r"<(?:PERSON|PROVIDER|PATIENT)_[1-9][0-9]*>")


def _only_honorifics(fragment: str) -> bool:
    words = [word for word in re.split(r"[\s.]+", fragment) if word]
    return bool(words) and all(word.casefold() in _HONORIFICS for word in words)


def _person_surrogate_follows(
    text: str, position: int, provenance: "TransformationProvenance"
) -> bool:
    """Whether the next thing after ``position`` is a person token this run wrote.

    Only person replacements recorded by this text transformation qualify.
    """
    while position < len(text) and text[position] in " .\t":
        position += 1
    for region in provenance.regions:
        if region.start == position:
            written = text[region.start : region.end]
            return bool(_PERSON_SURROGATE.fullmatch(written))
    return False


def _remainder_is_accounted_for(
    text: str,
    start: int,
    end: int,
    provenance: "TransformationProvenance",
    category: str,
    source: str,
    agreeing: int,
) -> bool:
    """Whether a straddling prediction claims anything beyond our own token.

    A detector reading the redacted output routinely proposes a span one word
    wider than the surrogate it is really responding to - "Dr. PROVIDER_001",
    "PATIENT_001 was". The span is not wholly inside a generated region, so it
    cannot simply be discounted, and blocking on it would hold clean documents
    on the strength of a boundary.

    So the remainder is examined rather than ignored. Each part of the span
    that no generated region covers is put through the same evidence model as
    any other candidate, in full document context. Only when every remainder
    is punctuation, whitespace, or text the model reads as *not* an identifier
    is the finding an artefact of the boundary. A real name sitting beside a
    surrogate - "PATIENT_001 Balasubramanian" - assesses as an identifier and
    still blocks, which is the case this must never wave through.
    """
    for gap_start, gap_end in provenance.uncovered(start, end):
        fragment = text[gap_start:gap_end]
        stripped = fragment.strip(_BOUNDARY_CHARACTERS)
        if not stripped:
            continue
        offset = gap_start + fragment.index(stripped)
        if is_identifier_label_text(stripped) and provenance.touching(start, end):
            # "Driver licence <LICENSE_ID_1>": the words the sanitizer kept in
            # front of the value it replaced. Every word is a field label or
            # connective from a closed list, so none can be a name; a name, a
            # code or an unknown word fails this test and goes to the evidence
            # model below.
            continue
        if (
            category == "PERSON"
            and _only_honorifics(stripped)
            and _person_surrogate_follows(text, offset + len(stripped), provenance)
        ):
            # "Dr <PROVIDER_1>": the title the sanitizer kept in front of the
            # surrogate it wrote for the name. Nothing but honorific words -
            # a closed list - is waved through here; "Dr Zelquin" still goes
            # to the evidence model below and blocks.
            continue
        assessment = assess_finding(
            text,
            offset,
            offset + len(stripped),
            category,
            source,
            agreeing_detectors=agreeing,
        )
        if assessment.action != "preserve":
            return False
    return True


def residual_findings(
    anonymized_text: str,
    provenance: "TransformationProvenance | None" = None,
) -> List[Dict[str, Any]]:
    """Re-scan the **exact** serialized output and classify what is found.

    The previous implementation blanked every generated token with spaces
    before re-scanning. That rewrote the sentence it was checking: "Dr.
    PROVIDER_001 at FACILITY_001" became "Dr.<spaces>at<spaces>", and both
    models predicted a name belonged in the hole. Sixty per cent of clean
    documents were blocked by artefacts of the masking itself.

    Nothing is masked now. The scan runs on the real output, and the
    provenance map says which character ranges the sanitizer wrote. A finding
    is discounted only when its span lies **wholly** inside a generated
    region - a prediction that merely touches a surrogate still covers text we
    did not generate, and that is exactly where a missed identifier would sit.

    Returns one record per finding carrying no value and no sentence.
    """
    if not isinstance(anonymized_text, str):
        raise TextAnonymizationError("Text input must be a string")
    if not anonymized_text.strip():
        return []
    if len(anonymized_text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise TextAnonymizationError(
            f"Text input exceeds the {MAX_TEXT_BYTES} byte limit",
            status_code=413,
        )

    provenance = provenance or TransformationProvenance()
    # Read with dash variants folded to a hyphen, as the first pass does; the
    # offsets are the same, so provenance and the original text still apply.
    scan_text = fold_dashes(anonymized_text)
    try:
        # The second production pass. This is the only place the validator
        # profile applies; the first pass above always runs the full chain.
        entities, _review = _detect_entities(
            scan_text,
            configured_model_name(),
            "strict",
            pass_kind=PASS_RESIDUAL,
        )
    except NerPhiDetectionError as exc:
        raise TextAnonymizationError(
            exc.error_code, status_code=exc.status_code
        ) from exc

    findings: List[Dict[str, Any]] = _malformed_output_findings(
        anonymized_text, provenance
    )
    # The validator's own structured scan, written separately from the
    # first pass's expressions so that one mistake cannot hide an
    # identifier from both passes. Its findings are deterministic: anything
    # it finds outside the sanitizer's own tokens blocks release.
    entities = (
        list(entities)
        + scan_structured_identifiers(anonymized_text)
        + foreign_token_entities(anonymized_text)
    )
    # A day left behind when a date was replaced in pieces sits between the
    # sanitizer's own tokens, so only the provenance map can place it.
    entities += date_fragments(anonymized_text, provenance.regions)
    proper_nouns = proper_noun_offsets(scan_text, configured_model_name())
    support_cache: Dict[str, Any] = {}

    def support_of(entity) -> Any:
        surface = scan_text[entity.start : entity.end].strip()
        if surface not in support_cache:
            support_cache[surface] = occurrence_support(
                entities, scan_text, entity.start, entity.end
            )
        return support_cache[surface]

    for entity in entities:
        location = provenance.location_type(entity.start, entity.end)
        covering = provenance.covering(entity.start, entity.end)
        evidence = evidence_type(entity.source)

        if covering is not None:
            # Wholly our own token: the detector is reading a surrogate.
            classification = (
                CLASSIFICATION_EXACT_SURROGATE
                if covering.kind == KIND_SURROGATE
                else CLASSIFICATION_PLACEHOLDER
            )
            blocking = False
        elif provenance.touching(entity.start, entity.end) and not any(
            _BRACKETED_TOKEN.sub("", anonymized_text[gap_start:gap_end]).strip(
                _BOUNDARY_CHARACTERS
            )
            for gap_start, gap_end in provenance.uncovered(entity.start, entity.end)
        ):
            # A detector read across adjacent tokens the sanitizer wrote
            # itself - "PATIENT_001PATIENT_002", or a model's PERSON over
            # "<PERSON_9> <PERSON_3>" - with nothing original between or
            # around them but spaces, punctuation and complete token shapes
            # copied from the source ("<PERSON_9>" names a category and a
            # number, nobody). No other letter or digit survives there to
            # block on; a single original character in a gap still blocks
            # below, whichever detector read it.
            classification, blocking = CLASSIFICATION_DETECTOR_ARTEFACT, False
        elif evidence == EVIDENCE_DETERMINISTIC:
            # Layer 1: an exact identifier match in the released text is a
            # genuine survivor, whatever else is true.
            classification, blocking = CLASSIFICATION_GENUINE_PHI, True
        else:
            agreeing = count_agreeing(
                entities,
                entity.start,
                entity.end,
                entity.entity_type,
                exclude_source=entity.source,
            )
            assessment = assess_finding(
                scan_text,
                entity.start,
                entity.end,
                entity.entity_type,
                entity.source,
                agreeing_detectors=agreeing,
                clinical_support=has_clinical_support(
                    entities, entity.start, entity.end
                ),
                contains_proper_noun=contains_proper_noun_token(
                    proper_nouns, entity.start, entity.end
                ),
                occurrence_support=lambda entity=entity: support_of(entity),
            )
            if assessment.action == "preserve":
                classification, blocking = CLASSIFICATION_USEFUL_CLINICAL, False
            elif location == "spans_generated_and_original" and (
                _remainder_is_accounted_for(
                    anonymized_text,
                    entity.start,
                    entity.end,
                    provenance,
                    entity.entity_type,
                    entity.source,
                    agreeing,
                )
            ):
                # The prediction reaches past our token, but everything it
                # reaches is examined and reads as non-identifying.
                classification, blocking = CLASSIFICATION_DETECTOR_ARTEFACT, False
            else:
                classification, blocking = CLASSIFICATION_PLAUSIBLE_PHI, True

        finding = {
            "detector": entity.source,
            "category": entity.entity_type,
            "evidence_type": evidence,
            "location_type": location,
            "overlaps_generated_region": bool(
                provenance.touching(entity.start, entity.end)
            ),
            "classification": classification,
            "blocking": blocking,
        }
        _trace_residual(
            finding,
            entity.start,
            entity.end,
            normalized_shape(anonymized_text[entity.start : entity.end]),
            text=anonymized_text[entity.start : entity.end],
            context=anonymized_text[max(0, entity.start - 40) : entity.end + 40],
        )
        findings.append(finding)
    return findings


def _residual_summary(findings: List[Dict[str, Any]]) -> Dict[str, int]:
    """Blocking findings only, counted by category."""
    summary: Dict[str, int] = {}
    for finding in findings:
        if finding["blocking"]:
            category = finding["category"]
            summary[category] = summary.get(category, 0) + 1
    return summary


def residual_phi_categories(
    anonymized_text: str,
    provenance: "TransformationProvenance | None" = None,
) -> Dict[str, int]:
    """Categories of genuinely surviving PHI. Non-empty must block release."""
    return _residual_summary(residual_findings(anonymized_text, provenance))
