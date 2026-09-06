"""Layered evidence model for detection findings.

Phase 10 protected clinical content with a closed vocabulary: if a word was in
the list it survived, otherwise it was redacted. That preserves the words
somebody thought of and destroys every unseen medication, condition or
eponym - and worse, it protects a term unconditionally once listed, so a
patient genuinely surnamed Wilson could be shielded by a rule meant for
Wilson's disease.

This module replaces "is it in the list?" with a layered decision in which the
vocabulary is one evidence source rather than the verdict:

1. **Exact high-confidence structured PHI always wins.** A regex that matched
   an SSN, an email or an MRN is not overridden by anything.
2. **Multiple agreeing detectors normally win.** Two independent models
   proposing the same span is strong evidence of an identifier.
3. **A clinical reading may block a weak, heuristic-only redaction.** This is
   where the vocabulary and the surrounding grammar are consulted.
4. **A clinical term never overrides strong contextual evidence that it is
   being used as a person, facility or location.** "Dr Parkinson" is a person
   even though "Parkinson" is a recorded eponym.
5. **Unknown and ambiguous terms escalate.** Neither deleted on suspicion nor
   preserved on hope: they go to a human.

Evidence strength, not word membership, decides. Nothing here inspects a
candidate against any expected-value list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from services.clinical_vocabulary import (
    CLINICAL_HEAD_NOUNS,
    contains_distinctive_clinical_token,
    is_clinical_eponym_use,
    is_non_name_label,
    is_unconditional_clinical_term,
)
from services.phi_detection import (
    SOURCE_AGE_RULE,
    SOURCE_CONTEXT_RULE,
    SOURCE_NER,
    SOURCE_STRICT_PROPER_NOUN,
    SOURCE_STRUCTURED_PATTERN,
)

#: Bumped when the rules move, not when the file is edited. A frozen
#: configuration is only meaningful if this string moves with the behaviour.
#:
#: v2 and v3 were both written from what the real chain did to the calibration
#: partition: a diagnosis is never an identifier; a single character carries no
#: identity; a span the tagger finds no proper noun in is not a name; Layer 1
#: belongs to the regex rather than to a model's choice of category label; and
#: a time reference with no calendar content is not a date element.
EVIDENCE_MODEL_VERSION = "detection-evidence-v3"

# -- evidence strength ------------------------------------------------------

EVIDENCE_DETERMINISTIC = "deterministic"   # a structured pattern or age rule
EVIDENCE_MODEL = "model"                   # a trained model proposed it
EVIDENCE_CONTEXT = "context_rule"          # an explicit contextual construction
EVIDENCE_HEURISTIC = "heuristic"           # capitalisation shape only

#: Sources whose findings are exact matches on identifier syntax.
_DETERMINISTIC_SOURCES = frozenset({SOURCE_STRUCTURED_PATTERN, SOURCE_AGE_RULE})
_MODEL_SOURCES = frozenset({SOURCE_NER, "stanford_deidentifier", "gliner_multi_pii"})

#: Categories whose syntax is self-evidencing. A regex hit on one of these is
#: never overridden by a clinical reading.
EXACT_STRUCTURED_CATEGORIES = frozenset(
    {
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "SSN",
        "URL",
        "IP_ADDRESS",
        "MEDICAL_RECORD_NUMBER",
        "PATIENT_ID",
        "HEALTH_PLAN_ID",
        "INSURANCE_ID",
        "ACCESSION_NUMBER",
        "DEVICE_ID",
        "AGE_OVER_89",
    }
)

#: Categories a clinical term can be misassigned to.
NAME_SHAPED_CATEGORIES = frozenset(
    {"PERSON", "ORGANIZATION", "FACILITY", "LOCATION", "ADDRESS", "MEDICAL_CONDITION"}
)

#: Categories that are never Safe Harbor identifiers, whichever detector
#: proposes them.
#:
#: A diagnosis is not one of the eighteen. GLiNER is asked for "medical
#: condition" because naming the label stops it reaching for PERSON when it
#: sees an eponym - the label earns its place as a *discriminator*. Treating
#: what it returns as PHI inverts the purpose of the pipeline: it redacted
#: "Vital signs remained stable" and "Fluid balance was maintained" and left
#: the note medically empty. The prediction is kept as evidence that the span
#: reads clinically; it is never grounds to remove it.
NON_IDENTIFIER_CATEGORIES = frozenset({"MEDICAL_CONDITION"})


#: Months and weekdays. A date element that Safe Harbor cares about names one
#: of these or carries digits; "the morning round" and "overnight" name
#: neither and are not date elements at all.
_CALENDAR_WORDS = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b",
    re.IGNORECASE,
)

#: Categories that are only themselves when the span carries the syntax.
_STRUCTURAL_HALLMARK = {
    "EMAIL_ADDRESS": lambda span: "@" in span,
    "URL": lambda span: "://" in span or "www." in span.lower(),
    "IP_ADDRESS": lambda span: span.count(".") >= 3 or ":" in span,
}


def _has_structural_syntax(category: str, span: str) -> bool:
    """Whether a span actually looks like the category it was labelled.

    Layer 1 exists for a regex that matched identifier syntax. Extending its
    authority to a *model* saying "PHONE_NUMBER" gives a prediction the
    standing of a match: GLiNER labelled the bare word "Fax" a phone number,
    and Layer 1 then blocked the document with no further examination. A model
    finding has to look like the thing before it gets that authority.
    """
    hallmark = _STRUCTURAL_HALLMARK.get(category)
    if hallmark is not None:
        return bool(hallmark(span))
    return sum(character.isdigit() for character in span) >= 2


def is_relative_time_expression(category: str, span: str) -> bool:
    """A time reference with no calendar content is not a date element.

    Safe Harbor removes elements of dates. "the morning round", "overnight"
    and "the next appointment" carry no date, and replacing them with
    <REDACTED_TIME> removes clinical sequencing for no privacy gain.
    """
    if category not in {"DATE_TIME", "DATE", "TIME"}:
        return False
    if any(character.isdigit() for character in span):
        return False
    return not _CALENDAR_WORDS.search(span)


def evidence_type(source: str) -> str:
    if source in _DETERMINISTIC_SOURCES:
        return EVIDENCE_DETERMINISTIC
    if source == SOURCE_CONTEXT_RULE:
        return EVIDENCE_CONTEXT
    if source == SOURCE_STRICT_PROPER_NOUN:
        return EVIDENCE_HEURISTIC
    if source in _MODEL_SOURCES:
        return EVIDENCE_MODEL
    return EVIDENCE_MODEL


# -- contextual evidence that a term is being used as an identifier ---------

#: Titles and constructions that mark the following words as a person, whatever
#: the dictionary says about them.
_PERSON_CONTEXT_BEFORE = re.compile(
    r"(?:\b(?:dr|dr\.|doctor|prof|prof\.|professor|mr|mr\.|mrs|mrs\.|ms|ms\.|"
    r"miss|sir|dame|nurse|sister|consultant|attending|surgeon|registrar)\s*|"
    r"\b(?:patient|resident|client|subject|donor|recipient)\s+|"
    r"\b(?:seen|treated|reviewed|referred|examined|admitted|discharged|"
    r"signed|countersigned|accompanied)\s+by\s+|"
    r"\bname\s*[:\-]\s*)$",
    re.IGNORECASE,
)
_PERSON_CONTEXT_AFTER = re.compile(
    r"^\s*(?:\b(?:attended|visited|called|arrived|consented|reported|stated|"
    r"complained|presented|was\s+admitted|was\s+discharged|was\s+seen|"
    r"signed|telephoned|is\s+the\s+(?:patient|next\s+of\s+kin))\b)",
    re.IGNORECASE,
)
_FACILITY_CONTEXT_BEFORE = re.compile(
    r"(?:\bat\s+|\badmitted\s+to\s+|\btransferred\s+to\s+|\breferred\s+to\s+)$",
    re.IGNORECASE,
)
#: A possessive immediately after the term, with a clinical head noun soon
#: after, is the classic eponym construction.
_EPONYM_AFTER = re.compile(
    r"^['’]?s?\s+(?:" + "|".join(sorted(CLINICAL_HEAD_NOUNS)) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceAssessment:
    """Why a finding was kept, dropped or escalated. Carries no value."""

    category: str
    source: str
    evidence: str
    #: "identifier" | "clinical" | "ambiguous"
    reading: str
    #: "redact" | "preserve" | "review"
    action: str
    rationale: str
    agreeing_detectors: int = 1

    def to_public_dict(self) -> Dict[str, object]:
        return {
            "category": self.category,
            "detector": self.source,
            "evidence_type": self.evidence,
            "reading": self.reading,
            "action": self.action,
            "rationale": self.rationale,
            "agreeing_detectors": self.agreeing_detectors,
        }


def _context_says_person(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 48) : start]
    after = text[end : end + 48]
    return bool(_PERSON_CONTEXT_BEFORE.search(before) or _PERSON_CONTEXT_AFTER.match(after))


def _context_says_facility(text: str, start: int, end: int) -> bool:
    return bool(_FACILITY_CONTEXT_BEFORE.search(text[max(0, start - 48) : start]))


#: Characters that end a sentence or a labelled line. `rstrip` removes the
#: newline itself, so a line break shows up as the punctuation before it or as
#: an empty prefix.
_SENTENCE_ENDINGS = frozenset(".!?:;")


def _is_sentence_initial(text: str, start: int) -> bool:
    """True when a span opens a sentence.

    English capitalises the first word of every sentence, so a capitalised
    token there is not evidence of a name. Without this the proper-noun
    heuristic flags ordinary openers - "Care was provided", "Assessment
    follows" - and either destroys them or floods manual review.
    """
    before = text[:start].rstrip()
    if not before:
        return True
    return before[-1] in _SENTENCE_ENDINGS


def _is_single_token(text: str, start: int, end: int) -> bool:
    return not re.search(r"\s", text[start:end].strip())


def _context_says_clinical(text: str, start: int, end: int) -> bool:
    """A clinical construction around the span: possessive plus a head noun."""
    return bool(_EPONYM_AFTER.match(text[end : end + 48]))


#: Words that exist to mark the thing before them as an organisation. A span
#: ending in one is a named body, whatever clinical words it contains: "Blood
#: Pressure Diagnostics Ltd" is a company, not a vital sign. This is a
#: grammatical designator test, not a directory of known organisations, so an
#: institution nobody listed is caught the same way.
_ORG_DESIGNATOR = re.compile(
    r"\b(?:ltd|limited|inc|incorporated|llc|llp|plc|pvt|gmbh|bv|nv|ag|sa|"
    r"corp|corporation|company|holdings|ventures|enterprises|"
    r"hospital|hospitals|clinic|clinics|infirmary|institute|institution|"
    r"laborator(?:y|ies)|labs?|diagnostics|imaging|radiology|pathology|"
    r"healthcare|health(?:care)?\s+system|trust|foundation|charity|"
    r"university|college|academy|"
    r"hospice|dispensary|polyclinic|nursing\s+home|"
    r"associates|partners|practice|centre|center)\.?$",
    re.IGNORECASE,
)

#: A quantity with a unit immediately after a term is prescribing grammar.
#: "Ranolazatib 250 mg twice daily" reads as a medication even though no
#: vocabulary contains that invented name - which is the point: the evidence
#: is the construction, so an unseen drug survives on the same footing as a
#: familiar one.
_DOSAGE_AFTER = re.compile(
    r"^[\s,:-]*\d+(?:[.,]\d+)?\s*"
    r"(?:mg|mcg|µg|ug|g|kg|ml|l|iu|units?|mmol|meq|mg/kg|mg/dl|mmhg|%)\b",
    re.IGNORECASE,
)


def _has_organisation_designator(span: str) -> bool:
    """True when the span ends in a word that declares it an organisation."""
    cleaned = span.strip().rstrip(".,;:")
    if not re.search(r"\s", cleaned):
        # A single word is its own designator only by coincidence; requiring
        # a modifier before it keeps "Practice" or "Centre" standing alone in
        # ordinary prose from reading as a company name.
        return False
    return bool(_ORG_DESIGNATOR.search(cleaned))


def _context_says_dosage(text: str, start: int, end: int) -> bool:
    return bool(_DOSAGE_AFTER.match(text[end : end + 32]))


#: Markers burned into a medical image that carry clinical meaning: laterality,
#: patient orientation, scale, and measurements with units. These are the
#: things a blanket "black out every OCR box" destroys, and losing the
#: laterality marker on a radiograph is not a small loss.
_CLINICAL_MARKER = re.compile(
    r"^(?:"
    r"[LR]|LT|RT|LEFT|RIGHT|AP|PA|LAT|OBL|"
    r"SUPINE|PRONE|ERECT|UPRIGHT|DECUBITUS|PORTABLE|"
    r"\d+(?:[.,]\d+)?\s*(?:mm|cm|m|kg|g|mg|ml|s|ms|mgy|kvp|mas|hu|t|mt)|"
    r"\d+(?:[.,]\d+)?\s*(?:x|×)\s*\d+(?:[.,]\d+)?\s*(?:mm|cm)?"
    r")$",
    re.IGNORECASE,
)


def is_confirmed_clinical_marker(text: str) -> bool:
    """Whether a short extracted string is recognisably clinical, not a name.

    Deliberately narrow. Anything this does not recognise is *uncertain*, not
    safe: the caller redacts uncertain text and records that it did, so an
    unrecognised marker costs a little utility rather than leaking a value.
    """
    candidate = text.strip().strip(":;,.")
    if not candidate:
        return False
    if _CLINICAL_MARKER.match(candidate):
        return True
    return is_unconditional_clinical_term(candidate) or contains_distinctive_clinical_token(
        candidate
    )


def clinical_reading_strength(text: str, start: int, end: int) -> Tuple[str, str]:
    """How strongly the surrounding text reads as clinical rather than naming.

    Returns (reading, rationale). The vocabulary contributes evidence here; it
    does not decide alone, and it is checked *after* the naming context so a
    listed eponym used as a surname is not shielded.
    """
    span = text[start:end]

    # Layer 4 first: naming context beats the dictionary.
    if _context_says_person(text, start, end):
        return "identifier", "person_context_present"
    if _context_says_facility(text, start, end):
        return "identifier", "facility_context_present"
    if _has_organisation_designator(span):
        return "identifier", "organisation_designator_present"

    # Vocabulary and grammar as evidence.
    if is_unconditional_clinical_term(span):
        return "clinical", "term_never_a_name"
    if _context_says_dosage(text, start, end):
        # Checked before the vocabulary so an unseen medication is protected
        # by the construction rather than by having been listed.
        return "clinical", "dosage_construction_grammar"
    if contains_distinctive_clinical_token(span):
        return "clinical", "contains_drug_or_analyte_token"
    if is_non_name_label(span):
        return "clinical", "field_label_or_role_noun"
    if _context_says_clinical(text, start, end):
        # Grammar alone, no dictionary needed: "<Word>'s syndrome" is a
        # condition whether or not the eponym is one we have heard of. This is
        # what removes the dependence on the closed list.
        return "clinical", "clinical_construction_grammar"
    if is_clinical_eponym_use(span, re.findall(r"\S+", text[end : end + 64])[:4]):
        return "clinical", "recorded_eponym_in_clinical_use"

    return "ambiguous", "no_decisive_evidence"


def assess_finding(
    text: str,
    start: int,
    end: int,
    category: str,
    source: str,
    agreeing_detectors: int = 1,
    clinical_support: bool = False,
    contains_proper_noun: Optional[bool] = None,
) -> EvidenceAssessment:
    """Decide what to do with one proposed span, by evidence strength."""
    evidence = evidence_type(source)

    # Layer 0: some categories are not identifiers at all. No amount of model
    # confidence turns a diagnosis into one of the eighteen.
    if category in NON_IDENTIFIER_CATEGORIES:
        return EvidenceAssessment(
            category, source, evidence, "clinical", "preserve",
            "category_is_not_a_safe_harbor_identifier", agreeing_detectors,
        )

    # A single character carries no identity on its own - the "C" in "37.2°C"
    # was being replaced with a patient surrogate. An initial in a naming
    # construction ("Patient R.") is still caught, because the context check
    # below runs on it like any other span.
    if (
        category in NAME_SHAPED_CATEGORIES
        and len(text[start:end].strip()) <= 1
        and not _context_says_person(text, start, end)
    ):
        return EvidenceAssessment(
            category, source, evidence, "clinical", "preserve",
            "single_character_span_carries_no_identity", agreeing_detectors,
        )

    # Layer 1: exact structured identifiers always win. A deterministic source
    # is the regex itself and needs no further proof; a model claiming one of
    # these categories has to produce a span that looks like one.
    span = text[start:end]
    if evidence == EVIDENCE_DETERMINISTIC or (
        category in EXACT_STRUCTURED_CATEGORIES and _has_structural_syntax(category, span)
    ):
        return EvidenceAssessment(
            category, source, evidence, "identifier", "redact",
            "exact_structured_match", agreeing_detectors,
        )

    if is_relative_time_expression(category, span):
        return EvidenceAssessment(
            category, source, evidence, "clinical", "preserve",
            "relative_time_expression_carries_no_date", agreeing_detectors,
        )

    if category not in NAME_SHAPED_CATEGORIES and category not in (
        EXACT_STRUCTURED_CATEGORIES
    ):
        return EvidenceAssessment(
            category, source, evidence, "identifier", "redact",
            "category_not_confusable_with_clinical_text", agreeing_detectors,
        )

    reading, rationale = clinical_reading_strength(text, start, end)

    if reading == "ambiguous" and clinical_support:
        # Another detector read this same span as a medical condition. That is
        # exactly the corroboration the vocabulary cannot give for a term
        # nobody listed, and it arrives from a model rather than a list.
        # Naming context still wins: it returns "identifier" above.
        reading = "clinical"
        rationale = "another_detector_read_the_span_as_clinical"

    if (
        reading != "identifier"
        and contains_proper_noun is False
        and agreeing_detectors < 2
    ):
        # A tagger found no proper noun anywhere in this span. "nursing
        # staff", "multidisciplinary team" and "Enrolment" are not names, and
        # a single model calling one a PERSON is not enough to replace it with
        # a patient surrogate.
        #
        # Three guards keep this from becoming a hole. Naming context still
        # wins, because it returns "identifier" above. Two agreeing detectors
        # still win, because a name the tagger mis-labels is exactly what
        # corroboration is for. And an unavailable parse arrives as None, not
        # False, so a tagger that failed exempts nothing.
        return EvidenceAssessment(
            category, source, evidence, "clinical", "preserve",
            "no_proper_noun_token_in_span", agreeing_detectors,
        )

    if reading == "identifier":
        # Layer 4: contextual naming evidence beats a clinical reading.
        return EvidenceAssessment(
            category, source, evidence, "identifier", "redact", rationale,
            agreeing_detectors,
        )

    if reading == "clinical":
        # Layer 3: a clinical reading blocks weak redaction, but does not
        # overrule several detectors agreeing.
        if agreeing_detectors >= 2 and evidence == EVIDENCE_MODEL:
            return EvidenceAssessment(
                category, source, evidence, "ambiguous", "review",
                f"{rationale}_but_multiple_detectors_agree", agreeing_detectors,
            )
        return EvidenceAssessment(
            category, source, evidence, "clinical", "preserve", rationale,
            agreeing_detectors,
        )

    # Ambiguous.
    if evidence == EVIDENCE_HEURISTIC:
        if agreeing_detectors >= 2:
            # Another detector saw the same span: no longer heuristic-only.
            return EvidenceAssessment(
                category, source, evidence, "identifier", "redact",
                "heuristic_supported_by_another_detector", agreeing_detectors,
            )
        # Layer 5, resolved fail-closed. An unexplained name-shaped span is
        # replaced by a surrogate rather than escalated-and-left-in-place: the
        # proper-noun fallback exists precisely to catch names the models miss,
        # and leaving one in the released text to await review would be a
        # privacy regression. Replacement is also the recoverable direction -
        # a surrogate costs one token of context, a leaked name costs the
        # patient.
        #
        # What stops this from destroying ordinary prose is upstream: the
        # detector no longer proposes a capitalised span whose proper-noun
        # reading is an artifact of sentence position (see
        # ner_phi_detector._is_case_artifact), so spans reaching here are
        # name-shaped in a way sentence capitalisation does not explain.
        return EvidenceAssessment(
            category, source, evidence, "identifier", "redact",
            "heuristic_name_shape_without_explanation", agreeing_detectors,
        )

    # Layer 2: a model or context rule with no clinical reading is an
    # identifier. This is the ordinary path for a real name.
    return EvidenceAssessment(
        category, source, evidence, "identifier", "redact",
        "model_evidence_without_clinical_reading", agreeing_detectors,
    )


def _same_reading(category: str, other: str) -> bool:
    """Whether two labels are two detectors saying the same thing."""
    if category == other:
        return True
    return category in NAME_SHAPED_CATEGORIES and other in NAME_SHAPED_CATEGORIES


def contains_proper_noun_token(
    proper_nouns,
    start: int,
    end: int,
):
    """Whether any proper-noun token overlaps the span.

    ``None`` in, ``None`` out: an unavailable parse must stay distinguishable
    from a parse that found nothing.
    """
    if proper_nouns is None:
        return None
    return any(
        token_start < end and start < token_end
        for token_start, token_end in proper_nouns
    )


def has_clinical_support(
    entities: Sequence[object],
    start: int,
    end: int,
) -> bool:
    """Whether some detector read an overlapping span as a medical condition."""
    return any(
        getattr(e, "entity_type", "") in NON_IDENTIFIER_CATEGORIES
        and getattr(e, "start", -1) < end
        and start < getattr(e, "end", -1)
        for e in entities
    )


def count_agreeing(
    entities: Sequence[object],
    start: int,
    end: int,
    category: str,
    exclude_source: Optional[str] = None,
) -> int:
    """How many distinct detectors made the *same* claim about this span.

    Agreement means two detectors independently reading the span the same
    way - both calling it a name, both calling it a facility. Bare overlap is
    not agreement: a structured-pattern hit on "MRN: 123456" overlaps a model
    calling the label "MRN" an organisation, but the two are describing
    different things, and counting them as corroboration escalated every
    ordinary field label to manual review.

    Deterministic sources are excluded for the same reason. A regex is
    authoritative about its own category - it decides on its own under Layer 1
    - and is silent about whether a neighbouring word is a name.
    """
    sources = set()
    for entity in entities:
        source = getattr(entity, "source", None)
        if source is None or source == exclude_source:
            continue
        if source in _DETERMINISTIC_SOURCES:
            continue
        if getattr(entity, "start", -1) >= end or start >= getattr(entity, "end", -1):
            continue
        if not _same_reading(category, getattr(entity, "entity_type", "")):
            continue
        sources.add(source)
    return len(sources) + (1 if exclude_source else 0)
