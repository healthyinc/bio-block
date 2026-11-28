"""The shared clinical-preservation policy: what may be kept, and only in context.

One policy, read by the anonymizer, the in-handler verifiers and the
independent release audit alike: all three reach it through
``services.detection_evidence``, so a term the sanitizer keeps is a term the
audit accepts, on the same evidence, re-proved on the finished artifact.

It answers one question about one span of text: is this a clinical invariant
that a generic model has misread as an identifier? Three structural
recognisers answer it, each needing both terminology *and* context:

* **Measurement units.** A complete unit expression - parsed by a small
  UCUM-style grammar, not matched as a literal - written directly after a
  numeric result: ``3.1 mU/L``, ``64 mmol/mol``, ``48 mL/min/1.73m²``,
  ``7.2 ×10⁹/L``. The number and unit may be split across lines or text runs.
* **Pathology stains**, in pathology context: ``PAS-D stain``, ``Ki-67
  immunohistochemistry``, a slide description with objective and pixel size.
* **Imaging sequences**, in imaging context: ``Sagittal STIR``, ``axial SWI``.

Every recogniser refuses when privacy evidence touches the span: an
identifier label before it (passport, MRN, registration, licence, phone...),
a title, a capitalised non-clinical word beside it, a token-shaped neighbour
(forged or not), an email or URL character, or a longer alphanumeric run.
Deterministic identifier rules are decided before this policy is consulted
and are never overruled by it.

Nothing here returns or logs a matched value. ``PreservationDecision`` holds
the term for the caller's own comparison and exposes only value-free fields in
``public_dict``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Tuple

POLICY_VERSION = "clinical-preservation-v2"


class PolicyError(RuntimeError):
    """The policy data failed its own validation. Raised at import."""


class TermCategory(str, Enum):
    UNIT = "measurement_unit"
    ANALYTE = "laboratory_analyte"
    CLINICAL_CODE = "clinical_code"
    VACCINE = "vaccine_term"
    MEDICATION = "medication_term"
    DEVICE = "medical_device"
    STAIN = "pathology_stain"
    MEDICAL_TERM = "medical_term"
    IMAGING_SEQUENCE = "imaging_sequence"


class Decision(str, Enum):
    PRESERVE = "preserve"
    REDACT = "redact"
    REVIEW = "review"
    NOT_APPLICABLE = "not_applicable"


# -- value-free reason codes -----------------------------------------------------------

REASON_UNIT = "measurement_unit_after_result"
REASON_STAIN = "pathology_stain_in_pathology_context"
REASON_SEQUENCE = "imaging_sequence_in_imaging_context"
REASON_LAB = "laboratory_structure"
REASON_UNIT_TOKEN = "unit_expression_token"
REASON_IMAGING = "imaging_structure"
REASON_PATHOLOGY = "pathology_structure"
REASON_MORPHOLOGY = "medical_term_morphology"
REASON_SERVICE = "clinical_service_structure"
REASON_NO_TERM = "no_structural_clinical_term"
REASON_CONFLICT = "privacy_evidence_touches_term"
REASON_NO_CONTEXT = "term_without_clinical_context"

#: Reasons that count as *structural* clinical evidence: a term the policy
#: recognised whole, in a clinical construction, with nothing identifying
#: beside it. Two model detectors agreeing on such a span are agreeing on a
#: domain-specific misreading, not on a name.
STRUCTURAL_REASONS: FrozenSet[str] = frozenset(
    {
        REASON_UNIT,
        REASON_STAIN,
        REASON_SEQUENCE,
        REASON_LAB,
        REASON_UNIT_TOKEN,
        REASON_IMAGING,
        REASON_PATHOLOGY,
        REASON_MORPHOLOGY,
        REASON_SERVICE,
    }
)

#: Release-gate reason codes: a protected clinical invariant did not survive.
CLINICAL_UNIT_CHANGED = "clinical_unit_changed"
CLINICAL_TERM_CHANGED = "clinical_term_changed"
CLINICAL_RELATION_CHANGED = "clinical_relation_changed"
CLINICAL_UTILITY_UNVERIFIED = "clinical_utility_unverified"


@dataclass(frozen=True)
class PreservationDecision:
    """One decision about one span. ``term`` never leaves the process."""

    term: str
    start: int
    end: int
    category: Optional[TermCategory]
    evidence_source: str
    surface: str
    structural_context: str
    exact_required: bool
    conflicting_phi: bool
    decision: Decision
    reason: str
    policy_version: str = POLICY_VERSION

    @property
    def preserve(self) -> bool:
        return self.decision is Decision.PRESERVE

    def public_dict(self) -> Dict[str, object]:
        return {
            "category": self.category.value if self.category else None,
            "evidence_source": self.evidence_source,
            "surface": self.surface,
            "structural_context": self.structural_context,
            "exact_required": self.exact_required,
            "conflicting_phi": self.conflicting_phi,
            "decision": self.decision.value,
            "reason": self.reason,
            "policy_version": self.policy_version,
        }


# -- measurement units: a UCUM-style grammar -------------------------------------------

_MICRO = "uµμ"
#: Metric prefixes clinical units actually carry.
_PREFIXES: Tuple[str, ...] = ("k", "d", "c", "m", "u", "µ", "μ", "n", "p", "f")
#: Atoms that take a prefix: gram, litre, mole, equivalent, unit, katal,
#: metre, second, osmole.
_PREFIXABLE: Tuple[str, ...] = (
    "g",
    "L",
    "l",
    "mol",
    "eq",
    "Eq",
    "U",
    "IU",
    "kat",
    "m",
    "s",
    "osm",
    "Osm",
)
#: Atoms written whole.
_WHOLE: Tuple[str, ...] = (
    "%",
    "mmHg",
    "cmH2O",
    "cmH₂O",
    "h",
    "hr",
    "hrs",
    "min",
    "sec",
    "d",
    "wk",
    "mo",
    "bpm",
    "IU",
    "iU",
    "units",
    "unit",
    "cells",
    "copies",
    "drops",
    "puffs",
    "breaths",
    "Hz",
    "kPa",
    "°C",
    "degC",
    "mSv",
    "mGy",
    "Gy",
    "Bq",
    "MBq",
)
_SUPERSCRIPT = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"
_EXPONENT = rf"(?:\^?[-−]?\d{{1,2}}|[{_SUPERSCRIPT}]+)"
#: A numeric factor inside a unit: ``10^9``, ``x10^9``, ``×10⁹``, ``10E9``, and a
#: body-surface divisor such as ``1.73m2``.
_FACTOR = (
    rf"(?:[x×]\s?)?10(?:[\^*E][-−]?\d{{1,2}}|[{_SUPERSCRIPT}]+)"
    rf"|\d(?:\.\d{{1,2}})?\s?m(?:2|²)"
)


def _alternation(items) -> str:
    return "|".join(
        re.escape(item) for item in sorted(set(items), key=len, reverse=True)
    )


_ATOM = (
    rf"(?:(?:{_alternation(_PREFIXES)})?(?:{_alternation(_PREFIXABLE)})"
    rf"|(?:{_alternation(_WHOLE)}))"
)
_TERM = rf"(?:(?:{_FACTOR})|(?:{_ATOM}){_EXPONENT}?)"
_SEPARATOR = r"\s?[/·*]\s?"
#: A complete unit expression: terms joined by ``/``, ``·`` or ``*``, at most
#: four of them. A factor alone (``10^9``) is not a unit.
UNIT_EXPRESSION = re.compile(rf"(?:{_TERM})(?:{_SEPARATOR}(?:{_TERM})){{0,3}}")
_UNIT_ONLY = re.compile(rf"^(?:{UNIT_EXPRESSION.pattern})$")

_NUMBER = r"(?:[<>≤≥]=?\s?)?\d{1,6}(?:[.,]\d{1,4})?"
#: A systolic/diastolic pair is one result: "118/74 mmHg". Only a pressure unit
#: accepts the pair (see ``_measurements``); any other unit after "a/b" is not one.
_PAIRED_NUMBER = rf"(?:\d{{2,3}}[ \t]?/[ \t]?)?{_NUMBER}"
_PRESSURE_UNIT = re.compile(r"(?:mm[ ]?Hg|kPa|cmH2O|cmH₂O)")
#: A result and its unit, possibly split across a line or text-run break.
MEASUREMENT = re.compile(
    rf"(?<![\w.@/+#-])(?P<number>{_PAIRED_NUMBER})"
    rf"(?P<gap>[ \t  ]*(?:\r?\n[ \t  ]*)?)"
    rf"(?P<unit>{UNIT_EXPRESSION.pattern})(?![\w@/.:-]*[\w@])"
)

#: Labels whose value is an identifier, however it is written. A number after
#: one of these is never a result, and a unit-like tail is never a unit.
_IDENTIFIER_LABEL = re.compile(
    r"(?i)(?:\b(?:mrn|id|no|nos|number|record|hospital|accession|licen[cs]e|gmc|nmc|hcpc|"
    r"reg(?:istration)?|passport|account|acct|nhs|ssn|sin|insurance|policy|member|"
    r"tel|telephone|phone|mobile|cell|fax|ext|extension|bleep|pager|room|bed|zip|"
    r"post\s?code|dob|serial|lot|batch|ref|reference|case|specimen\s+id|order)"
    r"\b|#)[\s:#.=-]*$"
)
_TITLE = re.compile(
    r"(?i)\b(?:dr|mr|mrs|ms|miss|mx|prof|professor|doctor|sir|dame|nurse|"
    r"sister|rev|fr)\.?\s*$"
)
_TOKEN_SHAPED = re.compile(
    r"<[A-Z][A-Z_]*(?:_[0-9]+)?>|\b[A-Z][A-Z]*(?:_[A-Z]+)*_\d{3,}\b"
)


def normalize_unit(unit: str) -> str:
    """Internal comparison key only; the released text keeps its own spelling."""
    folded = unicodedata.normalize("NFKC", unit)
    folded = (
        folded.replace("μ", "u").replace("µ", "u").replace("×", "x").replace("−", "-")
    )
    return re.sub(r"\s+", "", folded)


_FACTOR_ONLY = re.compile(_FACTOR)


def _has_atom(unit: str) -> bool:
    """A factor alone ("10^9", "1.73m2") is not a unit; an atom must remain."""
    return bool(re.search(r"[A-Za-z%°µμ]", _FACTOR_ONLY.sub("", unit)))


def is_unit_expression(text: str) -> bool:
    """Whether ``text`` is one complete unit expression, and nothing else."""
    candidate = (text or "").strip()
    return (
        bool(candidate)
        and bool(_UNIT_ONLY.fullmatch(candidate))
        and _has_atom(candidate)
    )


def _identifier_before(text: str, start: int) -> bool:
    window = text[max(0, start - 32) : start]
    line = window.rsplit("\n", 1)[-1]
    return bool(_IDENTIFIER_LABEL.search(line)) or bool(_TITLE.search(line))


def measurements(text: str) -> List[Tuple[int, int, int]]:
    """(start, end, unit_start) of every guarded result-with-unit in ``text``."""
    return list(_measurements(text or ""))


@lru_cache(maxsize=16)
def _measurements(text: str) -> Tuple[Tuple[int, int, int], ...]:
    # Cached per text: the span policy asks once per word, and a whole-text
    # scan per word is quadratic on a long note.
    found: List[Tuple[int, int, int]] = []
    for match in MEASUREMENT.finditer(text):
        start, end = match.span()
        if not _has_atom(match.group("unit")) or _identifier_before(text, start):
            continue
        if "/" in match.group("number") and not _PRESSURE_UNIT.fullmatch(
            match.group("unit").strip()
        ):
            continue
        # A number glued to a longer digit group ("0161 496 0558 mL") is part
        # of a phone number or an identifier, not a result.
        before = text[max(0, start - 6) : start]
        if re.search(r"\d[\s-]?$", before):
            continue
        tail = text[end : end + 2]
        if tail.startswith("@") or (tail[:1] == "." and tail[1:2].isalnum()):
            continue
        found.append((start, end, match.start("unit")))
    return tuple(found)


# -- pathology stains and imaging sequences --------------------------------------------

#: Stains and immunohistochemistry markers safe as standalone words.
_STAIN_TERMS: Tuple[str, ...] = (
    "h&e",
    "pas",
    "pas-d",
    "pasd",
    "d-pas",
    "dpas",
    "trichrome",
    "reticulin",
    "mucicarmine",
    "gms",
    "evg",
    "zn",
    "afb",
    "orcein",
    # dyes named with a colour: whole phrases only, never the first word alone
    "oil red o",
    "sudan black",
    "toluidine blue",
    "prussian blue",
    "congo red",
    "alcian blue",
    "methylene blue",
    "crystal violet",
    "orange g",
    "light green",
    "eosin",
    "haematoxylin",
    "hematoxylin",
    "safranin",
    "ki-67",
    "ki67",
    "mib-1",
    "her2",
    "er",
    "pr",
    "pd-l1",
    "alk",
    "ros1",
    "mlh1",
    "msh2",
    "msh6",
    "pms2",
    "s100",
    "s-100",
    "sox10",
    "ttf-1",
    "ttf1",
    "gata3",
    "wt1",
    "cdx2",
    "sma",
    "desmin",
    "vimentin",
    "synaptophysin",
    "chromogranin",
    "calretinin",
    "ema",
    "cea",
    "ae1/ae3",
    "cam5.2",
    "bcl2",
    "bcl6",
    "mum1",
    "p16",
    "p53",
    "p63",
    "p40",
    "ck7",
    "ck20",
    "ck5/6",
    "cd3",
    "cd4",
    "cd8",
    "cd10",
    "cd20",
    "cd30",
    "cd31",
    "cd34",
    "cd45",
    "cd56",
    "cd68",
    "cd117",
    "cd138",
    "dog1",
    "inhibin",
    "melan-a",
    "hmb45",
)
#: Stains named after people: kept only as part of a stain phrase ("Gram
#: stain", "Masson trichrome"), never as a lone capitalised word.
_EPONYM_STAINS: Tuple[str, ...] = (
    "gram",
    "giemsa",
    "masson",
    "grocott",
    "warthin-starry",
    "verhoeff",
    "fontana-masson",
    "ziehl-neelsen",
    "wright",
    "papanicolaou",
    "jones",
    "perls",
    "van gieson",
    "elastic van gieson",
)
_STAIN_HEADS = (
    "stain",
    "stains",
    "staining",
    "stained",
    "trichrome",
    "silver",
    "method",
)
_PATHOLOGY_CONTEXT = re.compile(
    r"(?i)\b(?:stain(?:s|ing|ed)?|immuno(?:histochemistry|stain(?:s|ing)?)|ihc|histolog\w*|"
    r"biopsy|biopsies|section|sections|slide|slides|objective|magnification|pixel\s+size|"
    r"microscop\w*|patholog\w*|specimen|block|tissue|mpp|appmag|positive|negative)\b|"
    r"\b\d+(?:\.\d+)?\s?[x×]\b|h\s?&\s?e"
)

#: MRI sequence names safe as standalone words.
_SEQUENCE_TERMS: Tuple[str, ...] = (
    "flair",
    "mprage",
    "stir",
    "swi",
    "dwi",
    "dti",
    "adc",
    "fmri",
    "t1",
    "t2",
    "t1w",
    "t2w",
    "t1-weighted",
    "t2-weighted",
    "pd-weighted",
    "gre",
    "epi",
    "tof",
    "spgr",
    "tse",
    "fse",
    "haste",
    "fiesta",
    "ciss",
    "vibe",
    "dixon",
    "mra",
    "mrv",
    "dce",
    "asl",
    "bold",
)
_IMAGING_CONTEXT = re.compile(
    r"(?i)\b(?:mri|mr|magnetic\s+resonance|sequence|sequences|axial|sagittal|coronal|"
    r"oblique|weighted|contrast|gadolinium|imaging|scan|protocol|series|acquisition|"
    r"diffusion|fat[-\s]suppressed|echo|slice|slices|3t|1\.5t)\b"
)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9&/+.\-]*[A-Za-z0-9]|[A-Za-z0-9]")


def _fold(term: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", term or "").casefold().split())


def _is_stain(term: str) -> bool:
    folded = _fold(term).strip(".,;:()")
    if folded in _STAIN_SET:
        return True
    parts = folded.split()
    # "PAS-D stain", "Gram stain", "Masson trichrome", "Ki-67 staining".
    if len(parts) >= 2 and parts[-1] in _STAIN_HEADS:
        head = " ".join(parts[:-1])
        return head in _STAIN_SET or head in _EPONYM_SET
    return False


def _is_sequence(term: str) -> bool:
    return _fold(term).strip(".,;:()") in _SEQUENCE_SET


def _neighbours(text: str, start: int, end: int) -> Tuple[str, str]:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end < 0 else line_end
    before = re.search(r"(\S+)[ \t]+$", text[line_start:start])
    after = re.match(r"[ \t]+(\S+)", text[end:line_end])
    return (before.group(1) if before else ""), (after.group(1) if after else "")


def _conflict(text: str, start: int, end: int, context: re.Pattern) -> bool:
    """Privacy evidence touching a term: labels, titles, tokens, name-like neighbours."""
    if _identifier_before(text, start):
        return True
    span = text[start:end]
    if "@" in span or "://" in span or _TOKEN_SHAPED.search(span):
        return True
    before, after = _neighbours(text, start, end)
    for word in (before, after):
        bare = word.strip(".,;:()|")
        if not bare:
            continue
        if _TOKEN_SHAPED.search(bare):
            return True
        if bare[:1].isupper() and not (
            context.search(bare)
            or _is_stain(bare)
            or _is_sequence(bare)
            or bare.casefold() in _NEUTRAL_WORDS
        ):
            if word is before and before.endswith((".", ":", ";", "|", ")")):
                continue
            return True
    # An alphanumeric identifier running into the term: "PAS-D7731", "STIR44".
    if re.match(r"[A-Za-z0-9]", text[end : end + 1]) or re.search(
        r"[A-Za-z0-9]$", text[max(0, start - 1) : start]
    ):
        return True
    return False


#: Capitalised words that are neither names nor evidence either way.
_NEUTRAL_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "with",
        "on",
        "in",
        "of",
        "for",
        "no",
        "not",
        "shows",
        "showed",
        "show",
        "was",
        "is",
        "are",
        "were",
        "see",
        "also",
        "both",
        "images",
        "positive",
        "negative",
        "stain",
        "staining",
        "sequence",
        "sequences",
        "axial",
        "sagittal",
        "coronal",
        "mri",
        "impression",
        "findings",
        "result",
        "results",
    }
)


def _context_window(text: str, start: int, end: int) -> str:
    """The line around a term, with the term itself blanked: context must be external."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end < 0 else line_end
    low, high = max(line_start, start - 120), min(line_end, end + 120)
    return text[low:start] + " " * (end - start) + text[end:high]


def _term_bounds(text: str, start: int, end: int) -> Tuple[int, int]:
    """Widen a model span to whole words, so "PAS" inside "PAS-D" is judged whole."""
    while (
        start > 0
        and re.match(r"[A-Za-z0-9&\-/.]", text[start - 1])
        and text[start - 1] != "."
    ):
        start -= 1
    while end < len(text) and re.match(r"[A-Za-z0-9&\-/]", text[end]):
        end += 1
    return start, end


def assess_clinical_span(
    text: str,
    start: int,
    end: int,
    evidence_source: str = "model",
    surface: str = "text",
) -> PreservationDecision:
    """Whether ``text[start:end]`` is a structurally proven clinical invariant."""
    text = text or ""
    span = text[start:end]

    def decide(
        decision, reason, category=None, context="", conflict=False, exact=False
    ):
        return PreservationDecision(
            _fold(span),
            start,
            end,
            category,
            evidence_source,
            surface,
            context,
            exact,
            conflict,
            decision,
            reason,
        )

    # A measurement unit: the span lies inside one guarded result-with-unit,
    # and touches its unit.
    for m_start, m_end, unit_start in measurements(text):
        if m_start <= start and end <= m_end and end > unit_start:
            return decide(
                Decision.PRESERVE,
                REASON_UNIT,
                TermCategory.UNIT,
                "numeric_result_then_unit",
                exact=True,
            )

    w_start, w_end = _term_bounds(text, start, end)
    words = text[w_start:w_end]
    # A stain phrase may run past the span, or the span may be part of one:
    # "PAS-D" + " stain", "Congo" + " red stain". Up to three following
    # words are tried, longest first; a first word alone ("Congo") is not a
    # stain unless it is listed on its own.
    extended = words
    following = re.match(r"(?:[ \t]+[A-Za-z]+){1,3}", text[w_end : w_end + 40])
    if following:
        pieces = re.findall(r"[ \t]+[A-Za-z]+", following.group(0))
        for count in range(len(pieces), 0, -1):
            candidate = words + "".join(pieces[:count])
            if _is_stain(candidate):
                extended = candidate
                break
    if _is_stain(words) or _is_stain(extended):
        stain_end = w_start + len(extended) if _is_stain(extended) else w_end
        if not _PATHOLOGY_CONTEXT.search(_context_window(text, w_start, stain_end)):
            return decide(
                Decision.NOT_APPLICABLE, REASON_NO_CONTEXT, TermCategory.STAIN
            )
        eponym_alone = _fold(words) in _EPONYM_SET and not _is_stain(extended)
        if eponym_alone or _conflict(text, w_start, stain_end, _PATHOLOGY_CONTEXT):
            return decide(
                Decision.NOT_APPLICABLE,
                REASON_CONFLICT,
                TermCategory.STAIN,
                conflict=True,
            )
        return decide(
            Decision.PRESERVE,
            REASON_STAIN,
            TermCategory.STAIN,
            "pathology_context",
            exact=True,
        )
    if _is_sequence(words):
        if not _IMAGING_CONTEXT.search(_context_window(text, w_start, w_end)):
            return decide(
                Decision.NOT_APPLICABLE,
                REASON_NO_CONTEXT,
                TermCategory.IMAGING_SEQUENCE,
            )
        if _conflict(text, w_start, w_end, _IMAGING_CONTEXT):
            return decide(
                Decision.NOT_APPLICABLE,
                REASON_CONFLICT,
                TermCategory.IMAGING_SEQUENCE,
                conflict=True,
            )
        return decide(
            Decision.PRESERVE,
            REASON_SEQUENCE,
            TermCategory.IMAGING_SEQUENCE,
            "imaging_context",
            exact=True,
        )

    # Structure, for terms no list names. Each rule needs a clinical
    # construction around the span, and each is refused where a title, an
    # identifier label or a token touches it; deterministic identifiers were
    # decided before this policy was asked.
    for rule, reason, category, context in _STRUCTURAL_RULES:
        region = rule(text, start, end)
        if region is None:
            continue
        low, high = region
        if _structure_conflict(text, low, high, start, end):
            return decide(
                Decision.NOT_APPLICABLE, REASON_CONFLICT, category, conflict=True
            )
        return decide(Decision.PRESERVE, reason, category, context, exact=True)
    return decide(Decision.NOT_APPLICABLE, REASON_NO_TERM)


# -- structural clinical context -------------------------------------------------------

_SPECIMENS = (
    "serum",
    "plasma",
    "blood",
    "whole blood",
    "venous blood",
    "arterial blood",
    "capillary blood",
    "cord blood",
    "urine",
    "24 hour urine",
    "cerebral spinal fluid",
    "cerebrospinal fluid",
    "spinal fluid",
    "csf",
    "saliva",
    "sputum",
    "stool",
    "faeces",
    "feces",
    "synovial fluid",
    "pleural fluid",
    "peritoneal fluid",
    "ascitic fluid",
    "body fluid",
    "tissue",
    "bone marrow",
    "semen",
    "sweat",
    "breath",
    "red blood cells",
    "red cells",
    "platelets",
    "leukocytes",
    "dried blood spot",
    "amniotic fluid",
    "vitreous fluid",
    "nasopharynx",
    "throat",
    "wound",
    "specimen",
)
_SPECIMEN = _alternation(_SPECIMENS)
#: "... in Serum or Plasma", "... in Urine by Test strip": a component measured
#: in a named specimen, the grammar of a laboratory test description.
_SPECIMEN_PHRASE = re.compile(
    rf"(?i)\b(?:in|from|of)\s+(?:{_SPECIMEN})(?:\s*(?:or|and|,)\s*(?:{_SPECIMEN}))*\b"
)
#: A LOINC-style property in brackets: "[Moles/volume]", "[Mass/volume]",
#: "[#/volume]", "[Presence]", "[Entitic volume]".
_PROPERTY_BRACKET = re.compile(
    r"\[[A-Za-z#][A-Za-z# ]{0,30}(?:/[A-Za-z][A-Za-z ]{0,20})?\]"
)
_SEGMENT_BREAK = re.compile(r"[\r\n|\t;\"]|,(?!\d)|\.(?=\s)")
_TITLE_WORDS = re.compile(
    r"(?i)\b(?:dr|mr|mrs|ms|miss|mx|prof|professor|doctor|nurse|sister|sir|dame|rev|"
    r"consultant|registrar)\b\.?"
)


def _segment(text: str, start: int, end: int, limit: int = 160) -> Tuple[int, int]:
    """The cell, line or sentence around a span, bounded."""
    low = max(0, start - limit)
    before = list(_SEGMENT_BREAK.finditer(text, low, start))
    seg_start = before[-1].end() if before else low
    after = _SEGMENT_BREAK.search(text, end, min(len(text), end + limit))
    seg_end = after.start() if after else min(len(text), end + limit)
    return seg_start, seg_end


def _lab_description(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    """A span inside a laboratory test description: property or specimen grammar."""
    low, high = _segment(text, start, end)
    if not (low <= start and end <= high):
        return None
    segment = text[low:high]
    if len(segment.split()) > 24:
        return None
    if not (_PROPERTY_BRACKET.search(segment) or _SPECIMEN_PHRASE.search(segment)):
        return None
    return low, high


_ANALYTE_WORD = r"[A-Za-z][A-Za-z0-9'\-+]*"
#: Internal capitals, a digit, a lower-case part after a hyphen, or all capitals:
#: "NT-proBNP", "HbA1c", "eGFR", "PSA" - a shape personal names do not have.
_NON_NAME_ANALYTE_SHAPE = re.compile(r"[a-z][A-Z]|\d|-[a-z]|\b[A-Z]{2,}\b")
_RESULT_AFTER = re.compile(
    r"\s*(?:[:=,;|\t]|\bof\b|\bwas\b|\bis\b|\bat\b)?\s*(?:[<>≤≥]=?\s?)?\d{1,6}(?:[.,]\d{1,4})?"
    r"\s*(?:[,;:|\t]\s*)?(?:\r?\n\s*)?(?P<unit>"
    + UNIT_EXPRESSION.pattern
    + r")(?![\w@/.:-]*[\w@])"
)


_ANALYTE_STOP = frozenset(
    {
        "and",
        "or",
        "with",
        "then",
        "but",
        "while",
        "plus",
        "vs",
        "versus",
        "after",
        "before",
        "was",
        "were",
        "is",
        "are",
    }
)


#: Words that qualify an analyte and are never a name on their own.
_ANALYTE_QUALIFIERS = frozenset(
    {
        "total",
        "free",
        "ionised",
        "ionized",
        "corrected",
        "conjugated",
        "unconjugated",
        "direct",
        "indirect",
        "fasting",
        "random",
        "venous",
        "arterial",
        "capillary",
        "serum",
        "plasma",
        "urine",
        "urinary",
        "whole",
        "blood",
        "high-sensitivity",
        "hs",
        "specific",
        "glycated",
        "bound",
        "unbound",
        "reverse",
        "intact",
        "bioavailable",
        "estimated",
        "calculated",
        "adjusted",
        "non-fasting",
        "trough",
        "peak",
        "baseline",
    }
)


def _analyte_before_result(
    text: str, start: int, end: int
) -> Optional[Tuple[int, int]]:
    """A span naming the analyte of a result that follows with a unit.

    "Testosterone 11.2 nmol/L", "Osmolality, 288, mOsm/kg", a unit on the next
    line. The analyte is at most four words from the start of its line or
    cell, and the result's unit must parse as a whole unit expression.
    """
    # A model span may reach into its own value ("NT-proBNP 1840" before " ng/L"):
    # the analyte is judged without the number it labels, and the region returned
    # below still contains the whole span.
    own_value = re.search(r"[ \t]+\d{1,6}(?:[.,]\d{1,4})?$", text[start:end])
    if own_value is not None:
        end = start + own_value.start()
        # A plain capitalised word ("Harriet 4.1") is how a name beside a figure
        # looks; only an analyte written in a shape no name has is read through.
        if end <= start or not _NON_NAME_ANALYTE_SHAPE.search(text[start:end]):
            return None
    token_end = end
    while token_end < len(text) and re.match(r"[A-Za-z0-9'\-+]", text[token_end]):
        token_end += 1
    match = _RESULT_AFTER.match(text, token_end)
    # The span may be a qualifier of the analyte phrase ("Total" in "Total
    # IgE 180 kU/L"): up to three more words may sit between it and the
    # result. Only a qualifier - "Quenbarrow potassium 4.1" is a name.
    hops = 0
    qualifier = all(
        word.casefold() in _ANALYTE_QUALIFIERS
        for word in re.findall(r"[A-Za-z][A-Za-z\-]*", text[start:end])
    )
    while (
        (match is None or not _has_atom(match.group("unit"))) and hops < 3 and qualifier
    ):
        following = re.match(
            r"[ \t]+([A-Za-z][A-Za-z0-9'\-+]*)", text[token_end : token_end + 40]
        )
        if following is None:
            return None
        token_end += following.end()
        hops += 1
        match = _RESULT_AFTER.match(text, token_end)
    if match is None or not _has_atom(match.group("unit")):
        return None
    seg_low, _high = _segment(text, start, end)
    # The analyte is the run of words directly before the result, back to a
    # number, a unit or a joining word: "... pmol/L and Fructosamine 281".
    low = start
    for match_word in reversed(list(re.finditer(r"\S+", text[seg_low:start]))):
        word = match_word.group(0).strip(",;:")
        if (
            not word
            or any(c.isdigit() for c in word)
            or is_unit_expression(word)
            or word.casefold() in _ANALYTE_STOP
            or len(text[seg_low + match_word.start() : token_end].split()) > 4
        ):
            break
        low = seg_low + match_word.start()
    phrase = text[low:token_end]
    words = phrase.split()
    if not 1 <= len(words) <= 4 or not all(
        re.fullmatch(_ANALYTE_WORD + r"[:,]?", w) for w in words
    ):
        return None
    # A pair of capitalised words is how a name looks; an analyte phrase of
    # that shape is kept only when one of its words is plainly not a name.
    titled = [w for w in words if re.fullmatch(r"[A-Z][a-z'\-]+", w)]
    if len(titled) >= 2 and not any(
        w.islower() or w.isupper() or any(c.isdigit() for c in w) for w in words
    ):
        return None
    if any(re.fullmatch(r"[A-Z][a-z']+(?:-[A-Z][a-z']+)+", w) for w in words):
        return None
    return low, match.end()


def _unit_token(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    """A span inside a whole token that is a compound unit expression ("mOsm/kg")."""
    low = start
    while low > 0 and not text[low - 1].isspace() and text[low - 1] not in ',;|"':
        low -= 1
    high = end
    while high < len(text) and not text[high].isspace() and text[high] not in ',;|"':
        high += 1
    token = text[low:high].strip("()[].:")
    if "@" in token or "://" in token:
        return None
    # A unit written after a number is a measurement: the measurement rule,
    # with its identifier-label guard, decides that one ("MRN 3.1 mU/L").
    if re.search(r"\d\s*$", text[max(0, low - 12) : low]):
        return None
    if not (is_unit_expression(token) and re.search(r"[/^⁰¹²³⁴⁵⁶⁷⁸⁹]|\d", token)):
        return None
    return low, high


_BODY_REGIONS = frozenset(
    {
        "knee",
        "hip",
        "shoulder",
        "elbow",
        "wrist",
        "ankle",
        "foot",
        "hand",
        "spine",
        "lumbar",
        "cervical",
        "thoracic",
        "sacral",
        "lumbosacral",
        "pelvis",
        "pelvic",
        "abdomen",
        "abdominal",
        "chest",
        "head",
        "neck",
        "brain",
        "cardiac",
        "heart",
        "liver",
        "kidney",
        "renal",
        "prostate",
        "breast",
        "orbit",
        "orbits",
        "sinus",
        "sinuses",
        "femur",
        "tibia",
        "humerus",
        "pituitary",
        "thigh",
        "calf",
        "forearm",
        "arm",
        "leg",
        "whole",
        "body",
        "left",
        "right",
        "bilateral",
        "joint",
        "internal",
        "auditory",
        "meati",
        "iam",
    }
)
_MODALITY = (
    r"(?:MRI|MR|CT|PET(?:-CT)?|SPECT|ultrasound|US|X-?ray|radiograph|MRA|MRV|DEXA|DXA)"
)
_REGION_THEN_MODALITY = re.compile(
    rf"(?P<region>[A-Za-z]+(?:\s+[A-Za-z]+){{0,3}})\s+{_MODALITY}\b(?=\s*[:,.\n]|\s+(?:of|with|shows|showed))"
)
_MODALITY_THEN_REGION = re.compile(
    rf"\b{_MODALITY}\s+(?:of\s+(?:the\s+)?)?(?P<region>[A-Za-z]+(?:\s+[A-Za-z]+){{0,3}})\b"
)
_PLANES = (
    r"(?:sagittal|axial|coronal|oblique|3D|2D|dynamic|delayed|post-?contrast|"
    r"pre-?contrast|fat[- ]suppressed|high[- ]resolution|volumetric|isotropic|"
    r"diffusion[- ]weighted|T1[- ]weighted|T2[- ]weighted)"
)
_SEQUENCE_WORD = r"[A-Za-z0-9][A-Za-z0-9\-+*/]{1,15}"
#: A series description as scanners write it: a plane, then short acquisition
#: acronyms and nothing else - "SAG PD FS", "AX T2 FLAIR", "COR STIR".
_PLANE_SERIES = re.compile(
    r"(?:SAG|SAGITTAL|AX|AXIAL|COR|CORONAL|OBL|OBLIQUE|3D|2D|TRA|TRANS)"
    r"(?P<rest>(?:\s+[A-Z0-9][A-Z0-9+*/\-]{0,7}){1,4})"
)
#: Acquisition words a series description is built from. A word after the
#: plane must be one of these or carry a digit ("T2", "3D"): "SAG JOHN SMITH"
#: is not a series.
_ACQUISITION_WORDS = frozenset(
    {
        "PD",
        "FS",
        "FAT",
        "SAT",
        "STIR",
        "FLAIR",
        "DWI",
        "ADC",
        "SWI",
        "GRE",
        "TSE",
        "FSE",
        "SE",
        "IR",
        "EPI",
        "SPGR",
        "FSPGR",
        "MPRAGE",
        "CUBE",
        "VIBE",
        "DIXON",
        "SPACE",
        "HASTE",
        "TRUFI",
        "FIESTA",
        "BLADE",
        "PROPELLER",
        "TOF",
        "MRA",
        "DTI",
        "BH",
        "NAV",
        "HR",
        "THIN",
        "FFE",
        "TFE",
        "MERGE",
        "CISS",
        "BRAVO",
        "IDEAL",
        "WATER",
        "IN",
        "OUT",
        "OPP",
        "PHASE",
        "POST",
        "PRE",
        "GAD",
        "C+",
        "C-",
        "FSE-IR",
        "TIRM",
        "MEDIC",
        "DESS",
        "SPIR",
        "SPAIR",
        "LOC",
        "SCOUT",
        "CAL",
        "MIP",
        "MPR",
        "RECON",
        "ANGIO",
    }
)
#: Anatomy that is never a surname, so "MR KNEE" is a modality and region
#: while "Mr Hand" or "Mr Head" stays a titled name.
_NON_SURNAME_REGIONS = frozenset(
    {
        "knee",
        "hip",
        "shoulder",
        "elbow",
        "wrist",
        "ankle",
        "spine",
        "lumbar",
        "cervical",
        "thoracic",
        "sacral",
        "lumbosacral",
        "pelvis",
        "pelvic",
        "abdomen",
        "abdominal",
        "brain",
        "cardiac",
        "kidney",
        "renal",
        "prostate",
        "orbit",
        "orbits",
        "sinus",
        "sinuses",
        "femur",
        "tibia",
        "humerus",
        "pituitary",
        "thigh",
        "forearm",
        "iam",
    }
)
_SEQUENCE_PHRASE = re.compile(
    rf"(?i:{_PLANES})\s+(?P<seq>{_SEQUENCE_WORD}(?:\s+{_SEQUENCE_WORD}){{0,2}})\s+"
    rf"(?i:sequences?|images?|series|acquisitions?|scans?|imaging|weighting)\b"
)
_ACRONYM_SEQUENCE = re.compile(
    r"\b(?P<seq>[A-Z][A-Z0-9\-+*]{1,11})\s+(?i:sequences?|acquisitions?|weighting)\b"
)


def _imaging_structure(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    """A span that is a body region in an imaging heading, or a named sequence."""
    low, high = _segment(text, start, end)
    window = text[low:high]
    span_end = min(end, len(text))
    while span_end > start and text[span_end - 1] in ".,;:":
        span_end -= 1
    for pattern in (_REGION_THEN_MODALITY, _MODALITY_THEN_REGION):
        for match in pattern.finditer(window):
            # The span may cover the modality too: "MR KNEE LEFT".
            m_start, m_end = low + match.start(), low + match.end()
            if (
                m_start <= start
                and span_end <= m_end
                and all(
                    word.casefold() in _BODY_REGIONS
                    for word in match.group("region").split()
                )
            ):
                return m_start, m_end
    # A series description is the whole segment, or the whole value after a
    # label's colon ("Series description: SAG PD FS").
    lead = text[low:start]
    candidates = [(low, high, text[low:high])]
    if re.search(r":\s*$", lead) and not text[span_end:high].strip(" .,;:"):
        candidates.append((start, span_end, text[start:span_end]))
    for c_low, c_high, value in candidates:
        plane = _PLANE_SERIES.fullmatch(value.strip(" .,;:"))
        if plane is not None and all(
            word in _ACQUISITION_WORDS or re.search(r"\d", word)
            for word in plane.group("rest").split()
        ):
            return c_low, c_high
    for pattern, needs_context in (
        (_SEQUENCE_PHRASE, False),
        (_ACRONYM_SEQUENCE, True),
    ):
        for match in pattern.finditer(window):
            s_start, s_end = low + match.start("seq"), low + match.end("seq")
            if not (s_start <= start and end <= s_end):
                continue
            words = match.group("seq").split()
            # A sequence is written as an acronym or in lower case; a pair of
            # capitalised words there is a name, not a protocol.
            if (
                any(re.fullmatch(r"[A-Z][a-z'\-]+", w) for w in words)
                and len(words) > 1
            ):
                continue
            if needs_context and not _IMAGING_CONTEXT.search(
                _context_window(text, s_start, s_end)
            ):
                continue
            return low + match.start(), low + match.end()
    return None


#: A word of a stain's name: letter-led (a marker such as "CD20" or "PAS-D"
#: included, a bare number never), and never a function word, so the phrase
#: stops at "on", "of", "shows" instead of running back over a sentence.
_STAIN_STOPWORDS = (
    r"(?:on|of|the|and|or|with|by|in|at|for|to|from|was|were|is|are|a|an|"
    r"shows?|showed|using|after|before|no|not)\b"
)
_STAIN_NAME_WORD = (
    rf"(?!(?i:{_STAIN_STOPWORDS}))"
    r"(?:[A-Z][A-Za-z\-]*|[a-z][a-z\-]+|[A-Z][A-Z0-9\-]{1,7})(?![\w])"
)
#: An identifier label inside a would-be stain name ends the claim.
_IDENTIFIER_WORD = re.compile(
    r"(?i)\b(?:mrn|id|no|number|record|accession|licen[cs]e|gmc|reg(?:istration)?|passport|"
    r"account|acct|nhs|ssn|tel|phone|mobile|fax|ref|reference|case|specimen\s+id)\b"
)
_STAIN_PHRASE_RE = re.compile(
    rf"(?P<name>{_STAIN_NAME_WORD}(?:\s+{_STAIN_NAME_WORD}){{0,3}})\s+"
    r"(?i:stain|stains|staining|stained|immunostain|immunostaining|histochemistry)\b"
)


def _pathology_structure(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    """A span naming a stain in a stain phrase, in pathology context."""
    low, high = _segment(text, start, end)
    window = text[low:high]
    for match in _STAIN_PHRASE_RE.finditer(window):
        # The span may be the stain's name or the whole phrase ("Von Kossa
        # stain"), but nothing outside it.
        if not (low + match.start() <= start and end <= low + match.end()):
            continue
        if _TITLE_WORDS.search(match.group("name")) or _IDENTIFIER_WORD.search(
            match.group("name")
        ):
            continue
        # A stain phrase is strong on its own; its pathology context may sit
        # a line or two away ("bone biopsy, decalcified sections.").
        p_start, p_end = low + match.start(), low + match.end()
        around = text[max(0, p_start - 300) : p_start] + " " + text[p_end : p_end + 300]
        if not _PATHOLOGY_CONTEXT.search(around):
            continue
        return low + match.start(), low + match.end()
    return None


_ABBREVIATED_TITLES = frozenset({"dr", "mr", "mrs", "ms", "mx", "prof", "rev", "fr"})


def _structure_conflict(text: str, low: int, high: int, start: int, end: int) -> bool:
    """Privacy evidence inside or at the edge of a structural region."""
    region = text[low:high]
    if "@" in region or "://" in region or _TOKEN_SHAPED.search(region):
        return True
    # A token-shaped word right before the construction may stand for a
    # replaced given name ("<PERSON_1> Movat stain"); forged or genuine, it
    # earns the construction nothing.
    previous = re.split(r"[.;:!?]\s|[\r\n]", text[max(0, low - 24) : low])[-1]
    if _TOKEN_SHAPED.search(previous):
        return True
    # A title directly before the span or inside it - not a role noun in the
    # previous sentence ("for the registrar. Tracheostomy").
    before = re.search(r"(\S+)\s*$", text[max(0, start - 24) : start])
    word = before.group(1).rstrip(",;:") if before else ""
    if word.endswith(".") and word[:-1].casefold() not in _ABBREVIATED_TITLES:
        word = ""  # "registrar." ends a sentence; only "Dr." abbreviates a title
    inside = [
        m
        for m in _TITLE_WORDS.finditer(text[start:end])
        if not _modality_mr(text, start + m.start(), start + m.end())
    ]
    # "MR" before upper-case anatomy is the modality of the region itself,
    # neither a title nor a medical-record label.
    modality = bool(word) and _modality_mr(
        text,
        max(0, start - 24) + before.start(1),
        max(0, start - 24) + before.start(1) + len(word),
    )
    if inside or (word and not modality and _TITLE_WORDS.fullmatch(word)):
        return True
    return _identifier_before(text, low) or (
        not modality and _identifier_before(text, start)
    )


def _modality_mr(text: str, low: int, high: int) -> bool:
    """Upper-case "MR" followed by upper-case anatomy that is never a surname."""
    if text[low:high] != "MR":
        return False
    following = re.match(r"\s+([A-Z]+)\b", text[high : high + 24])
    return bool(following) and following.group(1).casefold() in _NON_SURNAME_REGIONS


def modality_before(text: str, start: int) -> bool:
    """Whether the word right before ``start`` is the modality "MR" of anatomy."""
    before = re.search(r"(?<![A-Za-z])MR[ \t]+$", text[max(0, start - 8) : start])
    if before is None:
        return False
    low = start - len(before.group(0))
    return _modality_mr(text, low, low + 2)


_SPECIMEN_WORDS = frozenset(term for term in _SPECIMENS if " " not in term)


def _specimen_word(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    """A span that is one specimen word ("serum", "Plasma"): specimen nomenclature."""
    low, high = _term_bounds(text, start, end)
    if text[low:high].casefold() not in _SPECIMEN_WORDS:
        return None
    return low, high


#: Word endings of medical terminology no personal name carries: "myalgia",
#: "cholangitis", "splenomegaly", "leukopenia". Minimum lengths keep short
#: words and names with a look-alike ending ("Paloma") out.
_MORPHOLOGY = (
    (
        (
            "algia",
            "itis",
            "osis",
            "aemia",
            "emia",
            "penia",
            "uria",
            "ectomy",
            "otomy",
            "ostomy",
            "plasty",
            "scopy",
            "graphy",
            "pathy",
            "trophy",
            "plasia",
            "megaly",
            "rrhoea",
            "rrhea",
            "lysis",
            "iasis",
            "ology",
            "cytosis",
            "emesis",
            "ptysis",
        ),
        6,
    ),
    (("oma", "cele", "cyte"), 8),
    # compound anatomical adjectives: "nasogastric", "intravenous"
    (
        (
            "gastric",
            "venous",
            "cranial",
            "cutaneous",
            "muscular",
            "thoracic",
            "abdominal",
            "vesical",
            "urethral",
            "tracheal",
            "enteric",
            "pulmonary",
            "arterial",
            "osseous",
            "dermal",
            "ocular",
            "nasal",
        ),
        9,
    ),
)


def _medical_morphology(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    """A span that is one word with the ending of a medical term."""
    low, high = _term_bounds(text, start, end)
    # "Nasogastric tube": the term may carry up to two lower-case head nouns.
    region_end = high
    first, *rest = text[low:high].split(" ") or [""]
    if rest and len(rest) <= 2 and all(re.fullmatch(r"[a-z]+", w) for w in rest):
        high = low + len(first)
    word = text[low:high].strip("-").casefold()
    if not word.isalpha():
        return None
    for suffixes, minimum in _MORPHOLOGY:
        if len(word) >= minimum and word.endswith(suffixes):
            return low, region_end
    return None


#: Words that name what a clinical service treats. A department is not an
#: identifier; the words before its noun must all come from here (or read
#: as medical terms), so "Smith clinic" or "Royal Free clinic" is not one.
_SERVICE_SUBJECTS = frozenset(
    {
        "thyroid",
        "diabetes",
        "diabetic",
        "endocrine",
        "renal",
        "kidney",
        "liver",
        "breast",
        "fracture",
        "memory",
        "pain",
        "sleep",
        "stroke",
        "heart",
        "cardiac",
        "chest",
        "lung",
        "respiratory",
        "asthma",
        "allergy",
        "eye",
        "ear",
        "nose",
        "throat",
        "skin",
        "bone",
        "joint",
        "spine",
        "spinal",
        "antenatal",
        "postnatal",
        "maternity",
        "paediatric",
        "pediatric",
        "fertility",
        "sexual",
        "mental",
        "health",
        "surgical",
        "medical",
        "orthopaedic",
        "orthopedic",
        "vascular",
        "lipid",
        "obesity",
        "weight",
        "anticoagulation",
        "falls",
        "continence",
        "wound",
        "stoma",
        "dialysis",
        "transplant",
        "cancer",
        "tumour",
        "tumor",
        "general",
        "acute",
        "emergency",
        "day",
        "intensive",
        "care",
        "high",
        "dependency",
        "outpatient",
        "inpatient",
        "pre-assessment",
        "preassessment",
        "rapid",
        "access",
        "headache",
        "epilepsy",
        "movement",
        "disorders",
        "infectious",
        "diseases",
        "sickle",
        "cell",
        "blood",
        "pressure",
        "hypertension",
        "failure",
        "arrhythmia",
        "chronic",
        "disease",
        "palliative",
        "rehabilitation",
        "physiotherapy",
        "occupational",
        "therapy",
        "speech",
        "language",
        "dietetic",
        "dietetics",
        "pharmacy",
        "nutrition",
        "gastro",
        "bowel",
        "colorectal",
        "upper",
        "lower",
        "gi",
    }
)
_SERVICE_NOUN = (
    r"(?:clinic|clinics|department|dept|ward|unit|service|services|team|"
    r"outpatients?|mdt|centre|center|suite|theatres?)"
)
_SERVICE_PHRASE = re.compile(
    rf"(?P<subject>[A-Za-z][A-Za-z\-]*(?:[ \t]+[A-Za-z][A-Za-z\-]*){{0,3}})[ \t]+"
    rf"(?P<noun>(?i:{_SERVICE_NOUN}))\b"
)


def clinical_service_region(
    text: str, start: int, end: int
) -> Optional[Tuple[int, int]]:
    """The clinical service name ("Thyroid clinic", "renal unit") holding a span.

    Only the run of subject words directly before the service noun counts, so
    a facility or a person before it stays outside the region.
    """
    low, high = _segment(text, start, end)
    for match in _SERVICE_PHRASE.finditer(text, low, high):
        words = list(re.finditer(r"[A-Za-z][A-Za-z\-]*", match.group("subject")))
        phrase_start = None
        for word in reversed(words):
            value = word.group(0)
            if value.casefold() in _SERVICE_SUBJECTS or _medical_morphology(
                value, 0, len(value)
            ):
                phrase_start = match.start("subject") + word.start()
            else:
                break
        if phrase_start is not None and phrase_start <= start and end <= match.end():
            return phrase_start, match.end()
    return None


_STRUCTURAL_RULES = (
    (_specimen_word, REASON_LAB, TermCategory.ANALYTE, "specimen_word"),
    (
        _medical_morphology,
        REASON_MORPHOLOGY,
        TermCategory.MEDICAL_TERM,
        "term_morphology",
    ),
    (_unit_token, REASON_UNIT_TOKEN, TermCategory.UNIT, "whole_unit_expression"),
    (_lab_description, REASON_LAB, TermCategory.ANALYTE, "laboratory_description"),
    (_analyte_before_result, REASON_LAB, TermCategory.ANALYTE, "analyte_result_unit"),
    (
        _imaging_structure,
        REASON_IMAGING,
        TermCategory.IMAGING_SEQUENCE,
        "imaging_structure",
    ),
    (_pathology_structure, REASON_PATHOLOGY, TermCategory.STAIN, "stain_phrase"),
    (
        clinical_service_region,
        REASON_SERVICE,
        TermCategory.MEDICAL_TERM,
        "clinical_service_name",
    ),
)


# -- the clinical-utility ledger -------------------------------------------------------


@dataclass(frozen=True)
class ClinicalInvariant:
    """A protected clinical fact found in text. ``key`` never leaves the process."""

    start: int
    end: int
    category: TermCategory
    key: str


def clinical_invariants(text: str) -> List[ClinicalInvariant]:
    """Every protected invariant the policy would keep, in document order."""
    text = text or ""
    found: List[ClinicalInvariant] = []
    for start, end, unit_start in measurements(text):
        number = re.sub(r"\s+", "", text[start:unit_start])
        found.append(
            ClinicalInvariant(
                start,
                end,
                TermCategory.UNIT,
                number + " " + normalize_unit(text[unit_start:end]),
            )
        )
    covered = sorted((item.start, item.end) for item in found)
    cursor = 0
    for match in _WORD.finditer(text):
        start, end = match.span()
        # Words come in order, so one moving cursor finds the covering result.
        while cursor < len(covered) and covered[cursor][1] <= start:
            cursor += 1
        if cursor < len(covered) and covered[cursor][0] <= start:
            continue
        decision = assess_clinical_span(text, start, end)
        if decision.preserve and decision.category is not TermCategory.UNIT:
            found.append(
                ClinicalInvariant(start, end, decision.category, decision.term)
            )
    found.sort(key=lambda item: item.start)
    return found


def invariant_keys(text: str) -> Dict[Tuple[str, str], int]:
    """A multiset of (category, key): what must survive, compared by content."""
    counts: Dict[Tuple[str, str], int] = {}
    for item in clinical_invariants(text):
        counts[(item.category.value, item.key)] = (
            counts.get((item.category.value, item.key), 0) + 1
        )
    return counts


def lost_invariants(before: str, after: str) -> Tuple[str, ...]:
    """Value-free reason codes for invariants present before and missing after."""
    expected = invariant_keys(before)
    present = invariant_keys(after)
    reasons = set()
    for (category, key), count in expected.items():
        if present.get((category, key), 0) < count:
            reasons.add(
                CLINICAL_UNIT_CHANGED
                if category == TermCategory.UNIT.value
                else CLINICAL_TERM_CHANGED
            )
    return tuple(sorted(reasons))


def unattributed_losses(text: str, removals) -> Tuple[str, ...]:
    """Reason codes for invariants a non-deterministic replacement removed.

    ``removals`` is (start, end, deterministic) for every replaced span. A
    deterministic identifier overlapping an invariant is an approved
    transformation - a record number beside "mg" is still a record number.
    Anything else that removed a protected invariant did so on a model's or a
    heuristic's word alone, and the file must not be released automatically.
    """
    reasons = set()
    spans = [
        (start, end) for start, end, deterministic in removals if not deterministic
    ]
    hard = [(start, end) for start, end, deterministic in removals if deterministic]
    if not spans:
        return ()
    for item in clinical_invariants(text):
        if any(start < item.end and item.start < end for start, end in hard):
            continue
        if any(start < item.end and item.start < end for start, end in spans):
            reasons.add(
                CLINICAL_UNIT_CHANGED
                if item.category is TermCategory.UNIT
                else CLINICAL_TERM_CHANGED
            )
    return tuple(sorted(reasons))


#: Markers that are also given names or common words. Kept by the evidence
#: model only with context and guards, never by a whole-string match.
_NAME_LIKE_STAINS = frozenset({"ema", "alk", "cea", "alcian", "congo", "perls"})


def stain_atoms_pattern() -> str:
    """Standalone stain atoms as a regex alternation, for whole slide annotations.

    A slide annotation made only of these is kept without a model pass, so
    anything that could also be a name is left out and goes through the
    guarded evidence model instead.
    """
    return _alternation(
        term
        for term in _STAIN_TERMS
        if " " not in term and term not in _NAME_LIKE_STAINS
    )


# -- validation at import --------------------------------------------------------------

_TERM_SHAPE = re.compile(r"^[a-z0-9][a-z0-9&/+.\- ]*[a-z0-9]$|^[a-z0-9]$")


def _validate() -> Tuple[FrozenSet[str], FrozenSet[str], FrozenSet[str]]:
    tables = {
        "stains": _STAIN_TERMS,
        "eponym_stains": _EPONYM_STAINS,
        "sequences": _SEQUENCE_TERMS,
    }
    seen: Dict[str, str] = {}
    for name, terms in tables.items():
        if not terms:
            raise PolicyError(f"{name}_empty")
        if len(set(terms)) != len(terms):
            raise PolicyError(f"{name}_duplicate_entry")
        for term in terms:
            if not _TERM_SHAPE.fullmatch(term):
                raise PolicyError(f"{name}_malformed_entry")
            if term in seen:
                raise PolicyError(f"{name}_entry_also_in_{seen[term]}")
            seen[term] = name
    for sample in (
        "mU/L",
        "mmol/mol",
        "mL/min/1.73m2",
        "x10^9/L",
        "g/dL",
        "µmol/L",
        "%",
        "mmHg",
        "IU/L",
        "ng/mL",
    ):
        if not _UNIT_ONLY.fullmatch(sample):
            raise PolicyError("unit_grammar_unreachable")
    return (
        frozenset(_STAIN_TERMS),
        frozenset(_EPONYM_STAINS),
        frozenset(_SEQUENCE_TERMS),
    )


_STAIN_SET, _EPONYM_SET, _SEQUENCE_SET = _validate()
for _term in _STAIN_TERMS:
    if not _is_stain(_term):
        raise PolicyError("stain_entry_unreachable")
for _term in _SEQUENCE_TERMS:
    if not _is_sequence(_term):
        raise PolicyError("sequence_entry_unreachable")
del _term
