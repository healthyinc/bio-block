import os
import re
from functools import lru_cache
from typing import Any, List

from services import numeric_context
from services.clinical_vocabulary import (
    is_race_ethnicity_category,
    is_unconditional_clinical_term,
    protects_from_person_label,
)
from services.detection_evidence import line_carries_no_case_evidence
from services.phi_detection import (
    SOURCE_CONTEXT_RULE,
    SOURCE_NER,
    SOURCE_STRICT_PROPER_NOUN,
    DetectedEntity,
)
from services.request_deadline import check as check_request_deadline

DEFAULT_NER_MODEL = "en_core_web_sm"
NER_MODEL_ENV_VAR = "PHI_NER_MODEL"
_SAFE_MODEL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

SPACY_PHI_LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "FACILITY",
    "DATE": "DATE_TIME",
    "TIME": "TIME",
}
_NON_PHI_CLINICAL_TERMS = {
    "ct",
    "ecg",
    "ekg",
    "mri",
    "pet",
    "scan",
    "x-ray",
    "xray",
}
_NON_PHI_PERSON_TERMS = {"parkinson", "alzheimer", "crohn", "hodgkin", "covid-19"}
_DATE_LIKE_TEXT = re.compile(r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})")
# The optional "+" covers the aggregated form. Safe Harbor replaces an age
# above 89 with "90+", so "90+ years old" appears in our own output and in
# source documents that were already aggregated; without it the date label
# swallows the aggregate and the second pass reports the redaction as PHI.
_AGE_PHRASE = re.compile(
    r"\d{1,3}\+?\s*(?:-|\s)?\s*(?:years?|yrs?|y/?o|year[- ]old)\b.*"
)
_CLINICAL_SUBJECT_PREDICATES = {
    "admit",
    "complain",
    "diagnose",
    "discharge",
    "experience",
    "have",
    "present",
    "report",
    "suffer",
    "undergo",
}
_NON_NAME_PROPER_NOUNS = {
    "address",
    "attending",
    "contact",
    "date",
    "dob",
    "dr",
    "email",
    "mr",
    "mrs",
    "ms",
    "name",
    "patient",
    "phone",
    "telephone",
    "tel",
    "mobile",
    "fax",
    "mrn",
    "nhs",
    "physician",
    "professor",
}


class NerPhiDetectionError(RuntimeError):
    def __init__(self, error_code: str, status_code: int):
        super().__init__(error_code)
        self.error_code = error_code
        self.status_code = status_code


def configured_model_name() -> str:
    model_name = (os.getenv(NER_MODEL_ENV_VAR) or DEFAULT_NER_MODEL).strip()
    if not _SAFE_MODEL_NAME.fullmatch(model_name):
        raise NerPhiDetectionError("ner_model_unavailable", status_code=503)
    return model_name


@lru_cache(maxsize=4)
def load_spacy_pipeline(model_name: str) -> Any:
    try:
        import spacy

        nlp = spacy.load(model_name)
    except Exception as exc:
        raise NerPhiDetectionError("ner_model_unavailable", status_code=503) from exc

    if "ner" not in nlp.pipe_names:
        raise NerPhiDetectionError("ner_model_unavailable", status_code=503)
    return nlp


#: Personal titles as they are written: Title case or capitals. Lowercase
#: "ms", "dr" and "mr" are units and abbreviations far more often than titles.
_TITLES_AS_WRITTEN = (
    "Dr",
    "Dr.",
    "DR",
    "DR.",
    "Mr",
    "Mr.",
    "MR",
    "MR.",
    "Mrs",
    "Mrs.",
    "MRS",
    "MRS.",
    "Ms",
    "Ms.",
    "Professor",
    "PROFESSOR",
    "Prof",
    "Prof.",
    # Lowercase only where the word is never a unit or a finding: "mr" is
    # mitral regurgitation and "ms" milliseconds or multiple sclerosis.
    "dr",
    "dr.",
    "mrs",
    "mrs.",
    "professor",
)


def _name_token_pattern() -> dict:
    return {"IS_ALPHA": True, "IS_TITLE": True, "IS_STOP": False, "OP": "{1,3}"}


@lru_cache(maxsize=4)
def _context_matcher(model_name: str) -> Any:
    try:
        from spacy.matcher import Matcher

        nlp = load_spacy_pipeline(model_name)
        matcher = Matcher(nlp.vocab)
        matcher.add(
            "TITLE_PERSON",
            [
                [
                    # A title as written: "Ms" or "MR", never the unit "ms"
                    # in "7182 ms on the", which made "on the" a name. "MS"
                    # is left out: in a note it is multiple sclerosis.
                    {"ORTH": {"IN": list(_TITLES_AS_WRITTEN)}},
                    {"ORTH": ".", "OP": "?"},
                    _name_token_pattern(),
                ]
            ],
        )
        matcher.add(
            "PATIENT_PERSON",
            [[{"LOWER": "patient"}, {"ORTH": ":", "OP": "?"}, _name_token_pattern()]],
        )
        matcher.add(
            "NAME_FIELD_PERSON",
            [[{"LOWER": "name"}, {"ORTH": ":", "OP": "?"}, _name_token_pattern()]],
        )
        matcher.add(
            "ATTENDING_PERSON",
            [
                [
                    {"LOWER": "attending"},
                    {"LOWER": "physician"},
                    {"ORTH": ":", "OP": "?"},
                    _name_token_pattern(),
                ]
            ],
        )
        matcher.add(
            "CLINICAL_VERB_PERSON",
            [
                [
                    {"LOWER": {"IN": ["treated", "examined"]}},
                    {"LOWER": "by", "OP": "?"},
                    _name_token_pattern(),
                ]
            ],
        )
        matcher.add(
            "CLINICAL_SUBJECT_PERSON",
            [
                [
                    {
                        "IS_ALPHA": True,
                        "IS_TITLE": True,
                        "LOWER": {"NOT_IN": sorted(_NON_NAME_PROPER_NOUNS)},
                        "POS": "PROPN",
                        "OP": "{1,3}",
                    },
                    {"POS": "AUX", "OP": "?"},
                    {"LEMMA": {"IN": sorted(_CLINICAL_SUBJECT_PREDICATES)}},
                ]
            ],
        )
        return matcher
    except NerPhiDetectionError:
        raise
    except Exception as exc:
        raise NerPhiDetectionError("ner_model_unavailable", status_code=503) from exc


def _context_name_start(rule_name: str, doc: Any, start: int) -> int:
    if rule_name == "CLINICAL_SUBJECT_PERSON":
        return start
    if rule_name == "TITLE_PERSON":
        name_start = start + 1
        if name_start < len(doc) and doc[name_start].text == ".":
            name_start += 1
        return name_start
    if rule_name == "CLINICAL_VERB_PERSON":
        name_start = start + 1
        if name_start < len(doc) and doc[name_start].lower_ == "by":
            name_start += 1
        return name_start
    if rule_name == "ATTENDING_PERSON":
        name_start = start + 2
    else:
        name_start = start + 1
    if name_start < len(doc) and doc[name_start].text == ":":
        name_start += 1
    return name_start


def _context_name_end(rule_name: str, doc: Any, start: int, end: int) -> int:
    if rule_name != "CLINICAL_SUBJECT_PERSON":
        return end

    name_end = start
    while name_end < end and doc[name_end].pos_ == "PROPN":
        name_end += 1
    return name_end


def noun_evidence(text: str, model_name: str):
    """(proper-noun ranges, opening-modifier ranges) from one parse, or None.

    An opening modifier is the first word of a line or sentence that sits
    directly in front of a lowercase common noun - "Metabolic screen",
    "Genetics clinic", "Respiratory examination". Its capital is explained by
    its position and the construction is modifier-plus-noun. A word followed by
    a verb, a colon or a capital ("Kartik went", "Castellowe: follow-up") is not
    one, so an unseen name that opens a sentence is never mistaken for it.
    """
    try:
        nlp = load_spacy_pipeline(model_name)
        doc = nlp(text)
    except Exception:
        return None
    proper = tuple(
        (token.idx, token.idx + len(token.text))
        for token in doc
        if token.pos_ == "PROPN"
    )
    modifiers = []
    for token in doc[:-1]:
        following = doc[token.i + 1]
        opens = (
            token.i == 0
            or token.is_sent_start
            or text[: token.idx].rstrip(" \t").endswith("\n")
        )
        if (
            opens
            and token.is_alpha
            and token.text[:1].isupper()
            and not token.text.isupper()
            and following.pos_ == "NOUN"
            and following.text.islower()
            and following.idx == token.idx + len(token.text) + 1
        ):
            modifiers.append((token.idx, token.idx + len(token.text)))
    return proper, tuple(modifiers)


def proper_noun_offsets(text: str, model_name: str):
    """Character ranges of every proper-noun token, or None if unavailable.

    A tagger is the cheapest open-vocabulary evidence about whether a span is
    a name at all. "nursing staff", "multidisciplinary team" and "Enrolment"
    contain no proper noun; "rukmini balasubramanian" contains two even when
    it is typed in lower case, which is exactly the case a capitalisation
    rule would miss.

    Returns None rather than an empty tuple when the parse fails. The two mean
    opposite things: an empty tuple says "no proper nouns here", and treating
    a failed parse as that would exempt every span in the document.
    """
    try:
        nlp = load_spacy_pipeline(model_name)
        doc = nlp(text)
    except Exception:
        return None
    # No case-artifact probe and no rewriting of the sanitizer's tokens here:
    # this is evidence *for* a name. "Rose has pain." lower-cased reads as a
    # flower, and an initial between two tokens stops reading as a name once
    # its neighbours are disguised. Removing evidence is not the same as
    # declining to add a candidate.
    return tuple(
        (token.idx, token.idx + len(token.text))
        for token in doc
        if token.pos_ == "PROPN"
    )


def _should_keep_ner_entity(entity: Any) -> bool:
    normalized = entity.text.strip().casefold()
    if entity.label_ not in {"DATE", "TIME"} and _DATE_LIKE_TEXT.fullmatch(normalized):
        return False

    # Name-shaped labels are *not* filtered against the clinical vocabulary
    # here. A detector that deletes its own findings on dictionary grounds
    # makes the dictionary the final decision-maker: "Blood Pressure
    # Diagnostics Ltd" disappeared because it contains a vital sign, and the
    # evidence that it is a company - the designator "Ltd", the "sent to"
    # construction - never got a hearing. The finding is now emitted and
    # weighed in services.detection_evidence, where the vocabulary is one
    # source among several and naming context outranks it.

    if entity.label_ == "DATE" and normalized.isdigit():
        return len(normalized) == 4 and 1800 <= int(normalized) <= 2199
    # "62 years old" is an age, not a date. Ages are handled by the dedicated
    # age rule, which knows the Safe Harbor 90+ threshold; letting the date
    # label swallow them destroys an ordinary clinical fact.
    if entity.label_ == "DATE" and _AGE_PHRASE.fullmatch(normalized):
        return False
    return True


def _spans_overlap(start: int, end: int, entities: List[DetectedEntity]) -> bool:
    return any(start < entity.end and entity.start < end for entity in entities)


@lru_cache(maxsize=4)
def _invariant_spans(text: str) -> tuple:
    """Where the preservation layer finds protected clinical terms, once per text."""
    from services.clinical_preservation import clinical_invariants

    return tuple((item.start, item.end) for item in clinical_invariants(text))


def _is_safe_strict_proper_noun(doc: Any, start: int, end: int) -> bool:
    """True when this proper noun is clinical vocabulary rather than a name."""
    span_text = doc[start:end].text
    normalized = span_text.strip().casefold()
    if normalized in _NON_NAME_PROPER_NOUNS:
        return True
    if is_unconditional_clinical_term(span_text):
        return True
    # The capitalisation heuristic reads "Asian" as a name once an adjacent
    # identifier has been replaced by a placeholder. Only a whole standard
    # category label is exempt; "Asian Jordan" is not one.
    if is_race_ethnicity_category(span_text):
        return True
    # A run that is, or holds, a protected clinical term the preservation
    # layer recognises in place ("sagittal proton density series").
    span_start, span_end = doc[start].idx, doc[end - 1].idx + len(doc[end - 1].text)
    if any(
        low < span_end and span_start < high for low, high in _invariant_spans(doc.text)
    ):
        return True

    # Lookahead skips the possessive: "Parkinson's disease" tokenizes as
    # Parkinson + 's + disease, so a single-token lookahead lands on the
    # apostrophe and wrongly concludes the eponym is a surname.
    following = [doc[i].text for i in range(end, min(end + 4, len(doc)))]
    return protects_from_person_label(span_text, following)


#: Punctuation after which the next capitalised word is explained by ordinary
#: sentence or label capitalisation rather than by being a name.
_LINE_OPENING_PUNCTUATION = frozenset(".!?:;")
#: Spelt out rather than escaped so the characters survive every editor
#: and shell that has mangled them in this file before.
_HORIZONTAL_SPACE = chr(32) + chr(9)
_LINE_BREAKS = (chr(10), chr(13))

# Explicit person fields are strong evidence even when casing gives the
# tagger nothing to work with.  Keep the expression structural: it recognizes
# a field/title plus ordinary Unicode name tokens and contains no name list.
_LETTER = r"[^\W\d_]"
_NAME_ATOM = rf"(?:{_LETTER}\.|{_LETTER}+(?:[-'’]{_LETTER}+)*)"
_INITIALS_AND_SURNAME = rf"(?:(?:{_LETTER})\.[ \t]+){{1,3}}{_NAME_ATOM}"
_TWO_NAME_TOKENS = rf"{_NAME_ATOM}[ \t]+{_NAME_ATOM}"
_EXPLICIT_PERSON_CONTEXTS = (
    re.compile(
        rf"(?im)\b(?:patient(?:[ \t]+name)?|name|attending[ \t]+physician)"
        rf"[ \t]*(?::|\t)[ \t]*(?P<name>{_INITIALS_AND_SURNAME}|{_TWO_NAME_TOKENS})"
    ),
    # The title is matched as written and never straight after a number:
    # "7182 ms on the" is a duration, and "on the" is not a name. The name
    # after a real title may still be in any case.
    re.compile(
        rf"(?m)(?<!\d)(?<!\d )\b(?:Dr|DR|dr|Mr|MR|Mrs|MRS|mrs|Ms|Professor|PROFESSOR|professor|Prof)\.?[ \t]+"
        rf"(?P<name>{_INITIALS_AND_SURNAME}|{_TWO_NAME_TOKENS})"
    ),
)


def _explicit_person_context_entities(
    text: str, existing: List[DetectedEntity]
) -> List[DetectedEntity]:
    """Return complete names from explicit fields regardless of letter case."""
    entities: List[DetectedEntity] = []
    for pattern in _EXPLICIT_PERSON_CONTEXTS:
        for match in pattern.finditer(text):
            start, end = match.span("name")
            if any(
                entity.entity_type == "PERSON"
                and entity.start <= start
                and entity.end >= end
                for entity in existing + entities
            ):
                continue
            entities.append(
                DetectedEntity(
                    entity_type="PERSON",
                    start=start,
                    end=end,
                    source=SOURCE_CONTEXT_RULE,
                    score=None,
                    original_label="EXPLICIT_PERSON_CONTEXT",
                )
            )
    return entities


#: After a label word: up to two short code tokens ("A", "[Mass/volume]" is
#: not one), then the number it labels.
_LABELLED_VALUE_AFTER = re.compile(
    r"[ \t]+(?:[A-Za-z0-9][A-Za-z0-9\-]{0,5}[ \t]+){0,2}(\d+(?:\.\d+)?)"
)


def _labels_a_measurement(doc: Any, start: int, end: int) -> bool:
    """A single line-opening word that names a measured value on its own line.

    "Chromogranin A 112 ng/mL": the word is capitalised because it opens the
    line, and the number it labels carries a measurement unit. Only a unit
    counts - a count or a duration does not ("Jordan 2 days ago" is still a
    name) - and only at a line or sentence opening, where capitalisation
    carries no evidence of a name in the first place.
    """
    if end - start != 1:
        return False
    token = doc[start]
    text = doc.text
    preceding = text[: token.idx].rstrip(_HORIZONTAL_SPACE)
    if (
        preceding
        and not preceding.endswith(_LINE_BREAKS)
        and (preceding.rstrip()[-1:] not in _LINE_OPENING_PUNCTUATION)
    ):
        return False
    tail = token.idx + len(token.text)
    line_end = min(
        (i for i in (text.find(chr(10), tail), text.find(chr(13), tail)) if i >= 0),
        default=len(text),
    )
    match = _LABELLED_VALUE_AFTER.match(text, tail, line_end)
    if not match:
        return False
    context = numeric_context.classify(text, match.start(1), match.end(1))
    return context.reason == "value_with_unit"


def _is_case_artifact(doc: Any, nlp: Any, start: int, end: int) -> bool:
    """True when a token only looks like a proper noun because of its position.

    English capitalises the first word of every sentence and of many document
    labels, so a tagger seeing "Care was provided by ..." or a PDF title
    "Chart for ..." has no case evidence to work with and frequently guesses
    PROPN. The proper-noun fallback then proposes an ordinary noun as a name,
    and the sanitizer replaces a word carrying clinical meaning.

    Rather than exempting such words by listing them - which would only ever
    protect the words somebody thought of - this asks the tagger a second
    question: with the capitalisation removed, is it still a proper noun?
    "care" and "chart" fall back to NOUN; "kartik" and "jordan" stay PROPN,
    because their proper-noun reading comes from the word itself. The evidence
    is linguistic and open-vocabulary, so an unseen surname is still caught.

    Only single tokens at a sentence or line opening are probed. Anywhere else
    the capitalisation is already unexplained, and a multi-token run of
    capitals is not ordinary sentence case.
    """
    if end - start != 1 or nlp is None:
        return False

    token = doc[start]
    if not token.is_sent_start and token.idx != 0:
        preceding = doc.text[: token.idx]
        # A line break is itself an opening, and it must be tested before
        # the whitespace is stripped away. Stripping first loses it: a
        # label on the line after a placeholder then looks like it follows
        # a closing angle bracket rather than starting a line, and the
        # probe never runs on exactly the words - Enrolment, Claim,
        # Scanner - that a line-opening label is.
        opens_line = preceding.rstrip(_HORIZONTAL_SPACE).endswith(_LINE_BREAKS)
        stripped = preceding.rstrip()
        if (
            not opens_line
            and stripped
            and (stripped[-1] not in _LINE_OPENING_PUNCTUATION)
        ):
            return False

    sentence = token.sent if doc.has_annotation("SENT_START") else doc[:]
    sent_text = sentence.text
    offset = token.idx - sentence.start_char
    lowered = sent_text[:offset] + token.text[0].lower() + sent_text[offset + 1 :]
    if lowered == sent_text:
        return False

    try:
        probe_doc = nlp(lowered)
    except Exception:
        # A failed probe must not silently exempt a span; treat it as no
        # evidence, which leaves the finding in place.
        return False

    for probe_token in probe_doc:
        if probe_token.idx == offset:
            return probe_token.pos_ != "PROPN"
    return False


def _is_all_caps_case_artifact(doc: Any, nlp: Any, start: int, end: int) -> bool:
    """True when a run is capitalised only because its whole line is.

    A banner or heading in capitals ("SYNTHETIC DEMO RECORD - NOT A REAL
    PATIENT") or in title case ("Clinical Discharge Summary") gives a tagger no
    case evidence: every word looks like a proper noun. :func:`_is_case_artifact`
    lowers one initial, which leaves an upper-case word upper case and cannot
    answer for a run, so it cannot speak for these lines. This asks the same
    open-vocabulary question of the whole line: lowered, is the run still
    proper nouns? "synthetic" falls back to an adjective; "jordan lee" stays
    PROPN and is still proposed. Only a line that
    :func:`line_carries_no_case_evidence` is read this way, so an unexplained
    capital anywhere else is unaffected, and the models and the naming-context
    rules still read every span.
    """
    if nlp is None or end <= start:
        return False
    text = doc.text
    first, last = doc[start].idx, doc[end - 1].idx + len(doc[end - 1].text)
    line_start = text.rfind(chr(10), 0, first) + 1
    line_end = text.find(chr(10), last)
    line_end = len(text) if line_end < 0 else line_end
    line = text[line_start:line_end]
    if not line_carries_no_case_evidence(line):
        return False
    try:
        probe = nlp(line.lower())
    except Exception:
        # A failed probe is no evidence, which leaves the finding in place.
        return False
    seen = False
    for token in probe:
        position = line_start + token.idx
        if first <= position < last and token.is_alpha:
            seen = True
            if token.pos_ == "PROPN":
                return False
    return seen


def _has_no_letter_case(text: str) -> bool:
    """Whether this span is written in a script that does not have case.

    The strict proper-noun rule is a capitalisation argument wearing a
    tagger's clothes: a run of PROPN tokens is evidence because English
    capitalises names. Chinese, Japanese, Korean, Arabic, Hebrew, Thai and
    Devanagari have no capitals at all, so an English pipeline's opinion that
    a run of them is PROPN carries none of that evidence - it is the tagger
    having nothing to say.

    Measured: a Chinese sentence whose identifiers had already been replaced
    came back with 的记录 - "the record of" - proposed as a person, which held
    a workbook whose every identifier was gone.
    """
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    cased = sum(1 for character in letters if character.lower() != character.upper())
    return cased == 0


def _strict_proper_noun_entities(
    doc: Any,
    existing: List[DetectedEntity],
    nlp: Any = None,
) -> List[DetectedEntity]:
    entities: List[DetectedEntity] = []
    token_index = 0
    while token_index < len(doc):
        token = doc[token_index]
        if token.pos_ != "PROPN" or not token.is_alpha:
            token_index += 1
            continue
        if _has_no_letter_case(token.text):
            # Nothing was capitalised, because nothing in this script can be.
            # The rule has no evidence here; the models still see the span.
            token_index += 1
            continue
        if (
            not token.text[0].isupper()
            and not (
                token_index + 1 < len(doc)
                and doc[token_index + 1].pos_ == "PROPN"
                and doc[token_index + 1].is_alpha
            )
            and not (token_index > 0 and doc[token_index - 1].pos_ == "PROPN")
        ):
            # A lone lowercase word has no capitalisation and no neighbouring
            # name part: "kin" in "Next of kin" is the tagger's guess. A
            # lowercase run ("rukmini balasubramanian") is still a candidate,
            # and the models still read every span.
            token_index += 1
            continue

        start = token_index
        end = start + 1
        while end < len(doc) and doc[end].pos_ == "PROPN" and doc[end].is_alpha:
            end += 1

        # An existing detector may cover only the first token of a proper-name
        # run. Dropping the whole run on any overlap leaves the uncovered
        # surname in output. Emit each still-uncovered contiguous part; overlap
        # resolution can then join it back to the confirmed PERSON candidate.
        all_entities = existing + entities
        run_start = doc[start].idx
        run_end = doc[end - 1].idx + len(doc[end - 1].text)
        overlapping = [
            entity
            for entity in all_entities
            if run_start < entity.end and entity.start < run_end
        ]
        # Preserve the former clinical safety behavior unless an existing
        # PERSON finding establishes that this is a partially covered name.
        # A non-person model span over "Urine" must not turn the remaining
        # word "dipstick" into a surname.
        if overlapping and not any(
            entity.entity_type == "PERSON" and entity.source == SOURCE_CONTEXT_RULE
            for entity in overlapping
        ):
            token_index = end
            continue
        uncovered = []
        group_start = None
        for index in range(start, end):
            token = doc[index]
            covered = _spans_overlap(
                token.idx, token.idx + len(token.text), all_entities
            )
            if not covered and group_start is None:
                group_start = index
            if covered and group_start is not None:
                uncovered.append((group_start, index))
                group_start = None
        if group_start is not None:
            uncovered.append((group_start, end))
        for part_start, part_end in uncovered:
            span = doc[part_start:part_end]
            if (
                _is_safe_strict_proper_noun(doc, part_start, part_end)
                or _is_case_artifact(doc, nlp, part_start, part_end)
                or _is_all_caps_case_artifact(doc, nlp, part_start, part_end)
                or _labels_a_measurement(doc, part_start, part_end)
            ):
                continue
            entities.append(
                DetectedEntity(
                    entity_type="PERSON",
                    start=span.start_char,
                    end=span.end_char,
                    source=SOURCE_STRICT_PROPER_NOUN,
                    score=None,
                    original_label="STRICT_PROPER_NOUN",
                )
            )
        token_index = end

    return entities


class SpacyNerPhiDetector:
    def __init__(
        self,
        model_name: str,
        high_recall_proper_nouns: bool = False,
    ):
        self.model_name = model_name
        self.high_recall_proper_nouns = high_recall_proper_nouns

    def detect(self, text: str) -> List[DetectedEntity]:
        check_request_deadline()
        nlp = load_spacy_pipeline(self.model_name)
        try:
            doc = nlp(text)
        except Exception as exc:
            raise NerPhiDetectionError("ner_inference_failed", status_code=500) from exc

        entities = [
            DetectedEntity(
                entity_type=SPACY_PHI_LABEL_MAP[entity.label_],
                start=entity.start_char,
                end=entity.end_char,
                source=SOURCE_NER,
                score=None,
                original_label=entity.label_,
            )
            for entity in doc.ents
            if entity.label_ in SPACY_PHI_LABEL_MAP and _should_keep_ner_entity(entity)
        ]

        matcher = _context_matcher(self.model_name)
        try:
            context_matches = matcher(doc)
        except Exception as exc:
            raise NerPhiDetectionError("ner_inference_failed", status_code=500) from exc

        for match_id, start, end in context_matches:
            rule_name = doc.vocab.strings[match_id]
            name_start = _context_name_start(rule_name, doc, start)
            name_end = _context_name_end(rule_name, doc, start, end)
            if name_start >= name_end:
                continue
            name_span = doc[name_start:name_end]
            entities.append(
                DetectedEntity(
                    entity_type="PERSON",
                    start=name_span.start_char,
                    end=name_span.end_char,
                    source=SOURCE_CONTEXT_RULE,
                    score=None,
                    original_label=rule_name,
                )
            )

        entities.extend(_explicit_person_context_entities(text, entities))

        if self.high_recall_proper_nouns:
            entities.extend(_strict_proper_noun_entities(doc, entities, nlp))

        return entities
