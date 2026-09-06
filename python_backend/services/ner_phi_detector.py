import os
import re
from functools import lru_cache
from typing import Any, List

from services.clinical_vocabulary import (
    is_unconditional_clinical_term,
    protects_from_person_label,
)
from services.phi_detection import (
    SOURCE_CONTEXT_RULE,
    SOURCE_NER,
    SOURCE_STRICT_PROPER_NOUN,
    DetectedEntity,
)

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
                    {
                        "LOWER": {
                            "IN": [
                                "dr", "dr.", "mr", "mr.", "mrs", "mrs.",
                                "ms", "ms.", "professor",
                            ]
                        }
                    },
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


def _is_safe_strict_proper_noun(doc: Any, start: int, end: int) -> bool:
    """True when this proper noun is clinical vocabulary rather than a name."""
    span_text = doc[start:end].text
    normalized = span_text.strip().casefold()
    if normalized in _NON_NAME_PROPER_NOUNS:
        return True
    if is_unconditional_clinical_term(span_text):
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
        if not opens_line and stripped and (
            stripped[-1] not in _LINE_OPENING_PUNCTUATION
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

        start = token_index
        end = start + 1
        while end < len(doc) and doc[end].pos_ == "PROPN" and doc[end].is_alpha:
            end += 1

        span = doc[start:end]
        all_entities = existing + entities
        if (
            not _is_safe_strict_proper_noun(doc, start, end)
            and not _is_case_artifact(doc, nlp, start, end)
            and not _spans_overlap(span.start_char, span.end_char, all_entities)
        ):
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

        if self.high_recall_proper_nouns:
            entities.extend(_strict_proper_noun_entities(doc, entities, nlp))

        return entities
