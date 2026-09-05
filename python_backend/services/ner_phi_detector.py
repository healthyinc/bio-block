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
_AGE_PHRASE = re.compile(r"\d{1,3}\s*(?:-|\s)?\s*(?:years?|yrs?|y/?o|year[- ]old)\b.*")
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


def _following_words(entity: Any, count: int = 4) -> List[str]:
    """Surface tokens after an entity, for clinical-construction lookahead."""
    doc = entity.doc
    return [doc[i].text for i in range(entity.end, min(entity.end + count, len(doc)))]


def _should_keep_ner_entity(entity: Any) -> bool:
    normalized = entity.text.strip().casefold()
    if entity.label_ not in {"DATE", "TIME"} and _DATE_LIKE_TEXT.fullmatch(normalized):
        return False

    # The vocabulary is consulted for every name-shaped label, not only
    # PERSON. spaCy routinely tags eponyms such as Addison, Bell and Cushing
    # as ORG, and the previous PERSON-only check never saw them.
    if entity.label_ in {"PERSON", "ORG", "FAC", "NORP", "LOC", "GPE"}:
        if protects_from_person_label(entity.text, _following_words(entity)):
            return False

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


def _strict_proper_noun_entities(
    doc: Any,
    existing: List[DetectedEntity],
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
            entities.extend(_strict_proper_noun_entities(doc, entities))

        return entities
