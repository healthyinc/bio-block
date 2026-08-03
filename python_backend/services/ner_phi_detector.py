import os
import re
from functools import lru_cache
from typing import Any, List

from services.phi_detection import (
    SOURCE_CONTEXT_RULE,
    SOURCE_NER,
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
    "DATE": "DATE",
    "TIME": "TIME",
}
_NON_PHI_CLINICAL_TERMS = {"ct", "mri", "pet", "ecg", "ekg", "x-ray", "xray"}


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
        return matcher
    except NerPhiDetectionError:
        raise
    except Exception as exc:
        raise NerPhiDetectionError("ner_model_unavailable", status_code=503) from exc


def _context_name_start(rule_name: str, doc: Any, start: int) -> int:
    if rule_name == "TITLE_PERSON":
        name_start = start + 1
        if name_start < len(doc) and doc[name_start].text == ".":
            name_start += 1
        return name_start
    if rule_name == "ATTENDING_PERSON":
        name_start = start + 2
    else:
        name_start = start + 1
    if name_start < len(doc) and doc[name_start].text == ":":
        name_start += 1
    return name_start


def _should_keep_ner_entity(entity: Any) -> bool:
    normalized = entity.text.strip().casefold()
    if entity.label_ == "ORG" and normalized in _NON_PHI_CLINICAL_TERMS:
        return False
    if entity.label_ == "DATE" and normalized.isdigit():
        return len(normalized) == 4 and 1800 <= int(normalized) <= 2199
    return True


class SpacyNerPhiDetector:
    def __init__(self, model_name: str):
        self.model_name = model_name

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
            if name_start >= end:
                continue
            name_span = doc[name_start:end]
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

        return entities
