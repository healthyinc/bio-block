"""Consistent, study-local surrogate identifiers.

Replacing every name with the same ``<REDACTED_NAME>`` is safe but destroys
coreference: a reader can no longer tell whether two mentions are the same
person, and "the patient" versus "the referring physician" collapses into one
token. Surrogates keep the sentence readable and the document internally
consistent:

    Patient PATIENT_001 was seen by PROVIDER_001 at FACILITY_001.
    PATIENT_001 will follow up with PROVIDER_001 next week.

Three properties matter, and each is a deliberate constraint:

* **Study-local.** An allocator is created per authorized upload bundle and
  dies with it. Two studies that contain the same person produce unrelated
  surrogates, so no cross-study linkage is created.
* **Not derived from the value.** The surrogate is assigned by order of first
  appearance, never computed from the original text. A hash - salted or not -
  is a function of the identifier, and an attacker with a candidate list can
  confirm a guess by recomputing it. Order of appearance carries no such
  information.
* **Never persisted.** The mapping lives in memory for the duration of one
  request. It is not written to logs, reports, manifests, blockchain metadata
  or any public artifact. ``SurrogateAllocator`` deliberately exposes no
  method that returns the original values.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

#: Entity type -> surrogate prefix. A type absent from this map has no
#: meaningful identity to preserve and is handled by fixed redaction instead.
SURROGATE_PREFIXES: Mapping[str, str] = {
    "PERSON": "PATIENT",
    "PERSON_PROVIDER": "PROVIDER",
    "FACILITY": "FACILITY",
    "ORGANIZATION": "ORG",
    "LOCATION": "PLACE",
    "ADDRESS": "ADDRESS",
    "MEDICAL_RECORD_NUMBER": "RECORD",
    "PATIENT_ID": "PATIENTID",
    "HEALTH_PLAN_ID": "PLAN",
    "INSURANCE_ID": "PLAN",
    "ACCESSION_NUMBER": "ACCESSION",
    "DEVICE_ID": "DEVICE",
    "IDENTIFIER": "IDENTIFIER",
    "USERNAME": "USER",
}

#: Matches any surrogate this module can emit, so the residual validator can
#: mask them out rather than re-detecting our own output as PHI.
#: No trailing word boundary: two adjacent names produce
#: "PATIENT_001PATIENT_002", and a trailing boundary would leave both unmasked,
#: so the residual scan would flag our own output as surviving PHI.
SURROGATE_PATTERN = re.compile(
    r"\b(?:" + "|".join(sorted(set(SURROGATE_PREFIXES.values()))) + r")_\d{3,}"
)

#: Titles that identify a person as a care provider rather than the patient.
#: Deterministic evidence only: a title actually present in the text.
_PROVIDER_TITLES = frozenset(
    {
        "dr",
        "dr.",
        "doctor",
        "prof",
        "prof.",
        "professor",
        "consultant",
        "attending",
        "physician",
        "surgeon",
        "registrar",
        "resident",
        "nurse",
        "practitioner",
        "radiologist",
        "pathologist",
        "oncologist",
        "cardiologist",
    }
)
_PROVIDER_CONTEXT = re.compile(
    r"(?:seen|treated|reviewed|referred|signed|countersigned|examined)\s+by\s*$",
    re.IGNORECASE,
)


def looks_like_provider(text: str, start: int) -> bool:
    """Deterministic check for whether a PERSON span is a care provider.

    Uses only evidence present in the text: an immediately preceding title, or
    an explicit "seen by"/"referred by" construction. Absent that evidence the
    person is treated as the patient, which is the conservative default - a
    patient surrogate is never mistaken for a provider surrogate in a way that
    would imply the patient is staff.
    """
    prefix = text[max(0, start - 60) : start]
    if _PROVIDER_CONTEXT.search(prefix):
        return True
    tokens = re.findall(r"[A-Za-z.]+", prefix)
    if not tokens:
        return False
    return tokens[-1].strip().casefold() in _PROVIDER_TITLES


def _normalize(value: str) -> str:
    """Fold a value so that "Dr. Jane Doe" and "Jane Doe" share a surrogate."""
    cleaned = re.sub(r"[^\w\s]", " ", value or "").strip().casefold()
    tokens = [t for t in cleaned.split() if t and t not in _PROVIDER_TITLES]
    return " ".join(tokens)


@dataclass
class SurrogateAllocator:
    """Assigns stable surrogates within one study, and nothing beyond it.

    Not thread-shared by design: one allocator belongs to one upload bundle.
    A lock is held anyway so a single bundle processed concurrently still
    yields consistent numbering.
    """

    #: (prefix, normalized value) -> surrogate. In memory only.
    _assigned: Dict[Tuple[str, str], str] = field(default_factory=dict, repr=False)
    _counters: Dict[str, int] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def surrogate_for(self, entity_type: str, value: str) -> Optional[str]:
        """Return the study-local surrogate for this value, allocating if new.

        Returns ``None`` when the type has no surrogate form, so the caller
        falls back to fixed redaction.
        """
        prefix = SURROGATE_PREFIXES.get(entity_type)
        if prefix is None:
            return None
        key = (prefix, _normalize(value))
        if not key[1]:
            return None
        with self._lock:
            existing = self._assigned.get(key)
            if existing is not None:
                return existing
            nxt = self._counters.get(prefix, 0) + 1
            self._counters[prefix] = nxt
            surrogate = f"{prefix}_{nxt:03d}"
            self._assigned[key] = surrogate
            return surrogate

    def counts(self) -> Dict[str, int]:
        """How many distinct entities of each kind were replaced.

        Counts only. There is deliberately no accessor that returns the
        original values or the mapping itself, so a manifest or log cannot
        obtain them even by accident.
        """
        with self._lock:
            return dict(sorted(self._counters.items()))

    def distinct_entities(self) -> int:
        with self._lock:
            return len(self._assigned)

    def __repr__(self) -> str:  # pragma: no cover - defensive
        return f"SurrogateAllocator(distinct={self.distinct_entities()})"
