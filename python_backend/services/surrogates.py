"""Neutral, deterministic, dataset-scoped pseudonyms.

Replacing every name with one fixed placeholder is safe but destroys
coreference: a reader can no longer tell whether two mentions are the same
person. A realistic fake name keeps coreference but can itself look like PHI,
which confuses the residual scan and every human reviewer. Neutral numbered
tokens do neither:

    <PERSON_1> was seen at <FACILITY_1>; call <PHONE_1> if <PERSON_1> worsens.

Four properties are deliberate constraints:

* **Consistent within one dataset.** One allocator belongs to one upload or
  study - one text, one table, one workbook, one batch. The same value always
  gets the same token in it; different values get different tokens. Values are
  compared after normalisation - Unicode form, case, diacritics and titles for
  names, digits for telephone numbers, letters and digits for identifiers - so
  formatting does not split one value in two.
* **Unrelated across datasets.** A new allocator starts again at 1. The same
  person in two uploads gets tokens with no relationship, so no linkage is
  created across users or datasets.
* **Not derived from the value.** Numbers are assigned in order of first
  appearance, never computed from the value. A hash - salted or not - is a
  function of the identifier, and anyone with a candidate list could confirm a
  guess by recomputing it.
* **Never exposed.** The mapping lives in memory for one request. There is no
  method that returns it or an original value; only counts leave.

A token already present in the input is reserved before anything is
allocated, so a genuine ``<PERSON_1>`` in the source is never given to a
different person. Entity types without a family - dates, times, uncertain
ages - keep their fixed redaction.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Set, Tuple

#: Entity type -> token family. A type absent from this map has no identity
#: worth preserving and is handled by fixed redaction instead.
SURROGATE_FAMILIES: Mapping[str, str] = {
    "PERSON": "PERSON",
    "PERSON_PROVIDER": "PROVIDER",
    "FACILITY": "FACILITY",
    "ORGANIZATION": "ORGANIZATION",
    "LOCATION": "LOCATION",
    "ADDRESS": "LOCATION",
    "MEDICAL_RECORD_NUMBER": "RECORD_ID",
    "PATIENT_ID": "RECORD_ID",
    "IDENTIFIER": "RECORD_ID",
    "OTHER_UNIQUE_ID": "RECORD_ID",
    "HEALTH_PLAN_ID": "HEALTH_PLAN_ID",
    "INSURANCE_ID": "HEALTH_PLAN_ID",
    "ACCOUNT_NUMBER": "ACCOUNT_ID",
    "ACCESSION_NUMBER": "ACCESSION_ID",
    "DEVICE_ID": "DEVICE_ID",
    "VEHICLE_ID": "VEHICLE_ID",
    "LICENSE_NUMBER": "LICENSE_ID",
    "CERTIFICATE_NUMBER": "CERTIFICATE_ID",
    "NATIONAL_ID": "NATIONAL_ID",
    "BIOMETRIC_ID": "BIOMETRIC_ID",
    "USERNAME": "USERNAME",
    "PHONE_NUMBER": "PHONE",
    "FAX_NUMBER": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "URL": "URL",
    "IP_ADDRESS": "IP_ADDRESS",
    "US_SSN": "SSN",
    "SSN": "SSN",
    "AMBIGUOUS_NUMBER": "NUMBER",
}

_FAMILIES = tuple(sorted(set(SURROGATE_FAMILIES.values()), key=lambda f: (-len(f), f)))

#: Every token this module can emit. Angle brackets delimit a token, so two
#: adjacent tokens ("<PERSON_1><PERSON_2>") are still two matches.
SURROGATE_PATTERN = re.compile(r"<(" + "|".join(_FAMILIES) + r")_([1-9][0-9]*)>")

#: The study-local surrogates written before neutral tokens. Nothing emits
#: them any more; they are recognised so text released earlier is still read
#: as generated output rather than as surviving identifiers.
LEGACY_SURROGATE_PATTERN = re.compile(
    r"\b(?:PATIENT|PROVIDER|FACILITY|ORG|PLACE|ADDRESS|RECORD|PATIENTID|PLAN"
    r"|ACCESSION|DEVICE|IDENTIFIER|USER)_\d{3,}"
)

_DIGITS_ONLY = frozenset({"PHONE"})
_CASE_ONLY = frozenset({"EMAIL", "URL", "IP_ADDRESS"})
_ALPHANUMERIC = frozenset(
    {
        "RECORD_ID",
        "HEALTH_PLAN_ID",
        "ACCOUNT_ID",
        "ACCESSION_ID",
        "DEVICE_ID",
        "VEHICLE_ID",
        "LICENSE_ID",
        "CERTIFICATE_ID",
        "NATIONAL_ID",
        "BIOMETRIC_ID",
        "SSN",
        "USERNAME",
    }
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
    person is treated as the patient.
    """
    prefix = text[max(0, start - 60) : start]
    if _PROVIDER_CONTEXT.search(prefix):
        return True
    tokens = re.findall(r"[A-Za-z.]+", prefix)
    if not tokens:
        return False
    return tokens[-1].strip().casefold() in _PROVIDER_TITLES


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize(
        "NFKD", unicodedata.normalize("NFKC", value or "")
    )
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def normalize_value(entity_type: str, value: str) -> str:
    """The form two values must share to receive one token. Never logged."""
    family = SURROGATE_FAMILIES.get(entity_type, entity_type)
    folded = _fold(value)
    if family in _DIGITS_ONLY:
        return "".join(ch for ch in folded if ch.isdigit())
    if family in _CASE_ONLY:
        return folded.strip()
    if family in _ALPHANUMERIC:
        return "".join(ch for ch in folded if ch.isalnum())
    cleaned = re.sub(r"[^\w\s]", " ", folded)
    return " ".join(token for token in cleaned.split() if token not in _PROVIDER_TITLES)


def canonical_identity(
    secret: bytes, family: str, normalized: str, scope: str = "document"
) -> str:
    """A typed identity for one semantically classified value: a keyed digest.

    Created only after an occurrence has been classified, so the same digits
    as a contact and as a laboratory result are different identities (only
    the contact ever reaches here). The value itself is never a key.
    """
    message = "\0".join((family, scope, normalized)).encode("utf-8", "surrogatepass")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


@dataclass
class SurrogateAllocator:
    """Assigns neutral tokens within one dataset, and nothing beyond it."""

    #: canonical identity (a job-secret keyed digest) -> token. In memory only;
    #: no plaintext value is ever a key.
    _assigned: Dict[str, str] = field(default_factory=dict, repr=False)
    _families: Dict[str, str] = field(default_factory=dict, repr=False)
    _secret: bytes = field(default_factory=lambda: secrets.token_bytes(32), repr=False)
    _next: Dict[str, int] = field(default_factory=dict, repr=False)
    _reserved: Dict[str, Set[int]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reserve_existing(self, text: str) -> None:
        """Keep the numbers of tokens already present in the input out of use."""
        if not text or "<" not in text:
            return
        with self._lock:
            for match in SURROGATE_PATTERN.finditer(text):
                self._reserved.setdefault(match.group(1), set()).add(
                    int(match.group(2))
                )

    def surrogate_for(self, entity_type: str, value: str) -> Optional[str]:
        """The dataset-local token for this value, allocating one if it is new.

        Returns ``None`` when the type has no family, or the value normalises
        to nothing, so the caller falls back to fixed redaction.
        """
        family = SURROGATE_FAMILIES.get(entity_type)
        if family is None:
            return None
        normalized = normalize_value(entity_type, value)
        if not normalized:
            return None
        key = canonical_identity(self._secret, family, normalized)
        with self._lock:
            existing = self._assigned.get(key)
            if existing is not None:
                return existing
            reserved = self._reserved.get(family, ())
            number = self._next.get(family, 1)
            while number in reserved:
                number += 1
            self._next[family] = number + 1
            token = f"<{family}_{number}>"
            self._assigned[key] = token
            self._families[key] = family
            return token

    def emitted(self) -> frozenset:
        """Every token this allocator actually produced.

        A residual scan that discounts anything token-shaped skips the span
        an attacker picked when the upload arrives already carrying one.
        Callers check against this instead.
        """
        with self._lock:
            return frozenset(self._assigned.values())

    def counts(self) -> Dict[str, int]:
        """How many distinct values of each family were replaced. Counts only."""
        with self._lock:
            totals: Dict[str, int] = {}
            for family in self._families.values():
                totals[family] = totals.get(family, 0) + 1
            return dict(sorted(totals.items()))

    def distinct_entities(self) -> int:
        with self._lock:
            return len(self._assigned)

    def __repr__(self) -> str:  # pragma: no cover - defensive
        return f"SurrogateAllocator(distinct={self.distinct_entities()})"


HASH_SURROGATE_PATTERN = re.compile(
    r"\b(?:PERSON|MRN|PATIENT_ID|HEALTH_PLAN|ACCESSION|DEVICE)_[0-9A-F]{8}\b"
)
