"""Clinical vocabulary that must survive de-identification.

De-identification that removes "Parkinson" from "Parkinson's disease" is
technically safe and medically useless. Phase 9 measured useful-text
preservation at 0.214, and the cause was here rather than in either pinned
model:

* the eponym list held five terms and was consulted only for spaCy ``PERSON``
  labels and the proper-noun heuristic, so an eponym that spaCy tagged ``ORG``
  (Addison, Bell, Cushing, Wilson, Graves) was never checked at all;
* the "is this followed by 'disease'?" test looked at the immediately next
  token, which for the standard possessive form *Parkinson's disease* is
  ``'s``, not ``disease``. Every possessive eponym therefore failed the test
  that exists to protect it.

This module is the single source of truth for "this word is medicine, not a
person". It is deliberately a closed vocabulary of clinical terms, never a
pattern that could admit an arbitrary name: every entry is a term that appears
in a medical dictionary, and no entry is a surname on its own account.

An eponym only earns protection **in a clinical construction** - "Parkinson's
disease", "Crohn disease", "Bell's palsy". A bare "Parkinson" with no clinical
head noun is still treated as a possible surname, because a patient really can
be called Parkinson.
"""

from __future__ import annotations

import re
from typing import Sequence, Tuple

VOCABULARY_VERSION = "clinical-vocab-v1"

#: Head nouns that turn an eponym into a condition rather than a person.
#: "Parkinson's *disease*", "Hodgkin *lymphoma*", "Bell's *palsy*".
CLINICAL_HEAD_NOUNS: frozenset[str] = frozenset(
    {
        "disease",
        "diseases",
        "syndrome",
        "syndromes",
        "lymphoma",
        "lymphomas",
        "palsy",
        "sign",
        "signs",
        "reflex",
        "test",
        "manoeuvre",
        "maneuver",
        "phenomenon",
        "anaemia",
        "anemia",
        "dementia",
        "colitis",
        "arthritis",
        "sarcoma",
        "carcinoma",
        "tumour",
        "tumor",
        "cyst",
        "fracture",
        "deformity",
        "murmur",
        "node",
        "nodes",
        "gland",
        "duct",
        "space",
        "triangle",
        "canal",
        "ligament",
        "classification",
        "score",
        "scale",
        "index",
        "criteria",
        "stain",
        "procedure",
        "operation",
        "incision",
        "approach",
        "position",
        "grade",
        "staging",
    }
)

#: Eponymous conditions, signs, scores and anatomical structures. Protected
#: only in a clinical construction (see ``is_clinical_eponym_use``).
CLINICAL_EPONYMS: frozenset[str] = frozenset(
    {
        "addison",
        "alzheimer",
        "asperger",
        "babinski",
        "barrett",
        "bell",
        "bowman",
        "broca",
        "brudzinski",
        "buerger",
        "castleman",
        "charcot",
        "cheyne",
        "chiari",
        "colles",
        "conn",
        "cooper",
        "crohn",
        "cushing",
        "down",
        "dupuytren",
        "ehlers",
        "epstein",
        "ewing",
        "fallot",
        "glasgow",
        "graves",
        "guillain",
        "hashimoto",
        "hirschsprung",
        "hodgkin",
        "horner",
        "hunt",
        "huntington",
        "kaposi",
        "kawasaki",
        "kernig",
        "klinefelter",
        "krukenberg",
        "langerhans",
        "lewy",
        "marfan",
        "mallory",
        "meckel",
        "meniere",
        "menzies",
        "murphy",
        "paget",
        "pancoast",
        "parkinson",
        "peyronie",
        "purkinje",
        "raynaud",
        "reed",
        "reynolds",
        "romberg",
        "sjogren",
        "stevens",
        "still",
        "tinel",
        "turner",
        "virchow",
        "wernicke",
        "wilms",
        "wilson",
        "wolff",
    }
)

#: Imaging modalities and study abbreviations. Never a person, so protected
#: unconditionally.
MODALITY_TERMS: frozenset[str] = frozenset(
    {
        "ct",
        "cta",
        "mri",
        "mra",
        "mrcp",
        "pet",
        "spect",
        "ecg",
        "ekg",
        "eeg",
        "emg",
        "us",
        "ultrasound",
        "xray",
        "x-ray",
        "dexa",
        "dxa",
        "fluoroscopy",
        "mammogram",
        "mammography",
        "angiogram",
        "angiography",
        "echocardiogram",
        "echo",
        "endoscopy",
        "colonoscopy",
        "bronchoscopy",
        "scan",
        "radiograph",
    }
)

#: Units, laboratory analytes and measurement vocabulary. Never a person.
CLINICAL_MEASUREMENT_TERMS: frozenset[str] = frozenset(
    {
        "hba1c",
        "wbc",
        "rbc",
        "hgb",
        "hb",
        "hct",
        "mcv",
        "plt",
        "inr",
        "aptt",
        "ldl",
        "hdl",
        "tsh",
        "psa",
        "bnp",
        "crp",
        "esr",
        "egfr",
        "bun",
        "alt",
        "ast",
        "alp",
        "ggt",
        "ldh",
        "ck",
        "troponin",
        "creatinine",
        "glucose",
        "sodium",
        "potassium",
        "chloride",
        "bicarbonate",
        "calcium",
        "magnesium",
        "phosphate",
        "albumin",
        "bilirubin",
        "mg",
        "mcg",
        "kg",
        "ml",
        "dl",
        "mmol",
        "mmhg",
        "bpm",
        "iu",
        "meq",
        "cm",
        "mm",
        "bmi",
        "spo2",
        "fio2",
        "blood",
        "pressure",
        "pulse",
        "rate",
        "temperature",
        "saturation",
        "oxygen",
        "respiratory",
        "systolic",
        "diastolic",
        "weight",
        "height",
        "dose",
        "daily",
        "nightly",
        "twice",
        "once",
        "oral",
        "intravenous",
        "subcutaneous",
    }
)

#: Terms so distinctive that no personal name contains one. Used for the
#: token-wise check on multi-word spans: a detector that labels
#: "Atorvastatin 40 mg nightly" a PERSON has misclassified a prescription, and
#: checking the span as a whole against a single-term vocabulary never catches
#: it. Restricted to drugs and analytes on purpose - generic anatomy words
#: like "Heart" or "Rice" really are surnames.
DISTINCTIVE_CLINICAL_TOKENS: frozenset[str] = frozenset()

#: Anatomical and directional vocabulary that appears on images and in notes.
ANATOMY_TERMS: frozenset[str] = frozenset(
    {
        "anterior",
        "posterior",
        "lateral",
        "medial",
        "superior",
        "inferior",
        "proximal",
        "distal",
        "dorsal",
        "ventral",
        "cranial",
        "caudal",
        "sagittal",
        "coronal",
        "axial",
        "transverse",
        "left",
        "right",
        "bilateral",
        "supine",
        "prone",
        "erect",
        "abdomen",
        "thorax",
        "chest",
        "pelvis",
        "cranium",
        "skull",
        "spine",
        "cervical",
        "thoracic",
        "lumbar",
        "sacral",
        "femur",
        "tibia",
        "fibula",
        "humerus",
        "radius",
        "ulna",
        "clavicle",
        "scapula",
        "liver",
        "spleen",
        "kidney",
        "pancreas",
        "lung",
        "heart",
        "aorta",
        "brain",
        "cerebellum",
        "ventricle",
        "atrium",
    }
)

#: Common generic medication names. A closed list rather than a suffix rule:
#: "-in"/"-ol" suffixes match surnames as readily as drugs, and a rule that
#: could protect a surname is exactly what must not exist here.
MEDICATION_TERMS: frozenset[str] = frozenset(
    {
        "paracetamol", "acetaminophen", "ibuprofen", "naproxen", "aspirin",
        "codeine", "morphine", "fentanyl", "tramadol", "oxycodone",
        "amoxicillin", "penicillin", "flucloxacillin", "cefalexin",
        "ceftriaxone", "azithromycin", "clarithromycin", "erythromycin",
        "doxycycline", "gentamicin", "vancomycin", "metronidazole",
        "ciprofloxacin", "trimethoprim", "nitrofurantoin",
        "metformin", "gliclazide", "insulin", "sitagliptin", "empagliflozin",
        "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
        "ramipril", "lisinopril", "enalapril", "perindopril", "losartan",
        "candesartan", "amlodipine", "nifedipine", "diltiazem", "verapamil",
        "bisoprolol", "atenolol", "metoprolol", "carvedilol", "propranolol",
        "furosemide", "bendroflumethiazide", "spironolactone", "indapamide",
        "warfarin", "apixaban", "rivaroxaban", "dabigatran", "clopidogrel",
        "heparin", "enoxaparin",
        "omeprazole", "lansoprazole", "pantoprazole", "ranitidine",
        "salbutamol", "albuterol", "beclometasone", "budesonide",
        "prednisolone", "prednisone", "dexamethasone", "hydrocortisone",
        "levothyroxine", "carbimazole", "allopurinol", "colchicine",
        "sertraline", "fluoxetine", "citalopram", "escitalopram",
        "amitriptyline", "mirtazapine", "venlafaxine", "diazepam",
        "lorazepam", "gabapentin", "pregabalin", "levetiracetam",
        "lamotrigine", "carbamazepine", "phenytoin", "sodium valproate",
        "methotrexate", "azathioprine", "ciclosporin", "tacrolimus",
        "cisplatin", "carboplatin", "doxorubicin", "cyclophosphamide",
        "tamoxifen", "anastrozole", "rituximab", "trastuzumab",
        "ondansetron", "metoclopramide", "loperamide", "senna", "lactulose",
    }
)

#: Everything that is never a personal name, in one set.
UNCONDITIONAL_CLINICAL_TERMS: frozenset[str] = frozenset(
    MODALITY_TERMS | CLINICAL_MEASUREMENT_TERMS | ANATOMY_TERMS | MEDICATION_TERMS
)

#: Rebound now that the source sets exist.
DISTINCTIVE_CLINICAL_TOKENS = frozenset(
    MEDICATION_TERMS | CLINICAL_MEASUREMENT_TERMS | MODALITY_TERMS
)

#: Words that are proper-noun shaped but are labels, not names.
NON_NAME_LABELS: frozenset[str] = frozenset(
    {
        "patient",
        "provider",
        "doctor",
        "dr",
        "mr",
        "mrs",
        "ms",
        "miss",
        "prof",
        "professor",
        "attending",
        "resident",
        "consultant",
        "nurse",
        "clinic",
        "ward",
        "unit",
        "department",
        "history",
        "assessment",
        "plan",
        "impression",
        "findings",
        "diagnosis",
        "medications",
        "allergies",
        "procedure",
        "indication",
        "technique",
        "comparison",
        "portal",
        "contact",
        "address",
        "email",
        "phone",
        "fax",
        "date",
        "dob",
        "mrn",
        "record",
        "number",
        "referral",
        "follow",
        "note",
        # Relationship and role nouns. A relative's *name* is an identifier;
        # the word "mother" is not, and redacting it destroys the clinical
        # relationship the note is recording.
        "mother",
        "father",
        "parent",
        "spouse",
        "wife",
        "husband",
        "partner",
        "son",
        "daughter",
        "child",
        "sibling",
        "brother",
        "sister",
        "guardian",
        "carer",
        "caregiver",
        "next-of-kin",
        "relative",
        "family",
        "household",
        "employer",
        "reviewer",
        "signatory",
        "witness",
        "interpreter",
        "student",
        "trainee",
        "birth",
        "home",
        "primary",
        "secondary",
        "correspondence",
        "block",
        "portal",
        "login",
        "reference",
        "combined",
        "legacy",
        "study",
        "code",
        "escalated",
        "opinion",
        "spelling",
        "preferred",
        "retained",
        "case",
        "results",
        "details",
        "list",
        "summary",
    }
)

#: Function words that carry no identity. Present so a multi-word span made
#: only of labels and these ("Date of birth", "second reviewer") is recognised
#: as a field label rather than a name.
_LABEL_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or",
        "the", "to", "with", "was", "were", "is", "are", "second", "first",
        "third", "next", "new", "old", "other", "his", "her", "their", "its",
        "this", "that", "s",
    }
)

_POSSESSIVE = re.compile(r"['’]s?$")
_TRAILING_HEAD = re.compile(
    r"\s+(?:" + "|".join(sorted(CLINICAL_HEAD_NOUNS)) + r")$",
    re.IGNORECASE,
)


def normalize(term: str) -> str:
    """Casefold and strip a trailing possessive so lookups are stable."""
    cleaned = (term or "").strip().casefold()
    return _POSSESSIVE.sub("", cleaned).strip()


def is_unconditional_clinical_term(term: str) -> bool:
    """True for vocabulary that is never a personal name in any context."""
    normalized = normalize(term)
    if not normalized:
        return False
    if normalized in UNCONDITIONAL_CLINICAL_TERMS:
        return True
    tokens = [t for t in re.split(r"[\s/\-]+", normalized) if t]
    return bool(tokens) and all(
        token in UNCONDITIONAL_CLINICAL_TERMS for token in tokens
    )


def is_known_eponym(term: str) -> bool:
    """True if the bare term is a recorded clinical eponym."""
    stripped = _TRAILING_HEAD.sub("", normalize(term)).strip()
    return normalize(stripped) in CLINICAL_EPONYMS


def is_non_name_label(term: str) -> bool:
    """True when a span is a field label or role phrase, not a name.

    Checked token-wise, because a model that cannot see the value - it has
    been masked - reliably labels the words beside the hole instead. "Date of
    birth", "Home address" and "second reviewer" are all detected as PERSON or
    ADDRESS in that situation, and none of them is an identifier.
    """
    normalized = normalize(term)
    if not normalized:
        return False
    if normalized in NON_NAME_LABELS:
        return True
    tokens = [t for t in re.split(r"[^\w]+", normalized) if t]
    if not tokens:
        return False
    if not any(token in NON_NAME_LABELS for token in tokens):
        # At least one real label word must be present, or an ordinary
        # two-word name made of stopwords alone would be let through.
        return False
    return all(
        token in NON_NAME_LABELS or token in _LABEL_STOPWORDS for token in tokens
    )


def is_clinical_eponym_use(
    term: str,
    following_words: Sequence[str] = (),
) -> bool:
    """True when an eponym appears in a clinical construction.

    ``following_words`` are the next few surface tokens after the term. The
    possessive marker is skipped, because "Parkinson's disease" tokenizes with
    ``'s`` between the eponym and its head noun and the previous
    implementation's single-token lookahead landed on the apostrophe and
    concluded the eponym was a surname.

    A bare eponym with no clinical head noun is deliberately **not** protected:
    a patient can genuinely be named Parkinson.
    """
    if not is_known_eponym(term):
        return False

    # The head noun may already be inside the term ("Crohn disease").
    if _TRAILING_HEAD.search(normalize(term)):
        return True

    for word in following_words:
        candidate = (word or "").strip().casefold()
        if not candidate:
            continue
        # Skip the possessive, whether tokenized separately or attached.
        if candidate in {"'s", "’s", "'", "’", "s"}:
            continue
        stripped = _POSSESSIVE.sub("", candidate).strip(".,;:()")
        return stripped in CLINICAL_HEAD_NOUNS
    return False


def contains_distinctive_clinical_token(term: str) -> bool:
    """True when any token is a drug, analyte or modality.

    A multi-word span such as "Atorvastatin 40 mg nightly" never matches a
    single-term lookup, so a detector that labels it a PERSON slips through a
    whole-span check. No personal name contains one of these tokens.
    """
    tokens = [t for t in re.split(r"[^\w]+", normalize(term)) if t]
    return any(token in DISTINCTIVE_CLINICAL_TOKENS for token in tokens)


def protects_from_person_label(
    term: str,
    following_words: Sequence[str] = (),
) -> bool:
    """Should this span be kept out of a PERSON-style redaction?

    True for vocabulary that is never a name, and for an eponym used in a
    clinical construction. Everything else is left to the detectors.
    """
    if is_unconditional_clinical_term(term):
        return True
    if contains_distinctive_clinical_token(term):
        return True
    if is_non_name_label(term):
        return True
    return is_clinical_eponym_use(term, following_words)


def following_words_from_text(text: str, end: int, count: int = 4) -> Tuple[str, ...]:
    """Cheap lookahead for callers that hold raw text rather than a doc."""
    tail = text[end : end + 64]
    return tuple(re.findall(r"[^\s]+", tail))[:count]
