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
        # Instruments and formulae named after the person who devised them.
        # Nobody is described as "Smith catheter"; "Foley catheter" is a device.
        "catheter",
        "catheters",
        "cannula",
        "forceps",
        "speculum",
        "tourniquet",
        "retractor",
        "needle",
        "tube",
        "clamp",
        "curve",
        "formula",
        "equation",
        "nomogram",
        "valve",
        "flap",
        "graft",
    }
)

#: Heads accepted only after an eponym the vocabulary already knows ("Broca
#: area"). They are too ordinary to protect an unknown capitalised word: "the
#: Jordan area" names a place, and a place is an identifier.
EPONYM_ONLY_HEADS: frozenset[str] = frozenset(
    {
        "area",
        "areas",
        "gyrus",
        "fasciculus",
        "cortex",
        "nucleus",
        "plexus",
        "pouch",
        "fossa",
        "foramen",
        "notch",
        "point",
        "line",
        "angle",
        "zone",
    }
)

#: Words that sit between an eponym and its head in a named scale or chart:
#: "Bristol stool chart", "Glasgow coma scale", "Karnofsky performance status".
NAMED_SCALE_MODIFIERS: frozenset[str] = frozenset(
    {
        "stool",
        "coma",
        "pain",
        "performance",
        "depression",
        "anxiety",
        "rating",
        "dyspnoea",
        "dyspnea",
        "disability",
        "severity",
        "faces",
        "apnoea",
        "apnea",
        "outcome",
        "risk",
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
        "paracetamol",
        "acetaminophen",
        "ibuprofen",
        "naproxen",
        "aspirin",
        "codeine",
        "morphine",
        "fentanyl",
        "tramadol",
        "oxycodone",
        "amoxicillin",
        "penicillin",
        "flucloxacillin",
        "cefalexin",
        "ceftriaxone",
        "azithromycin",
        "clarithromycin",
        "erythromycin",
        "doxycycline",
        "gentamicin",
        "vancomycin",
        "metronidazole",
        "ciprofloxacin",
        "trimethoprim",
        "nitrofurantoin",
        "metformin",
        "gliclazide",
        "insulin",
        "sitagliptin",
        "empagliflozin",
        "atorvastatin",
        "simvastatin",
        "rosuvastatin",
        "pravastatin",
        "ramipril",
        "lisinopril",
        "enalapril",
        "perindopril",
        "losartan",
        "candesartan",
        "amlodipine",
        "nifedipine",
        "diltiazem",
        "verapamil",
        "bisoprolol",
        "atenolol",
        "metoprolol",
        "carvedilol",
        "propranolol",
        "furosemide",
        "bendroflumethiazide",
        "spironolactone",
        "indapamide",
        "warfarin",
        "apixaban",
        "rivaroxaban",
        "dabigatran",
        "clopidogrel",
        "heparin",
        "enoxaparin",
        "omeprazole",
        "lansoprazole",
        "pantoprazole",
        "ranitidine",
        "salbutamol",
        "albuterol",
        "beclometasone",
        "budesonide",
        "prednisolone",
        "prednisone",
        "dexamethasone",
        "hydrocortisone",
        "levothyroxine",
        "carbimazole",
        "allopurinol",
        "colchicine",
        "sertraline",
        "fluoxetine",
        "citalopram",
        "escitalopram",
        "amitriptyline",
        "mirtazapine",
        "venlafaxine",
        "diazepam",
        "lorazepam",
        "gabapentin",
        "pregabalin",
        "levetiracetam",
        "lamotrigine",
        "carbamazepine",
        "phenytoin",
        "sodium valproate",
        "methotrexate",
        "azathioprine",
        "ciclosporin",
        "tacrolimus",
        "cisplatin",
        "carboplatin",
        "doxorubicin",
        "cyclophosphamide",
        "tamoxifen",
        "anastrozole",
        "rituximab",
        "trastuzumab",
        "ondansetron",
        "metoclopramide",
        "loperamide",
        "senna",
        "lactulose",
    }
)

#: Finding nouns that are not personal names.  Membership is terminology
#: evidence; the response-to-treatment grammar supplies structural evidence
#: when several identity models share the same misreading.
CLINICAL_FINDING_TERMS: frozenset[str] = frozenset(
    {
        "plaque",
        "plaques",
        "lesion",
        "lesions",
        "rash",
        "rashes",
        "erythema",
        "oedema",
        "edema",
        "ulcer",
        "ulcers",
        "nodule",
        "nodules",
        "effusion",
        "effusions",
        "opacity",
        "opacities",
        "wheeze",
        "wheezes",
        "crackles",
        "cyanosis",
    }
)

#: Everything that is never a personal name, in one set.
UNCONDITIONAL_CLINICAL_TERMS: frozenset[str] = frozenset(
    MODALITY_TERMS
    | CLINICAL_MEASUREMENT_TERMS
    | ANATOMY_TERMS
    | MEDICATION_TERMS
    | CLINICAL_FINDING_TERMS
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
        "telephone",
        "tel",
        "mobile",
        "fax",
        "date",
        "dob",
        "mrn",
        "nhs",
        # contact and identifier field labels: the label, never the value
        "ext",
        "extension",
        "bleep",
        "bleeper",
        "pager",
        "switchboard",
        "hospital",
        "gmc",
        "nmc",
        "hcpc",
        "registration",
        "licence",
        "license",
        # spreadsheet and laboratory structure words
        "range",
        "target",
        "limit",
        "lower",
        "upper",
        "interval",
        "threshold",
        "baseline",
        "median",
        "average",
        "minimum",
        "maximum",
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
        "review",
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
        # Role, group and service nouns. "Ward staff" and "Pharmacy team"
        # identify nobody, and a model calling one a PERSON replaced it with a
        # patient surrogate. Common surnames are deliberately left out -
        # porter, day, night, carter, taylor, cook, mason, bishop, marshal -
        # because a word that is often somebody's name cannot be exempt here.
        "staff",
        "team",
        "teams",
        "personnel",
        "crew",
        "colleagues",
        "registrar",
        "registrars",
        "physician",
        "physicians",
        "surgeon",
        "surgeons",
        "clinician",
        "clinicians",
        "practitioner",
        "practitioners",
        "therapist",
        "therapists",
        "physiotherapist",
        "physiotherapists",
        "pharmacist",
        "pharmacists",
        "midwife",
        "midwives",
        "paramedic",
        "paramedics",
        "anaesthetist",
        "anesthesiologist",
        "radiographer",
        "sonographer",
        "dietitian",
        "matron",
        "nurses",
        "doctors",
        "theatre",
        "theater",
        "pharmacy",
        "nursing",
        "surgical",
        "medical",
        "community",
        "reception",
        "outpatient",
        "inpatient",
        "triage",
        "ambulance",
        # Documentation and form words. A wristband tag, an alert flagged on a
        # chart, a reference field: none of them names anybody, and the
        # proper-noun tagger reads each as a name whenever it is capitalised
        # or stands alone on a line. Common surnames stay out, as above -
        # "Field" and "Mark" are somebody's name.
        "pt",
        "id",
        "ref",
        "wristband",
        "tag",
        "label",
        "alert",
        "flagged",
        "flag",
        "confirmed",
        "verified",
        "noted",
        "callback",
        "copy",
        "filing",
        "urgent",
        "pending",
        "awaiting",
        "attached",
        "enclosed",
        "referral",
        "heading",
    }
)

#: Function words that carry no identity. Present so a multi-word span made
#: only of labels and these ("Date of birth", "second reviewer") is recognised
#: as a field label rather than a name.
_LABEL_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "was",
        "were",
        "is",
        "are",
        "second",
        "first",
        "third",
        "next",
        "new",
        "old",
        "other",
        "his",
        "her",
        "their",
        "its",
        "this",
        "that",
        "s",
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


#: Immunization, vaccine-product, clinical-device and MRI-sequence vocabulary: a closed list,
#: matched token by token. A vaccine code is kept only when it is one of these
#: exact codes, so "PCV20" is clinical while "PCV20-Alice", "MRN20" or a per-row
#: identifier never are. Words that are common surnames ("Walker", "Brace") are
#: deliberately absent. Membership is evidence, not a verdict: the caller also
#: requires that nothing around the span reads as a name.
CLINICAL_PRODUCT_TERMS: frozenset[str] = frozenset(
    {
        # immunization vocabulary
        "vaccine",
        "vaccines",
        "vaccination",
        "vaccinations",
        "immunization",
        "immunizations",
        "immunisation",
        "immunisations",
        "booster",
        "influenza",
        "flu",
        "pneumococcal",
        "meningococcal",
        "pertussis",
        "diphtheria",
        "tetanus",
        "measles",
        "mumps",
        "rubella",
        "varicella",
        "zoster",
        "shingles",
        "hepatitis",
        "rotavirus",
        "polio",
        "poliovirus",
        "papillomavirus",
        "rabies",
        "typhoid",
        "cholera",
        "seasonal",
        "injectable",
        "preservative",
        "conjugate",
        "valent",
        "quadrivalent",
        "trivalent",
        "recombinant",
        "inactivated",
        "attenuated",
        "toxoid",
        "adjuvanted",
        "intranasal",
        "intramuscular",
        "subcutaneous",
        # exact vaccine product and schedule codes
        "pcv13",
        "pcv15",
        "pcv20",
        "pcv21",
        "ppsv23",
        "mmr",
        "mmrv",
        "dtap",
        "tdap",
        "td",
        "ipv",
        "hepa",
        "hepb",
        "hib",
        "hpv",
        "hpv9",
        "menacwy",
        "menb",
        "rsv",
        "bcg",
        "laiv",
        "iiv4",
        "riv4",
        "rzv",
        "zvl",
        "pcv",
        "ppsv",
        # clinical devices
        "inhaler",
        "inhalers",
        "spacer",
        "nebuliser",
        "nebulizer",
        "catheter",
        "cannula",
        "stent",
        "pacemaker",
        "defibrillator",
        "glucometer",
        "oximeter",
        "syringe",
        "splint",
        # MRI sequence acronyms that are not also words or names ("pet", "se", "pd",
        # "map" and "bold" are left out)
        "flair",
        "mprage",
        "dwi",
        "dti",
        "swi",
        "adc",
        "stir",
        "fmri",
    }
)


def is_clinical_product_term(term: str) -> bool:
    """Every token of ``term`` is on the closed product, device and sequence list.

    A vaccine's valence written apart ("PCV 20 valent") is allowed as a
    number of at most two digits, and only beside a listed word: a longer
    number is never covered, so "Influenza 5528" does not qualify.
    """
    tokens = [token.strip(".,;:()") for token in re.split(r"[\s/]+", normalize(term))]
    tokens = [token for token in tokens if token]
    words = [token for token in tokens if not token.isdigit()]
    return (
        bool(words)
        and all(token in CLINICAL_PRODUCT_TERMS for token in words)
        and all(len(token) <= 2 for token in tokens if token.isdigit())
    )


def is_clinical_finding_term(term: str) -> bool:
    """Whether every word is a non-name clinical finding noun."""
    tokens = [token for token in re.split(r"[\s/\-]+", normalize(term)) if token]
    return bool(tokens) and all(token in CLINICAL_FINDING_TERMS for token in tokens)


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


#: Units and scales named after the people who devised them. A tagger is not
#: wrong to see a surname here - Anders Celsius and Herman Snellen were people
#: - so membership alone protects nothing. It is one half of the evidence; the
#: other half is that the token is being used to measure something, which
#: ``services.detection_evidence`` establishes from what surrounds it.
UNIT_EPONYMS: frozenset[str] = frozenset(
    {
        "celsius",
        "fahrenheit",
        "kelvin",
        "rankine",
        "snellen",
        "jaeger",
        "landolt",
        "hounsfield",
        "gray",
        "sievert",
        "becquerel",
        "curie",
        "roentgen",
        "pascal",
        "newton",
        "joule",
        "watt",
        "hertz",
        "ampere",
        "volt",
        "ohm",
        "tesla",
        "weber",
        "henry",
        "siemens",
        "farad",
        "coulomb",
        "angstrom",
        "dalton",
        "svedberg",
        "poise",
        "stokes",
        "french",
        "charriere",
    }
)

#: Words that turn an eponym into a measurement: a unit head noun, or the
#: word that introduces one.
UNIT_HEAD_WORDS: frozenset[str] = frozenset(
    {
        "degree",
        "degrees",
        "deg",
        "unit",
        "units",
        "scale",
        "score",
        "metre",
        "metres",
        "meter",
        "meters",
        "line",
        "lines",
        "chart",
        "number",
        "size",
        "gauge",
        "grade",
        "index",
    }
)


def is_unit_eponym(term: str) -> bool:
    """True if the bare term names a unit or scale after its deviser."""
    return normalize(term).strip(".,;:()") in UNIT_EPONYMS


def is_unit_head_word(word: str) -> bool:
    return normalize(word).strip(".,;:()") in UNIT_HEAD_WORDS


def is_known_eponym(term: str) -> bool:
    """True if the bare term is a recorded clinical eponym."""
    stripped = _TRAILING_HEAD.sub("", normalize(term)).strip()
    return normalize(stripped) in CLINICAL_EPONYMS


#: Nouns that name a group of staff and can head a shift phrase.
GROUP_HEAD_NOUNS: frozenset[str] = frozenset(
    {
        "staff",
        "team",
        "teams",
        "crew",
        "personnel",
        "colleagues",
        "nurses",
        "doctors",
        "registrars",
        "physicians",
        "surgeons",
        "clinicians",
        "practitioners",
        "therapists",
        "physiotherapists",
        "pharmacists",
        "midwives",
        "paramedics",
    }
)

#: One member of staff named by role: "Outreach nurse", "stroke consultant".
#: Heads a role phrase exactly as a group noun does; the modifiers must still
#: all be shift or service words, so "Ravensholt nurse" is still a name.
ROLE_HEAD_NOUNS: frozenset[str] = GROUP_HEAD_NOUNS | frozenset(
    {
        "nurse",
        "doctor",
        "registrar",
        "physician",
        "surgeon",
        "clinician",
        "practitioner",
        "therapist",
        "physiotherapist",
        "pharmacist",
        "midwife",
        "paramedic",
        "consultant",
        "dietitian",
        "dietician",
        "radiographer",
        "sonographer",
        "coordinator",
        "specialist",
        "technician",
        "lead",
    }
)

#: Common nouns that open a sentence in a clinical note ("Mobility had improved",
#: "Imaging was reported", "Nutrition advice was reinforced"): an assessed
#: function, system, test or treatment. A sentence opens with a capital, so a
#: capitalisation-driven tagger reads them as proper nouns. Only an opening
#: single word of this closed list is kept, never beside naming context, and no
#: given name or surname is on it ("Grace", "Hope", "Page", "Wood" are absent).
CLINICAL_DOMAIN_NOUNS: frozenset[str] = frozenset(
    {
        "mobility",
        "cognition",
        "nutrition",
        "appetite",
        "sleep",
        "mood",
        "pain",
        "swelling",
        "breathing",
        "respiration",
        "circulation",
        "continence",
        "hydration",
        "swallowing",
        "speech",
        "vision",
        "hearing",
        "balance",
        "gait",
        "strength",
        "sensation",
        "behaviour",
        "behavior",
        "orientation",
        "memory",
        "concentration",
        "energy",
        "weight",
        "temperature",
        "pulse",
        "bloods",
        "urine",
        "stool",
        "bowels",
        "bladder",
        "skin",
        "wound",
        "wounds",
        "dressing",
        "drain",
        "catheter",
        "cannula",
        "imaging",
        "ultrasound",
        "radiology",
        "pathology",
        "histology",
        "cytology",
        "microbiology",
        "culture",
        "cultures",
        "swab",
        "observations",
        "examination",
        "auscultation",
        "palpation",
        "inspection",
        "investigations",
        "results",
        "history",
        "impression",
        "assessment",
        "management",
        "treatment",
        "therapy",
        "physiotherapy",
        "medication",
        "medications",
        "analgesia",
        "antibiotics",
        "oxygen",
        "fluids",
        "feeding",
        "diet",
        "activity",
        "exercise",
        "rehabilitation",
        "discharge",
        "review",
        "referral",
        "consent",
        "capacity",
        "risk",
        "falls",
        "allergies",
        "symptoms",
        "signs",
        "vitals",
        "anaesthesia",
        "surgery",
        "operation",
        "biopsy",
        "endoscopy",
        "colonoscopy",
        "echocardiography",
        "ecg",
        "respirations",
        "hygiene",
        "elimination",
        "oedema",
        "edema",
        "fever",
        "cough",
        "nausea",
        "vomiting",
        "diarrhoea",
        "constipation",
        "headache",
        "dizziness",
        "fatigue",
        "anxiety",
        "confusion",
        "agitation",
        "delirium",
        "sedation",
        "ventilation",
        "perfusion",
        "compliance",
        "adherence",
        "education",
        "counselling",
        "counseling",
        "safeguarding",
        "dentition",
        "dysphagia",
        "continence",
        "posture",
        "tone",
        "reflexes",
        "coordination",
    }
)

#: Singular staff and carer role nouns that stand alone in lower case, with the
#: determiners and numerals that introduce them ("one assistant", "the carer").
BARE_ROLE_NOUNS: frozenset[str] = frozenset(
    {
        "assistant",
        "assistants",
        "aide",
        "aides",
        "carer",
        "carers",
        "orderly",
        "porter",
        "attendant",
        "helper",
        "volunteer",
        "interpreter",
        "chaperone",
        "escort",
        "sitter",
        "nurse",
        "nurses",
        "doctor",
        "doctors",
        "therapist",
        "therapists",
        "clinician",
        "clinicians",
        "physician",
        "physicians",
        "midwife",
        "paramedic",
        "paramedics",
        "pharmacist",
        "technician",
        "staff",
    }
)
_ROLE_DETERMINERS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "one",
        "two",
        "three",
        "another",
        "his",
        "her",
        "their",
        "our",
        "my",
        "your",
        "some",
        "any",
        "each",
        "no",
    }
)


def is_bare_role_noun_phrase(term: str) -> bool:
    """A lower-case role noun with at most a determiner or numeral before it.

    "assistant", "one assistant", "the carer". Capital letters anywhere defeat
    it, so "Carer Smith" and "Nurse" at the head of a name are never excused here.
    """
    stripped = (term or "").strip()
    if not stripped or stripped != stripped.lower():
        return False
    tokens = [t for t in re.split(r"[^\w]+", stripped) if t]
    if not 1 <= len(tokens) <= 2 or tokens[-1] not in BARE_ROLE_NOUNS:
        return False
    return all(token in _ROLE_DETERMINERS for token in tokens[:-1])


#: Words that name a shift or a time of day. Never label words on their own -
#: "Day" and "Night" are surnames - and only ever modifiers of a staff group:
#: "Night nursing team", "day staff", "weekend registrars".
SHIFT_MODIFIERS: frozenset[str] = frozenset(
    {
        "night",
        "day",
        "early",
        "late",
        "evening",
        "morning",
        "weekend",
        "twilight",
        "overnight",
        "daytime",
        "nighttime",
    }
)

#: Laboratory tests a specimen qualifier can head: "Urine microscopy",
#: "Sputum culture". Evidence only after a qualifier, never on their own -
#: "Gram" and "Screen" can be surnames.
LABORATORY_TEST_NAMES: frozenset[str] = frozenset(
    {
        "microscopy",
        "culture",
        "cultures",
        "dipstick",
        "cytology",
        "histology",
        "histopathology",
        "pcr",
        "serology",
        "toxicology",
        "biochemistry",
        "haematology",
        "hematology",
        "microbiology",
        "urinalysis",
        "electrophoresis",
        "osmolality",
        "sensitivities",
        "sensitivity",
        "gram",
        "stain",
        "screen",
        "analysis",
        "output",
        "protein",
        "ketones",
        "sample",
        "specimen",
        "gas",
        "smear",
    }
)


#: Clinical services and specialties that name what a staff group does:
#: "Physiotherapy team", "palliative care team", "stroke registrars".
CLINICAL_SERVICE_TERMS: frozenset[str] = frozenset(
    {
        "physiotherapy",
        "occupational",
        "therapy",
        "speech",
        "language",
        "dietetic",
        "dietetics",
        "nutrition",
        "palliative",
        "care",
        "critical",
        "intensive",
        "outreach",
        "rehabilitation",
        "stroke",
        "renal",
        "respiratory",
        "orthopaedic",
        "orthopedic",
        "paediatric",
        "pediatric",
        "obstetric",
        "maternity",
        "neonatal",
        "geriatric",
        "elderly",
        "frailty",
        "mental",
        "health",
        "psychiatric",
        "social",
        "work",
        "pain",
        "acute",
        "emergency",
        "infection",
        "control",
        "tissue",
        "viability",
        "falls",
        "diabetes",
        "stoma",
        "wound",
        "dialysis",
        "continence",
        "allergy",
        "fertility",
        "lipid",
        "obesity",
        "colorectal",
        "discharge",
        "safeguarding",
        "liaison",
        "specialist",
        "clinical",
        "transplant",
        "endoscopy",
        "anaesthetic",
        "anesthetic",
        "dental",
        "vascular",
        "cardiac",
        "trauma",
        "ward",
        "theatres",
        "district",
    }
)

#: Word endings that make a specialty: cardiology, radiotherapy, paediatrics.
_SERVICE_ENDING = re.compile(
    r"[a-z]{3,}(?:ology|ologies|otherapy|therapies|iatrics|iatry|paedics|pedics)"
)


def _is_service_word(token: str) -> bool:
    return token in CLINICAL_SERVICE_TERMS or bool(_SERVICE_ENDING.fullmatch(token))


def _is_staff_group_phrase(tokens: Sequence[str]) -> bool:
    """A group of staff named by its shift or its service.

    "Night nursing team", "Physiotherapy team", "stroke registrars": the head
    is a group noun and at least one modifier is a shift word or a clinical
    service; every other modifier is itself a label or service word.
    "Ravensholt team" and "Brightwater Physiotherapy team" both fail, so a
    name in front of the group noun is still a name.
    """
    if len(tokens) < 2 or tokens[-1] not in ROLE_HEAD_NOUNS:
        return False
    modifiers = tokens[:-1]
    return any(
        token in SHIFT_MODIFIERS or _is_service_word(token) for token in modifiers
    ) and all(
        token in SHIFT_MODIFIERS
        or _is_service_word(token)
        or token in NON_NAME_LABELS
        or token in _LABEL_STOPWORDS
        for token in modifiers
    )


def is_staff_role_phrase(term: str) -> bool:
    """A role or staff group named by its service or shift, nothing else.

    "Outreach nurse", "Diabetes specialist nurse", "Night nursing team":
    structure, not a list of names, so two models agreeing on a PERSON
    reading of one are agreeing on the same misreading.
    """
    tokens = [t for t in re.split(r"[^\w]+", normalize(term)) if t]
    return _is_staff_group_phrase(tokens)


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
    if _is_staff_group_phrase(tokens):
        # The group noun is the label; its modifiers say which shift or service.
        return True
    if not any(token in NON_NAME_LABELS for token in tokens):
        # At least one real label word must be present, or an ordinary
        # two-word name made of stopwords alone would be let through.
        return False
    if all(token in NON_NAME_LABELS or token in _LABEL_STOPWORDS for token in tokens):
        return True
    return False


def is_label_word(word: str) -> bool:
    """One word that is a field label, role noun or function word, never a name."""
    normalized = normalize(word)
    return normalized in NON_NAME_LABELS or normalized in _LABEL_STOPWORDS


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
        return stripped in CLINICAL_HEAD_NOUNS or stripped in EPONYM_ONLY_HEADS
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


#: Words that head a field in a clinical document rather than naming anyone.
#: A record is written as a sequence of labelled fields, and the label is part
#: of the record's meaning: strip "Dx" and the reader no longer knows that
#: K80.20 is a diagnosis, which is how the code beside it came to be read as
#: an account number.
#:
#: Membership alone protects nothing. The label has to be standing in label
#: position - followed by a colon - before it is treated as one, because
#: several of these are also perfectly good surnames. "Baker:" at the head of
#: a field is a label; "Baker" in a sentence is a name, and the rule that uses
#: this set never sees the second case.
CLINICAL_FIELD_LABELS: frozenset[str] = frozenset(
    {
        # abbreviations, which is where the damage is worst
        "dx",
        "ddx",
        "hx",
        "fhx",
        "shx",
        "pmh",
        "pmhx",
        "psh",
        "hpi",
        "ros",
        "px",
        "rx",
        "sx",
        "tx",
        "imp",
        "a/p",
        "o/e",
        "c/o",
        "inr",
        "obs",
        # administrative heads
        "patient",
        "name",
        "record",
        "account",
        "mrn",
        "nhs",
        "chart",
        "case",
        "dob",
        "sex",
        "gender",
        "age",
        "telephone",
        "phone",
        "mobile",
        "fax",
        "email",
        "address",
        "ward",
        "bed",
        "clinic",
        "consultant",
        "gp",
        "referrer",
        "referred",
        "reviewed",
        "author",
        "signed",
        "dictated",
        # clinical heads
        "diagnosis",
        "diagnoses",
        "impression",
        "indication",
        "history",
        "presentation",
        "complaint",
        "examination",
        "findings",
        "finding",
        "result",
        "results",
        "investigation",
        "investigations",
        "plan",
        "management",
        "treatment",
        "medication",
        "medications",
        "drug",
        "drugs",
        "allergy",
        "allergies",
        "procedure",
        "operation",
        "technique",
        "comparison",
        "specimen",
        "sample",
        "site",
        "laterality",
        "stage",
        "grade",
        "severity",
        "assessment",
        "summary",
        "conclusion",
        "advice",
        "follow",
        "prognosis",
        "outcome",
        "observation",
        "observations",
        "vitals",
        "score",
        "scores",
        "microscopy",
        "macroscopy",
        "report",
        "note",
        "notes",
        # Column heads a spreadsheet uses where prose would use a sentence.
        "analyte",
        "analytes",
        "unit",
        "units",
        "value",
        "values",
        "dose",
        "doses",
        "reference",
        "range",
        "test",
        "tests",
        "panel",
        "method",
    }
)


def is_clinical_field_label(term: str) -> bool:
    """Whether this word heads a field, ignoring case and trailing marks."""
    return normalize(term).strip(".,;:()") in CLINICAL_FIELD_LABELS


#: The subset of field heads that introduce clinical narrative rather than
#: administrative detail. "Procedure:" and "Findings:" are followed by what
#: was done and what was seen; "Patient:", "Record:" and "Telephone:" are
#: followed by who and which, and nothing under those heads is ever narrative
#: worth keeping. Splitting them matters because the narrative rule is only
#: allowed to speak for the first group.
CLINICAL_NARRATIVE_HEADS: frozenset[str] = frozenset(
    {
        "diagnosis",
        "diagnoses",
        "impression",
        "indication",
        "history",
        "presentation",
        "complaint",
        "examination",
        "findings",
        "finding",
        "result",
        "results",
        "investigation",
        "investigations",
        "plan",
        "management",
        "treatment",
        "procedure",
        "operation",
        "technique",
        "comparison",
        "specimen",
        "note",
        "notes",
        "assessment",
        "summary",
        "conclusion",
        "advice",
        "prognosis",
        "outcome",
        "observation",
        "observations",
        "microscopy",
        "macroscopy",
        "dx",
        "ddx",
        "imp",
        "hpi",
        "ros",
        "px",
        "sx",
        "tx",
        # The same column heads. A spreadsheet writes "Analyte | haemoglobin"
        # where a letter writes "the analyte was haemoglobin", and the value
        # beside one of these is a finding rather than a party.
        "analyte",
        "analytes",
        "unit",
        "units",
        "dose",
        "doses",
        "test",
        "tests",
        "panel",
        "method",
        "reference",
        "range",
        "medication",
        "medications",
        "drug",
        "drugs",
        "allergy",
        "allergies",
    }
)


#: Standard race words that are also ordinary surnames. A capitalised "White"
#: or "Black" beside a name is never exempted as a demographic category.
_SURNAME_LIKE_CATEGORY_TERMS: frozenset[str] = frozenset({"white", "black"})


# Closed clinical demographic labels, separated from deferred table policy.
# Source: csv_sanitizer.CATEGORY_VOCABULARIES at 4114cac, race/ethnicity only.
RACE_ETHNICITY_CATEGORY_LABELS = frozenset(
    [
        "african american",
        "alaska native",
        "american indian",
        "american indian or alaska native",
        "asian",
        "black",
        "black or african american",
        "declined",
        "hawaiian",
        "hispanic",
        "hispanic or latino",
        "latino",
        "multiracial",
        "native",
        "native american",
        "native hawaiian",
        "native hawaiian or other pacific islander",
        "non-hispanic",
        "nonhispanic",
        "not hispanic",
        "not hispanic or latino",
        "other",
        "other race",
        "pacific islander",
        "two or more races",
        "unknown",
        "white",
    ]
)


def is_demographic_category_label(term: str) -> bool:
    """Whole-label evidence only; a category followed by a name does not match."""
    key = " ".join(term.strip().casefold().replace("_", " ").split())
    return key in RACE_ETHNICITY_CATEGORY_LABELS


def is_race_ethnicity_category(term: str) -> bool:
    """Whole category evidence, excluding words also commonly used as surnames."""
    key = " ".join(term.strip().casefold().replace("_", " ").split())
    return key not in _SURNAME_LIKE_CATEGORY_TERMS and is_demographic_category_label(
        term
    )


#: Words that label an identifier field or join a label to its value. They
#: carry no identity: the identity was in the value, which a generated token now
#: holds. A span of only these words next to such a token is a label, not a
#: person, place or number. Any word outside this closed list (a name, a
#: digit-bearing code, an unknown noun) makes the phrase something else.
IDENTIFIER_LABEL_WORDS: frozenset[str] = frozenset(
    {
        "zip",
        "postal",
        "post",
        "code",
        "postcode",
        "county",
        "residence",
        "address",
        "home",
        "street",
        "city",
        "town",
        "state",
        "licence",
        "license",
        "driver",
        "drivers",
        "driver's",
        "plate",
        "registration",
        "reg",
        "vehicle",
        "vin",
        "number",
        "no",
        "num",
        "id",
        "identifier",
        "serial",
        "device",
        "pacemaker",
        "pump",
        "insulin",
        "implant",
        "iris",
        "scan",
        "voiceprint",
        "fingerprint",
        "template",
        "face",
        "photo",
        "photograph",
        "headshot",
        "file",
        "image",
        "ref",
        "reference",
        "case",
        "tracking",
        "participant",
        "study",
        "client",
        "patient",
        "member",
        "subscriber",
        "beneficiary",
        "health",
        "plan",
        "insurance",
        "account",
        "acct",
        "billing",
        "chart",
        "medical",
        "record",
        "mrn",
        "ssn",
        "social",
        "security",
        "phone",
        "mobile",
        "fax",
        "facsimile",
        "telephone",
        "email",
        "website",
        "url",
        "link",
        "ip",
        "contact",
        "call",
        "reachable",
        "portal",
        "of",
        "on",
        "at",
        "to",
        "for",
        "from",
        "the",
        "a",
        "an",
        "in",
        "by",
        "via",
        "see",
        "logged",
        "wrote",
        "recorded",
        "noted",
        "with",
    }
)


def is_identifier_label_text(text: str) -> bool:
    """Whether every word is a field label or connective, and there is at least one."""
    # Whole whitespace-delimited words only: "MRN20" is a code, not the label
    # "MRN", so a digit or any other character inside a word declines.
    words = [word.strip('.,;:()[]"') for word in text.split()]
    words = [word for word in words if word]
    return bool(words) and all(
        re.fullmatch(r"[A-Za-z']+", word) is not None
        and word.casefold() in IDENTIFIER_LABEL_WORDS
        for word in words
    )


def is_clinical_narrative_head(term: str) -> bool:
    """Whether this word introduces clinical narrative rather than a party."""
    return normalize(term).strip(".,;:()") in CLINICAL_NARRATIVE_HEADS
