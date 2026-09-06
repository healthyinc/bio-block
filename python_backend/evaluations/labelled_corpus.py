"""Labelled synthetic PHI corpus, v2, with three non-overlapping partitions.

Every value in this file is invented. Names are fictional, email domains use
`.invalid` (reserved and unresolvable by RFC 2606), telephone numbers use the
555-01xx range reserved for fiction, and identifiers carry synthetic prefixes.
No real patient information appears here or anywhere in the evaluation path.
Social-security numbers use the 900 range, which the SSA has never issued, so
no value here can collide with a real one.

Three partitions, with **disjoint value pools** so a value seen while choosing
thresholds cannot reappear in the held-out measurement:

* ``development``  - used while building and debugging detectors.
* ``calibration``  - the only partition thresholds may be selected on.
* ``test``         - held out. Run once, after the configuration is locked.

Ground truth is span-level. A document records the exact character offsets of
every value that must be removed, plus the ordinary terms that must survive.
Offsets are recorded by the builder as the text is assembled, so they cannot
drift out of sync with the prose.

Two recall figures come out of this, and they answer different questions:

* **span recall** - was the gold span overlapped by *any* detection, so the
  value would be redacted? This is the privacy-critical number.
* **typed recall** - was it overlapped by a detection of the right category?
  Lower typed recall with full span recall means the value is removed but
  labelled differently, which is a reporting problem, not a leak.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

from evaluations.corpus_generator import DOCUMENTS_PER_PARTITION, generate

#: Phase 11's corpus - the third distinct evaluation corpus, and the fourth
#: revision of this file. Phase 10's held-out partition was inspected before
#: the vocabulary was extended, so it is diagnostic data now and `heldout_v3`
#: is the untouched one.
CORPUS_VERSION = "canary-v4.0"

PARTITION_DEV = "development"
PARTITION_CALIB = "calibration"
#: Phase 9's held-out partition. It has been inspected, so it is diagnostic
#: data now and must never again be used as evidence of generalisation.
PARTITION_DIAGNOSTIC = "test"
#: Phase 10's held-out partition. It was run once and then reported, but the
#: clinical vocabulary was extended afterwards, so it too is diagnostic now.
PARTITION_DIAGNOSTIC_V2 = "heldout_v2"
#: Phase 11's held-out partition. Untouched while the evidence model, the
#: residual validator and the thresholds were settled, then run once.
PARTITION_HELDOUT = "heldout_v3"
PARTITIONS = (
    PARTITION_DEV,
    PARTITION_CALIB,
    PARTITION_DIAGNOSTIC,
    PARTITION_DIAGNOSTIC_V2,
    PARTITION_HELDOUT,
)
#: Kept for callers written before the Phase 9 partition was retired.
PARTITION_TEST = PARTITION_DIAGNOSTIC


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

#: Gold categories, mapped to the internal taxonomy where one exists.
#: A gold category with no internal equivalent is a coverage gap by
#: construction, and the evaluation reports it as such rather than hiding it.
CATEGORY_ALIASES: Dict[str, Tuple[str, ...]] = {
    # People
    "PERSON": ("PERSON",),
    "PERSON_RELATIVE": ("PERSON",),
    "PERSON_CLINICIAN": ("PERSON",),
    # Organisations and places
    "EMPLOYER": ("ORGANIZATION", "FACILITY"),
    "HOSPITAL": ("FACILITY", "ORGANIZATION", "LOCATION"),
    "ORGANIZATION": ("ORGANIZATION", "FACILITY"),
    "ADDRESS": ("ADDRESS", "LOCATION"),
    "GEOGRAPHY": ("LOCATION", "ADDRESS"),
    "POSTAL_CODE": ("POSTAL_CODE", "LOCATION", "ADDRESS"),
    # Temporal
    "DATE": ("DATE_TIME", "DATE"),
    "AGE_OVER_89": ("AGE", "AGE_OVER_89", "DATE_TIME"),
    # Contact
    "PHONE": ("PHONE_NUMBER",),
    "FAX": ("PHONE_NUMBER", "FAX_NUMBER"),
    "EMAIL": ("EMAIL_ADDRESS",),
    # Numbers
    "SSN": ("US_SSN", "SSN"),
    "MRN": ("MEDICAL_RECORD_NUMBER", "PATIENT_ID", "IDENTIFIER"),
    "PATIENT_ID": ("PATIENT_ID", "MEDICAL_RECORD_NUMBER", "IDENTIFIER"),
    "HEALTH_PLAN": ("HEALTH_PLAN_ID", "INSURANCE_ID", "IDENTIFIER"),
    "ACCOUNT_NUMBER": ("ACCOUNT_NUMBER", "IDENTIFIER"),
    "LICENSE_NUMBER": ("DRIVER_LICENSE", "LICENSE_NUMBER", "IDENTIFIER"),
    "CERTIFICATE_NUMBER": ("CERTIFICATE_NUMBER", "IDENTIFIER", "LICENSE_NUMBER"),
    "DEVICE_ID": ("DEVICE_ID", "IDENTIFIER"),
    "VEHICLE_ID": ("VEHICLE_ID", "DEVICE_ID", "IDENTIFIER"),
    "ACCESSION": ("ACCESSION_NUMBER", "IDENTIFIER"),
    "BIOMETRIC_ID": ("BIOMETRIC_ID", "IDENTIFIER"),
    "UNUSUAL_ID": ("IDENTIFIER", "DEVICE_ID", "PATIENT_ID"),
    # Network
    "URL": ("URL",),
    "IP_ADDRESS": ("IP_ADDRESS",),
    "USERNAME": ("USERNAME", "IDENTIFIER"),
}

#: Every gold category the corpus is required to cover.
REQUIRED_CATEGORIES: Tuple[str, ...] = tuple(sorted(CATEGORY_ALIASES))


@dataclass(frozen=True)
class GoldSpan:
    """One value that must be removed, with exact offsets into the document."""

    start: int
    end: int
    category: str
    #: Stress features this span exercises, e.g. "unicode", "chunk_boundary".
    tags: Tuple[str, ...] = ()

    def overlaps(self, start: int, end: int) -> bool:
        return start < self.end and self.start < end


@dataclass(frozen=True)
class LabelledDocument:
    doc_id: str
    partition: str
    text: str
    spans: Tuple[GoldSpan, ...]
    #: Ordinary terms that must survive: clinical vocabulary that looks like a
    #: name, units, eponymous conditions. Recorded as offsets so an
    #: over-redaction can be counted precisely.
    negatives: Tuple[Tuple[int, int, str], ...] = ()
    tags: Tuple[str, ...] = ()
    notes: str = ""

    def value(self, span: GoldSpan) -> str:
        """The gold value. Used only for corpus self-checks, never reported."""
        return self.text[span.start : span.end]


class _Builder:
    """Assembles a document while recording exact offsets as it goes."""

    def __init__(self, doc_id: str, partition: str):
        self.doc_id = doc_id
        self.partition = partition
        self._parts: List[str] = []
        self._len = 0
        self._spans: List[GoldSpan] = []
        self._negatives: List[Tuple[int, int, str]] = []
        self._tags: List[str] = []

    def t(self, text: str) -> "_Builder":
        """Append ordinary prose."""
        self._parts.append(text)
        self._len += len(text)
        return self

    def phi(self, value: str, category: str, *tags: str) -> "_Builder":
        """Append a value that must be removed, recording its span."""
        start = self._len
        self._parts.append(value)
        self._len += len(value)
        self._spans.append(GoldSpan(start, self._len, category, tuple(tags)))
        self._tags.extend(tags)
        return self

    def keep(self, value: str, label: str) -> "_Builder":
        """Append a term that must survive redaction."""
        start = self._len
        self._parts.append(value)
        self._len += len(value)
        self._negatives.append((start, self._len, label))
        return self

    def done(self, notes: str = "", *tags: str) -> LabelledDocument:
        self._tags.extend(tags)
        return LabelledDocument(
            doc_id=self.doc_id,
            partition=self.partition,
            text="".join(self._parts),
            spans=tuple(self._spans),
            negatives=tuple(self._negatives),
            tags=tuple(sorted(set(self._tags))),
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Disjoint value pools, one per partition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValuePool:
    """Every invented value a partition may use. No value repeats across pools."""

    person: Sequence[str]
    person_unicode: Sequence[str]
    relative: Sequence[str]
    clinician: Sequence[str]
    employer: str
    hospital: str
    organization: str
    address: str
    geography: str
    postal_code: str
    date_iso: str
    date_us: str
    date_long: str
    age_over_89: str
    phone: str
    phone_intl: str
    fax: str
    email: str
    email_unicode: str
    ssn: str
    mrn: str
    mrn_misspelt_label: str
    patient_id: str
    health_plan: str
    account_number: str
    license_number: str
    certificate_number: str
    device_id: str
    vehicle_id: str
    accession: str
    biometric_id: str
    unusual_id: str
    url: str
    ip_address: str
    username: str


POOLS: Dict[str, ValuePool] = {
    PARTITION_DEV: ValuePool(
        person=("Ananya Krishnamurthy", "Tobias Ravensworth"),
        person_unicode=("José Peñaloza-Ferrán", "Śrīnivāsan Ayyangār"),
        relative=("Deepa Krishnamurthy", "Harold Ravensworth"),
        clinician=("Dr. Vikram Chatterjee", "Dr. Elspeth Marchetti"),
        employer="Quillfeather Textiles Ltd",
        hospital="Northmarsh General Hospital",
        organization="Vermillion Research Collective",
        address="418 Bramblewick Lane, Apartment 7C",
        geography="Thornbury Heath, Wexcombe County",
        postal_code="49182",
        date_iso="2019-04-02",
        date_us="03/15/1985",
        date_long="14 March 2021",
        age_over_89="94",
        phone="555-0102",
        phone_intl="+91 98765 43210",
        fax="555-0117",
        email="ananya.k@example.invalid",
        email_unicode="josé.peñaloza@example.invalid",
        ssn="911-45-6789",
        mrn="SYN-4820193",
        mrn_misspelt_label="Medcial Record Numbr",
        patient_id="SYN-PT-0099",
        health_plan="SYN-PLAN-556677",
        account_number="ACCT-99120345",
        license_number="DL-K4820193X",
        certificate_number="CERT-MB-77120",
        device_id="SYN-DEV-330091",
        vehicle_id="VIN-1HGCM82633A004352",
        accession="SYN-ACC-771201",
        biometric_id="FP-TEMPLATE-A0091",
        unusual_id="ZZ//4471-Q8//KX",
        url="https://records.example.invalid/patient/9931",
        ip_address="203.0.113.42",
        username="a.krishnamurthy91",
    ),
    PARTITION_CALIB: ValuePool(
        person=("Rukmini Balasubramanian", "Gareth Ollivander"),
        person_unicode=("Zoë Müller-Thorvald", "Ravīndra Śarmā"),
        relative=("Sundar Balasubramanian", "Marjorie Ollivander"),
        clinician=("Dr. Priyanka Venkataraman", "Dr. Aloysius Brackenridge"),
        employer="Windlass Aeronautics Pvt Ltd",
        hospital="Saint Corwin Memorial Institute",
        organization="Halcyon Biosciences Trust",
        address="72 Pendlemere Crescent, Flat 3B",
        geography="Ashcombe Vale, Merridew District",
        postal_code="60731",
        date_iso="2020-11-19",
        date_us="07/22/1978",
        date_long="9 September 2018",
        age_over_89="97",
        phone="555-0123",
        phone_intl="+44 20 7946 0958",
        fax="555-0138",
        email="r.balasubramanian@example.invalid",
        email_unicode="zoë.müller@example.invalid",
        ssn="922-65-4321",
        mrn="SYN-6610284",
        mrn_misspelt_label="Medicial Recrd No",
        patient_id="SYN-PT-4417",
        health_plan="SYN-PLAN-889900",
        account_number="ACCT-45077318",
        license_number="DL-R6610284Y",
        certificate_number="CERT-NP-31908",
        device_id="SYN-DEV-771043",
        vehicle_id="VIN-5YJ3E1EA7KF317654",
        accession="SYN-ACC-330877",
        biometric_id="IRIS-TEMPLATE-B7734",
        unusual_id="QX--9083/TT--LM",
        url="https://portal.example.invalid/chart/4417",
        ip_address="198.51.100.77",
        username="r.balasub2020",
    ),
    PARTITION_DIAGNOSTIC_V2: ValuePool(
        person=("Padmavathi Venkataraghavan", "Algernon Fitzwilliam-Crewe"),
        person_unicode=("Åsa Lindqvist-Öberg", "Bhāskara Rāmānujan"),
        relative=("Ganesan Venkataraghavan", "Hortensia Fitzwilliam-Crewe"),
        clinician=("Dr. Shalini Muthukrishnan", "Dr. Peregrine Wolstenholme"),
        employer="Larkspur Instrumentation GmbH",
        hospital="Duskwater Priory Teaching Hospital",
        organization="Ardent Meridian Research Society",
        address="93 Quillon Hollow Way, Unit 12D",
        geography="Umberfield Reach, Calderstone Hundred",
        postal_code="27615",
        date_iso="2015-08-14",
        date_us="02/09/1969",
        date_long="5 November 2013",
        age_over_89="102",
        phone="555-0161",
        phone_intl="+81 3 5550 2244",
        fax="555-0179",
        email="p.venkataraghavan@example.invalid",
        email_unicode="åsa.lindqvist@example.invalid",
        ssn="944-54-9876",
        mrn="SYN-2748903",
        mrn_misspelt_label="Medical Recrod Numbr",
        patient_id="SYN-PT-6612",
        health_plan="SYN-PLAN-332211",
        account_number="ACCT-61204497",
        license_number="DL-P2748903Q",
        certificate_number="CERT-PT-90214",
        device_id="SYN-DEV-554120",
        vehicle_id="VIN-WBA3A5C56DF123789",
        accession="SYN-ACC-611903",
        biometric_id="RETINA-TEMPLATE-D2290",
        unusual_id="NN;;3391-V7;;RR",
        url="https://notes.example.invalid/case/6612",
        ip_address="198.18.51.203",
        username="p.venkat_1969",
    ),
    PARTITION_TEST: ValuePool(
        person=("Meenakshi Raghunathan", "Cornelius Ashdown-Blythe"),
        person_unicode=("François Lécuyer", "Kṛṣṇa Mūrti Iyer"),
        relative=("Lakshmi Raghunathan", "Beatrice Ashdown-Blythe"),
        clinician=("Dr. Nandini Sundaresan", "Dr. Bartholomew Quiller"),
        employer="Everdene Marine Logistics",
        hospital="Fallowfield Regional Medical Centre",
        organization="Cinderhall Genomics Foundation",
        address="1156 Marlowe Ridge Road, Suite 210",
        geography="Kestrelmoor, Barrowdale Parish",
        postal_code="83094",
        date_iso="2017-06-28",
        date_us="11/03/1992",
        date_long="23 January 2016",
        age_over_89="91",
        phone="555-0145",
        phone_intl="+61 2 5550 0134",
        fax="555-0159",
        email="m.raghunathan@example.invalid",
        email_unicode="françois.lécuyer@example.invalid",
        ssn="933-78-9012",
        mrn="SYN-9037461",
        mrn_misspelt_label="Medical Recod Numbe",
        patient_id="SYN-PT-8823",
        health_plan="SYN-PLAN-114477",
        account_number="ACCT-70318842",
        license_number="DL-M9037461Z",
        certificate_number="CERT-RN-55021",
        device_id="SYN-DEV-118204",
        vehicle_id="VIN-JH4KA9650MC012345",
        accession="SYN-ACC-902314",
        biometric_id="VOICE-TEMPLATE-C5518",
        unusual_id="WW::7710-Z3::PP",
        url="https://ehr.example.invalid/record/8823",
        ip_address="192.0.2.155",
        username="m.raghu_1992",
    ),
    PARTITION_HELDOUT: ValuePool(
        person=("Yashodhara Sathyanarayanan", "Montgomery Wintersgill"),
        person_unicode=("Ómar Björnsdóttir", "Lakṣmī Tirumalāchārya"),
        relative=("Vaikuntam Sathyanarayanan", "Rosalind Wintersgill"),
        clinician=("Dr. Devayani Ponnambalam", "Dr. Cuthbert Ravencourt"),
        employer="Ravencourt Aviation Systems Ltd",
        hospital="Highbarrow District Hospital",
        organization="Standerwick Bioinformatics Trust",
        address="204 Oldencastle Row, Unit 9A",
        geography="Winterscombe Reach, Highbarrow District",
        postal_code="85412",
        date_iso="2012-02-27",
        date_us="09/18/1974",
        date_long="17 June 2011",
        age_over_89="96",
        phone="555-0183",
        phone_intl="+49 30 5550 7712",
        fax="555-0191",
        email="y.sathyanarayanan@example.invalid",
        email_unicode="ómar.björnsdóttir@example.invalid",
        ssn="955-31-8842",
        mrn="SYN-V-8512740",
        mrn_misspelt_label="Medcal Recordd Num",
        patient_id="SYN-V-PT-8531",
        health_plan="SYN-V-PLAN-854120",
        account_number="ACCT-85219034",
        license_number="DL-Z8512740W",
        certificate_number="CERT-RN-85177",
        device_id="SYN-V-DEV-853301",
        vehicle_id="VIN-KM8J3CA46JU887215",
        accession="SYN-V-ACC-851904",
        biometric_id="PALM-TEMPLATE-E8544",
        unusual_id="VV..8531-W9..TT",
        url="https://ravencourt.example.invalid/chart/8531",
        ip_address="203.0.113.221",
        username="y.sathya_1974",
    ),
}


#: Clinical vocabulary that resembles a personal name but is not PHI. Shared
#: across partitions on purpose: these are real medical terms, not values, and
#: over-redacting them is the same defect in every partition.
NON_PHI_CLINICAL_TERMS: Tuple[str, ...] = (
    "Parkinson",
    "Alzheimer",
    "Crohn",
    "Hodgkin",
    "Bell",
    "Graves",
    "Paget",
    "Wilson",
    "Addison",
    "Cushing",
)


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------

_FILLER = (
    "The patient tolerated the procedure without complication. Vital signs "
    "remained stable throughout the observation period. No adverse reaction "
    "was recorded by nursing staff. "
)


def _build_partition(partition: str) -> List[LabelledDocument]:
    p = POOLS[partition]
    docs: List[LabelledDocument] = []

    # --- 1. Names: Indian, international, relatives, clinicians, employers ---
    b = _Builder(f"{partition}_names", partition)
    b.t("Patient ").phi(p.person[0], "PERSON", "indian_name")
    b.t(" was accompanied by ").phi(p.relative[0], "PERSON_RELATIVE", "relative")
    b.t(", the patient's mother. Care was provided by ")
    b.phi(p.clinician[0], "PERSON_CLINICIAN", "clinician")
    b.t(" at ").phi(p.hospital, "HOSPITAL", "facility")
    b.t(".\nThe patient is employed by ").phi(p.employer, "EMPLOYER", "employer")
    b.t(" and was referred through ").phi(p.organization, "ORGANIZATION")
    b.t(".\nA second reviewer, ").phi(p.person[1], "PERSON", "international_name")
    b.t(", countersigned. Spouse ").phi(p.relative[1], "PERSON_RELATIVE", "relative")
    b.t(" was present.")
    docs.append(b.done("Names across origins, relatives, clinicians, employers."))

    # --- 2. Unicode and mixed casing ---
    b = _Builder(f"{partition}_unicode_casing", partition)
    b.t("Referral received for ").phi(p.person_unicode[0], "PERSON", "unicode")
    b.t(" (preferred spelling retained).\nSecond opinion from ")
    b.phi(p.person_unicode[1], "PERSON", "unicode", "transliteration")
    b.t(".\nContact address: ").phi(p.email_unicode, "EMAIL", "unicode")
    b.t("\nCASE ESCALATED BY ").phi(p.person[0].upper(), "PERSON", "mixed_case", "uppercase")
    b.t(" and reviewed by ").phi(p.person[1].lower(), "PERSON", "mixed_case", "lowercase")
    b.t(".")
    docs.append(b.done("Unicode names, transliteration, upper and lower casing."))

    # --- 3. Geography, address, postal code ---
    b = _Builder(f"{partition}_geography", partition)
    b.t("Home address on file: ").phi(p.address, "ADDRESS", "address")
    b.t(", ").phi(p.geography, "GEOGRAPHY", "geography")
    b.t(" ").phi(p.postal_code, "POSTAL_CODE", "postal_code")
    b.t(".\nTransport arranged from the residence to ")
    b.phi(p.hospital, "HOSPITAL", "facility")
    b.t(". Mail returned from ZIP ").phi(p.postal_code, "POSTAL_CODE", "postal_code")
    b.t(" twice.")
    docs.append(b.done("Street address, region, and postal code."))

    # --- 4. Dates and age over 89 ---
    b = _Builder(f"{partition}_dates_age", partition)
    b.t("Date of birth ").phi(p.date_us, "DATE", "date_us")
    b.t("; admitted ").phi(p.date_iso, "DATE", "date_iso")
    b.t("; discharged ").phi(p.date_long, "DATE", "date_written")
    b.t(".\nThe patient is ").phi(p.age_over_89, "AGE_OVER_89", "age_over_89")
    b.t(" years old, which exceeds the Safe Harbor aggregation threshold.\n")
    b.t("A sibling aged ").keep("62", "age_under_90")
    b.t(" years is not an identifier at that age.")
    docs.append(
        b.done(
            "Three date formats plus an age above 89, which Safe Harbor "
            "requires be aggregated rather than reported."
        )
    )

    # --- 5. Contact details, including fax and international phone ---
    b = _Builder(f"{partition}_contact", partition)
    b.t("Primary contact ").phi(p.phone, "PHONE", "phone")
    b.t(", international ").phi(p.phone_intl, "PHONE", "phone_international")
    b.t(".\nFax results to ").phi(p.fax, "FAX", "fax")
    b.t(".\nEmail ").phi(p.email, "EMAIL", "email")
    b.t("\nPortal login ").phi(p.username, "USERNAME", "username")
    b.t(" at ").phi(p.url, "URL", "url")
    b.t(" accessed from ").phi(p.ip_address, "IP_ADDRESS", "ip_end_of_sentence")
    b.t(".")
    docs.append(b.done("Phone, fax, email, username, URL, and a sentence-final IP."))

    # --- 6. Numeric identifiers, including uncommon ones ---
    b = _Builder(f"{partition}_identifiers", partition)
    b.t("MRN: ").phi(p.mrn, "MRN", "mrn")
    b.t("\nPatient ID ").phi(p.patient_id, "PATIENT_ID")
    b.t("\nHealth Plan ID ").phi(p.health_plan, "HEALTH_PLAN")
    b.t("\nAccession Number ").phi(p.accession, "ACCESSION")
    b.t("\nAccount number ").phi(p.account_number, "ACCOUNT_NUMBER", "account")
    b.t("\nDriver's license ").phi(p.license_number, "LICENSE_NUMBER", "license")
    b.t("\nCertificate ").phi(p.certificate_number, "CERTIFICATE_NUMBER", "certificate")
    b.t("\nDevice ID ").phi(p.device_id, "DEVICE_ID")
    b.t("\nVehicle VIN ").phi(p.vehicle_id, "VEHICLE_ID", "vehicle")
    b.t("\nBiometric template reference ").phi(p.biometric_id, "BIOMETRIC_ID", "biometric")
    b.t("\nLegacy study code ").phi(p.unusual_id, "UNUSUAL_ID", "unusual_format")
    b.t("\nSSN ").phi(p.ssn, "SSN", "ssn")
    docs.append(b.done("The full identifier sweep, including rarely handled kinds."))

    # --- 7. Misspellings and abbreviations around identifier labels ---
    b = _Builder(f"{partition}_misspelling_abbrev", partition)
    b.t("Pt. ").phi(p.person[0], "PERSON", "abbreviation_context")
    b.t(", DOB ").phi(p.date_us, "DATE", "abbreviation_context")
    b.t(".\n").t(p.mrn_misspelt_label).t(": ")
    b.phi(p.mrn, "MRN", "misspelt_label")
    b.t("\nPh. ").phi(p.phone, "PHONE", "abbreviation_context")
    b.t("\nAddr: ").phi(p.address, "ADDRESS", "abbreviation_context")
    b.t("\nHosp: ").phi(p.hospital, "HOSPITAL", "abbreviation_context")
    b.t("\nSurname mis-typed as ")
    b.phi(p.person[1].replace("o", "0", 1), "PERSON", "misspelt_value")
    b.t(" in the referral letter.")
    docs.append(
        b.done("Abbreviated labels and misspellings, of both labels and values.")
    )

    # --- 8. Multiline PHI and overlapping identifiers ---
    b = _Builder(f"{partition}_multiline_overlap", partition)
    b.t("Correspondence block:\n")
    b.phi(p.person[0], "PERSON", "multiline")
    b.t("\n").phi(p.address, "ADDRESS", "multiline")
    b.t("\n").phi(p.geography, "GEOGRAPHY", "multiline")
    b.t(" ").phi(p.postal_code, "POSTAL_CODE", "multiline")
    b.t("\n").phi(p.phone, "PHONE", "multiline")
    b.t("\n\nCombined reference ")
    # An identifier embedded inside a URL: two gold spans over the same region.
    combined = f"{p.url}?mrn={p.mrn}"
    start_before = len("".join(b._parts))
    b.phi(combined, "URL", "overlapping")
    b._spans.append(
        GoldSpan(
            start_before + len(p.url) + len("?mrn="),
            start_before + len(combined),
            "MRN",
            ("overlapping", "nested"),
        )
    )
    b.t(" was quoted in the request.")
    docs.append(
        b.done("A multi-line correspondence block and an MRN nested inside a URL.")
    )

    # --- 9. Non-PHI clinical vocabulary that resembles names ---
    b = _Builder(f"{partition}_negative_controls", partition)
    b.t("Assessment: the patient has ").keep("Parkinson", "eponym")
    b.t("'s disease with features of ").keep("Alzheimer", "eponym")
    b.t("'s dementia.\nHistory of ").keep("Crohn", "eponym")
    b.t("'s colitis and ").keep("Hodgkin", "eponym")
    b.t(" lymphoma in remission.\n").keep("Bell", "eponym")
    b.t("'s palsy resolved. ").keep("Graves", "eponym")
    b.t("' disease excluded. ").keep("Paget", "eponym")
    b.t("'s disease of bone noted on imaging.\n")
    b.t("Screened for ").keep("Wilson", "eponym").t("'s disease, ")
    b.keep("Addison", "eponym").t("'s disease and ")
    b.keep("Cushing", "eponym").t("'s syndrome; all negative.\n")
    b.t("Imaging modalities used: ").keep("CT", "modality").t(", ")
    b.keep("MRI", "modality").t(" and ").keep("ECG", "modality").t(".\n")
    b.t("Treating physician ").phi(p.clinician[1], "PERSON_CLINICIAN", "clinician")
    b.t(" signed off.")
    docs.append(
        b.done(
            "Eponymous conditions and modality abbreviations must survive; the "
            "one real name in the note must not.",
            "negative_controls",
        )
    )

    # --- 10. PHI straddling a chunk boundary ---
    # Placed so the value sits across the default 2000-character window edge.
    b = _Builder(f"{partition}_chunk_boundary", partition)
    lead = _FILLER * 11  # comfortably past 2000 characters
    trimmed = lead[: 2000 - 8]
    b.t(trimmed).t("Name: ")
    b.phi(p.person[1], "PERSON", "chunk_boundary")
    b.t(" MRN ").phi(p.mrn, "MRN", "chunk_boundary")
    b.t(". ").t(_FILLER * 3)
    b.t("Second-window contact ").phi(p.email, "EMAIL", "second_window")
    b.t(".")
    docs.append(
        b.done(
            "A name and MRN positioned across the default 2000-character "
            "window edge, to exercise overlapping-chunk inference.",
            "chunk_boundary",
        )
    )

    return docs


def build_corpus() -> Dict[str, List[LabelledDocument]]:
    """Build every partition, hand-authored documents first."""
    return {partition: partition_documents(partition) for partition in PARTITIONS}


@lru_cache(maxsize=len(PARTITIONS))
def _partition_documents(partition: str) -> Tuple[LabelledDocument, ...]:
    """Hand-authored documents, then generated ones up to the target size.

    Ten documents per partition measured nothing: a manual-review rate over
    ten documents moves in steps of ten per cent. The hand-authored ten stay
    at the front because they are the ones a reader can check by eye; the
    generator supplies the volume behind them.

    Cached because every integrity test rebuilds all five partitions, and the
    documents are immutable once built.
    """
    handwritten = _build_partition(partition)
    remaining = max(0, DOCUMENTS_PER_PARTITION - len(handwritten))
    generated = generate(partition, _Builder, remaining)
    return tuple(handwritten) + tuple(generated)


def partition_documents(partition: str) -> List[LabelledDocument]:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition: {partition}")
    return list(_partition_documents(partition))


def all_values(partition: str) -> List[str]:
    """Every gold value in a partition, for disjointness self-checks."""
    values: List[str] = []
    for document in partition_documents(partition):
        for span in document.spans:
            values.append(document.value(span))
    return values


def corpus_statistics() -> Dict[str, Dict[str, int]]:
    """Counts only. Never returns a value."""
    stats: Dict[str, Dict[str, int]] = {}
    for partition, documents in build_corpus().items():
        by_category: Dict[str, int] = {}
        for document in documents:
            for span in document.spans:
                by_category[span.category] = by_category.get(span.category, 0) + 1
        stats[partition] = {
            "documents": len(documents),
            "gold_spans": sum(len(d.spans) for d in documents),
            "negative_terms": sum(len(d.negatives) for d in documents),
            "categories_covered": len(by_category),
            **{f"cat_{name}": count for name, count in sorted(by_category.items())},
        }
    return stats
