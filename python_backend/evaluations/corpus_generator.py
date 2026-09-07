"""Seeded document generator for the labelled corpus.

Ten hand-authored documents per partition were enough to prove a detector
runs. They are not enough to measure one: a manual-review rate computed over
ten documents moves in steps of ten per cent, and a single unlucky sentence
looks like a systematic defect. Phase 11 needs partitions large enough that
the numbers mean something, so this module composes each partition up to two
hundred documents from templates and per-partition value pools.

Two properties matter more than variety here.

**Disjointness.** Every generated value is built from stems belonging to
exactly one partition, so a value used while choosing a threshold cannot
reappear in the held-out measurement. The property holds by construction
rather than by inspection, and ``test_labelled_corpus`` asserts it anyway.

**Determinism.** The generator is seeded from the corpus version and the
partition name, never from the clock or from iteration order, so the corpus a
report describes is the corpus anybody else rebuilds.

Every value here is invented. Names are composed from synthetic stems, email
and web addresses use the reserved ``.invalid`` domain, telephone numbers come
from the 555-01xx fiction range, addresses use documentation network blocks,
and social-security numbers use the 900 range that was never issued.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

#: Documents per partition. The requirement is at least two hundred; the
#: hand-authored templates are counted towards it, so the generator makes up
#: the difference.
DOCUMENTS_PER_PARTITION = 200


@dataclass(frozen=True)
class StemPool:
    """Per-partition building blocks. No stem appears in two pools."""

    given: Tuple[str, ...]
    family: Tuple[str, ...]
    unicode_given: Tuple[str, ...]
    unicode_family: Tuple[str, ...]
    place: Tuple[str, ...]
    street: Tuple[str, ...]
    org_head: Tuple[str, ...]
    org_tail: Tuple[str, ...]
    #: Leading digits reserved to this partition, keeping every generated
    #: number disjoint from every other partition's.
    numeric_block: str
    #: Phone suffixes inside the 555-01xx fiction range, twenty per partition.
    phone_block: int
    #: Fourth octet range inside a documentation network.
    ip_network: str
    ip_block: int
    ssn_group: str
    id_prefix: str
    #: Encounter years, birth years and aggregated ages reserved to this
    #: partition. Bare numbers are values too: an age of "96" appearing in two
    #: partitions is the same leak as a name appearing in two partitions, and
    #: the disjointness test counts it as one.
    encounter_years: Tuple[int, int]
    birth_years: Tuple[int, int]
    ages_over_89: Tuple[int, ...]
    ages_under_90: Tuple[int, ...]


STEMS: Dict[str, StemPool] = {
    "development": StemPool(
        given=(
            "Arvind", "Bhavana", "Chetanya", "Dhruvika", "Eshaan", "Farhana",
            "Gautami", "Harishwar", "Ingvild", "Joakim", "Katarzyna", "Lennart",
        ),
        family=(
            "Thistlewood", "Barrowmede", "Culverhaye", "Draymont", "Everwilde",
            "Fenwarden", "Glaisdale", "Hollowbeck", "Ironvale", "Jarrowmoor",
            "Kestrelbourne", "Lindenshaw",
        ),
        unicode_given=("Ílvaro", "Ķrishnā", "Ǫlafur"),
        unicode_family=("Peñaloza-Ferrán", "Ayyangār-Śāstri", "Þorvaldsdóttir"),
        place=("Wrenmarsh", "Coldbarrow", "Fernhollow", "Aldergate"),
        street=("Bramblewick", "Quillon", "Harrowmere", "Ashvale"),
        org_head=("Quillfeather", "Vermillion", "Northmarsh", "Larchgate"),
        org_tail=("Textiles Ltd", "Research Collective", "General Hospital",
                  "Diagnostics Ltd"),
        numeric_block="41",
        phone_block=100,
        ip_network="203.0.113",
        ip_block=1,
        ssn_group="11",
        id_prefix="SYN-D",
        encounter_years=(1996, 2000),
        birth_years=(1930, 1940),
        ages_over_89=(90, 93),
        ages_under_90=(61, 63),
    ),
    "calibration": StemPool(
        given=(
            "Rukmini", "Gareth", "Sundaram", "Marjolein", "Prathibha", "Aloysius",
            "Naveenya", "Bertrand", "Chandrika", "Desmond", "Ellammal", "Fitzroy",
        ),
        family=(
            "Balasubramanian", "Ollivander", "Brackenridge", "Windlass",
            "Halcyonmere", "Pendlemere", "Ashcombe", "Merridew", "Corwinshaw",
            "Dunwallow", "Estermont", "Farrowgate",
        ),
        unicode_given=("Zoë", "Ravīndra", "Åsvald"),
        unicode_family=("Müller-Thorvald", "Śarmā-Nāyar", "Øksendal"),
        place=("Ashcombe Vale", "Merridew Cross", "Pendlemere", "Halcyon Reach"),
        street=("Pendlemere", "Corwin", "Halcyon", "Windlass"),
        org_head=("Windlass", "Halcyon", "Saint Corwin", "Pendlemere"),
        org_tail=("Aeronautics Pvt Ltd", "Biosciences Trust",
                  "Memorial Institute", "Imaging Laboratory"),
        numeric_block="52",
        phone_block=120,
        ip_network="198.51.100",
        ip_block=1,
        ssn_group="22",
        id_prefix="SYN-C",
        encounter_years=(2000, 2004),
        birth_years=(1940, 1950),
        ages_over_89=(95, 98),
        ages_under_90=(64, 73),
    ),
    "test": StemPool(
        given=(
            "Meenakshi", "Cornelius", "Lakshmipathy", "Beatrix", "Nandini",
            "Bartholomew", "Vaidehi", "Percival", "Sharmila", "Reginald",
            "Thangaraj", "Ursula",
        ),
        family=(
            "Raghunathan", "Ashdown-Blythe", "Sundaresan", "Quillerton",
            "Everdene", "Cinderhall", "Fallowfield", "Kestrelmoor", "Barrowdale",
            "Marlowridge", "Nettlefold", "Oakenshott",
        ),
        unicode_given=("François", "Kṛṣṇa", "Ingrídur"),
        unicode_family=("Lécuyer-Mounier", "Mūrti-Iyer", "Sigurðardóttir"),
        place=("Kestrelmoor", "Barrowdale", "Fallowfield", "Marlowe Ridge"),
        street=("Marlowe Ridge", "Cinderhall", "Everdene", "Nettlefold"),
        org_head=("Everdene", "Cinderhall", "Fallowfield", "Nettlefold"),
        org_tail=("Marine Logistics", "Genomics Foundation",
                  "Regional Medical Centre", "Pathology Laboratory"),
        numeric_block="63",
        phone_block=140,
        ip_network="192.0.2",
        ip_block=1,
        ssn_group="33",
        id_prefix="SYN-T",
        encounter_years=(2004, 2008),
        birth_years=(1950, 1960),
        ages_over_89=(99, 101),
        ages_under_90=(66, 75),
    ),
    "heldout_v2": StemPool(
        given=(
            "Padmavathi", "Algernon", "Ganesaraj", "Hortensia", "Shalini",
            "Peregrine", "Kanchana", "Wolstan", "Yamunadevi", "Zachariah",
            "Bhuvaneswari", "Casimir",
        ),
        family=(
            "Venkataraghavan", "Fitzwilliam-Crewe", "Muthukrishnan",
            "Wolstenholme", "Larkspurne", "Duskwater", "Umberfield",
            "Calderstone", "Quillonhollow", "Ardentmere", "Bramblecote",
            "Cindermoor",
        ),
        unicode_given=("Åsa", "Bhāskara", "Élodie"),
        unicode_family=("Lindqvist-Öberg", "Rāmānujan-Aiyar", "Ødegård"),
        place=("Umberfield Reach", "Calderstone", "Duskwater", "Larkspur Vale"),
        street=("Quillon Hollow", "Larkspur", "Duskwater", "Umberfield"),
        org_head=("Larkspur", "Ardent Meridian", "Duskwater Priory",
                  "Calderstone"),
        org_tail=("Instrumentation GmbH", "Research Society",
                  "Teaching Hospital", "Radiology Clinic"),
        numeric_block="74",
        phone_block=160,
        ip_network="198.18.51",
        ip_block=1,
        ssn_group="44",
        id_prefix="SYN-H",
        encounter_years=(2008, 2011),
        birth_years=(1960, 1969),
        ages_over_89=(103, 105),
        ages_under_90=(68, 77),
    ),
    "heldout_v3": StemPool(
        given=(
            "Yashodhara", "Montgomery", "Vaikuntam", "Rosalind", "Anantharaman",
            "Cuthbert", "Devayani", "Ferdinand", "Girijadevi", "Humphrey",
            "Ilangovan", "Jocasta",
        ),
        family=(
            "Sathyanarayanan", "Wintersgill", "Ponnambalam", "Ravencourt",
            "Thiruvengadam", "Standerwick", "Muthuswamy", "Highbarrow",
            "Chidambaram", "Fairweather-Voss", "Nagarathinam", "Oldencastle",
        ),
        unicode_given=("Ómar", "Lakṣmī", "Sébastien"),
        unicode_family=("Björnsdóttir", "Tirumalāchārya", "Grüneberg-Vasseur"),
        place=("Highbarrow", "Ravencourt", "Standerwick", "Winterscombe"),
        street=("Ravencourt", "Highbarrow", "Standerwick", "Oldencastle"),
        org_head=("Ravencourt", "Highbarrow", "Standerwick", "Winterscombe"),
        org_tail=("Aviation Systems Ltd", "Bioinformatics Trust",
                  "District Hospital", "Screening Laboratory"),
        numeric_block="85",
        phone_block=180,
        ip_network="203.0.113",
        ip_block=200,
        ssn_group="55",
        id_prefix="SYN-V",
        encounter_years=(2022, 2025),
        birth_years=(1986, 1992),
        ages_over_89=(106, 108),
        ages_under_90=(69, 79),
    ),
}


#: Ordinary clinical language that must survive. Shared across partitions on
#: purpose: these are medical terms rather than values, and destroying them is
#: the same defect wherever it happens.
NEGATIVE_TERMS: Tuple[Tuple[str, str], ...] = (
    ("Parkinson", "eponym"),
    ("Alzheimer", "eponym"),
    ("Crohn", "eponym"),
    ("Hodgkin", "eponym"),
    ("Bell", "eponym"),
    ("Graves", "eponym"),
    ("Wilson", "eponym"),
    ("Addison", "eponym"),
    ("Cushing", "eponym"),
    ("Paget", "eponym"),
    ("Blood pressure", "vital_sign"),
    ("temporal lobe", "anatomy"),
    ("left atrium", "anatomy"),
    ("iliac crest", "anatomy"),
    ("Metformin", "medication"),
    ("Atorvastatin", "medication"),
    ("Levothyroxine", "medication"),
    ("Amoxicillin", "medication"),
    ("Creatinine", "analyte"),
    ("Haemoglobin", "analyte"),
)

#: Conditions written as eponyms, for the possessive construction.
EPONYM_CONDITIONS: Tuple[Tuple[str, str], ...] = (
    ("Parkinson", "disease"),
    ("Alzheimer", "disease"),
    ("Crohn", "disease"),
    ("Hodgkin", "lymphoma"),
    ("Bell", "palsy"),
    ("Wilson", "disease"),
    ("Addison", "disease"),
    ("Cushing", "syndrome"),
)

#: Deliberate misspellings of field labels. Real notes contain them, and a
#: detector that only matches the correct spelling of "Medical Record Number"
#: is measuring its own regex rather than the document.
MISSPELT_LABELS: Tuple[str, ...] = (
    "Medcial Record Numbr",
    "Medicial Recrd No",
    "Medical Recrod Numbr",
    "Medical Recod Numbe",
    "Medcal Recordd Num",
)

FILLER_SENTENCES: Tuple[str, ...] = (
    "The patient tolerated the procedure without complication. ",
    "Vital signs remained stable throughout the observation period. ",
    "No adverse reaction was recorded by nursing staff. ",
    "Fluid balance was maintained and oral intake resumed overnight. ",
    "The multidisciplinary team reviewed the plan at the morning round. ",
)


# ---------------------------------------------------------------------------
# Value construction
# ---------------------------------------------------------------------------


class _Values:
    """Draws invented values for one document, from one partition's stems."""

    def __init__(self, stems: StemPool, rng: random.Random, index: int):
        self.stems = stems
        self.rng = rng
        self.index = index

    def person(self) -> str:
        return f"{self.rng.choice(self.stems.given)} {self.rng.choice(self.stems.family)}"

    def unicode_person(self) -> str:
        return (
            f"{self.rng.choice(self.stems.unicode_given)} "
            f"{self.rng.choice(self.stems.unicode_family)}"
        )

    def clinician(self) -> str:
        return f"Dr. {self.person()}"

    def hospital(self) -> str:
        return f"{self.rng.choice(self.stems.org_head)} {self.stems.org_tail[2]}"

    def organization(self) -> str:
        return f"{self.rng.choice(self.stems.org_head)} {self.rng.choice(self.stems.org_tail)}"

    def address(self) -> str:
        number = self.rng.randrange(10, 999)
        street = self.rng.choice(self.stems.street)
        kind = self.rng.choice(("Lane", "Crescent", "Way", "Road"))
        return f"{number} {street} {kind}, Unit {self.rng.randrange(1, 40)}"

    def geography(self) -> str:
        return f"{self.rng.choice(self.stems.place)} District"

    def postal_code(self) -> str:
        return f"{self.stems.numeric_block}{self.rng.randrange(100, 999)}"

    def _encounter_year(self) -> int:
        return self.rng.randrange(*self.stems.encounter_years)

    def date_iso(self) -> str:
        return (
            f"{self._encounter_year()}-"
            f"{self.rng.randrange(1, 13):02d}-{self.rng.randrange(1, 29):02d}"
        )

    def date_us(self) -> str:
        return (
            f"{self.rng.randrange(1, 13):02d}/{self.rng.randrange(1, 29):02d}/"
            f"{self.rng.randrange(*self.stems.birth_years)}"
        )

    def date_long(self) -> str:
        month = self.rng.choice(
            ("January", "March", "April", "June", "September", "November")
        )
        return f"{self.rng.randrange(1, 29)} {month} {self._encounter_year()}"

    def phone(self) -> str:
        return f"555-0{self.stems.phone_block + self.rng.randrange(0, 20):03d}"

    def fax(self) -> str:
        return f"555-0{self.stems.phone_block + self.rng.randrange(0, 20):03d}"

    def phone_international(self) -> str:
        country = self.rng.choice(("+91", "+44", "+61", "+81", "+49"))
        return f"{country} {self.stems.numeric_block} 5550 {self.rng.randrange(1000, 9999)}"

    def email(self, person: str) -> str:
        local = person.lower().replace(" ", ".").replace("dr..", "")
        return f"{local}@example.invalid"

    def ssn(self) -> str:
        # The 900 range was never issued, so no real number can collide.
        return f"9{self.stems.ssn_group}-{self.rng.randrange(10, 99)}-{self.rng.randrange(1000, 9999)}"

    def mrn(self) -> str:
        return f"{self.stems.id_prefix}-{self.stems.numeric_block}{self.rng.randrange(10000, 99999)}"

    def patient_id(self) -> str:
        return f"{self.stems.id_prefix}-PT-{self.stems.numeric_block}{self.rng.randrange(10, 99)}"

    def health_plan(self) -> str:
        return f"{self.stems.id_prefix}-PLAN-{self.stems.numeric_block}{self.rng.randrange(1000, 9999)}"

    def account_number(self) -> str:
        return f"ACCT-{self.stems.numeric_block}{self.rng.randrange(100000, 999999)}"

    def license_number(self) -> str:
        letter = self.rng.choice("KRMPQZ")
        return f"DL-{letter}{self.stems.numeric_block}{self.rng.randrange(10000, 99999)}X"

    def certificate_number(self) -> str:
        role = self.rng.choice(("MB", "NP", "RN", "PT"))
        return f"CERT-{role}-{self.stems.numeric_block}{self.rng.randrange(100, 999)}"

    def device_id(self) -> str:
        return f"{self.stems.id_prefix}-DEV-{self.stems.numeric_block}{self.rng.randrange(1000, 9999)}"

    def vehicle_id(self) -> str:
        return f"VIN-{self.stems.numeric_block}{self.rng.randrange(10**13, 10**14)}"

    def accession(self) -> str:
        return f"{self.stems.id_prefix}-ACC-{self.stems.numeric_block}{self.rng.randrange(1000, 9999)}"

    def biometric_id(self) -> str:
        modality = self.rng.choice(("FP", "IRIS", "RETINA", "VOICE"))
        return f"{modality}-TEMPLATE-{self.stems.numeric_block}{self.rng.randrange(100, 999)}"

    def unusual_id(self) -> str:
        separator = self.rng.choice(("//", "--", "::", ";;"))
        return (
            f"{self.stems.id_prefix[-1]}{self.stems.numeric_block}{separator}"
            f"{self.rng.randrange(1000, 9999)}{separator}"
            f"{self.rng.choice(('QX', 'KX', 'LM', 'RR'))}"
        )

    def url(self) -> str:
        host = self.rng.choice(self.stems.org_head).lower().replace(" ", "-")
        return (
            f"https://{host}.example.invalid/chart/"
            f"{self.stems.numeric_block}{self.rng.randrange(1000, 9999)}"
        )

    def ip_address(self) -> str:
        octet = self.stems.ip_block + self.rng.randrange(0, 50)
        return f"{self.stems.ip_network}.{octet}"

    def username(self, person: str) -> str:
        first, _, last = person.partition(" ")
        return f"{first[0].lower()}.{last.lower()}{self.stems.numeric_block}"

    def age_over_89(self) -> str:
        return str(self.rng.choice(self.stems.ages_over_89))

    def age_under_90(self) -> str:
        return str(self.rng.choice(self.stems.ages_under_90))

    def misspell(self, value: str) -> str:
        """Transpose two letters, the way a typed note goes wrong."""
        letters = [c for i, c in enumerate(value) if c.isalpha()]
        if len(letters) < 4:
            return value
        chars = list(value)
        positions = [i for i, c in enumerate(chars) if c.isalpha()]
        cut = self.rng.choice(positions[1:-1])
        chars[cut], chars[cut - 1] = chars[cut - 1], chars[cut]
        return "".join(chars)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
#
# Each template appends prose and values to a builder. `b.phi` records a gold
# span, `b.keep` records a term that must survive. Offsets are recorded as the
# text is assembled, so they cannot drift out of step with the prose.


def _admission(b, v: _Values) -> str:
    patient = v.person()
    relative = v.person()
    clinician = v.clinician()
    b.t("Admission note\n\nPatient ").phi(patient, "PERSON", "indian_name")
    b.t(" was admitted on ").phi(v.date_iso(), "DATE", "date_iso")
    b.t(" accompanied by ").phi(relative, "PERSON_RELATIVE", "relative")
    b.t(", the patient's next of kin. Care was provided by ")
    b.phi(clinician, "PERSON_CLINICIAN", "clinician")
    b.t(" at ").phi(v.hospital(), "HOSPITAL", "facility")
    b.t(".\n").keep("Blood pressure", "vital_sign")
    b.t(" was 128/76 and the patient remained afebrile. ")
    b.t("A history of ").keep("Crohn", "eponym").t(" disease is recorded.")
    return "Admission with names, a facility, a date and clinical negatives."


def _discharge(b, v: _Values) -> str:
    patient = v.person()
    b.t("Discharge summary for ").phi(patient, "PERSON", "international_name")
    b.t("\nDischarged ").phi(v.date_long(), "DATE", "date_written")
    b.t(" to the home address at ").phi(v.address(), "ADDRESS", "address")
    b.t(", ").phi(v.geography(), "GEOGRAPHY", "geography")
    b.t(" ").phi(v.postal_code(), "POSTAL_CODE", "postal_code")
    b.t(".\nFollow-up on ").phi(v.phone(), "PHONE", "phone")
    b.t(". Continue ").keep("Metformin", "medication")
    b.t(" 500 mg twice daily and ").keep("Atorvastatin", "medication")
    b.t(" 40 mg nightly.")
    return "Discharge with address, postal code, telephone and medications."


def _laboratory(b, v: _Values) -> str:
    b.t("Laboratory report\nAccession ").phi(v.accession(), "ACCESSION", "accession")
    b.t("\nMedical Record Number ").phi(v.mrn(), "MRN", "record_number")
    b.t("\nCollected ").phi(v.date_iso(), "DATE", "date_iso")
    b.t("\n").keep("Creatinine", "analyte").t(" 84 umol/L; ")
    b.keep("Haemoglobin", "analyte").t(" 13.1 g/dL. Results reviewed.")
    return "Laboratory identifiers beside analyte names that must survive."


def _imaging(b, v: _Values) -> str:
    b.t("Imaging report\nPatient identifier ")
    b.phi(v.patient_id(), "PATIENT_ID", "patient_id")
    b.t("\nScanner ").phi(v.device_id(), "DEVICE_ID", "device")
    b.t("\nFindings: a small lesion in the ").keep("temporal lobe", "anatomy")
    b.t(" with no involvement of the ").keep("left atrium", "anatomy")
    b.t(". Reported by ").phi(v.clinician(), "PERSON_CLINICIAN", "clinician")
    b.t(".")
    return "Imaging identifiers beside anatomical terms."


def _referral(b, v: _Values) -> str:
    clinician = v.clinician()
    b.t("Referral letter\nFrom ").phi(clinician, "PERSON_CLINICIAN", "clinician")
    b.t(" at ").phi(v.organization(), "ORGANIZATION", "organization")
    b.t("\nReply to ").phi(v.email(clinician.replace("Dr. ", "")), "EMAIL", "email")
    b.t(" or via ").phi(v.url(), "URL", "url")
    b.t(".\nThe patient is employed by ").phi(v.organization(), "EMPLOYER", "employer")
    b.t(" and requires a workplace assessment.")
    return "Referral with organisation, employer, email and web address."


def _insurance(b, v: _Values) -> str:
    b.t("Claim record\nHealth plan ")
    b.phi(v.health_plan(), "HEALTH_PLAN", "health_plan")
    b.t("\nAccount ").phi(v.account_number(), "ACCOUNT_NUMBER", "account")
    b.t("\nSocial security ").phi(v.ssn(), "SSN", "ssn")
    b.t("\nClaim submitted ").phi(v.date_us(), "DATE", "date_us")
    b.t(" for outpatient physiotherapy.")
    return "Claim identifiers, including a never-issued SSN range."


def _medication_review(b, v: _Values) -> str:
    patient = v.person()
    b.t("Medication review for ").phi(patient, "PERSON", "indian_name")
    b.t("\n").keep("Levothyroxine", "medication").t(" 75 mcg each morning.\n")
    b.keep("Amoxicillin", "medication").t(" 500 mg three times daily for 7 days.\n")
    b.t("Temperature 37.2°C; trace element dose 250µg.\n")
    b.t("Reviewed by ").phi(v.clinician(), "PERSON_CLINICIAN", "clinician")
    b.t(" on ").phi(v.date_iso(), "DATE", "date_iso").t(".")
    return "Medications beside dosages, with non-ASCII units in the prose."


def _telemetry(b, v: _Values) -> str:
    patient = v.person()
    b.t("Device telemetry\nUploaded from ")
    b.phi(v.ip_address(), "IP_ADDRESS", "ip_end_of_sentence")
    b.t(".\nOperator account ").phi(v.username(patient), "USERNAME", "username")
    b.t("\nDevice ").phi(v.device_id(), "DEVICE_ID", "device")
    b.t(" reported no fault. Battery telemetry within range.")
    return "Network and account identifiers from a device upload."


def _consent(b, v: _Values) -> str:
    patient = v.person()
    b.t("Consent form\nSigned by ").phi(patient, "PERSON", "international_name")
    b.t(" on ").phi(v.date_long(), "DATE", "date_written")
    b.t("\nWitness licence ").phi(v.license_number(), "LICENSE_NUMBER", "license")
    b.t("\nPractitioner certificate ")
    b.phi(v.certificate_number(), "CERTIFICATE_NUMBER", "certificate")
    b.t("\nThe procedure and its risks were explained in full.")
    return "Consent with licence and certificate numbers."


def _transport(b, v: _Values) -> str:
    b.t("Transport record\nVehicle ").phi(v.vehicle_id(), "VEHICLE_ID", "vehicle")
    b.t("\nCollected from ").phi(v.address(), "ADDRESS", "address")
    b.t(", ").phi(v.geography(), "GEOGRAPHY", "geography")
    b.t("\nDelivered to ").phi(v.hospital(), "HOSPITAL", "facility")
    b.t(". Journey uneventful.")
    return "Transport with a vehicle identifier and two places."


def _biometric(b, v: _Values) -> str:
    b.t("Biometric enrolment\nTemplate ")
    b.phi(v.biometric_id(), "BIOMETRIC_ID", "biometric")
    b.t("\nLocal reference ").phi(v.unusual_id(), "UNUSUAL_ID", "unusual_format")
    b.t("\nFax confirmation to ").phi(v.fax(), "FAX", "fax")
    b.t("\nEnrolment completed without difficulty.")
    return "A biometric template beside an identifier in no standard format."


def _age_boundary(b, v: _Values) -> str:
    b.t("Geriatric review\nThe patient is ")
    b.phi(v.age_over_89(), "AGE_OVER_89", "age_over_89")
    b.t(" years old, above the Safe Harbor aggregation threshold.\n")
    b.t("A sibling aged ").keep(v.age_under_90(), "age_under_90")
    b.t(" years is not an identifier at that age.\n")
    b.t("Mobility is unchanged and cognition is intact.")
    return "Ages either side of the ninety-year boundary."


def _misspelt(b, v: _Values) -> str:
    patient = v.person()
    b.t("Ward note\n").t(MISSPELT_LABELS[v.index % len(MISSPELT_LABELS)])
    b.t(": ").phi(v.mrn(), "MRN", "misspelt_label")
    b.t("\nPateint name ").phi(patient, "PERSON", "misspelt_label")
    b.t("\nSeen by ").phi(v.misspell(v.clinician()), "PERSON_CLINICIAN", "misspelt_value")
    b.t("\nPt afebrile, obs stable, r/v in a.m. per usual protocol.")
    return "Misspelt field labels and a misspelt clinician name."


def _unicode(b, v: _Values) -> str:
    unicode_person = v.unicode_person()
    b.t("Referral received for ").phi(unicode_person, "PERSON", "unicode")
    b.t(" (preferred spelling retained).\nSecond opinion from ")
    b.phi(v.unicode_person(), "PERSON", "unicode", "transliteration")
    b.t("\nContact ").phi(v.email(unicode_person), "EMAIL", "unicode")
    b.t("\nCASE ESCALATED BY ").phi(v.person().upper(), "PERSON", "mixed_case", "uppercase")
    b.t(" and reviewed by ").phi(v.person().lower(), "PERSON", "mixed_case", "lowercase")
    b.t(".")
    return "Unicode names, transliteration and mixed casing."


def _repeated(b, v: _Values) -> str:
    patient = v.person()
    mrn = v.mrn()
    b.t("Progress notes\nDay 1: ").phi(patient, "PERSON", "repeated_entity")
    b.t(" reviewed on the ward. Record ").phi(mrn, "MRN", "repeated_entity")
    b.t(".\nDay 2: ").phi(patient, "PERSON", "repeated_entity")
    b.t(" remained stable. Record ").phi(mrn, "MRN", "repeated_entity")
    b.t(".\nDay 3: ").phi(patient, "PERSON", "repeated_entity")
    b.t(" was discharged. ").keep("Blood pressure", "vital_sign")
    b.t(" normal throughout.")
    return "One patient and one record number repeated across three days."


def _chunk_boundary(b, v: _Values) -> str:
    lead = "".join(FILLER_SENTENCES) * 8
    b.t(lead[: 2000 - 8]).t("Name: ")
    b.phi(v.person(), "PERSON", "chunk_boundary")
    b.t(" MRN ").phi(v.mrn(), "MRN", "chunk_boundary")
    b.t(". ").t("".join(FILLER_SENTENCES))
    b.t("Second-window contact ").phi(v.email(v.person()), "EMAIL", "second_window")
    b.t(".")
    return "A name and record number straddling the 2000-character window."


def _abbreviations(b, v: _Values) -> str:
    b.t("A&E note\nPt seen in ED at 03:40. Hx of ")
    b.keep("Addison", "eponym").t(" disease and ")
    b.keep("Graves", "eponym").t(" disease.\n")
    b.t("MRN ").phi(v.mrn(), "MRN", "abbreviation_context")
    b.t("; DOB ").phi(v.date_us(), "DATE", "abbreviation_context")
    b.t("\nObs: BP 118/74, HR 82, SpO2 97% RA. D/C home with GP f/u.")
    return "Heavy abbreviation with identifiers in a compressed layout."


def _overlapping(b, v: _Values) -> str:
    patient = v.person()
    b.t("Encounter header\n").phi(patient, "PERSON", "overlapping")
    b.t(" (").phi(v.mrn(), "MRN", "overlapping")
    b.t("), seen ").phi(v.date_iso(), "DATE", "overlapping")
    b.t(" at ").phi(v.hospital(), "HOSPITAL", "overlapping")
    b.t(".\nMultiline continuation:\n  contact ")
    b.phi(v.phone_international(), "PHONE", "phone_international", "multiline")
    b.t("\n  alternate ").phi(v.email(patient), "EMAIL", "multiline")
    return "Identifiers packed adjacently and continued across lines."


def _no_phi_eponyms(b, v: _Values) -> str:
    eponym, condition = EPONYM_CONDITIONS[v.index % len(EPONYM_CONDITIONS)]
    b.t("Clinical impression\n").keep(eponym, "eponym")
    b.t(f"'s {condition} remains the working diagnosis. ")
    b.t("Response to treatment has been satisfactory.\n")
    b.keep("Blood pressure", "vital_sign").t(" and pulse are within range. ")
    b.keep("Creatinine", "analyte").t(" is stable at 79 umol/L.\n")
    b.t("The ").keep("iliac crest", "anatomy").t(" graft site has healed.")
    return "No identifiers at all: only clinical language that resembles one."


def _no_phi_plan(b, v: _Values) -> str:
    b.t("Management plan\nContinue current therapy and review in six weeks. ")
    b.t("Encourage graded exercise and a reduced-salt diet.\n")
    b.t("Repeat ").keep("Haemoglobin", "analyte").t(" and renal function before ")
    b.t("the next appointment. Escalate if symptoms recur.\n")
    b.t("No further imaging is indicated at this stage.")
    return "A plan containing no identifiers of any kind."


def _long_narrative(b, v: _Values) -> str:
    patient = v.person()
    b.t("Extended narrative\n").phi(patient, "PERSON", "long_text")
    b.t(" was followed for twelve months. ")
    b.t("".join(FILLER_SENTENCES) * 4)
    b.t("Medication was adjusted; ").keep("Levothyroxine", "medication")
    b.t(" was increased. ").t("".join(FILLER_SENTENCES) * 4)
    b.t("The final review was on ").phi(v.date_long(), "DATE", "long_text")
    b.t(" at ").phi(v.hospital(), "HOSPITAL", "long_text")
    b.t(". ").t("".join(FILLER_SENTENCES) * 3)
    b.t("Contact ").phi(v.phone(), "PHONE", "long_text").t(" for queries.")
    return "A long note with identifiers spread across its whole length."


TEMPLATES: Tuple[Callable, ...] = (
    _admission,
    _discharge,
    _laboratory,
    _imaging,
    _referral,
    _insurance,
    _medication_review,
    _telemetry,
    _consent,
    _transport,
    _biometric,
    _age_boundary,
    _misspelt,
    _unicode,
    _repeated,
    _chunk_boundary,
    _abbreviations,
    _overlapping,
    _no_phi_eponyms,
    _no_phi_plan,
    _long_narrative,
)


def generate(
    partition: str,
    builder_factory: Callable[[str, str], object],
    count: int,
    first_index: int = 0,
) -> List[object]:
    """Build `count` documents for `partition`, deterministically.

    The seed is the partition name alone, so a document's content depends on
    nothing but where it lives. Rebuilding the corpus on another machine, or
    after any unrelated edit to this file's prose, produces the same values.
    """
    if partition not in STEMS:
        raise ValueError(f"no stem pool for partition: {partition}")
    stems = STEMS[partition]

    documents: List[object] = []
    for offset in range(count):
        index = first_index + offset
        rng = random.Random(f"{partition}:{index}")
        values = _Values(stems, rng, index)
        template = TEMPLATES[index % len(TEMPLATES)]
        builder = builder_factory(f"{partition}_gen_{index:04d}", partition)
        notes = template(builder, values)
        documents.append(builder.done(notes, "generated"))
    return documents
