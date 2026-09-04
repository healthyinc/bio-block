"""Synthetic canary corpus for PHI-hardening evaluation.

Every value here is invented. Names are obviously fictional, the email domain
is `.invalid` (reserved by RFC 2606 and unresolvable), phone numbers use the
555-01xx range reserved for fiction, and identifiers carry a `SYN-` prefix. No
real patient information appears in this file or anywhere in the evaluation
path.

A canary is a value we expect the pipeline to find and remove. The harness
measures two separate things:

* **Detection** - did the detector propose the canary's category at all?
* **Residual** - did the canary's literal value survive into anything the
  pipeline was willing to release, index, or preview?

Residual is the one that matters. Zero residual canaries is an acceptance
condition for this suite, not proof that a deployment has zero PHI leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Canary values
# --------------------------------------------------------------------------

CANARY_PERSON = "Jordan Fictional"
CANARY_PERSON_ALT = "Marisol Nonexistent"
CANARY_CLINICIAN = "Dr. Wendell Imaginary"
CANARY_EMAIL = "jordan.fictional@example.invalid"
CANARY_PHONE = "555-0100"
CANARY_PHONE_FULL = "555-321-7654"
CANARY_SSN = "123-45-6789"
CANARY_MRN = "SYN-4820193"
CANARY_PATIENT_ID = "SYN-PT-0099"
CANARY_ACCESSION = "SYN-ACC-771201"
CANARY_HEALTH_PLAN = "SYN-PLAN-556677"
CANARY_DEVICE = "SYN-DEV-330091"
CANARY_DATE = "2019-04-02"
CANARY_URL = "https://records.example.invalid/patient/9931"
CANARY_IP = "203.0.113.42"

#: Every canary the corpus can contain, for residual scanning.
ALL_CANARIES: Tuple[str, ...] = (
    CANARY_PERSON,
    CANARY_PERSON_ALT,
    CANARY_CLINICIAN,
    CANARY_EMAIL,
    CANARY_PHONE,
    CANARY_PHONE_FULL,
    CANARY_SSN,
    CANARY_MRN,
    CANARY_PATIENT_ID,
    CANARY_ACCESSION,
    CANARY_HEALTH_PLAN,
    CANARY_DEVICE,
    CANARY_DATE,
    CANARY_URL,
    CANARY_IP,
)


@dataclass(frozen=True)
class CanaryCase:
    """One evaluation input plus the categories it is expected to surface."""

    case_id: str
    modality: str
    #: Built lazily so a missing optional dependency skips one case, not all.
    build: Callable[[], bytes]
    filename: str
    content_type: Optional[str]
    expected_categories: Tuple[str, ...]
    #: Canary values present in this case, used for residual scanning.
    canaries: Tuple[str, ...]
    #: What the release decision must be. None means "not asserted here".
    expect_releasable: Optional[bool] = False
    notes: str = ""


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

CLINICAL_NOTE = (
    f"Patient {CANARY_PERSON} (MRN: {CANARY_MRN}) was seen by {CANARY_CLINICIAN} "
    f"on {CANARY_DATE}.\n"
    f"Patient ID {CANARY_PATIENT_ID}. Accession Number {CANARY_ACCESSION}.\n"
    f"Health Plan ID {CANARY_HEALTH_PLAN}. Device ID {CANARY_DEVICE}.\n"
    f"Contact {CANARY_EMAIL} or {CANARY_PHONE_FULL}. SSN {CANARY_SSN}.\n"
    f"Portal {CANARY_URL} accessed from {CANARY_IP}.\n"
    f"Follow-up with {CANARY_PERSON_ALT} arranged."
)

NOTE_CANARIES: Tuple[str, ...] = (
    CANARY_PERSON,
    CANARY_PERSON_ALT,
    CANARY_MRN,
    CANARY_PATIENT_ID,
    CANARY_ACCESSION,
    CANARY_HEALTH_PLAN,
    CANARY_DEVICE,
    CANARY_EMAIL,
    CANARY_PHONE_FULL,
    CANARY_SSN,
    CANARY_DATE,
    CANARY_URL,
    CANARY_IP,
)


def _build_text() -> bytes:
    return CLINICAL_NOTE.encode("utf-8")


def _build_long_text() -> bytes:
    """A note long enough to force overlapping-chunk inference."""
    filler = "The patient tolerated the procedure well. " * 200
    return (filler + CLINICAL_NOTE + filler).encode("utf-8")


def _build_csv() -> bytes:
    return (
        "name,email,phone,mrn,age,gender,diagnosis\n"
        f"{CANARY_PERSON},{CANARY_EMAIL},{CANARY_PHONE_FULL},{CANARY_MRN},31,F,flu\n"
        f"{CANARY_PERSON_ALT},m.n@example.invalid,555-0101,SYN-4820194,32,F,cold\n"
        "Avery Imagined,a.i@example.invalid,555-0102,SYN-4820195,33,M,flu\n"
        "Rowan Invented,r.i@example.invalid,555-0103,SYN-4820196,34,M,cold\n"
    ).encode("utf-8")


def _build_pdf() -> bytes:
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 720), f"Patient: {CANARY_PERSON}", fontsize=11)
    page.insert_text((72, 700), f"MRN: {CANARY_MRN}", fontsize=11)
    page.insert_text((72, 680), f"SSN: {CANARY_SSN}", fontsize=11)
    # Metadata is a surface the old text-only path never looked at.
    document.set_metadata({"title": f"Chart for {CANARY_PERSON}", "author": CANARY_CLINICIAN})
    payload = document.tobytes()
    document.close()
    return payload


def _build_pdf_image_only() -> bytes:
    """A page with raster content and no text layer: must not read as clean."""
    import fitz

    document = fitz.open()
    page = document.new_page()
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 16))
    pixmap.clear_with(180)
    page.insert_image(fitz.Rect(100, 100, 200, 200), pixmap=pixmap)
    payload = document.tobytes()
    document.close()
    return payload


def _build_workbook() -> bytes:
    import openpyxl
    from openpyxl.comments import Comment

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Cohort"
    sheet.append(["name", "email", "mrn"])
    sheet.append([CANARY_PERSON, CANARY_EMAIL, CANARY_MRN])
    sheet["A1"].comment = Comment(f"verified by {CANARY_CLINICIAN}", "reviewer")
    # A hidden sheet is a classic place for content a user does not see.
    hidden = workbook.create_sheet("Archive")
    hidden.append(["legacy", CANARY_PATIENT_ID])
    hidden.sheet_state = "hidden"
    workbook.properties.creator = CANARY_PERSON_ALT
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _build_dicom() -> bytes:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

    pixels = np.full((16, 16), 90, dtype=np.uint8)
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.8.498.51"
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.52"

    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = CANARY_PERSON
    dataset.PatientID = CANARY_MRN
    dataset.AccessionNumber = CANARY_ACCESSION
    dataset.ReferringPhysicianName = CANARY_CLINICIAN
    dataset.StudyDate = "20190402"
    dataset.Rows = 16
    dataset.Columns = 16
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixels.tobytes()

    buffer = BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def _build_raster_with_metadata() -> bytes:
    from PIL import Image, PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("Patient", CANARY_PERSON)
    info.add_text("Contact", CANARY_EMAIL)
    buffer = BytesIO()
    Image.new("RGB", (48, 48), (200, 200, 200)).save(
        buffer, format="PNG", pnginfo=info
    )
    return buffer.getvalue()


# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------

CORPUS: Tuple[CanaryCase, ...] = (
    CanaryCase(
        case_id="text_clinical_note",
        modality="text",
        build=_build_text,
        filename="note.txt",
        content_type="text/plain",
        expected_categories=(
            "PERSON",
            "MEDICAL_RECORD_NUMBER",
            "PATIENT_ID",
            "ACCESSION_NUMBER",
            "HEALTH_PLAN_ID",
            "DEVICE_ID",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "US_SSN",
            "DATE_TIME",
            "URL",
            "IP_ADDRESS",
        ),
        canaries=NOTE_CANARIES,
        expect_releasable=True,
        notes="The only modality that can release automatically.",
    ),
    CanaryCase(
        case_id="text_long_note_chunked",
        modality="text",
        build=_build_long_text,
        filename="long-note.txt",
        content_type="text/plain",
        expected_categories=("PERSON", "US_SSN", "EMAIL_ADDRESS"),
        canaries=NOTE_CANARIES,
        expect_releasable=True,
        notes="Long enough to exercise overlapping-chunk inference.",
    ),
    CanaryCase(
        case_id="csv_cohort",
        modality="csv",
        build=_build_csv,
        filename="cohort.csv",
        content_type="text/csv",
        expected_categories=(),
        canaries=(
            CANARY_PERSON,
            CANARY_PERSON_ALT,
            CANARY_EMAIL,
            CANARY_PHONE_FULL,
            CANARY_MRN,
        ),
        expect_releasable=False,
        notes="Column removal plus k-anonymity; release posture under review.",
    ),
    CanaryCase(
        case_id="pdf_text_and_metadata",
        modality="pdf",
        build=_build_pdf,
        filename="chart.pdf",
        content_type="application/pdf",
        expected_categories=("PERSON", "US_SSN"),
        canaries=(CANARY_PERSON, CANARY_MRN, CANARY_SSN, CANARY_CLINICIAN),
        expect_releasable=False,
        notes="Document metadata is a surface the text-only path ignored.",
    ),
    CanaryCase(
        case_id="pdf_image_only_page",
        modality="pdf",
        build=_build_pdf_image_only,
        filename="scanned.pdf",
        content_type="application/pdf",
        expected_categories=(),
        canaries=(),
        expect_releasable=False,
        notes="Zero entities here must not read as a clean document.",
    ),
    CanaryCase(
        case_id="workbook_cohort",
        modality="workbook",
        build=_build_workbook,
        filename="cohort.xlsx",
        content_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        expected_categories=("PERSON", "EMAIL_ADDRESS"),
        canaries=(
            CANARY_PERSON,
            CANARY_PERSON_ALT,
            CANARY_EMAIL,
            CANARY_MRN,
            CANARY_PATIENT_ID,
            CANARY_CLINICIAN,
        ),
        expect_releasable=False,
        notes="Cells, comments, hidden sheet, and document properties.",
    ),
    CanaryCase(
        case_id="dicom_metadata_and_pixels",
        modality="dicom",
        build=_build_dicom,
        filename="scan.dcm",
        content_type="application/dicom",
        expected_categories=(),
        canaries=(
            CANARY_PERSON,
            CANARY_MRN,
            CANARY_ACCESSION,
            CANARY_CLINICIAN,
        ),
        expect_releasable=False,
        notes="Standing facial-reconstruction blocker keeps this blocked.",
    ),
)

#: Cases used only for the preview/index gates, not for ingest routing.
GATE_CORPUS: Tuple[CanaryCase, ...] = (
    CanaryCase(
        case_id="raster_with_metadata",
        modality="raster",
        build=_build_raster_with_metadata,
        filename="scan.png",
        content_type="image/png",
        expected_categories=(),
        canaries=(CANARY_PERSON, CANARY_EMAIL),
        expect_releasable=True,
        notes="Input PNG text chunks must not survive into the preview.",
    ),
)


def residual_canaries(haystack: str, canaries: Sequence[str]) -> List[str]:
    """Return which canaries survived. Callers report names, never values."""
    return [canary for canary in canaries if canary and canary in haystack]


def canary_labels(values: Sequence[str]) -> List[str]:
    """Stable, non-revealing labels for a set of canaries.

    A canary value is synthetic, but reports still name it by index rather than
    by value so no report format ever normalizes carrying a matched string.
    """
    index = {value: position for position, value in enumerate(ALL_CANARIES)}
    return sorted(f"canary_{index.get(value, -1):02d}" for value in values)
