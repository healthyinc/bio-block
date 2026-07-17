from io import BytesIO
from typing import Any, Dict

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    pydicom = None
    InvalidDicomError = Exception


SUPPORTED_PROFILES = {"strict", "research"}
PIXEL_REDACTION_STATUS = "not_started_week4"
TEXT_REDACTION_VALUE = "REDACTED"
PERSON_NAME_REDACTION_VALUE = "ANONYMIZED"
DATE_REDACTION_VALUE = "19000101"
TIME_REDACTION_VALUE = "000000"
AGE_REDACTION_VALUE = "000Y"
SEX_REDACTION_VALUE = "O"

PHI_KEYWORDS = {
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientSex",
    "PatientAge",
    "OtherPatientIDs",
    "OtherPatientNames",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientMotherBirthName",
    "AccessionNumber",
    "StudyID",
    "StudyDate",
    "StudyTime",
    "SeriesDate",
    "SeriesTime",
    "AcquisitionDate",
    "AcquisitionTime",
    "ContentDate",
    "ContentTime",
    "InstanceCreationDate",
    "InstanceCreationTime",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "PhysiciansOfRecord",
    "InsurancePlanIdentification",
}


class DicomAnonymizationError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _normalize_profile(profile: str) -> str:
    normalized = (profile or "").strip().lower()
    if normalized not in SUPPORTED_PROFILES:
        raise DicomAnonymizationError(
            "Invalid privacy profile. Supported profiles: strict, research"
        )
    return normalized


def _redaction_value_for(element: Any) -> str:
    keyword = element.keyword

    if element.VR == "PN":
        return PERSON_NAME_REDACTION_VALUE
    if element.VR == "DA" or keyword.endswith("Date"):
        return DATE_REDACTION_VALUE
    if element.VR == "TM" or keyword.endswith("Time"):
        return TIME_REDACTION_VALUE
    if element.VR == "AS" or keyword.endswith("Age"):
        return AGE_REDACTION_VALUE
    if keyword == "PatientSex":
        return SEX_REDACTION_VALUE

    return TEXT_REDACTION_VALUE


def _scrub_dataset(dataset: Any) -> Dict[str, Any]:
    scrubbed_elements = 0
    field_counts: Dict[str, int] = {}

    for element in dataset:
        if element.VR == "SQ":
            for item in element.value:
                nested = _scrub_dataset(item)
                scrubbed_elements += nested["scrubbed_elements"]
                for keyword, count in nested["scrubbed_field_counts"].items():
                    field_counts[keyword] = field_counts.get(keyword, 0) + count
            continue

        keyword = element.keyword
        if keyword in PHI_KEYWORDS:
            element.value = _redaction_value_for(element)
            scrubbed_elements += 1
            field_counts[keyword] = field_counts.get(keyword, 0) + 1

    return {
        "scrubbed_elements": scrubbed_elements,
        "scrubbed_field_counts": field_counts,
    }


def _mark_patient_identity_removed(dataset: Any) -> None:
    dataset.PatientIdentityRemoved = "YES"
    dataset.DeidentificationMethod = "BioBlock metadata anonymization tokens"


def _remove_private_tags(dataset: Any) -> int:
    removed = 0

    for element in list(dataset):
        if element.tag.is_private:
            del dataset[element.tag]
            removed += 1
            continue

        if element.VR == "SQ":
            for item in element.value:
                removed += _remove_private_tags(item)

    return removed


def anonymize_dicom_metadata(
    file_bytes: bytes,
    profile: str = "strict",
) -> Dict[str, Any]:
    if pydicom is None:
        raise DicomAnonymizationError(
            "DICOM metadata anonymization dependency is not available.",
            status_code=503,
        )
    if not file_bytes:
        raise DicomAnonymizationError("DICOM file is empty")

    privacy_profile = _normalize_profile(profile)

    try:
        dataset = pydicom.dcmread(BytesIO(file_bytes), force=False)
    except InvalidDicomError as exc:
        raise DicomAnonymizationError("Invalid DICOM file format") from exc
    except Exception as exc:
        raise DicomAnonymizationError("Invalid DICOM file format") from exc

    original_pixel_data = bytes(dataset.PixelData) if "PixelData" in dataset else None

    scrubbed = _scrub_dataset(dataset)
    private_tags_removed = 0
    if privacy_profile == "strict":
        private_tags_removed = _remove_private_tags(dataset)

    current_pixel_data = bytes(dataset.PixelData) if "PixelData" in dataset else None

    return {
        "anonymization_status": "completed",
        "metadata_summary": {
            "profile": privacy_profile,
            "fields_scrubbed": scrubbed["scrubbed_elements"],
            "scrubbed_field_counts": scrubbed["scrubbed_field_counts"],
            "private_tags_removed": private_tags_removed,
            "pixel_data_present": original_pixel_data is not None,
            "pixel_data_preserved": original_pixel_data == current_pixel_data,
        },
        "pixel_redaction_status": PIXEL_REDACTION_STATUS,
    }

def anonymize_dicom_file_bytes(
    file_bytes: bytes,
    profile: str = "strict",
) -> Dict[str, Any]:
    if pydicom is None:
        raise DicomAnonymizationError(
            "DICOM metadata anonymization dependency is not available.",
            status_code=503,
        )
    if not file_bytes:
        raise DicomAnonymizationError("DICOM file is empty")

    privacy_profile = _normalize_profile(profile)

    try:
        dataset = pydicom.dcmread(BytesIO(file_bytes), force=False)
    except InvalidDicomError as exc:
        raise DicomAnonymizationError("Invalid DICOM file format") from exc
    except Exception as exc:
        raise DicomAnonymizationError("Invalid DICOM file format") from exc

    original_pixel_data = bytes(dataset.PixelData) if "PixelData" in dataset else None

    scrubbed = _scrub_dataset(dataset)
    private_tags_removed = 0
    if privacy_profile == "strict":
        private_tags_removed = _remove_private_tags(dataset)

    current_pixel_data = bytes(dataset.PixelData) if "PixelData" in dataset else None
    _mark_patient_identity_removed(dataset)

    output = BytesIO()
    dataset.save_as(output, enforce_file_format=True)

    return {
        "anonymization_status": "completed",
        "anonymized_dicom_bytes": output.getvalue(),
        "metadata_summary": {
            "profile": privacy_profile,
            "fields_scrubbed": scrubbed["scrubbed_elements"],
            "scrubbed_field_counts": scrubbed["scrubbed_field_counts"],
            "private_tags_removed": private_tags_removed,
            "pixel_data_present": original_pixel_data is not None,
            "pixel_data_preserved": original_pixel_data == current_pixel_data,
        },
        "pixel_redaction_status": PIXEL_REDACTION_STATUS,
    }
