import os
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any, Dict

from services.privacy_profiles import (
    PrivacyProfileError,
    get_privacy_profile,
    validate_privacy_profile,
)

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    pydicom = None
    InvalidDicomError = Exception


PIXEL_REDACTION_STATUS = "metadata_only"
TEXT_REDACTION_VALUE = "REDACTED"
PERSON_NAME_REDACTION_VALUE = "ANONYMIZED"
DATE_REDACTION_VALUE = "19000101"
TIME_REDACTION_VALUE = "000000"
AGE_REDACTION_VALUE = "000Y"
SEX_REDACTION_VALUE = "O"
STUDY_SALT_ENV_VAR = "BIOBLOCK_STUDY_SALT"

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
TECHNICAL_METADATA_KEYWORDS = {
    "Modality",
    "Rows",
    "Columns",
    "BitsAllocated",
    "BitsStored",
    "PixelSpacing",
    "SliceThickness",
    "ManufacturerModelName",
    "KVP",
    "RepetitionTime",
    "EchoTime",
}


class DicomAnonymizationError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _profile_settings(profile: str) -> tuple[str, Dict[str, Any]]:
    try:
        normalized = validate_privacy_profile(profile)
        return normalized, get_privacy_profile(normalized)
    except PrivacyProfileError as exc:
        raise DicomAnonymizationError(exc.detail, status_code=exc.status_code) from exc


def _date_shift_days(dataset: Any) -> int:
    seed = (
        os.getenv(STUDY_SALT_ENV_VAR)
        or str(getattr(dataset, "StudyInstanceUID", ""))
        or str(getattr(dataset, "SOPInstanceUID", ""))
        or "bioblock-date-shift"
    )
    value = sum((index + 1) * ord(char) for index, char in enumerate(seed))
    days = value % 731 - 365
    return days or 17


def _shift_dicom_date(value: Any, days: int) -> str:
    try:
        shifted = datetime.strptime(str(value), "%Y%m%d") + timedelta(days=days)
    except (TypeError, ValueError):
        return DATE_REDACTION_VALUE
    return shifted.strftime("%Y%m%d")


def _generalize_age(value: Any) -> str:
    match = str(value or "").strip()
    if len(match) < 4 or not match[:3].isdigit():
        return AGE_REDACTION_VALUE

    years = int(match[:3])
    decade = min((years // 10) * 10, 90)
    return f"{decade:03d}Y"


def _safe_patient_sex(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in {"M", "F", "O"}:
        return normalized
    return SEX_REDACTION_VALUE


def _redaction_value_for(element: Any, settings: Dict[str, Any], date_shift_days: int) -> tuple[str, str]:
    keyword = element.keyword

    if element.VR == "PN":
        return PERSON_NAME_REDACTION_VALUE, "redacted"
    if element.VR == "DA" or keyword.endswith("Date"):
        if settings["date_strategy"] == "shift":
            return _shift_dicom_date(element.value, date_shift_days), "shifted"
        return DATE_REDACTION_VALUE, "redacted"
    if element.VR == "TM" or keyword.endswith("Time"):
        return TIME_REDACTION_VALUE, "redacted"
    if element.VR == "AS" or keyword.endswith("Age"):
        if settings["allow_generalized_demographics"]:
            return _generalize_age(element.value), "generalized"
        return AGE_REDACTION_VALUE, "redacted"
    if keyword == "PatientSex":
        if settings["allow_generalized_demographics"]:
            return _safe_patient_sex(element.value), "generalized"
        return SEX_REDACTION_VALUE, "redacted"

    return TEXT_REDACTION_VALUE, "redacted"


def _scrub_dataset(dataset: Any, settings: Dict[str, Any], date_shift_days: int) -> Dict[str, Any]:
    scrubbed_elements = 0
    shifted_dates = 0
    generalized_demographics = 0
    field_counts: Dict[str, int] = {}

    for element in dataset:
        if element.VR == "SQ":
            for item in element.value:
                nested = _scrub_dataset(item, settings, date_shift_days)
                scrubbed_elements += nested["scrubbed_elements"]
                shifted_dates += nested["dates_shifted"]
                generalized_demographics += nested["generalized_demographics"]
                for keyword, count in nested["scrubbed_field_counts"].items():
                    field_counts[keyword] = field_counts.get(keyword, 0) + count
            continue

        keyword = element.keyword
        if keyword in PHI_KEYWORDS:
            replacement, action = _redaction_value_for(
                element,
                settings,
                date_shift_days,
            )
            element.value = replacement
            scrubbed_elements += 1
            field_counts[keyword] = field_counts.get(keyword, 0) + 1
            if action == "shifted":
                shifted_dates += 1
            elif action == "generalized":
                generalized_demographics += 1

    return {
        "scrubbed_elements": scrubbed_elements,
        "scrubbed_field_counts": field_counts,
        "dates_shifted": shifted_dates,
        "generalized_demographics": generalized_demographics,
    }


def _mark_patient_identity_removed(dataset: Any, profile: str, settings: Dict[str, Any]) -> None:
    dataset.PatientIdentityRemoved = "YES"
    private_policy = "rm-private" if settings["remove_dicom_private_tags"] else "keep-private"
    dataset.DeidentificationMethod = (
        f"BioBlock {profile}; dates={settings['date_strategy']}; {private_policy}"
    )


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



def _technical_metadata_present(dataset: Any) -> list[str]:
    return sorted(
        keyword
        for keyword in TECHNICAL_METADATA_KEYWORDS
        if hasattr(dataset, keyword)
    )


def _anonymize_dataset(dataset: Any, profile: str) -> Dict[str, Any]:
    privacy_profile, settings = _profile_settings(profile)
    original_pixel_data = bytes(dataset.PixelData) if "PixelData" in dataset else None
    technical_metadata_preserved = _technical_metadata_present(dataset)

    scrubbed = _scrub_dataset(dataset, settings, _date_shift_days(dataset))
    private_tags_removed = 0
    if settings["remove_dicom_private_tags"]:
        private_tags_removed = _remove_private_tags(dataset)

    current_pixel_data = bytes(dataset.PixelData) if "PixelData" in dataset else None
    _mark_patient_identity_removed(dataset, privacy_profile, settings)

    return {
        "profile": privacy_profile,
        "settings": settings,
        "scrubbed": scrubbed,
        "private_tags_removed": private_tags_removed,
        "pixel_data_present": original_pixel_data is not None,
        "pixel_data_preserved": original_pixel_data == current_pixel_data,
        "technical_metadata_preserved": technical_metadata_preserved
        if settings["preserve_dicom_technical_metadata"]
        else [],
    }


def _read_dataset(file_bytes: bytes) -> Any:
    try:
        return pydicom.dcmread(BytesIO(file_bytes), force=False)
    except InvalidDicomError as exc:
        raise DicomAnonymizationError("Invalid DICOM file format") from exc
    except Exception as exc:
        raise DicomAnonymizationError("Invalid DICOM file format") from exc


def _metadata_summary(anonymized: Dict[str, Any]) -> Dict[str, Any]:
    scrubbed = anonymized["scrubbed"]
    settings = anonymized["settings"]
    return {
        "profile": anonymized["profile"],
        "date_strategy": settings["date_strategy"],
        "fields_scrubbed": scrubbed["scrubbed_elements"],
        "scrubbed_field_counts": scrubbed["scrubbed_field_counts"],
        "dates_shifted": scrubbed["dates_shifted"],
        "generalized_demographics": scrubbed["generalized_demographics"],
        "private_tags_removed": anonymized["private_tags_removed"],
        "remove_dicom_private_tags": settings["remove_dicom_private_tags"],
        "preserve_dicom_technical_metadata": settings[
            "preserve_dicom_technical_metadata"
        ],
        "technical_metadata_preserved": anonymized["technical_metadata_preserved"],
        "pixel_data_present": anonymized["pixel_data_present"],
        "pixel_data_preserved": anonymized["pixel_data_preserved"],
    }


def _dataset_to_bytes(dataset: Any) -> bytes:
    output = BytesIO()
    dataset.save_as(output, enforce_file_format=True)
    return output.getvalue()


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

    dataset = _read_dataset(file_bytes)
    anonymized = _anonymize_dataset(dataset, profile)

    return {
        "anonymization_status": "completed",
        "metadata_summary": _metadata_summary(anonymized),
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

    dataset = _read_dataset(file_bytes)
    anonymized = _anonymize_dataset(dataset, profile)

    return {
        "anonymization_status": "completed",
        "anonymized_dicom_bytes": _dataset_to_bytes(dataset),
        "metadata_summary": _metadata_summary(anonymized),
        "pixel_redaction_status": PIXEL_REDACTION_STATUS,
    }


