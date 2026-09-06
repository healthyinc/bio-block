from typing import Any, Callable, Dict, Optional

from services.text_anonymization import (
    TextAnonymizationError,
    anonymize_clinical_text,
)
from services.dicom_anonymization import (
    DicomAnonymizationError,
    anonymize_dicom_file_bytes,
)
from services.nifti_anonymization import (
    NiftiAnonymizationError,
    anonymize_nifti_metadata,
)
from services.ocr_redaction import redact_dicom_pixels, safe_ocr_response
from services.wsi_tiling import scan_wsi_bytes
from services.privacy_profiles import (
    PrivacyProfileError,
    validate_privacy_profile as validate_config_privacy_profile,
)

SUPPORTED_PROFILES = {"strict", "research"}
SUPPORTED_MODALITIES = {"csv", "text", "dicom", "nifti", "wsi"}
HEADER_READ_LIMIT = 4096
TEXT_READ_LIMIT_BYTES = 256 * 1024


class IngestionError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def validate_privacy_profile(profile: str) -> str:
    try:
        return validate_config_privacy_profile(profile)
    except PrivacyProfileError as exc:
        raise IngestionError(exc.detail, status_code=exc.status_code) from exc


def _safe_filename(filename: str) -> str:
    cleaned = (filename or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    if not cleaned:
        raise IngestionError("Uploaded file must include a filename")
    return cleaned


def _extension(filename: str) -> str:
    lower_name = filename.lower()
    if lower_name.endswith(".nii.gz"):
        return ".nii.gz"
    if "." not in lower_name:
        return ""
    return f".{lower_name.rsplit('.', 1)[-1]}"


def _has_dicom_preamble(header: bytes) -> bool:
    return len(header) >= 132 and header[128:132] == b"DICM"


def detect_modality(
    filename: str,
    content_type: Optional[str],
    header: bytes,
) -> str:
    safe_name = _safe_filename(filename)
    ext = _extension(safe_name)
    mime = (content_type or "").strip().lower()

    if _has_dicom_preamble(header):
        return "dicom"

    if ext == ".csv":
        return "csv"
    if ext == ".txt":
        return "text"
    if ext in {".dcm", ".dicom"}:
        return "dicom"
    if ext in {".nii", ".nii.gz"}:
        return "nifti"
    if ext in {".svs", ".tif", ".tiff"}:
        return "wsi"

    if mime == "text/csv":
        return "csv"
    if mime == "text/plain":
        return "text"
    if mime in {"application/dicom", "application/x-dicom"} or "dicom" in mime:
        return "dicom"
    if mime in {"image/tiff", "image/tif"}:
        return "wsi"

    raise IngestionError("Unsupported file modality")


def _placeholder_result(handler_name: str) -> Dict[str, str]:
    return {
        "handler": handler_name,
        "routing_status": "handler_selected",
        "anonymization_status": "placeholder",
        "message": (
            "Week 1 routing scaffold selected this handler. "
            "Real anonymization will be added in later milestones."
        ),
    }


def anonymize_csv() -> Dict[str, str]:
    return _placeholder_result("anonymize_csv")


def anonymize_text(
    text_content: bytes,
    profile: str,
    study_salt: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        text = text_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IngestionError("Text uploads must be UTF-8 encoded") from exc

    try:
        result = anonymize_clinical_text(
            text=text,
            profile=profile,
            study_salt=study_salt,
        )
    except TextAnonymizationError as exc:
        raise IngestionError(exc.detail, status_code=exc.status_code) from exc

    return {
        "handler": "anonymize_text",
        "routing_status": "handler_selected",
        "anonymization_status": result["anonymization_status"],
        "message": "Text anonymization completed.",
        "anonymized_text": result["anonymized_text"],
        "date_strategy": result["date_strategy"],
        "text_identifier_strategy": result["text_identifier_strategy"],
        "detected_entities": result["detected_entities"],
    }


def anonymize_dicom(file_content: bytes, profile: str) -> Dict[str, Any]:
    try:
        metadata_result = anonymize_dicom_file_bytes(file_content, profile=profile)
    except DicomAnonymizationError as exc:
        raise IngestionError(exc.detail, status_code=exc.status_code) from exc

    metadata_scrubbed_content = metadata_result["anonymized_dicom_bytes"]
    pixel_result = safe_ocr_response(
        redact_dicom_pixels(metadata_scrubbed_content, profile=profile)
    )

    return {
        "handler": "anonymize_dicom",
        "routing_status": "handler_selected",
        "anonymization_status": metadata_result["anonymization_status"],
        "message": "DICOM metadata anonymization and pixel redaction completed.",
        "metadata_summary": metadata_result["metadata_summary"],
        **pixel_result,
    }


def anonymize_nifti(
    file_content: bytes,
    filename: str,
    profile: str,
) -> Dict[str, Any]:
    try:
        result = anonymize_nifti_metadata(
            file_content,
            filename=filename,
            profile=profile,
        )
    except NiftiAnonymizationError as exc:
        raise IngestionError(exc.detail, status_code=exc.status_code) from exc

    return {
        "handler": "anonymize_nifti",
        "routing_status": "handler_selected",
        "anonymization_status": result["anonymization_status"],
        "message": "NIfTI metadata anonymization completed.",
        "metadata_summary": result["metadata_summary"],
    }


def anonymize_wsi(
    file_content: bytes,
    filename: str,
    profile: str,
) -> Dict[str, Any]:
    result = scan_wsi_bytes(file_content, filename=filename)
    anonymization_status = (
        "redaction_plan_ready"
        if result["pixel_redaction_status"] == "redaction_plan_ready"
        else "pending"
    )

    return {
        "handler": "anonymize_wsi",
        "routing_status": "handler_selected",
        "anonymization_status": anonymization_status,
        "message": "WSI priority OCR tiling evaluated.",
        **result,
    }


HANDLER_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "csv": anonymize_csv,
    "text": anonymize_text,
    "dicom": anonymize_dicom,
    "nifti": anonymize_nifti,
    "wsi": anonymize_wsi,
}


def route_for_ingestion(
    filename: str,
    content_type: Optional[str],
    header: bytes,
    profile: str,
    text_content: Optional[bytes] = None,
    file_content: Optional[bytes] = None,
    study_salt: Optional[str] = None,
) -> Dict[str, Any]:
    safe_name = _safe_filename(filename)
    privacy_profile = validate_privacy_profile(profile)
    modality = detect_modality(safe_name, content_type, header)

    handler = HANDLER_REGISTRY.get(modality)
    if handler is None:
        raise IngestionError(
            "No ingestion handler is available for detected modality",
            status_code=500,
        )

    if modality == "text":
        if text_content is None:
            raise IngestionError(
                "Text content was not provided for anonymization",
                status_code=500,
            )
        handler_result = handler(text_content, privacy_profile, study_salt)
    elif modality == "dicom":
        if file_content is None:
            raise IngestionError(
                "DICOM content was not provided for anonymization",
                status_code=500,
            )
        handler_result = handler(file_content, privacy_profile)
    elif modality == "nifti":
        if file_content is None:
            raise IngestionError(
                "NIfTI content was not provided for anonymization",
                status_code=500,
            )
        handler_result = handler(file_content, safe_name, privacy_profile)
    elif modality == "wsi":
        if file_content is None:
            raise IngestionError(
                "WSI content was not provided for OCR scan planning",
                status_code=500,
            )
        handler_result = handler(file_content, safe_name, privacy_profile)
    else:
        handler_result = handler()

    response = {
        "status": "success",
        "filename": safe_name,
        "detected_modality": modality,
        "privacy_profile": privacy_profile,
        "handler": handler_result["handler"],
        "routing_status": handler_result["routing_status"],
        "anonymization_status": handler_result["anonymization_status"],
        "message": handler_result["message"],
        "downstream": {
            "ipfs_chunking": "pending",
            "cid_encryption": "pending",
            "metadata_indexing": "pending",
            "blockchain_transaction": "pending",
        },
    }
    if "anonymized_text" in handler_result:
        response["anonymized_text"] = handler_result["anonymized_text"]
    if "date_strategy" in handler_result:
        response["date_strategy"] = handler_result["date_strategy"]
    if "text_identifier_strategy" in handler_result:
        response["text_identifier_strategy"] = handler_result["text_identifier_strategy"]
    if "detected_entities" in handler_result:
        response["detected_entities"] = handler_result["detected_entities"]
    if "metadata_summary" in handler_result:
        response["metadata_summary"] = handler_result["metadata_summary"]
    if "pixel_redaction_status" in handler_result:
        response["pixel_redaction_status"] = handler_result["pixel_redaction_status"]
    for safe_key in (
        "ocr_boxes_detected",
        "boxes_redacted",
        "frames_processed",
        "scanned_regions",
        "ocr_engine_status",
        "ocr_confidence_threshold",
        "tiles_scanned",
        "priority_regions_scanned",
        "image_dimensions",
        "tile_size",
        "wsi_rewrite_status",
        "redaction_plan_boxes",
    ):
        if safe_key in handler_result:
            response[safe_key] = handler_result[safe_key]

    return response

