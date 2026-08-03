from typing import Any, Callable, Dict, Optional

from services.text_anonymization import (
    MAX_TEXT_BYTES,
    TextAnonymizationError,
    anonymize_clinical_text,
)

SUPPORTED_PROFILES = {"strict", "research"}
SUPPORTED_MODALITIES = {"csv", "text", "dicom", "nifti", "wsi"}
HEADER_READ_LIMIT = 4096
TEXT_READ_LIMIT_BYTES = MAX_TEXT_BYTES


class IngestionError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def validate_privacy_profile(profile: str) -> str:
    normalized = (profile or "").strip().lower()
    if normalized not in SUPPORTED_PROFILES:
        raise IngestionError(
            "Invalid privacy profile. Supported profiles: strict, research"
        )
    return normalized


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
        "detected_entities": result["detected_entities"],
        "entity_count": result["entity_count"],
        "detection_sources": result["detection_sources"],
        "ner_model": result["ner_model"],
        "trained_ner_active": result["trained_ner_active"],
    }


def anonymize_dicom() -> Dict[str, str]:
    return _placeholder_result("anonymize_dicom")


def anonymize_nifti() -> Dict[str, str]:
    return _placeholder_result("anonymize_nifti")


def anonymize_wsi() -> Dict[str, str]:
    return _placeholder_result("anonymize_wsi")


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
    if "detected_entities" in handler_result:
        response["detected_entities"] = handler_result["detected_entities"]

    if "entity_count" in handler_result:
        response["entity_count"] = handler_result["entity_count"]
    if "detection_sources" in handler_result:
        response["detection_sources"] = handler_result["detection_sources"]
    if "ner_model" in handler_result:
        response["ner_model"] = handler_result["ner_model"]
    if "trained_ner_active" in handler_result:
        response["trained_ner_active"] = handler_result["trained_ner_active"]
    return response
