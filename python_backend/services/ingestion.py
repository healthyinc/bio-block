from typing import Any, Callable, Dict, Optional


SUPPORTED_PROFILES = {"strict", "research"}
SUPPORTED_MODALITIES = {"csv", "text", "dicom", "nifti", "wsi"}
HEADER_READ_LIMIT = 4096


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


def anonymize_text() -> Dict[str, str]:
    return _placeholder_result("anonymize_text")


def anonymize_dicom() -> Dict[str, str]:
    return _placeholder_result("anonymize_dicom")


def anonymize_nifti() -> Dict[str, str]:
    return _placeholder_result("anonymize_nifti")


def anonymize_wsi() -> Dict[str, str]:
    return _placeholder_result("anonymize_wsi")


HANDLER_REGISTRY: Dict[str, Callable[[], Dict[str, str]]] = {
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

    handler_result = handler()
    return {
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
