from typing import Any, Callable, Dict, Optional, Tuple

from services.text_anonymization import (
    MAX_TEXT_BYTES,
    TextAnonymizationError,
    anonymize_clinical_text,
    residual_phi_categories,
)
from services.document_sanitization import (
    REASON_NO_VALIDATED_WRITER,
    DocumentSanitizationError,
    scan_pdf_for_ingestion,
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
from services.tabular_anonymization import (
    TabularAnonymizationError,
    anonymize_tabular_csv,
)
from services.wsi_tiling import scan_wsi_bytes
from services.privacy_profiles import (
    PrivacyProfileError,
    validate_privacy_profile as validate_config_privacy_profile,
)
from services.privacy_contracts import (
    SanitizedArtifact,
    expert_determination_decision,
    issue_release,
    manual_review_decision,
)
from services.privacy_policy import PrivacyPolicyError, resolve_privacy_policy

SUPPORTED_PROFILES = {"safe_harbor_v1", "strict", "research"}
SUPPORTED_MODALITIES = {"csv", "text", "pdf", "dicom", "nifti", "wsi"}
HEADER_READ_LIMIT = 4096
TEXT_READ_LIMIT_BYTES = min(256 * 1024, MAX_TEXT_BYTES)
TABULAR_SUMMARY_KEYS = (
    "rows_in",
    "rows_out",
    "k",
    "l",
    "direct_identifiers_removed",
    "precise_geography_columns_removed",
    "columns_removed",
    "quasi_identifiers_used",
    "sensitive_column",
    "output_columns",
    "safe_harbor_report",
    "equivalence_classes",
    "min_group_size",
    "k_anonymity_satisfied",
    "l_diversity_satisfied",
    "generalized_cells_count",
    "suppressed_cells_count",
    "generalization_rate",
    "suppression_rate",
    "warnings",
)


class IngestionError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def validate_privacy_profile(profile: str) -> str:
    try:
        resolved = resolve_privacy_policy(profile)
        validate_config_privacy_profile(resolved.config_profile)
        return resolved.requested_profile
    except (PrivacyProfileError, PrivacyPolicyError) as exc:
        detail = getattr(exc, "detail", str(exc))
        status_code = getattr(exc, "status_code", 400)
        raise IngestionError(detail, status_code=status_code) from exc


def _blocked_downstream(status: str) -> Dict[str, str]:
    return {
        "ipfs_chunking": status,
        "cid_encryption": status,
        "metadata_indexing": status,
        "blockchain_transaction": status,
    }


def _expert_determination_response(
    safe_name: str,
    modality: str,
) -> Dict[str, Any]:
    decision = expert_determination_decision()
    return {
        "status": "success",
        "filename": safe_name,
        "detected_modality": modality,
        "privacy_profile": "research",
        "privacy_policy": decision.policy.value,
        "handler": HANDLER_REGISTRY[modality].__name__,
        "routing_status": "release_blocked",
        "anonymization_status": decision.disposition.value,
        "message": "Research processing requires an expert determination.",
        "release_decision": decision.to_public_dict(),
        "downstream": _blocked_downstream("blocked"),
    }


FACIAL_RECONSTRUCTION_REASON = "facial_reconstruction_not_mitigated"


def _imaging_reason_codes(
    handler_result: Dict[str, Any],
    status_key: str,
    base_reason: str,
) -> Tuple[str, ...]:
    """Reason codes for volumetric imaging, always including the standing one."""
    reasons = [base_reason, FACIAL_RECONSTRUCTION_REASON]
    status = handler_result.get(status_key)
    if status and status not in {"completed", "completed_no_text_detected"}:
        reasons.append(str(status))
    return tuple(sorted(set(reasons)))


def _release_decision_for(
    modality: str,
    handler_result: Dict[str, Any],
    safe_name: str,
):
    if modality == "text" and handler_result.get("anonymization_status") == "completed":
        residual = handler_result.get("residual_phi_categories") or {}
        if residual:
            # Redaction did not clear the text. Categories only, never values.
            return manual_review_decision(
                "privacy_requirements_not_met",
                *sorted(f"residual_{category.lower()}" for category in residual),
            )
        artifact = SanitizedArtifact(
            content=handler_result["anonymized_text"].encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            filename=safe_name,
            validators=(
                "typed_phi_detection",
                "deterministic_redaction",
                "residual_phi_rescan",
            ),
        )
        return issue_release(artifact, "safe_harbor_technical_checks_passed")
    if modality == "pdf":
        # No validated PDF writer exists, so a PDF is never auto-releasable.
        return manual_review_decision(
            *(handler_result.get("unscannable_reasons") or [REASON_NO_VALIDATED_WRITER])
        )
    if modality == "csv":
        return manual_review_decision("serialized_output_validation_pending")
    if modality == "dicom":
        # Cross-sectional imaging permits facial reconstruction, which Safe
        # Harbor treats as a comparable image. No defacing step exists, so
        # this blocks regardless of how the pixel scan went.
        return manual_review_decision(
            *_imaging_reason_codes(
                handler_result,
                "pixel_redaction_status",
                "dicom_validation_incomplete",
            )
        )
    if modality == "nifti":
        return manual_review_decision(
            *_imaging_reason_codes(
                handler_result,
                "anonymization_status",
                "nifti_serialization_pending",
            )
        )
    if modality == "wsi":
        return manual_review_decision("validated_wsi_writer_unavailable")
    return manual_review_decision("validation_incomplete")


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


def _has_pdf_magic(header: bytes) -> bool:
    return header[:1024].lstrip().startswith(b"%PDF-")


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
    if _has_pdf_magic(header):
        return "pdf"

    if ext == ".csv":
        return "csv"
    if ext == ".txt":
        return "text"
    if ext == ".pdf":
        return "pdf"
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
    if mime == "application/pdf":
        return "pdf"
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


def anonymize_csv(
    file_content: bytes,
    k: int = 5,
    l: int = 2,
    direct_identifiers: Optional[list[str]] = None,
    quasi_identifiers: Optional[list[str]] = None,
    sensitive_column: Optional[str] = None,
    safe_harbor_mappings: Optional[dict[str, list[str]]] = None,
) -> Dict[str, Any]:
    try:
        result = anonymize_tabular_csv(
            file_content,
            k=k,
            l=l,
            direct_identifiers=direct_identifiers,
            quasi_identifiers=quasi_identifiers,
            sensitive_column=sensitive_column,
            safe_harbor_mappings=safe_harbor_mappings,
        )
    except TabularAnonymizationError as exc:
        raise IngestionError(exc.detail, status_code=exc.status_code) from exc
    return {
        "handler": "anonymize_csv",
        "routing_status": "handler_selected",
        "message": "CSV tabular anonymization completed.",
        **result,
    }


def anonymize_text(
    text_content: bytes,
    profile: str,
    study_salt: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        text = text_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IngestionError("Text uploads must be UTF-8 encoded") from exc
    if "\x00" in text:
        # A NUL byte can split a value the detector would otherwise match, so
        # the upload is unscannable rather than clean.
        raise IngestionError("Text uploads must not contain NUL bytes")

    try:
        result = anonymize_clinical_text(
            text=text,
            profile=profile,
            study_salt=study_salt,
        )
        residual = residual_phi_categories(result["anonymized_text"])
    except TextAnonymizationError as exc:
        raise IngestionError(exc.detail, status_code=exc.status_code) from exc

    return {
        "residual_phi_categories": residual,
        "handler": "anonymize_text",
        "routing_status": "handler_selected",
        "anonymization_status": result["anonymization_status"],
        "message": "Text anonymization completed.",
        "anonymized_text": result["anonymized_text"],
        "date_strategy": result["date_strategy"],
        "text_identifier_strategy": result["text_identifier_strategy"],
        "detected_entities": result["detected_entities"],
        "entity_count": result["entity_count"],
        "detection_sources": result["detection_sources"],
        "ner_model": result["ner_model"],
        "trained_ner_active": result["trained_ner_active"],
    }


def scan_pdf(file_content: bytes, profile: str) -> Dict[str, Any]:
    """Inventory and scan PDF surfaces. Never produces releasable bytes."""
    try:
        result = scan_pdf_for_ingestion(file_content, profile=profile)
    except DocumentSanitizationError as exc:
        raise IngestionError(exc.detail, status_code=exc.status_code) from exc

    return {
        "handler": "scan_pdf",
        "routing_status": "handler_selected",
        "anonymization_status": result["anonymization_status"],
        "message": result["message"],
        "pdf_summary": result["pdf_summary"],
        "unscannable_reasons": result["unscannable_reasons"],
        "detected_entities": result["detected_entities"],
        "entity_count": result["entity_count"],
        "detection_sources": result["detection_sources"],
        "residual_phi_categories": result["residual_phi_categories"],
        "text_layer_complete": result["text_layer_complete"],
        "pages": result["pages"],
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
    "pdf": scan_pdf,
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
    csv_k: int = 5,
    csv_l: int = 2,
    csv_direct_identifiers: Optional[list[str]] = None,
    csv_quasi_identifiers: Optional[list[str]] = None,
    csv_sensitive_column: Optional[str] = None,
    csv_safe_harbor_mappings: Optional[dict[str, list[str]]] = None,
) -> Dict[str, Any]:
    safe_name = _safe_filename(filename)
    privacy_profile = validate_privacy_profile(profile)
    resolved_policy = resolve_privacy_policy(privacy_profile)
    modality = detect_modality(safe_name, content_type, header)

    if not resolved_policy.automatic_release_allowed:
        return _expert_determination_response(safe_name, modality)

    handler = HANDLER_REGISTRY.get(modality)
    if handler is None:
        raise IngestionError(
            "No ingestion handler is available for detected modality",
            status_code=500,
        )

    if modality == "csv":
        if file_content is None:
            raise IngestionError(
                "CSV content was not provided for anonymization",
                status_code=500,
            )
        handler_result = handler(
            file_content,
            csv_k,
            csv_l,
            csv_direct_identifiers,
            csv_quasi_identifiers,
            csv_sensitive_column,
            csv_safe_harbor_mappings,
        )
    elif modality == "text":
        if text_content is None:
            raise IngestionError(
                "Text content was not provided for anonymization",
                status_code=500,
            )
        handler_result = handler(text_content, resolved_policy.config_profile, study_salt)
    elif modality == "pdf":
        if file_content is None:
            raise IngestionError(
                "PDF content was not provided for scanning",
                status_code=500,
            )
        handler_result = handler(file_content, resolved_policy.config_profile)
    elif modality == "dicom":
        if file_content is None:
            raise IngestionError(
                "DICOM content was not provided for anonymization",
                status_code=500,
            )
        handler_result = handler(file_content, resolved_policy.config_profile)
    elif modality == "nifti":
        if file_content is None:
            raise IngestionError(
                "NIfTI content was not provided for anonymization",
                status_code=500,
            )
        handler_result = handler(file_content, safe_name, resolved_policy.config_profile)
    elif modality == "wsi":
        if file_content is None:
            raise IngestionError(
                "WSI content was not provided for OCR scan planning",
                status_code=500,
            )
        handler_result = handler(file_content, safe_name, resolved_policy.config_profile)
    else:
        handler_result = handler()

    release_decision = _release_decision_for(modality, handler_result, safe_name)
    response = {
        "status": "success",
        "filename": safe_name,
        "detected_modality": modality,
        "privacy_profile": privacy_profile,
        "privacy_policy": resolved_policy.policy.value,
        "handler": handler_result["handler"],
        "routing_status": handler_result["routing_status"],
        "anonymization_status": handler_result["anonymization_status"],
        "message": handler_result["message"],
        "release_decision": release_decision.to_public_dict(),
        "downstream": _blocked_downstream(
            "pending" if release_decision.releasable else "blocked"
        ),
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
    if "residual_phi_categories" in handler_result:
        response["residual_phi_categories"] = handler_result["residual_phi_categories"]
    for pdf_key in (
        "pdf_summary",
        "unscannable_reasons",
        "text_layer_complete",
        "pages",
    ):
        if pdf_key in handler_result:
            response[pdf_key] = handler_result[pdf_key]
    if "rows_in" in handler_result:
        response["tabular_summary"] = {
            key: handler_result[key]
            for key in TABULAR_SUMMARY_KEYS
            if key in handler_result
        }
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

    if "entity_count" in handler_result:
        response["entity_count"] = handler_result["entity_count"]
    if "detection_sources" in handler_result:
        response["detection_sources"] = handler_result["detection_sources"]
    if "ner_model" in handler_result:
        response["ner_model"] = handler_result["ner_model"]
    if "trained_ner_active" in handler_result:
        response["trained_ner_active"] = handler_result["trained_ner_active"]
    return response

