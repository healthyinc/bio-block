import os
import tempfile
from typing import Any, Dict

import numpy as np

from services.privacy_profiles import (
    PrivacyProfileError,
    get_privacy_profile,
    validate_privacy_profile,
)

try:
    import nibabel as nib
except ImportError:
    nib = None


SUPPORTED_EXTENSIONS = {".nii", ".nii.gz"}
TEXT_HEADER_FIELDS = ("descrip", "aux_file", "intent_name", "db_name")
DEFACING_STATUS = "not_implemented"


class NiftiAnonymizationError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _profile_settings(profile: str) -> tuple[str, Dict[str, Any]]:
    try:
        normalized = validate_privacy_profile(profile)
        return normalized, get_privacy_profile(normalized)
    except PrivacyProfileError as exc:
        raise NiftiAnonymizationError(exc.detail, status_code=exc.status_code) from exc


def _nifti_suffix(filename: str) -> str:
    lower_name = (filename or "").strip().lower()
    if lower_name.endswith(".nii.gz"):
        return ".nii.gz"
    if lower_name.endswith(".nii"):
        return ".nii"
    raise NiftiAnonymizationError("NIfTI uploads must use .nii or .nii.gz")


def _header_value_has_text(value: Any) -> bool:
    if hasattr(value, "tobytes"):
        raw = value.tobytes()
    elif isinstance(value, bytes):
        raw = value
    else:
        raw = str(value).encode("utf-8", errors="ignore")

    return bool(raw.replace(b"\x00", b"").strip())


def _scrub_header_fields(header: Any) -> Dict[str, Any]:
    field_counts: Dict[str, int] = {}

    for field_name in TEXT_HEADER_FIELDS:
        if field_name not in header:
            continue

        if _header_value_has_text(header[field_name]):
            field_counts[field_name] = 1
        header[field_name] = b""

    return {
        "fields_scrubbed": sum(field_counts.values()),
        "scrubbed_field_counts": field_counts,
    }


def _validate_scrubbed_nifti(scrubbed_image: Any, original_shape: tuple) -> Dict[str, Any]:
    """Serialize the scrubbed volume, re-read it, and confirm the scrub held.

    Asserting that header fields are clear without re-reading the bytes we
    would hand on is how a scrub that did not survive serialization gets
    reported as complete.
    """
    failures = []
    image_data_preserved = False
    try:
        serialized = scrubbed_image.to_bytes()
        reread = nib.Nifti1Image.from_bytes(serialized)
    except Exception:
        return {
            "metadata_validation_status": "serialization_failed",
            "validation_failures": ["serialization_failed"],
            "image_data_preserved": False,
        }

    for field_name in TEXT_HEADER_FIELDS:
        if field_name not in reread.header:
            continue
        if _header_value_has_text(reread.header[field_name]):
            failures.append(f"{field_name}_not_cleared")

    if len(reread.header.extensions):
        failures.append("extensions_present_after_scrub")
    if tuple(reread.shape) != tuple(original_shape):
        failures.append("shape_changed")
    else:
        image_data_preserved = True

    return {
        "metadata_validation_status": "verified" if not failures else "verification_failed",
        "validation_failures": sorted(set(failures)),
        "image_data_preserved": image_data_preserved,
    }


def _load_nifti_from_bytes(file_bytes: bytes, suffix: str):
    temp_path = None
    temp_file = tempfile.NamedTemporaryFile(
        suffix=suffix,
        prefix="bioblock_nifti_",
        delete=False,
    )
    try:
        temp_path = temp_file.name
        temp_file.write(file_bytes)
        temp_file.close()
        return nib.load(temp_path), temp_path
    except Exception:
        temp_file.close()
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def anonymize_nifti_metadata(
    file_bytes: bytes,
    filename: str,
    profile: str = "strict",
) -> Dict[str, Any]:
    if nib is None:
        raise NiftiAnonymizationError(
            "NIfTI metadata anonymization dependency is not available.",
            status_code=503,
        )
    if not file_bytes:
        raise NiftiAnonymizationError("NIfTI file is empty")

    privacy_profile, settings = _profile_settings(profile)
    suffix = _nifti_suffix(filename)

    temp_path = None
    try:
        image, temp_path = _load_nifti_from_bytes(file_bytes, suffix)
    except Exception as exc:
        raise NiftiAnonymizationError("Invalid NIfTI file format") from exc

    try:
        original_shape = tuple(int(dimension) for dimension in image.shape)
        original_affine = np.array(image.affine, copy=True)
        original_dtype = str(image.get_data_dtype())
        original_extensions = len(image.header.extensions)

        scrubbed_header = image.header.copy()
        scrubbed = _scrub_header_fields(scrubbed_header)

        extensions_removed = 0
        if settings["remove_nifti_extensions"]:
            extensions_removed = len(scrubbed_header.extensions)
            scrubbed_header.extensions.clear()

        scrubbed_image = nib.Nifti1Image(
            image.dataobj,
            image.affine.copy(),
            header=scrubbed_header,
        )

        extensions_preserved = original_extensions - extensions_removed
        validation = _validate_scrubbed_nifti(scrubbed_image, original_shape)
        if extensions_preserved:
            # An extension can embed a whole DICOM header. A surviving one is
            # unscanned metadata, not a clean result.
            validation["metadata_validation_status"] = "verification_failed"
            validation["validation_failures"] = sorted(
                set(validation.get("validation_failures", []) + ["extensions_preserved"])
            )

        status = (
            "completed"
            if validation["metadata_validation_status"] == "verified"
            else "privacy_requirements_not_met"
        )

        return {
            "anonymization_status": status,
            "metadata_summary": {
                "profile": privacy_profile,
                "remove_nifti_extensions": settings["remove_nifti_extensions"],
                "fields_scrubbed": scrubbed["fields_scrubbed"],
                "scrubbed_field_counts": scrubbed["scrubbed_field_counts"],
                "extensions_present_before": original_extensions,
                "extensions_removed": extensions_removed,
                "extensions_preserved": extensions_preserved,
                "image_shape": list(original_shape),
                "shape_preserved": tuple(scrubbed_image.shape) == original_shape,
                "affine_preserved": np.array_equal(
                    scrubbed_image.affine,
                    original_affine,
                ),
                "datatype_preserved": (
                    str(scrubbed_image.get_data_dtype()) == original_dtype
                ),
                "image_data_preserved": validation["image_data_preserved"],
                "metadata_validation_status": validation["metadata_validation_status"],
                "validation_failures": validation.get("validation_failures", []),
                # Cross-sectional head imaging permits facial reconstruction,
                # which Safe Harbor treats as a comparable image. No defacing
                # step exists, so this stays a standing blocker.
                "defacing_status": DEFACING_STATUS,
                "safe_technical_metadata_preserved": [
                    "shape",
                    "affine",
                    "datatype",
                    "image_data",
                ],
            },
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
