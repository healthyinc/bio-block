import json
import os
import sys
import tempfile
from io import BytesIO

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "week3-test-salt")

from main import app  # noqa: E402
from services.nifti_anonymization import (  # noqa: E402
    NiftiAnonymizationError,
    anonymize_nifti_metadata,
)


client = TestClient(app)

HEADER_TEXT = b"SYNTHETIC_NIFTI_HEADER_TEXT"
AUX_TEXT = b"SYNTHETIC_AUX_FILE"
INTENT_TEXT = b"SYNTHETIC_INTENT_NAME"
EXTENSION_TEXT = b"SYNTHETIC_EXTENSION_TEXT"


def build_nifti_bytes(suffix: str = ".nii") -> bytes:
    data = np.arange(8, dtype=np.int16).reshape((2, 2, 2))
    affine = np.eye(4)
    image = nib.Nifti1Image(data, affine)
    image.header["descrip"] = HEADER_TEXT
    image.header["aux_file"] = AUX_TEXT
    image.header["intent_name"] = INTENT_TEXT
    image.header.extensions.append(nib.nifti1.Nifti1Extension(6, EXTENSION_TEXT))

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        nib.save(image, temp_path)
        with open(temp_path, "rb") as saved_file:
            return saved_file.read()
    finally:
        os.unlink(temp_path)


def test_nifti_header_scrubbing_preserves_image_properties():
    result = anonymize_nifti_metadata(build_nifti_bytes(".nii"), "scan.nii")
    response_text = json.dumps(result)
    summary = result["metadata_summary"]

    assert result["anonymization_status"] == "completed"
    assert summary["fields_scrubbed"] == 3
    assert summary["scrubbed_field_counts"] == {
        "descrip": 1,
        "aux_file": 1,
        "intent_name": 1,
    }
    assert summary["extensions_removed"] == 1
    assert summary["image_shape"] == [2, 2, 2]
    assert summary["shape_preserved"] is True
    assert summary["affine_preserved"] is True
    assert summary["datatype_preserved"] is True
    assert summary["image_data_preserved"] is True
    assert HEADER_TEXT.decode("ascii") not in response_text
    assert AUX_TEXT.decode("ascii") not in response_text
    assert EXTENSION_TEXT.decode("ascii") not in response_text


def test_invalid_nifti_is_rejected():
    with pytest.raises(NiftiAnonymizationError) as exc:
        anonymize_nifti_metadata(b"not nifti", "scan.nii")

    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid NIfTI file format"


@pytest.mark.parametrize(
    ("filename", "suffix"),
    [
        ("scan.nii", ".nii"),
        ("scan.nii.gz", ".nii.gz"),
    ],
)
def test_nifti_api_returns_completed_for_supported_extensions(filename, suffix):
    response = client.post(
        "/api/v1/ingest",
        files={
            "file": (
                filename,
                BytesIO(build_nifti_bytes(suffix)),
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    response_text = json.dumps(body)
    assert body["detected_modality"] == "nifti"
    assert body["handler"] == "anonymize_nifti"
    assert body["anonymization_status"] == "completed"
    assert body["metadata_summary"]["fields_scrubbed"] == 3
    assert HEADER_TEXT.decode("ascii") not in response_text
    assert AUX_TEXT.decode("ascii") not in response_text
    assert "file_bytes" not in body


def test_nifti_research_profile_removes_extensions_and_preserves_safe_metadata():
    result = anonymize_nifti_metadata(
        build_nifti_bytes(".nii"),
        "scan.nii",
        profile="research",
    )
    summary = result["metadata_summary"]

    assert summary["profile"] == "research"
    assert summary["remove_nifti_extensions"] is True
    assert summary["extensions_removed"] == 1
    assert summary["extensions_preserved"] == 0
    assert summary["shape_preserved"] is True
    assert summary["affine_preserved"] is True
    assert summary["datatype_preserved"] is True


