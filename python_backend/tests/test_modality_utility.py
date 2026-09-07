"""Per-modality utility measurement, and the manifest that reports it.

Phase 10 wrote a utility contract for every modality and measured one of them.
A verdict computed from no measurements reports "passed" because nothing
contradicted it, which is indistinguishable from a gate that has stopped
gating. These tests hold the measurements to two standards: they must actually
be taken, and they must carry no content.

Release posture is asserted alongside, because measuring a modality is not the
same as clearing it. DICOM and NIfTI stay blocked pending defacing, WSI
pending a validated writer, PDF and workbook under manual review.
"""

import json
import os
import sys
from io import BytesIO

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import modality_utility as mu  # noqa: E402
from services.transformation_manifest import (  # noqa: E402
    FORBIDDEN_KEYS,
    MANIFEST_VERSION,
    build_manifest,
    component_versions,
)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _rows(count):
    return [
        {"age": str(30 + index), "gender": "F" if index % 2 else "M", "dx": "flu"}
        for index in range(count)
    ]


def test_csv_measurement_reports_every_contract_term():
    original = _rows(6)
    generalised = [dict(row, age="30-39") for row in original]

    metrics = mu.measure_csv_utility(
        ["age", "gender", "dx"],
        original,
        ["age", "gender", "dx"],
        generalised,
        generalized_columns=["age"],
    )

    for term in (
        "rows_retained",
        "rows_suppressed",
        "cells_modified",
        "cells_generalized",
        "column_retention",
        "information_loss_inverse",
        "numeric_distribution_drift",
        "categorical_frequency_drift",
        "correlation_preservation",
        "k_verification_status",
        "l_verification_status",
    ):
        assert term in metrics, f"{term} is not measured"

    assert metrics["cells_generalized"] == 6
    assert metrics["rows_suppressed"] == 0


def test_csv_measurement_carries_no_cell_values():
    rows = [{"name": "Rukmini Balasubramanian", "age": "44"}]
    metrics = mu.measure_csv_utility(["name", "age"], rows, ["age"], rows)

    serialized = json.dumps(metrics)
    assert "Rukmini" not in serialized
    assert "Balasubramanian" not in serialized


def test_suppressed_rows_are_counted_not_hidden():
    metrics = mu.measure_csv_utility(
        ["age"], _rows(10), ["age"], _rows(4)
    )

    assert metrics["rows_suppressed"] == 6
    assert metrics["row_retention"] == 0.4


# ---------------------------------------------------------------------------
# Raster
# ---------------------------------------------------------------------------


def _png(colour=(200, 180, 160), size=(32, 24)):
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def test_raster_measurement_reports_geometry_and_pixel_fidelity():
    payload = _png()
    metrics = mu.measure_raster_utility(payload, payload)

    assert metrics["width"] == 32
    assert metrics["height"] == 24
    assert metrics["pixel_format"] == "RGB"
    assert metrics["dimensions_preserved"] is True
    assert metrics["pixel_equality_outside_redactions"] == 1.0


def test_raster_measurement_notices_pixels_changed_outside_a_redaction():
    from PIL import Image

    before = _png()
    changed = Image.new("RGB", (32, 24), (10, 10, 10))
    buffer = BytesIO()
    changed.save(buffer, format="PNG")

    metrics = mu.measure_raster_utility(
        before, buffer.getvalue(), redaction_boxes=[(0, 0, 4, 4)]
    )

    assert metrics["pixel_equality_outside_redactions"] < 1.0


def test_raster_measurement_separates_preserved_and_uncertain_regions():
    payload = _png()
    metrics = mu.measure_raster_utility(
        payload,
        payload,
        redaction_boxes=[(0, 0, 8, 8)],
        preserved_label_regions=2,
        review_regions=1,
    )

    assert metrics["preserved_label_regions"] == 2
    assert metrics["regions_requiring_review"] == 1
    assert metrics["redacted_area"] == 64


# ---------------------------------------------------------------------------
# Modalities with no validated writer
# ---------------------------------------------------------------------------


def test_wsi_measurement_never_claims_output_preservation():
    """There is no slide writer, so there is nothing to have preserved."""
    metrics = mu.measure_wsi_utility(
        width=2048,
        height=1024,
        level_count=3,
        tile_size=512,
        tiles_scanned=8,
        diagnostic_tiles_available=8,
        associated_images=["label", "macro", "thumbnail"],
        metadata_keys=["openslide.vendor"],
    )

    assert metrics["output_available"] is False
    assert metrics["rewritten_output_preservation"] == "no_validated_writer"
    assert metrics["associated_image_count"] == 3
    assert metrics["pyramid_levels"] == 3


def test_pdf_measurement_reports_every_surface_a_writer_would_have_to_handle():
    metrics = mu.measure_pdf_utility(
        ["Routine follow-up. Metformin 500 mg."],
        ["Routine follow-up. Metformin 500 mg."],
        image_count=2,
        annotation_count=1,
        form_field_count=3,
        link_count=4,
        attachment_count=1,
        metadata_fields=["title", "author"],
    )

    assert metrics["pages"] == 1
    assert metrics["annotations"] == 1
    assert metrics["form_fields"] == 3
    assert metrics["links"] == 4
    assert metrics["attachments"] == 1
    assert metrics["writer_status"] == "no_validated_writer"
    assert metrics["clinical_term_preservation"] == 1.0


def test_workbook_measurement_inventories_the_hidden_surfaces():
    metrics = mu.measure_workbook_utility(
        sheet_count=3,
        hidden_sheet_count=1,
        row_count=40,
        column_count=12,
        formula_count=5,
        comment_count=2,
        defined_name_count=4,
        document_property_count=3,
        macro_count=1,
        external_link_count=2,
    )

    assert metrics["hidden_sheets"] == 1
    assert metrics["macros"] == 1
    assert metrics["external_links"] == 2
    assert metrics["cell_type_preservation"] == mu.UNAVAILABLE
    assert metrics["output_available"] is False


# ---------------------------------------------------------------------------
# Wiring: the pipelines actually produce these
# ---------------------------------------------------------------------------


def test_dicom_pipeline_measures_geometry_and_stays_blocked():
    pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

    from services.dicom_anonymization import anonymize_dicom_file_bytes

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.8.498.51"
    meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.52"
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = "1.2.826.0.1.3680043.8.498.53"
    dataset.SeriesInstanceUID = "1.2.826.0.1.3680043.8.498.54"
    dataset.Modality = "OT"
    dataset.PatientName = "SYNTH^TEST^ID"
    dataset.Rows = 8
    dataset.Columns = 8
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.full((8, 8), 128, dtype=np.uint8).tobytes()
    buffer = BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)

    result = anonymize_dicom_file_bytes(buffer.getvalue())
    metrics = result["utility_metrics"]

    assert metrics["rows"] == 8
    assert metrics["columns"] == 8
    assert metrics["geometry_preserved"] is True
    assert metrics["output_decode_valid"] is True
    assert "transfer_syntax_changed" in metrics
    # Measuring the file did not make it releasable.
    assert result["pixel_redaction_status"]


def test_nifti_pipeline_measures_geometry_and_makes_no_facial_claim():
    nib = pytest.importorskip("nibabel")
    from services.nifti_anonymization import anonymize_nifti_metadata

    image = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4))
    payload = image.to_bytes()

    result = anonymize_nifti_metadata(payload, "scan.nii")
    metrics = result["utility_metrics"]

    assert metrics["shape_preserved"] is True
    assert metrics["affine_preserved"] is True
    assert metrics["defacing_applied"] is False
    assert metrics["facial_privacy_claim"] == "not_assessed"


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


def _manifest(**overrides):
    payload = dict(
        artifact_id="a" * 64,
        modality="text",
        policy="safe_harbor_v1",
        detected={"PERSON": 2, "AGE_OVER_89": 1},
        sources={"ner": 2, "age_rule": 1},
        surrogate_counts={"PATIENT": 2},
        utility={"clinical_term_preservation": 1.0},
        utility_passed=True,
        residual_categories={},
        review_reasons=[],
        model_mode="legacy_test",
        generated_regions={"surrogate": 2, "generalized": 1},
    )
    payload.update(overrides)
    return build_manifest(**payload)


def test_manifest_reports_every_required_field():
    manifest = _manifest()

    assert manifest["manifest_version"] == MANIFEST_VERSION
    assert manifest["modality"] == "text"
    assert manifest["policy_version"] == "safe_harbor_v1"
    assert manifest["component_versions"]["evidence_model"]
    assert manifest["component_versions"]["clinical_vocabulary"]
    assert manifest["transformation_types"]["replaced_with_surrogate"] == 2
    assert manifest["transformation_types"]["generalized"] == 1
    assert manifest["regions_modified"]["generated_region_total"] == 3
    assert manifest["privacy_checks"]["passed"] is True
    assert manifest["utility_checks"]["passed"] is True
    assert manifest["release_decision"] == "releasable"
    assert manifest["manual_review_reason"] is None


def test_manifest_names_the_reason_a_reviewer_must_act_on():
    manifest = _manifest(
        review_reasons=["uncertain_age_reference"],
        unsupported_reasons=["serialized_output_validation_passed"],
    )

    assert manifest["release_decision"] == "manual_review_required"
    # Not the code recording a check that passed on the way to the block.
    assert manifest["manual_review_reason"] == "uncertain_age_reference"


def test_residual_finding_blocks_on_privacy_not_utility():
    manifest = _manifest(residual_categories={"US_SSN": 1}, utility_passed=False)

    assert manifest["release_decision"] == "blocked_privacy"
    assert manifest["reason_codes"] == ["privacy_requirements_not_met"]


def test_manifest_carries_no_value_no_mapping_and_no_filename():
    manifest = _manifest()
    serialized = json.dumps(manifest)

    def _keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from _keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from _keys(item)

    assert FORBIDDEN_KEYS.isdisjoint(set(_keys(manifest)))
    assert "PATIENT_001" not in serialized
    assert ".txt" not in serialized


def test_component_versions_never_report_an_empty_configuration():
    """An unavailable pin is recorded as unavailable, not as no pins."""
    versions = component_versions("legacy_test")

    assert versions["detectors"] != {}
    assert versions["thresholds"] != {}


def test_every_ingested_modality_receives_a_manifest():
    from fastapi.testclient import TestClient

    import main

    client = TestClient(main.app)
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("note.txt", b"Patient attended clinic.", "text/plain")},
        data={"privacy_profile": "strict"},
    )

    assert response.status_code == 200
    manifest = response.json()["transformation_manifest"]
    assert manifest["manifest_version"] == MANIFEST_VERSION
    assert manifest["artifact_id"] != "note.txt"
    assert "note.txt" not in json.dumps(manifest)
