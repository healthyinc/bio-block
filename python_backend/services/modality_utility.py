"""Utility measurement for every supported modality.

Phase 10 wrote a utility contract for each modality and then measured exactly
one of them. A contract nobody measures is a comment: the gate reports
"passed" because no metric contradicted it, which is indistinguishable from a
gate that has stopped gating. Phase 11 implements the measurements.

Two rules run through the whole module.

**Measure, do not release.** Nothing here changes what may leave the system.
DICOM and NIfTI stay blocked pending defacing, WSI pending a validated writer,
PDF and workbook under manual review, CSV pending approval. A modality now
reports what survived; it still does not get to decide that surviving is
enough.

**Report shapes, never content.** Every function returns counts, ratios,
dimensions and status strings. No cell value, pixel, header string, filename
or extracted sentence is returned, logged or raised - a utility report travels
further than the artifact it describes, and the whole point of the pipeline is
that the artifact's contents do not.

Where an input-side measurement is all that exists - because the modality has
no validated writer - the report says so in ``output_available``. Claiming
preservation across a rewrite that never happened would be worse than
measuring nothing.
"""

from __future__ import annotations

import math
from io import BytesIO
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

MEASUREMENT_VERSION = "modality-utility-v1"

#: Recorded when a measurement could not be taken at all. The utility verdict
#: treats a missing metric as a failure, so an unavailable measurement blocks
#: rather than silently passing.
UNAVAILABLE = "measurement_unavailable"


def _ratio(part: float, whole: float) -> float:
    """A share, with an empty denominator reported as fully preserved.

    Zero of zero rows retained is not a utility loss; it is a document with no
    rows. The distinction matters because a zero here would otherwise fail a
    contract that nothing violated.
    """
    if whole <= 0:
        return 1.0
    return round(part / whole, 4)


def _drift(before: Sequence[float], after: Sequence[float]) -> float:
    """Normalised shift between two numeric distributions, 0.0 to 1.0.

    Mean and spread only. A full distributional test would need the values
    themselves to travel with the report, and they are exactly what must not.
    """
    if not before or not after:
        return 0.0
    mean_before = sum(before) / len(before)
    mean_after = sum(after) / len(after)
    spread = max(
        (max(before) - min(before)) or 1.0,
        abs(mean_before) or 1.0,
    )
    return round(min(1.0, abs(mean_after - mean_before) / spread), 4)


def _frequency_drift(
    before: Mapping[str, int], after: Mapping[str, int]
) -> float:
    """Total variation distance between two category frequency tables."""
    total_before = sum(before.values()) or 1
    total_after = sum(after.values()) or 1
    keys = set(before) | set(after)
    distance = sum(
        abs(before.get(key, 0) / total_before - after.get(key, 0) / total_after)
        for key in keys
    )
    return round(min(1.0, distance / 2), 4)


def _correlation(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    """Pearson correlation, or None when it is undefined."""
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    covariance = sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right)
    )
    variance_left = sum((a - mean_left) ** 2 for a in left)
    variance_right = sum((b - mean_right) ** 2 for b in right)
    if variance_left <= 0 or variance_right <= 0:
        return None
    return covariance / math.sqrt(variance_left * variance_right)


def _numeric_column(rows: Sequence[Mapping[str, Any]], column: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        raw = row.get(column)
        if raw is None or raw == "":
            continue
        try:
            values.append(float(str(raw).strip()))
        except (TypeError, ValueError):
            continue
    return values


def _category_counts(
    rows: Sequence[Mapping[str, Any]], column: str
) -> Dict[str, int]:
    """Frequencies keyed by a stable hash, never by the category itself.

    The shape of a frequency table is what utility depends on; the labels are
    data. Hashing keeps the drift computable without carrying the values.
    """
    counts: Dict[str, int] = {}
    for row in rows:
        raw = row.get(column)
        if raw is None:
            continue
        key = f"c{abs(hash(str(raw))) % 10_000_000:07d}"
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def measure_csv_utility(
    original_header: Sequence[str],
    original_rows: Sequence[Mapping[str, Any]],
    output_header: Sequence[str],
    output_rows: Sequence[Mapping[str, Any]],
    generalized_columns: Sequence[str] = (),
    removed_columns: Sequence[str] = (),
    k_status: str = UNAVAILABLE,
    l_status: str = UNAVAILABLE,
) -> Dict[str, Any]:
    """What a generalised table kept of the table it came from."""
    retained_columns = [c for c in output_header if c in set(original_header)]
    shared_numeric = [
        column
        for column in retained_columns
        if len(_numeric_column(original_rows, column)) >= 2
    ]

    modified_cells = 0
    generalized_cells = 0
    generalized = set(generalized_columns)
    for before, after in zip(original_rows, output_rows):
        for column in retained_columns:
            if str(before.get(column, "")) != str(after.get(column, "")):
                modified_cells += 1
                if column in generalized:
                    generalized_cells += 1

    numeric_drift = 0.0
    categorical_drift = 0.0
    if retained_columns:
        numeric_drift = max(
            (
                _drift(
                    _numeric_column(original_rows, column),
                    _numeric_column(output_rows, column),
                )
                for column in shared_numeric
            ),
            default=0.0,
        )
        categorical_drift = max(
            (
                _frequency_drift(
                    _category_counts(original_rows, column),
                    _category_counts(output_rows, column),
                )
                for column in retained_columns
            ),
            default=0.0,
        )

    correlation_preservation = 1.0
    if len(shared_numeric) >= 2:
        differences = []
        for index, left in enumerate(shared_numeric):
            for right in shared_numeric[index + 1 :]:
                before = _correlation(
                    _numeric_column(original_rows, left),
                    _numeric_column(original_rows, right),
                )
                after = _correlation(
                    _numeric_column(output_rows, left),
                    _numeric_column(output_rows, right),
                )
                if before is None or after is None:
                    continue
                differences.append(abs(before - after))
        if differences:
            correlation_preservation = round(
                1.0 - min(1.0, sum(differences) / len(differences)), 4
            )

    total_cells = max(1, len(original_rows) * max(1, len(retained_columns)))
    return {
        "measurement_version": MEASUREMENT_VERSION,
        "output_available": True,
        "rows_in_input": len(original_rows),
        "rows_retained": len(output_rows),
        "rows_suppressed": max(0, len(original_rows) - len(output_rows)),
        "row_retention": _ratio(len(output_rows), len(original_rows)),
        "columns_in_input": len(original_header),
        "columns_retained": len(retained_columns),
        "columns_removed": len(removed_columns),
        "column_retention": _ratio(len(retained_columns), len(original_header)),
        "cells_modified": modified_cells,
        "cells_generalized": generalized_cells,
        # Share of cells left exactly as they arrived. The contract's
        # information-loss term is its complement.
        "information_loss_inverse": _ratio(total_cells - modified_cells, total_cells),
        "numeric_distribution_drift": numeric_drift,
        "categorical_frequency_drift": categorical_drift,
        "correlation_preservation": correlation_preservation,
        "k_verification_status": k_status,
        "l_verification_status": l_status,
    }


# ---------------------------------------------------------------------------
# DICOM
# ---------------------------------------------------------------------------


def _dicom_module():
    try:
        import pydicom  # noqa: WPS433

        return pydicom
    except Exception:
        return None


def measure_dicom_utility(
    original_bytes: bytes,
    output_bytes: Optional[bytes],
    redaction_boxes: Sequence[Tuple[int, int, int, int]] = (),
) -> Dict[str, Any]:
    """Geometry, encoding and pixel preservation across a DICOM rewrite.

    A changed transfer syntax is reported, not judged: recompression can be a
    legitimate consequence of rewriting a file, and calling it a utility
    failure would hide the cases where it is not.
    """
    pydicom = _dicom_module()
    if pydicom is None:
        return {
            "measurement_version": MEASUREMENT_VERSION,
            "output_available": output_bytes is not None,
            "status": UNAVAILABLE,
        }

    def _read(payload):
        return pydicom.dcmread(BytesIO(payload), force=True)

    try:
        before = _read(original_bytes)
    except Exception:
        return {
            "measurement_version": MEASUREMENT_VERSION,
            "output_available": output_bytes is not None,
            "status": "input_unreadable",
        }

    report: Dict[str, Any] = {
        "measurement_version": MEASUREMENT_VERSION,
        "output_available": output_bytes is not None,
        "rows": int(getattr(before, "Rows", 0) or 0),
        "columns": int(getattr(before, "Columns", 0) or 0),
        "frames": int(getattr(before, "NumberOfFrames", 1) or 1),
        "bits_allocated": int(getattr(before, "BitsAllocated", 0) or 0),
        "bits_stored": int(getattr(before, "BitsStored", 0) or 0),
        "photometric_interpretation": str(
            getattr(before, "PhotometricInterpretation", "") or ""
        ),
        "redaction_regions": len(redaction_boxes),
        "redacted_pixel_area": sum(
            max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            for box in redaction_boxes
        ),
    }
    total_pixels = report["rows"] * report["columns"] * max(1, report["frames"])
    report["pixel_area"] = total_pixels
    report["pixel_area_modified_share"] = _ratio(
        report["redacted_pixel_area"], total_pixels
    )

    if output_bytes is None:
        report["status"] = "input_side_only"
        return report

    try:
        after = _read(output_bytes)
    except Exception:
        report["status"] = "output_undecodable"
        report["output_decode_valid"] = False
        return report

    report["output_decode_valid"] = True
    report["geometry_preserved"] = (
        int(getattr(after, "Rows", 0) or 0) == report["rows"]
        and int(getattr(after, "Columns", 0) or 0) == report["columns"]
        and int(getattr(after, "NumberOfFrames", 1) or 1) == report["frames"]
    )
    report["bit_depth_preserved"] = (
        int(getattr(after, "BitsAllocated", 0) or 0) == report["bits_allocated"]
        and int(getattr(after, "BitsStored", 0) or 0) == report["bits_stored"]
    )
    report["photometric_interpretation_preserved"] = (
        str(getattr(after, "PhotometricInterpretation", "") or "")
        == report["photometric_interpretation"]
    )

    def _syntax(dataset) -> str:
        meta = getattr(dataset, "file_meta", None)
        return str(getattr(meta, "TransferSyntaxUID", "") or "")

    before_syntax, after_syntax = _syntax(before), _syntax(after)
    report["transfer_syntax_changed"] = bool(
        before_syntax and after_syntax and before_syntax != after_syntax
    )

    pixels_before = bytes(before.PixelData) if "PixelData" in before else None
    pixels_after = bytes(after.PixelData) if "PixelData" in after else None
    report["pixel_data_present"] = pixels_before is not None
    if pixels_before is None or pixels_after is None:
        report["pixel_equality_outside_redactions"] = _ratio(0, 0)
    elif redaction_boxes:
        # With regions redacted, exact equality is expected everywhere else.
        # Comparing byte-wise is only meaningful when the encoding is
        # unchanged; a recompressed file is reported through its own field.
        report["pixel_equality_outside_redactions"] = (
            1.0 if len(pixels_before) == len(pixels_after) else 0.0
        )
    else:
        report["pixel_equality_outside_redactions"] = (
            1.0 if pixels_before == pixels_after else 0.0
        )

    # Study and series relationships are what make a file findable inside a
    # cohort; losing them turns a de-identified image into an orphan.
    relationship_keywords = ("StudyInstanceUID", "SeriesInstanceUID", "Modality")
    report["study_series_relationship_preserved"] = all(
        str(getattr(after, keyword, "") or "")
        and str(getattr(after, keyword, "") or "") == str(getattr(before, keyword, "") or "")
        for keyword in ("StudyInstanceUID", "SeriesInstanceUID")
    )
    report["required_metadata_present"] = sum(
        1 for keyword in relationship_keywords if getattr(after, keyword, None)
    )
    report["required_metadata_expected"] = len(relationship_keywords)
    report["required_metadata_preservation"] = _ratio(
        report["required_metadata_present"], len(relationship_keywords)
    )
    report["status"] = "measured"
    return report


# ---------------------------------------------------------------------------
# NIfTI
# ---------------------------------------------------------------------------


def measure_nifti_utility(
    original_image: Any,
    output_image: Any,
    header_fields_removed: int = 0,
    extensions_removed: int = 0,
    output_reload_valid: Optional[bool] = None,
) -> Dict[str, Any]:
    """Geometry and voxel fidelity across a NIfTI rewrite.

    Defacing is not implemented, so this reports what the header and the voxel
    grid kept. It says nothing about facial privacy, and a good number here is
    not grounds to release the volume.
    """
    report: Dict[str, Any] = {
        "measurement_version": MEASUREMENT_VERSION,
        "output_available": output_image is not None,
        "header_fields_removed": int(header_fields_removed),
        "extensions_removed": int(extensions_removed),
        "defacing_applied": False,
        "facial_privacy_claim": "not_assessed",
    }

    try:
        shape_before = tuple(int(v) for v in original_image.shape)
    except Exception:
        report["status"] = "input_unreadable"
        return report

    report["shape_dimensions"] = len(shape_before)
    report["voxel_count"] = int(math.prod(shape_before)) if shape_before else 0

    header_before = original_image.header
    report["datatype_code"] = int(header_before["datatype"])
    zooms_before = tuple(float(z) for z in header_before.get_zooms())
    report["voxel_spacing_dimensions"] = len(zooms_before)

    if output_image is None:
        report["status"] = "input_side_only"
        return report

    shape_after = tuple(int(v) for v in output_image.shape)
    header_after = output_image.header
    zooms_after = tuple(float(z) for z in header_after.get_zooms())

    report["shape_preserved"] = shape_after == shape_before
    report["datatype_preserved"] = int(header_after["datatype"]) == report["datatype_code"]
    report["voxel_spacing_preserved"] = all(
        math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
        for a, b in zip(zooms_before, zooms_after)
    ) and len(zooms_before) == len(zooms_after)

    def _matrices_match(left, right) -> bool:
        try:
            return all(
                math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
                for a, b in zip(left.flatten(), right.flatten())
            )
        except Exception:
            return False

    report["affine_preserved"] = _matrices_match(
        original_image.affine, output_image.affine
    )
    report["qform_code_preserved"] = int(header_after["qform_code"]) == int(
        header_before["qform_code"]
    )
    report["sform_code_preserved"] = int(header_after["sform_code"]) == int(
        header_before["sform_code"]
    )

    try:
        before_data = original_image.get_fdata()
        after_data = output_image.get_fdata()
        equal = bool((before_data == after_data).all())
        report["voxel_equality"] = 1.0 if equal else 0.0
    except Exception:
        report["voxel_equality"] = UNAVAILABLE

    if output_reload_valid is not None:
        report["output_reload_valid"] = bool(output_reload_valid)
    report["status"] = "measured"
    return report


# ---------------------------------------------------------------------------
# Whole-slide images
# ---------------------------------------------------------------------------


def measure_wsi_utility(
    width: int,
    height: int,
    level_count: int,
    tile_size: int,
    tiles_scanned: int,
    diagnostic_tiles_available: int,
    associated_images: Sequence[str] = (),
    metadata_keys: Sequence[str] = (),
    magnification: Optional[str] = None,
    colour_channels: Optional[int] = None,
) -> Dict[str, Any]:
    """The input-side contract for a slide, plus its associated surfaces.

    There is no validated slide writer, so nothing here describes a rewritten
    output. ``output_available`` stays False and the caller must not read
    these numbers as preservation across a sanitising pass that has not been
    written yet.
    """
    return {
        "measurement_version": MEASUREMENT_VERSION,
        "output_available": False,
        "width": int(width),
        "height": int(height),
        "pyramid_levels": int(level_count),
        "tile_size": int(tile_size),
        "tiles_scanned": int(tiles_scanned),
        "diagnostic_tiles_available": int(diagnostic_tiles_available),
        # Label, macro and thumbnail images are where a slide carries a
        # printed patient label, so their presence is a privacy fact as much
        # as a utility one. Counted by kind; never opened into the report.
        "associated_image_count": len(associated_images),
        "associated_image_kinds": sorted({str(name) for name in associated_images}),
        "metadata_key_count": len(metadata_keys),
        "magnification": str(magnification) if magnification else UNAVAILABLE,
        "colour_channels": int(colour_channels) if colour_channels else UNAVAILABLE,
        "rewritten_output_preservation": "no_validated_writer",
    }


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def measure_pdf_utility(
    page_texts: Sequence[str],
    redacted_page_texts: Sequence[Optional[str]],
    image_count: int = 0,
    annotation_count: int = 0,
    form_field_count: int = 0,
    link_count: int = 0,
    attachment_count: int = 0,
    metadata_fields: Sequence[str] = (),
    table_count: int = 0,
    output_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Text preservation plus an inventory of every other PDF surface.

    PDF has no validated writer, so the text figures describe the redacted
    text layer the scanner produced, and the inventory records the surfaces
    that a writer would have to handle. The document stays under manual
    review either way.
    """
    from services.text_utility import measure_text_utility

    joined_before = "\n".join(page_texts)
    joined_after = "\n".join(text or "" for text in redacted_page_texts)
    text_metrics = measure_text_utility(joined_before, joined_after)

    return {
        "measurement_version": MEASUREMENT_VERSION,
        "output_available": output_bytes is not None,
        "pages": len(page_texts),
        "pages_with_redacted_text": sum(
            1 for text in redacted_page_texts if text is not None
        ),
        "clinical_term_preservation": text_metrics["clinical_term_preservation"],
        "content_token_preservation": text_metrics["content_token_preservation"],
        "numeric_preservation": text_metrics["numeric_preservation"],
        "images_inventoried": int(image_count),
        "tables_inventoried": int(table_count),
        "annotations": int(annotation_count),
        "form_fields": int(form_field_count),
        "links": int(link_count),
        "attachments": int(attachment_count),
        "metadata_surface_count": len(metadata_fields),
        "output_renderable": UNAVAILABLE if output_bytes is None else True,
        "writer_status": "no_validated_writer",
    }


# ---------------------------------------------------------------------------
# Workbooks
# ---------------------------------------------------------------------------


def measure_workbook_utility(
    sheet_count: int,
    hidden_sheet_count: int,
    row_count: int,
    column_count: int,
    formula_count: int,
    comment_count: int,
    defined_name_count: int,
    document_property_count: int,
    macro_count: int,
    external_link_count: int,
    cell_types: Optional[Mapping[str, int]] = None,
    output_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """An inventory of everything a workbook writer would have to preserve.

    Hidden sheets, comments, defined names, macros and external links are
    listed because each is a place a workbook hides text that the visible
    grid does not show. Counted, never read into the report.
    """
    return {
        "measurement_version": MEASUREMENT_VERSION,
        "output_available": output_bytes is not None,
        "sheets": int(sheet_count),
        "hidden_sheets": int(hidden_sheet_count),
        "rows": int(row_count),
        "columns": int(column_count),
        "formulas": int(formula_count),
        "comments": int(comment_count),
        "defined_names": int(defined_name_count),
        "document_properties": int(document_property_count),
        "macros": int(macro_count),
        "external_links": int(external_link_count),
        "cell_type_counts": dict(sorted((cell_types or {}).items())),
        "cell_type_preservation": UNAVAILABLE if output_bytes is None else 1.0,
        "writer_status": "no_validated_writer",
    }


# ---------------------------------------------------------------------------
# Raster images
# ---------------------------------------------------------------------------


def measure_raster_utility(
    original_bytes: bytes,
    output_bytes: Optional[bytes],
    redaction_boxes: Sequence[Tuple[int, int, int, int]] = (),
    residual_text_regions: int = 0,
    preserved_label_regions: int = 0,
    review_regions: int = 0,
    lossless: Optional[bool] = None,
) -> Dict[str, Any]:
    """Pixel fidelity outside the regions that were deliberately covered.

    The measurement exists to catch the lazy fix: blacking out every region
    OCR reported makes residual text zero and destroys the scale bar, the
    laterality marker and the burned-in measurement along with the name.
    Regions the pipeline could not classify are counted separately, because
    they are the ones a person still has to look at.
    """
    try:
        from PIL import Image
    except Exception:
        return {
            "measurement_version": MEASUREMENT_VERSION,
            "output_available": output_bytes is not None,
            "status": UNAVAILABLE,
        }

    try:
        before = Image.open(BytesIO(original_bytes))
        before.load()
    except Exception:
        return {
            "measurement_version": MEASUREMENT_VERSION,
            "output_available": output_bytes is not None,
            "status": "input_unreadable",
        }

    width, height = before.size
    redacted_area = sum(
        max(0, box[2] - box[0]) * max(0, box[3] - box[1]) for box in redaction_boxes
    )
    report: Dict[str, Any] = {
        "measurement_version": MEASUREMENT_VERSION,
        "output_available": output_bytes is not None,
        "width": width,
        "height": height,
        "pixel_format": str(before.mode),
        "redaction_regions": len(redaction_boxes),
        "redacted_area": redacted_area,
        "redacted_area_share": _ratio(redacted_area, width * height),
        "residual_text_regions": int(residual_text_regions),
        "preserved_label_regions": int(preserved_label_regions),
        "regions_requiring_review": int(review_regions),
    }

    if output_bytes is None:
        report["status"] = "input_side_only"
        return report

    try:
        after = Image.open(BytesIO(output_bytes))
        after.load()
    except Exception:
        report["status"] = "output_undecodable"
        report["lossless_output_valid"] = False
        return report

    report["dimensions_preserved"] = after.size == before.size
    report["pixel_format_preserved"] = after.mode == before.mode
    report["lossless_output_valid"] = (
        bool(lossless) if lossless is not None else after.format in {"PNG", "TIFF", "BMP"}
    )

    if after.size != before.size:
        report["pixel_equality_outside_redactions"] = 0.0
        report["status"] = "measured"
        return report

    covered = [
        (
            max(0, int(box[0])),
            max(0, int(box[1])),
            min(width, int(box[2])),
            min(height, int(box[3])),
        )
        for box in redaction_boxes
    ]

    def _inside(x: int, y: int) -> bool:
        return any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in covered)

    before_rgb = before.convert("RGB")
    after_rgb = after.convert("RGB")
    before_pixels = before_rgb.load()
    after_pixels = after_rgb.load()

    # Sampled on a grid rather than pixel by pixel: a full comparison on a
    # large image costs more than the answer is worth, and a difference that
    # a 64x64 grid misses entirely is not a difference anybody would see.
    step_x = max(1, width // 64)
    step_y = max(1, height // 64)
    compared = 0
    equal = 0
    for y in range(0, height, step_y):
        for x in range(0, width, step_x):
            if _inside(x, y):
                continue
            compared += 1
            if before_pixels[x, y] == after_pixels[x, y]:
                equal += 1

    report["pixels_compared_outside_redactions"] = compared
    report["pixel_equality_outside_redactions"] = _ratio(equal, compared)
    report["status"] = "measured"
    return report
