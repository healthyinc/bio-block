"""PDF surface inventory and PHI scanning.

A PDF carries PHI on far more surfaces than the visible text layer: document
info and XMP metadata, annotations, form-field values, embedded attachments,
bookmarks, link targets, and raster images. Extracting only ``page.get_text()``
and reporting zero entities is a fail-open answer, because an image-only page
looks identical to a clean one.

This module inventories every surface it can read, scans the readable text with
the same typed detector pipeline the TXT path uses, and records every surface it
could not read. Two distinct conclusions come out of that:

* ``text_layer_complete`` - every text surface was read and scanned, and the
  redacted result carried no residual PHI. Only then is redacted page text
  returned at all.
* ``fully_scannable`` - the above, and no unreadable surface exists (no raster
  pixels, embedded files, active content, or encryption).

There is no validated PDF writer yet, so a PDF is never releasable: the best
outcome is ``manual_review_required`` with a complete inventory attached.
Original bytes are never returned, and only categories and counts are reported,
never values.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from services.text_anonymization import (
    MAX_TEXT_BYTES,
    TextAnonymizationError,
    anonymize_clinical_text,
    residual_phi_categories,
)

MAX_PDF_BYTES = 32 * 1024 * 1024
MAX_PDF_PAGES = 500
# Per-surface budget, well under the text pipeline's own limit.
MAX_SURFACE_TEXT_BYTES = min(128 * 1024, MAX_TEXT_BYTES)

STATUS_MANUAL_REVIEW = "manual_review_required"
STATUS_UNSCANNABLE = "unsupported_or_unscannable"

REASON_READER_UNAVAILABLE = "pdf_reader_unavailable"
REASON_UNPARSEABLE = "pdf_unparseable"
REASON_ENCRYPTED = "pdf_encrypted"
REASON_PAGE_LIMIT = "pdf_page_limit_exceeded"
REASON_RASTER_REQUIRES_OCR = "pdf_raster_requires_ocr"
REASON_EMBEDDED_FILES = "pdf_embedded_files_unscannable"
REASON_ACTIVE_CONTENT = "pdf_active_content_present"
REASON_SURFACE_TOO_LARGE = "pdf_surface_exceeds_scan_limit"
REASON_RESIDUAL_PHI = "privacy_requirements_not_met"
REASON_NO_VALIDATED_WRITER = "pdf_validated_writer_unavailable"

# Distinctive markers for embedded scripting, launch actions, or attachments.
# Their presence alone blocks; we do not try to reason about what they do.
# Short tokens such as "/JS" and "/AA" are deliberately excluded: they collide
# with ordinary bytes inside compressed streams often enough to be meaningless.
_BLOCKING_CONTENT_TOKENS = (
    b"/JavaScript",
    b"/Launch",
    b"/RichMedia",
    b"/EmbeddedFile",
    b"/Movie",
)
# Reported for operator context, but not blocking on their own.
_ADVISORY_CONTENT_TOKENS = (
    b"/OpenAction",
    b"/AcroForm",
)
_PDF_MAGIC = b"%PDF-"

_METADATA_TEXT_KEYS = (
    "title",
    "author",
    "subject",
    "keywords",
    "creator",
    "producer",
)

_XML_TAG = re.compile(r"<[^>]+>")


class DocumentSanitizationError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _load_pymupdf():
    try:
        import fitz  # PyMuPDF

        return fitz
    except Exception:
        return None


def _strip_xml(xmp: str) -> str:
    return _XML_TAG.sub(" ", xmp)


def _found_tokens(raw: bytes, tokens: Tuple[bytes, ...]) -> List[str]:
    return sorted(
        token.decode("ascii").lstrip("/") for token in tokens if token in raw
    )


def _merge_counts(target: Dict[str, int], addition: Dict[str, int]) -> None:
    for key, value in addition.items():
        target[key] = target.get(key, 0) + value


def _blocked(
    reasons: List[str],
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a fully blocked result. Never carries text or original bytes."""
    return {
        "handler": "scan_pdf",
        "routing_status": "handler_selected",
        "anonymization_status": STATUS_UNSCANNABLE,
        "message": "PDF could not be scanned and is not releasable.",
        "pdf_summary": summary or {},
        "unscannable_reasons": sorted(set(reasons + [REASON_NO_VALIDATED_WRITER])),
        "detected_entities": {},
        "entity_count": 0,
        "detection_sources": {},
        "residual_phi_categories": {},
        "scannable": False,
        "text_layer_complete": False,
        "pages": [],
    }


class _SurfaceScanner:
    """Accumulates category counts across every readable text surface."""

    def __init__(self, profile: str):
        self.profile = profile
        self.counts: Dict[str, int] = {}
        self.sources: Dict[str, int] = {}
        self.reasons: List[str] = []
        self.complete = True

    def scan(self, text: str) -> Optional[str]:
        """Scan one surface. Returns redacted text, or None if unreadable."""
        if not text or not text.strip():
            return ""
        if len(text.encode("utf-8")) > MAX_SURFACE_TEXT_BYTES:
            self.reasons.append(REASON_SURFACE_TOO_LARGE)
            self.complete = False
            return None
        result = anonymize_clinical_text(text, profile=self.profile)
        _merge_counts(self.counts, result["detected_entities"])
        _merge_counts(self.sources, result["detection_sources"])
        return result["anonymized_text"]


def _page_surfaces(page) -> Tuple[str, List[str], int]:
    """Return (page text, annotation/widget/link texts, raster image count).

    Any read failure propagates: an unreadable surface must not look empty.
    """
    page_text = page.get_text("text") or ""
    extras: List[str] = []

    for annotation in page.annots() or []:
        info = annotation.info or {}
        for key in ("content", "title", "subject"):
            value = info.get(key)
            if value:
                extras.append(str(value))

    for widget in page.widgets() or []:
        for value in (widget.field_name, widget.field_value):
            if value:
                extras.append(str(value))

    for link in page.get_links() or []:
        target = link.get("uri") or link.get("file")
        if target:
            extras.append(str(target))

    image_count = len(page.get_images(full=True) or [])
    return page_text, extras, image_count


def scan_pdf_bytes(content: bytes, profile: str = "strict") -> Dict[str, Any]:
    """Inventory a PDF's PHI-bearing surfaces and scan the readable ones."""
    if not isinstance(content, (bytes, bytearray)):
        raise DocumentSanitizationError("PDF content must be bytes", status_code=500)
    if not content:
        raise DocumentSanitizationError("Uploaded PDF is empty")
    if len(content) > MAX_PDF_BYTES:
        raise DocumentSanitizationError(
            f"PDF uploads must be {MAX_PDF_BYTES} bytes or smaller",
            status_code=413,
        )
    raw = bytes(content)
    if not raw[:1024].lstrip().startswith(_PDF_MAGIC):
        raise DocumentSanitizationError("File is not a PDF document")

    fitz = _load_pymupdf()
    if fitz is None:
        return _blocked([REASON_READER_UNAVAILABLE])

    blocking_tokens = _found_tokens(raw, _BLOCKING_CONTENT_TOKENS)
    advisory_tokens = _found_tokens(raw, _ADVISORY_CONTENT_TOKENS)

    scanner = _SurfaceScanner(profile)
    unreadable: List[str] = []
    if blocking_tokens:
        unreadable.append(REASON_ACTIVE_CONTENT)

    document = None
    try:
        try:
            document = fitz.open(stream=raw, filetype="pdf")
        except Exception:
            return _blocked([REASON_UNPARSEABLE] + unreadable)

        if document.needs_pass or document.is_encrypted:
            return _blocked([REASON_ENCRYPTED] + unreadable)

        page_count = document.page_count
        if page_count > MAX_PDF_PAGES:
            return _blocked(
                [REASON_PAGE_LIMIT] + unreadable, {"page_count": page_count}
            )

        try:
            embedded_count = document.embfile_count()
        except Exception:
            embedded_count = 0
            unreadable.append(REASON_UNPARSEABLE)
        if embedded_count:
            unreadable.append(REASON_EMBEDDED_FILES)

        pages: List[Dict[str, Any]] = []
        text_pages = 0
        image_only_pages = 0
        raster_pages = 0
        annotation_surfaces = 0

        for index in range(page_count):
            try:
                page_text, extras, image_count = _page_surfaces(document[index])
            except Exception:
                return _blocked(
                    [REASON_UNPARSEABLE] + unreadable, {"page_count": page_count}
                )

            annotation_surfaces += len(extras)
            has_text = bool(page_text.strip())
            if has_text:
                text_pages += 1
            if image_count:
                # Raster pixels are not scanned here; burned-in PHI would
                # survive. Pixel handling is a separate hardening step.
                raster_pages += 1
                unreadable.append(REASON_RASTER_REQUIRES_OCR)
                if not has_text:
                    image_only_pages += 1

            redacted = scanner.scan(page_text)
            for extra in extras:
                if scanner.scan(extra) is None:
                    redacted = None

            pages.append(
                {
                    "page_number": index + 1,
                    "has_text_layer": has_text,
                    "image_count": image_count,
                    "annotation_surfaces": len(extras),
                    "redacted_text": redacted,
                }
            )

        try:
            metadata = document.metadata or {}
        except Exception:
            metadata = {}
            unreadable.append(REASON_UNPARSEABLE)
            scanner.complete = False
        metadata_fields_present = sorted(
            key for key in _METADATA_TEXT_KEYS if str(metadata.get(key) or "").strip()
        )
        for key in metadata_fields_present:
            scanner.scan(str(metadata[key]))

        xmp_present = False
        try:
            xmp = document.get_xml_metadata() or ""
            xmp_present = bool(xmp.strip())
            if xmp_present:
                scanner.scan(_strip_xml(xmp))
        except Exception:
            unreadable.append(REASON_UNPARSEABLE)
            scanner.complete = False

        bookmark_entries = 0
        try:
            for entry in document.get_toc() or []:
                bookmark_entries += 1
                if len(entry) > 1 and entry[1]:
                    scanner.scan(str(entry[1]))
        except Exception:
            unreadable.append(REASON_UNPARSEABLE)
            scanner.complete = False

        summary = {
            "page_count": page_count,
            "text_pages": text_pages,
            "raster_pages": raster_pages,
            "image_only_pages": image_only_pages,
            "annotation_surfaces": annotation_surfaces,
            "embedded_file_count": embedded_count,
            "metadata_fields_present": metadata_fields_present,
            "xmp_metadata_present": xmp_present,
            "bookmark_entries": bookmark_entries,
            "blocking_content_indicators": blocking_tokens,
            "advisory_content_indicators": advisory_tokens,
        }
    finally:
        if document is not None:
            try:
                document.close()
            except Exception:
                pass

    text_layer_complete = scanner.complete and all(
        page["redacted_text"] is not None for page in pages
    )

    residual: Dict[str, int] = {}
    if text_layer_complete:
        for page in pages:
            _merge_counts(residual, residual_phi_categories(page["redacted_text"] or ""))
        if residual:
            # Redaction did not clear the text layer; withhold the output.
            text_layer_complete = False
            unreadable.append(REASON_RESIDUAL_PHI)

    reasons = sorted(set(unreadable + scanner.reasons))
    fully_scannable = not reasons and text_layer_complete
    # Even a fully scannable PDF is not releasable: there is no validated
    # writer that can produce sanitized PDF bytes yet.
    reasons = sorted(set(reasons + [REASON_NO_VALIDATED_WRITER]))

    return {
        "handler": "scan_pdf",
        "routing_status": "handler_selected",
        "anonymization_status": (
            STATUS_MANUAL_REVIEW if fully_scannable else STATUS_UNSCANNABLE
        ),
        "message": (
            "PDF surfaces were inventoried and the readable text was scanned. "
            "No validated PDF writer is available, so the document is not "
            "automatically releasable."
        ),
        "pdf_summary": summary,
        "unscannable_reasons": reasons,
        "detected_entities": scanner.counts,
        "entity_count": sum(scanner.counts.values()),
        "detection_sources": scanner.sources,
        "residual_phi_categories": residual,
        "scannable": fully_scannable,
        "text_layer_complete": text_layer_complete,
        "pages": [
            {
                "page_number": page["page_number"],
                "has_text_layer": page["has_text_layer"],
                "image_count": page["image_count"],
                "annotation_surfaces": page["annotation_surfaces"],
                # Redacted text is withheld entirely unless every text surface
                # in the document was read and came back clear, so a partial
                # scan can never be mistaken for a complete one.
                "anonymized_text": (
                    page["redacted_text"] if text_layer_complete else None
                ),
            }
            for page in pages
        ],
    }


def scan_pdf_for_ingestion(content: bytes, profile: str) -> Dict[str, Any]:
    """Ingestion-facing wrapper that normalizes detector failures to blocks."""
    try:
        return scan_pdf_bytes(content, profile=profile)
    except TextAnonymizationError as exc:
        # Detector failures block; they never degrade to unscanned output.
        raise DocumentSanitizationError(
            exc.detail, status_code=exc.status_code
        ) from exc
