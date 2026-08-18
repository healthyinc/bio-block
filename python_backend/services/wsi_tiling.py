from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from services.ocr_redaction import (
    OCRBackend,
    OCRBox,
    OCREngineUnavailable,
    get_default_ocr_backend,
)


DEFAULT_TILE_SIZE = 1024
DEFAULT_BORDER_FRACTION = 0.15


@dataclass(frozen=True)
class TileCoordinate:
    x: int
    y: int
    width: int
    height: int
    priority_region: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "priority_region": self.priority_region,
        }


def generate_priority_tiles(
    width: int,
    height: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    border_fraction: float = DEFAULT_BORDER_FRACTION,
) -> List[TileCoordinate]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("Tile size must be positive")
    if border_fraction <= 0:
        raise ValueError("Border fraction must be positive")

    tiles: List[TileCoordinate] = []
    seen = set()

    def add_tile(x: int, y: int, region: str):
        clamped_x = _clamp_start(x, width, tile_size)
        clamped_y = _clamp_start(y, height, tile_size)
        tile_width = min(tile_size, width - clamped_x)
        tile_height = min(tile_size, height - clamped_y)
        key = (clamped_x, clamped_y, tile_width, tile_height)
        if key in seen:
            return
        seen.add(key)
        tiles.append(
            TileCoordinate(
                x=clamped_x,
                y=clamped_y,
                width=tile_width,
                height=tile_height,
                priority_region=region,
            )
        )

    right_x = max(0, width - tile_size)
    bottom_y = max(0, height - tile_size)

    add_tile(0, 0, "corner_top_left")
    add_tile(right_x, 0, "corner_top_right")
    add_tile(0, bottom_y, "corner_bottom_left")
    add_tile(right_x, bottom_y, "corner_bottom_right")

    border_depth_x = max(1, int(round(width * border_fraction)))
    border_depth_y = max(1, int(round(height * border_fraction)))
    x_positions = _axis_tile_positions(width, tile_size)
    y_positions = _axis_tile_positions(height, tile_size)

    for y in _leading_border_positions(height, tile_size, border_depth_y):
        for x in x_positions:
            add_tile(x, y, "top_border")

    for y in _trailing_border_positions(height, tile_size, border_depth_y):
        for x in x_positions:
            add_tile(x, y, "bottom_border")

    for x in _leading_border_positions(width, tile_size, border_depth_x):
        for y in y_positions:
            add_tile(x, y, "left_border")

    for x in _trailing_border_positions(width, tile_size, border_depth_x):
        for y in y_positions:
            add_tile(x, y, "right_border")

    return tiles


def map_tile_boxes_to_slide(
    tile: TileCoordinate,
    boxes: Iterable[OCRBox],
) -> List[OCRBox]:
    return [
        OCRBox(
            text=box.text,
            confidence=box.confidence,
            x=tile.x + box.x,
            y=tile.y + box.y,
            width=box.width,
            height=box.height,
        )
        for box in boxes
    ]


def scan_wsi_slide(
    slide: Any,
    ocr_backend: Optional[OCRBackend] = None,
    tile_size: int = DEFAULT_TILE_SIZE,
    border_fraction: float = DEFAULT_BORDER_FRACTION,
) -> Dict[str, Any]:
    width, height = _slide_dimensions(slide)
    tiles = generate_priority_tiles(
        width=width,
        height=height,
        tile_size=tile_size,
        border_fraction=border_fraction,
    )

    backend = ocr_backend or get_default_ocr_backend()
    engine_status = _backend_status(backend)
    if engine_status == "unavailable":
        return _wsi_result(
            pixel_redaction_status="skipped_ocr_unavailable",
            ocr_engine_status=engine_status,
            image_dimensions={"width": width, "height": height},
            tile_size=tile_size,
        )

    boxes_detected = 0
    tiles_scanned = 0
    priority_regions: List[str] = []

    try:
        for tile in tiles:
            tile_image = slide.read_region(
                (tile.x, tile.y),
                0,
                (tile.width, tile.height),
            )
            if isinstance(tile_image, Image.Image) and tile_image.mode == "RGBA":
                tile_image = tile_image.convert("RGB")

            boxes = list(backend.detect_text_boxes(tile_image))
            boxes_detected += len(map_tile_boxes_to_slide(tile, boxes))
            tiles_scanned += 1
            if tile.priority_region not in priority_regions:
                priority_regions.append(tile.priority_region)
    except OCREngineUnavailable:
        return _wsi_result(
            pixel_redaction_status="skipped_ocr_unavailable",
            ocr_engine_status="unavailable",
            image_dimensions={"width": width, "height": height},
            tile_size=tile_size,
            tiles_scanned=tiles_scanned,
            priority_regions_scanned=priority_regions,
        )
    except Exception:
        return _wsi_result(
            pixel_redaction_status="ocr_failed",
            ocr_engine_status="error",
            image_dimensions={"width": width, "height": height},
            tile_size=tile_size,
            tiles_scanned=tiles_scanned,
            priority_regions_scanned=priority_regions,
        )

    return _wsi_result(
        pixel_redaction_status="redaction_plan_ready",
        ocr_boxes_detected=boxes_detected,
        boxes_redacted=0,
        redaction_plan_boxes=boxes_detected,
        ocr_engine_status=engine_status,
        image_dimensions={"width": width, "height": height},
        tile_size=tile_size,
        tiles_scanned=tiles_scanned,
        priority_regions_scanned=priority_regions,
    )


def scan_wsi_bytes(
    file_bytes: bytes,
    filename: str,
    ocr_backend: Optional[OCRBackend] = None,
    tile_size: int = DEFAULT_TILE_SIZE,
    border_fraction: float = DEFAULT_BORDER_FRACTION,
) -> Dict[str, Any]:
    openslide_module = _load_openslide_module()
    if openslide_module is None:
        return _wsi_result(
            pixel_redaction_status="redaction_plan_unavailable",
            ocr_engine_status=_backend_status(ocr_backend or get_default_ocr_backend()),
            tile_size=tile_size,
        )

    suffix = _safe_suffix(filename)
    temp_path = None
    try:
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix="bioblock_wsi_",
            delete=False,
        )
        temp_path = temp_file.name
        with temp_file:
            temp_file.write(file_bytes)

        slide = openslide_module.OpenSlide(temp_path)
    except Exception:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        return _wsi_result(
            pixel_redaction_status="unsupported_wsi_format",
            ocr_engine_status=_backend_status(ocr_backend or get_default_ocr_backend()),
            tile_size=tile_size,
        )

    try:
        return scan_wsi_slide(
            slide=slide,
            ocr_backend=ocr_backend,
            tile_size=tile_size,
            border_fraction=border_fraction,
        )
    finally:
        try:
            slide.close()
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


def _axis_tile_positions(length: int, tile_size: int) -> List[int]:
    if length <= tile_size:
        return [0]

    positions = list(range(0, length, tile_size))
    final_start = max(0, length - tile_size)
    if positions[-1] != final_start:
        positions.append(final_start)
    return positions


def _leading_border_positions(length: int, tile_size: int, depth: int) -> List[int]:
    positions = []
    current = 0
    limit = min(depth, length)
    while current < limit:
        positions.append(_clamp_start(current, length, tile_size))
        current += tile_size
    return _dedupe_preserve_order(positions)


def _trailing_border_positions(length: int, tile_size: int, depth: int) -> List[int]:
    positions = []
    current = _clamp_start(length - tile_size, length, tile_size)
    limit = max(0, length - depth)
    while current + min(tile_size, length - current) > limit:
        positions.append(_clamp_start(current, length, tile_size))
        current -= tile_size
        if current < 0:
            break
    return _dedupe_preserve_order(positions)


def _clamp_start(start: int, length: int, tile_size: int) -> int:
    return max(0, min(int(start), max(0, length - tile_size)))


def _dedupe_preserve_order(values: Sequence[int]) -> List[int]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _slide_dimensions(slide: Any) -> Tuple[int, int]:
    dimensions = getattr(slide, "dimensions", None)
    if not dimensions or len(dimensions) != 2:
        raise ValueError("WSI slide must expose width and height dimensions")
    return int(dimensions[0]), int(dimensions[1])


def _backend_status(backend: OCRBackend) -> str:
    status = getattr(backend, "ocr_engine_status", "available")
    if callable(status):
        status = status()
    return str(status or "available")


def _load_openslide_module():
    try:
        import openslide
    except Exception:
        return None
    return openslide


def _safe_suffix(filename: str) -> str:
    _, extension = os.path.splitext(filename or "")
    if not extension:
        return ".svs"
    return extension.lower()


def _wsi_result(
    pixel_redaction_status: str,
    ocr_boxes_detected: int = 0,
    boxes_redacted: int = 0,
    redaction_plan_boxes: int = 0,
    tiles_scanned: int = 0,
    priority_regions_scanned: Optional[List[str]] = None,
    image_dimensions: Optional[Dict[str, int]] = None,
    tile_size: int = DEFAULT_TILE_SIZE,
    ocr_engine_status: str = "not_applicable",
) -> Dict[str, Any]:
    return {
        "pixel_redaction_status": pixel_redaction_status,
        "ocr_boxes_detected": ocr_boxes_detected,
        "boxes_redacted": boxes_redacted,
        "redaction_plan_boxes": redaction_plan_boxes,
        "tiles_scanned": tiles_scanned,
        "priority_regions_scanned": priority_regions_scanned or [],
        "image_dimensions": image_dimensions,
        "tile_size": tile_size,
        "ocr_engine_status": ocr_engine_status,
        "wsi_rewrite_status": "not_supported_yet",
    }
