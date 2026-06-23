import json
import math
import os
import sys

from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ocr_redaction import OCRBox  # noqa: E402
from services.wsi_tiling import (  # noqa: E402
    TileCoordinate,
    generate_priority_tiles,
    map_tile_boxes_to_slide,
    scan_wsi_bytes,
    scan_wsi_slide,
)


class FakeOCRBackend:
    ocr_engine_status = "available"

    def __init__(self, boxes=None):
        self.boxes = boxes or []

    def detect_text_boxes(self, image):
        return self.boxes


class FakeSlide:
    def __init__(self, width, height):
        self.dimensions = (width, height)
        self.read_regions = []

    def read_region(self, location, level, size):
        self.read_regions.append((location, level, size))
        return Image.new("RGB", size, "white")


def test_priority_tile_generator_returns_corners_and_borders_first():
    tiles = generate_priority_tiles(4096, 3072, tile_size=1024)

    assert [tile.priority_region for tile in tiles[:4]] == [
        "corner_top_left",
        "corner_top_right",
        "corner_bottom_left",
        "corner_bottom_right",
    ]
    assert "top_border" in [tile.priority_region for tile in tiles]
    assert "bottom_border" in [tile.priority_region for tile in tiles]
    assert "left_border" in [tile.priority_region for tile in tiles]
    assert "right_border" in [tile.priority_region for tile in tiles]


def test_tile_coordinates_stay_inside_image_bounds():
    width = 2500
    height = 1800
    tiles = generate_priority_tiles(width, height, tile_size=700)

    assert tiles
    for tile in tiles:
        assert tile.x >= 0
        assert tile.y >= 0
        assert tile.width > 0
        assert tile.height > 0
        assert tile.x + tile.width <= width
        assert tile.y + tile.height <= height


def test_large_wsi_scan_reads_priority_tiles_not_full_image():
    slide = FakeSlide(10000, 8000)
    result = scan_wsi_slide(
        slide,
        ocr_backend=FakeOCRBackend(),
        tile_size=1024,
    )

    assert result["pixel_redaction_status"] == "redaction_plan_ready"
    assert result["tiles_scanned"] == len(slide.read_regions)
    full_tile_count = math.ceil(10000 / 1024) * math.ceil(8000 / 1024)
    assert result["tiles_scanned"] < full_tile_count
    for _location, _level, size in slide.read_regions:
        assert size[0] <= 1024
        assert size[1] <= 1024


def test_tile_ocr_boxes_map_to_global_coordinates():
    tile = TileCoordinate(
        x=1024,
        y=2048,
        width=512,
        height=512,
        priority_region="top_border",
    )
    mapped = map_tile_boxes_to_slide(
        tile,
        [OCRBox("SLIDE_LABEL", 0.95, 10, 20, 100, 30)],
    )

    assert mapped == [
        OCRBox("SLIDE_LABEL", 0.95, 1034, 2068, 100, 30),
    ]


def test_wsi_scan_is_honest_when_rewrite_is_not_supported():
    result = scan_wsi_slide(
        FakeSlide(2048, 2048),
        ocr_backend=FakeOCRBackend([OCRBox("WSI_LABEL", 0.99, 0, 0, 50, 20)]),
        tile_size=1024,
    )
    response_text = json.dumps(result)

    assert result["pixel_redaction_status"] == "redaction_plan_ready"
    assert result["wsi_rewrite_status"] == "not_supported_yet"
    assert result["ocr_boxes_detected"] > 0
    assert result["boxes_redacted"] == 0
    assert result["redaction_plan_boxes"] == result["ocr_boxes_detected"]
    assert "WSI_LABEL" not in response_text


def test_wsi_bytes_returns_honest_status_without_openslide(monkeypatch):
    monkeypatch.setattr("services.wsi_tiling._load_openslide_module", lambda: None)

    result = scan_wsi_bytes(
        b"not a real slide",
        filename="slide.svs",
        ocr_backend=FakeOCRBackend(),
    )

    assert result["pixel_redaction_status"] == "redaction_plan_unavailable"
    assert result["wsi_rewrite_status"] == "not_supported_yet"
    assert result["tiles_scanned"] == 0
