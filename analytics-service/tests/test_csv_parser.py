import pytest
from fastapi import HTTPException

from app.utils.csv_parser import (
    decode_csv_content,
    detect_delimiter,
    parse_csv,
    validate_file_size,
    MAX_FILE_SIZE_BYTES,
)


class TestValidateFileSize:

    def test_accepts_small_file(self):
        validate_file_size(b"x" * 100)

    def test_rejects_oversized(self):
        with pytest.raises(HTTPException) as exc:
            validate_file_size(b"x" * (MAX_FILE_SIZE_BYTES + 1))
        assert exc.value.status_code == 413


class TestDecodeCSVContent:

    def test_utf8(self):
        result = decode_csv_content("name,age\nAlice,25\n".encode("utf-8"))
        assert "Alice" in result

    def test_iso_fallback(self):
        result = decode_csv_content("name,age\nRené,30\n".encode("iso-8859-1"))
        assert "30" in result


class TestDetectDelimiter:

    def test_comma(self):
        assert detect_delimiter("a,b,c\n1,2,3\n") == ","

    def test_tab(self):
        assert detect_delimiter("a\tb\tc\n1\t2\t3\n") == "\t"

    def test_semicolon(self):
        assert detect_delimiter("a;b;c\n1;2;3\n") == ";"

    def test_fallback(self):
        assert detect_delimiter("") == ","


class TestParseCSV:

    def test_valid(self, sample_csv_bytes):
        df = parse_csv(sample_csv_bytes)
        assert len(df) == 5
        assert list(df.columns) == ["age", "glucose", "cholesterol"]

    def test_missing_values(self, sample_csv_with_missing):
        df = parse_csv(sample_csv_with_missing)
        assert df["glucose"].isna().sum() >= 1

    def test_empty_raises(self):
        with pytest.raises(HTTPException) as exc:
            parse_csv(b"")
        assert exc.value.status_code == 400

    def test_oversized_raises(self):
        big = b"a,b\n" + b"1,2\n" * (MAX_FILE_SIZE_BYTES // 4 + 1)
        with pytest.raises(HTTPException) as exc:
            parse_csv(big)
        assert exc.value.status_code == 413
