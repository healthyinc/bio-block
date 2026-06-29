

import csv
import io

import pandas as pd
from pandas.errors import EmptyDataError
from fastapi import HTTPException

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_ROWS = 10_000
SUPPORTED_ENCODINGS = ("utf-8", "iso-8859-1")

# Excel magic bytes: XLSX (PK zip) and XLS (OLE2 compound doc)
_XLSX_MAGIC = b"PK"
_XLS_MAGIC = b"\xd0\xcf\x11\xe0"


def validate_file_size(content: bytes) -> None:
    if len(content) > MAX_FILE_SIZE_BYTES:
        size_mb = round(len(content) / (1024 * 1024), 1)
        raise HTTPException(413, f"File is {size_mb}MB — max is 10MB.")


def _is_excel(content: bytes) -> bool:
    """Detect whether *content* is an Excel file (.xlsx or .xls) by magic bytes."""
    return content[:2] == _XLSX_MAGIC or content[:4] == _XLS_MAGIC


def decode_csv_content(content: bytes) -> str:
    for encoding in SUPPORTED_ENCODINGS:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise HTTPException(400, "Unable to decode CSV. Supported: UTF-8, ISO-8859-1.")


def detect_delimiter(text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(text[:4096])
        return dialect.delimiter
    except csv.Error:
        return ","


def _parse_excel(content: bytes) -> pd.DataFrame:
    """Parse an Excel file (.xlsx or .xls) from raw bytes."""
    try:
        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
    except Exception:
        # Fallback to xlrd for older .xls files
        try:
            df = pd.read_excel(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(
                400,
                f"Unable to parse Excel file: {exc}. "
                "Ensure the file is a valid .xlsx or .xls file.",
            )
    return df


def _parse_csv_text(content: bytes) -> pd.DataFrame:
    """Parse a CSV file from raw bytes."""
    text = decode_csv_content(content)
    sep = detect_delimiter(text)

    try:
        df = pd.read_csv(
            io.StringIO(text), sep=sep, on_bad_lines="skip", engine="python"
        )
    except EmptyDataError:
        raise HTTPException(400, "CSV is empty or has no parseable data.")

    return df


def parse_csv(content: bytes) -> pd.DataFrame:
    """Parse an uploaded file (CSV or Excel) into a DataFrame.

    The function auto-detects Excel files by magic bytes and routes
    to the appropriate parser.  The name ``parse_csv`` is kept for
    backward compatibility with existing callers.
    """
    validate_file_size(content)

    if _is_excel(content):
        df = _parse_excel(content)
    else:
        df = _parse_csv_text(content)

    if len(df) > MAX_ROWS:
        raise HTTPException(413, f"Dataset has {len(df):,} rows — max is {MAX_ROWS:,}.")

    if df.empty:
        raise HTTPException(400, "File is empty or has no parseable data.")

    # Strip whitespace from column names to prevent "not found" mismatches
    df.columns = df.columns.str.strip()

    return df
