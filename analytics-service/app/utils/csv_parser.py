

import csv
import io

import pandas as pd
from pandas.errors import EmptyDataError
from fastapi import HTTPException

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_ROWS = 10_000
SUPPORTED_ENCODINGS = ("utf-8", "iso-8859-1")


def validate_file_size(content: bytes) -> None:
    if len(content) > MAX_FILE_SIZE_BYTES:
        size_mb = round(len(content) / (1024 * 1024), 1)
        raise HTTPException(413, f"File is {size_mb}MB — max is 10MB.")


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


def parse_csv(content: bytes) -> pd.DataFrame:
    validate_file_size(content)
    text = decode_csv_content(content)
    sep = detect_delimiter(text)

    try:
        df = pd.read_csv(
            io.StringIO(text), sep=sep, on_bad_lines="skip", engine="python"
        )
    except EmptyDataError:
        raise HTTPException(400, "CSV is empty or has no parseable data.")

    if len(df) > MAX_ROWS:
        raise HTTPException(413, f"Dataset has {len(df):,} rows — max is {MAX_ROWS:,}.")

    if df.empty:
        raise HTTPException(400, "CSV is empty or has no parseable data.")

    return df
