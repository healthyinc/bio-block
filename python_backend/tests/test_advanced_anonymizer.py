import os
import sys
import pytest
import pydicom
import pytesseract
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
from reportlab.pdfgen import canvas

# Add parent directory to path to import advanced_anonymizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_anonymizer import anonymize_dicom, anonymize_pdf  # noqa: E402

# HARDCODED PATHS AS REQUESTED
POPPLER_PATH = (
    r'C:\Users\Titas Ghosh\Downloads\Release-25.12.0-0\poppler-25.12.0'
    r'\Library\bin'
)
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set tesseract cmd
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


@pytest.fixture
def test_dcm_path(tmp_path):
    """Generates a dummy DICOM file."""
    file_path = tmp_path / "test.dcm"

    # Create minimal DICOM
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.2")
    file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
    file_meta.TransferSyntaxUID = UID("1.2.840.10008.1.2.1")

    ds = FileDataset(
        str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128
    )
    ds.PatientName = "John^Doe"
    ds.PatientID = "123456"
    ds.PatientBirthDate = "20000101"

    ds.save_as(str(file_path))
    return str(file_path)


@pytest.fixture
def test_pdf_path(tmp_path):
    """Generates a dummy PDF file."""
    file_path = tmp_path / "test.pdf"

    c = canvas.Canvas(str(file_path))
    c.drawString(100, 750, "This is a test PDF.")
    c.drawString(100, 700, "Patient Name: John Doe")
    c.drawString(100, 650, "Patient ID: 123456")
    c.save()

    return str(file_path)


def test_dicom_anonymization(test_dcm_path, tmp_path):
    output_path = tmp_path / "anonymized_test.dcm"

    anonymize_dicom(str(test_dcm_path), str(output_path))

    assert os.path.exists(output_path)

    ds = pydicom.dcmread(str(output_path))
    assert ds.PatientName == "ANONYMIZED"
    assert ds.PatientID == "ANONYMIZED"
    assert ds.PatientBirthDate == "ANONYMIZED"


def test_pdf_anonymization(test_pdf_path, tmp_path):
    output_path = tmp_path / "anonymized_test.pdf"

    # Check if poppler path exists, if not skip or fail with message
    if not os.path.exists(POPPLER_PATH):
        pytest.fail(f"Poppler path not found: {POPPLER_PATH}")

    anonymize_pdf(
        str(test_pdf_path), str(output_path), poppler_path=POPPLER_PATH
    )

    assert os.path.exists(output_path)
