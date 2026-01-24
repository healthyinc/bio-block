import os
import shutil
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

# Dynamic path resolution config
# 1. Environment variables
# 2. System PATH
# 3. Known fallbacks (for specific local setups)

POPPLER_PATH = os.getenv("POPPLER_PATH")
TESSERACT_PATH = os.getenv("TESSERACT_PATH")

# Fallback for Tesseract
if not TESSERACT_PATH:
    if shutil.which("tesseract"):
        TESSERACT_PATH = "tesseract"
    elif os.name == 'nt' and os.path.exists(
        r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ):
        TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Fallback for Poppler (Windows specific usually)
if not POPPLER_PATH:
    # On Windows, poppler is often not in main PATH.
    # Check known local fallback
    local_fallback = (
        r'C:\Users\Titas Ghosh\Downloads\Release-25.12.0-0\poppler-25.12.0'
        r'\Library\bin'
    )
    if os.name == 'nt' and os.path.exists(local_fallback):
        POPPLER_PATH = local_fallback

# Configure Tesseract if found
if TESSERACT_PATH:
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


@pytest.mark.skipif(
    not POPPLER_PATH and not shutil.which("pdftoppm"),
    reason="Poppler not found in PATH or POPPLER_PATH env var"
)
def test_pdf_anonymization(test_pdf_path, tmp_path):
    output_path = tmp_path / "anonymized_test.pdf"

    # poppler_path argument is only needed if it's not in the system PATH
    # If shutil.which("pdftoppm") is True, we can pass None.
    # Otherwise, pass the resolved POPPLER_PATH.
    use_poppler_path = POPPLER_PATH
    if shutil.which("pdftoppm"):
        use_poppler_path = None

    anonymize_pdf(
        str(test_pdf_path), str(output_path), poppler_path=use_poppler_path
    )

    assert os.path.exists(output_path)
