import os
from typing import Optional

import img2pdf
import pydicom
from pdf2image import convert_from_path
from presidio_image_redactor import ImageRedactorEngine


def anonymize_dicom(input_path: str, output_path: str) -> None:
    """
    Anonymize a DICOM file by redacting patient information.

    This function reads a DICOM file, replaces sensitive patient attributes
    (PatientName, PatientID, PatientBirthDate) with 'ANONYMIZED', and saves
    the modified file to the specified output path.

    Args:
        input_path (str): The path to the input DICOM file.
        output_path (str): The path where the anonymized DICOM file will be
                           saved.

    Returns:
        None
    """
    try:
        ds = pydicom.dcmread(input_path)

        # Redact sensitive fields if they exist
        if "PatientName" in ds:
            ds.PatientName = "ANONYMIZED"
        if "PatientID" in ds:
            ds.PatientID = "ANONYMIZED"
        if "PatientBirthDate" in ds:
            ds.PatientBirthDate = "ANONYMIZED"

        ds.save_as(output_path)
        print(f"Successfully anonymized DICOM: {input_path} -> {output_path}")

    except Exception as e:
        print(f"Error anonymizing DICOM file {input_path}: {e}")
        raise


def anonymize_pdf(
    input_path: str, output_path: str, poppler_path: Optional[str] = None
) -> None:
    """
    Anonymize a PDF file by converting it to images, redacting PII, and
    rebuilding the PDF.

    This function uses:
    1. pdf2image to convert the PDF pages into images.
    2. presidio-image-redactor to detect and redact Personal Identifiable
       Information (PII) from the images.
    3. img2pdf to convert the redacted images back into a single PDF file.

    Args:
        input_path (str): The path to the input PDF file.
        output_path (str): The path where the anonymized PDF file will be
                           saved.
        poppler_path (Optional[str]): The path to the Poppler binary directory.
                                      Defaults to None.

    Returns:
        None
    """
    try:
        # 1. Convert PDF to images
        # poppler_path is required on Windows if not in PATH
        images = convert_from_path(input_path, poppler_path=poppler_path)

        redactor = ImageRedactorEngine()
        redacted_image_paths = []

        # Temp list for image cleanup (optional, or rely on tempfile)
        # For simplicity in this script, we'll save to a temp pattern or handle
        # in memory if possible.
        # img2pdf accepts bytes or file paths. Presidio returns a PIL Image.

        redacted_images_bytes = []

        for i, image in enumerate(images):
            # 2. Redact PII from the image
            # redactor.redact returns a new PIL Image
            redacted_image = redactor.redact(image, fill="black")

            # Convert PIL image to bytes for img2pdf
            # We need to save it to a temporary buffer or file?
            # img2pdf mostly works with files or raw bytes.
            # Let's save to a temporary file path to be safe and compatible
            # with img2pdf

            # Actually, img2pdf can take a list of PIL images if we convert
            # them?
            # No, img2pdf library documentation says it takes filenames or
            # file-like objects.
            # However, passing PIL objects directly isn't supported by standard
            # img2pdf.
            # We need to save them or convert to bytes.

            temp_img_path = f"{output_path}_temp_page_{i}.jpg"
            redacted_image.save(temp_img_path, format="JPEG")
            redacted_image_paths.append(temp_img_path)

        # 3. Recompile to PDF
        with open(output_path, "wb") as f:
            f.write(img2pdf.convert(redacted_image_paths))

        print(f"Successfully anonymized PDF: {input_path} -> {output_path}")

        # Cleanup temporary files
        for path in redacted_image_paths:
            if os.path.exists(path):
                os.remove(path)

    except Exception as e:
        print(f"Error anonymizing PDF file {input_path}: {e}")
        # Build cleanup in case of error
        if 'redacted_image_paths' in locals():
            for path in redacted_image_paths:
                if os.path.exists(path):
                    os.remove(path)
        raise
