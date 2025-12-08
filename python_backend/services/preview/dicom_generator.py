"""
DICOM preview generator for medical imaging files (.dcm, .dicom).
Converts DICOM files to PNG images for preview.
"""
import io
from typing import Tuple
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
from PIL import Image

# DICOM imports
try:
    import pydicom
    from pydicom.errors import InvalidDicomError
    pydicom_available = True
except ImportError:
    pydicom_available = False

from .base import PreviewGenerator


class DicomPreviewGenerator(PreviewGenerator):
    """
    Generator for DICOM medical imaging files.
    Converts DICOM pixel data to PNG images.
    """
    
    # Supported DICOM extensions
    SUPPORTED_EXTENSIONS = {'dcm', 'dicom'}
    
    # Supported MIME types (though DICOM files often don't have standard MIME types)
    SUPPORTED_MIME_TYPES = {
        'application/dicom',
        'application/x-dicom'
    }
    
    def __init__(self):
        """Initialize the DICOM generator."""
        if not pydicom_available:
            print("Warning: pydicom library not found. DICOM preview will not work.")
            print("Install with: pip install pydicom")
    
    def can_handle(self, filename: str = None, content_type: str = None) -> bool:
        """
        Check if this is a DICOM file.
        """
        if not pydicom_available:
            return False
        
        # Check by filename extension first (most reliable for DICOM)
        if filename:
            ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if ext in self.SUPPORTED_EXTENSIONS:
                return True
        
        # Check by content type
        if content_type:
            if content_type.lower() in self.SUPPORTED_MIME_TYPES:
                return True
            # Sometimes DICOM files are detected with 'dicom' in the content type
            if 'dicom' in content_type.lower():
                return True
        
        return False
    
    def generate_preview(self, file_contents: bytes, filename: str = None, content_type: str = None) -> Tuple[StreamingResponse, str]:
        """
        Convert DICOM file to PNG image for preview.
        """
        if not pydicom_available:
            raise HTTPException(
                status_code=503,
                detail="pydicom not available. Install with: pip install pydicom"
            )
        
        try:
            # Validate file type
            file_extension = filename.lower().split('.')[-1] if filename else ''
            if file_extension not in ['dcm', 'dicom']:
                # Check content type as fallback
                if not (content_type and 'dicom' in content_type.lower()):
                    raise HTTPException(
                        status_code=400,
                        detail="File must be a DICOM file (.dcm or .dicom)"
                    )
            
            # Read DICOM file from bytes
            dicom_dataset = pydicom.dcmread(io.BytesIO(file_contents))
            
            # Check if pixel data exists
            if 'PixelData' not in dicom_dataset:
                raise HTTPException(
                    status_code=400,
                    detail="DICOM file does not contain pixel data"
                )
            
            # Get pixel array
            pixel_array = dicom_dataset.pixel_array
            
            # Normalize pixel values to 0-255 range
            pixel_min = pixel_array.min()
            pixel_max = pixel_array.max()
            
            if pixel_max > 255:
                # Normalize if values are outside 0-255 range
                if pixel_max == pixel_min:
                    # Uniform image (all pixels same value) - set to middle gray
                    pixel_array = np.full_like(pixel_array, 128, dtype=np.uint8)
                else:
                    pixel_array = ((pixel_array - pixel_min) / 
                                 (pixel_max - pixel_min) * 255).astype(np.uint8)
            else:
                pixel_array = pixel_array.astype(np.uint8)
            
            # Handle grayscale and RGB images
            if len(pixel_array.shape) == 2:
                # Grayscale image
                pil_image = Image.fromarray(pixel_array, mode='L')
            elif len(pixel_array.shape) == 3:
                # RGB or color image
                pil_image = Image.fromarray(pixel_array, mode='RGB')
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported DICOM image format"
                )
            
            # Convert to RGB if grayscale for better compatibility
            if pil_image.mode == 'L':
                pil_image = pil_image.convert('RGB')
            
            # Save to bytes buffer as PNG
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            return StreamingResponse(
                io.BytesIO(img_buffer.read()),
                media_type="image/png"
            ), "image/png"
            
        except InvalidDicomError:
            raise HTTPException(
                status_code=400,
                detail="Invalid DICOM file format"
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing DICOM file: {str(e)}"
            )

