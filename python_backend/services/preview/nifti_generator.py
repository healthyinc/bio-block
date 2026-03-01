"""
NIfTI (Neuroimaging) preview generator for brain scan files.
Supports formats like .nii and .nii.gz using the NiBabel library.
"""
import os
import io
import tempfile
import numpy as np
from typing import Tuple
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

# Try to import nibabel
nibabel_available = False
nib = None


def _try_import_nibabel():
    """
    Try to import nibabel. This works if:
    - Installed via pip: pip install nibabel
    - Installed via conda: conda install -c conda-forge nibabel
    """
    global nib, nibabel_available
    try:
        import nibabel
        nib = nibabel
        nibabel_available = True
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"Warning: nibabel import failed: {str(e)}")
        return False


# Try to import nibabel at module load time
if not _try_import_nibabel():
    print("Warning: NiBabel not available. NIfTI preview will not work.")
    print("Install with: pip install nibabel")

from .base import PreviewGenerator


class NiftiPreviewGenerator(PreviewGenerator):
    """
    Generator for NIfTI neuroimaging files (MRI/fMRI brain scans).
    Supports formats: .nii, .nii.gz
    
    Extracts the middle axial slice from the 3D volume and returns
    it as a PNG image for preview.
    """

    # Supported NIfTI extensions
    SUPPORTED_EXTENSIONS = {'nii', 'nii.gz'}

    def __init__(self):
        """Initialize the NIfTI generator."""
        if not nibabel_available:
            print("Warning: NiBabel not available. NIfTI preview will not work.")
            print("Install with: pip install nibabel")

    def can_handle(self, filename: str = None, content_type: str = None) -> bool:
        """
        Check if this is a NIfTI file.
        """
        if not nibabel_available:
            return False

        # Check by filename extension (most reliable for NIfTI)
        if filename:
            lower_name = filename.lower()
            # Check for .nii.gz first (compound extension)
            if lower_name.endswith('.nii.gz'):
                return True
            # Then check for .nii
            ext = lower_name.split('.')[-1] if '.' in lower_name else ''
            if ext == 'nii':
                return True

        return False

    def generate_preview(self, file_contents: bytes, filename: str = None, content_type: str = None) -> Tuple[StreamingResponse, str]:
        """
        Generate preview image from NIfTI file.
        
        Extracts the middle axial slice from the 3D volume, normalizes
        pixel values to 0-255, and returns a PNG image.
        
        NiBabel requires a file path, so we save to a temporary file first.
        """
        if not nibabel_available or nib is None:
            raise HTTPException(
                status_code=503,
                detail="NiBabel not available. Install via: pip install nibabel"
            )

        # Create temporary file to store the NIfTI file
        # NiBabel needs a file path, not bytes
        temp_file = None
        try:
            # Determine file suffix for temp file
            if filename and filename.lower().endswith('.nii.gz'):
                suffix = '.nii.gz'
            else:
                suffix = '.nii'

            # Create temporary file with appropriate extension
            temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix='nifti_preview_')
            temp_file = temp_path

            # Write file contents to temporary file
            with os.fdopen(temp_fd, 'wb') as f:
                f.write(file_contents)

            # Load the NIfTI file using NiBabel
            try:
                img = nib.load(temp_path)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not open NIfTI file with NiBabel: {str(e)}. File may be corrupted or unsupported."
                )

            try:
                # Get image data as a numpy array
                data = img.get_fdata()

                # Handle 4D data (e.g., fMRI time series) by taking the first volume
                if data.ndim == 4:
                    data = data[..., 0]
                elif data.ndim < 3:
                    raise HTTPException(
                        status_code=400,
                        detail=f"NIfTI file has unexpected dimensions: {data.ndim}D. Expected 3D or 4D data."
                    )

                # Extract the middle axial slice (z-axis)
                middle_idx = data.shape[2] // 2
                middle_slice = data[:, :, middle_idx]

                # Rotate 90 degrees for correct anatomical orientation
                middle_slice = np.rot90(middle_slice)

                # Normalize pixel values to 0-255
                slice_min = np.min(middle_slice)
                slice_max = np.max(middle_slice)

                if slice_max == slice_min:
                    # Constant image - produce a blank (black) preview
                    normalized = np.zeros_like(middle_slice, dtype=np.uint8)
                else:
                    normalized = ((middle_slice - slice_min) / (slice_max - slice_min) * 255.0).astype(np.uint8)

                # Convert to PIL Image (grayscale)
                preview_image = Image.fromarray(normalized, mode='L')

                # Convert to RGB for consistency with other generators
                preview_image = preview_image.convert('RGB')

                # Save to PNG buffer
                img_buffer = io.BytesIO()
                preview_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)

                # Return streaming response
                return StreamingResponse(
                    io.BytesIO(img_buffer.read()),
                    media_type="image/png"
                ), "image/png"

            finally:
                # NiBabel may keep file handles; ensure cleanup
                del img

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing NIfTI file: {str(e)}"
            )
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    # Log but don't fail if cleanup fails
                    print(f"Warning: Could not delete temporary file {temp_file}: {str(e)}")
