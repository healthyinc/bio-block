"""
WSI (Whole Slide Image) preview generator for pathology images.
Supports formats like .svs, .ndpi, .scn using OpenSlide library.
"""
import os
import io
import tempfile
from typing import Tuple
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

# Dynamic DLL loading for OpenSlide on Windows
def _setup_openslide_dll_path():
    """
    Configure DLL path for OpenSlide on Windows.
    Looks for openslide_binaries in the python_backend directory.
    """
    try:
        # Get the directory where this file is located (services/preview/)
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels to reach python_backend/
        python_backend_dir = os.path.dirname(os.path.dirname(current_file_dir))
        
        # Path to openslide_binaries/bin
        openslide_bin_path = os.path.join(python_backend_dir, 'openslide_binaries', 'bin')
        
        if os.path.exists(openslide_bin_path):
            # Python 3.8+ preferred method
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(openslide_bin_path)
            
            # Also add to PATH for compatibility
            current_path = os.environ.get('PATH', '')
            if openslide_bin_path not in current_path:
                os.environ['PATH'] = openslide_bin_path + os.pathsep + current_path
                
            return True
        else:
            return False
    except Exception:
        return False

# Try to setup DLL path and import openslide
openslide_available = False
openslide = None

if _setup_openslide_dll_path():
    try:
        import openslide
        openslide_available = True
    except ImportError:
        openslide_available = False
        print("Warning: openslide-python library not found. WSI preview will not work.")
        print("Install with: pip install openslide-python")
    except Exception as e:
        openslide_available = False
        print(f"Warning: Could not load OpenSlide library: {str(e)}")
        print("Make sure openslide_binaries are properly set up.")
else:
    print("Warning: OpenSlide binaries not found. WSI preview will not work.")
    print("Run setup_openslide.ps1 to download and set up the binaries.")

from .base import PreviewGenerator


class WsiPreviewGenerator(PreviewGenerator):
    """
    Generator for Whole Slide Images (WSI) used in pathology.
    Supports formats: .svs (Aperio), .ndpi (Hamamatsu), .scn (Leica), etc.
    """
    
    # Supported WSI extensions
    SUPPORTED_EXTENSIONS = {'svs', 'ndpi', 'scn', 'vms', 'vmu', 'mrxs', 'tiff', 'tif', 'bif'}
    
    # Supported MIME types (though WSI files often don't have standard MIME types)
    SUPPORTED_MIME_TYPES = {
        'image/tiff',
        'image/tif'
    }
    
    def __init__(self):
        """Initialize the WSI generator."""
        if not openslide_available:
            print("Warning: OpenSlide not available. WSI preview will not work.")
            print("Install openslide-python and run setup_openslide.ps1")
    
    def can_handle(self, filename: str = None, content_type: str = None) -> bool:
        """
        Check if this is a WSI file.
        """
        if not openslide_available:
            return False
        
        # Check by filename extension first (most reliable for WSI)
        if filename:
            ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if ext in self.SUPPORTED_EXTENSIONS:
                return True
        
        # Check by content type (less reliable for WSI)
        if content_type:
            if content_type.lower() in self.SUPPORTED_MIME_TYPES:
                return True
        
        return False
    
    def generate_preview(self, file_contents: bytes, filename: str = None, content_type: str = None) -> Tuple[StreamingResponse, str]:
        """
        Generate preview thumbnail from WSI file.
        OpenSlide requires a file path, so we save to a temporary file first.
        """
        if not openslide_available or openslide is None:
            raise HTTPException(
                status_code=503,
                detail="OpenSlide not available. Install openslide-python and run setup_openslide.ps1 to download binaries."
            )
        
        # Create temporary file to store the WSI file
        # OpenSlide needs a file path, not bytes
        temp_file = None
        try:
            # Determine file extension for temp file
            ext = filename.lower().split('.')[-1] if filename and '.' in filename else 'svs'
            
            # Create temporary file with appropriate extension
            temp_fd, temp_path = tempfile.mkstemp(suffix=f'.{ext}', prefix='wsi_preview_')
            temp_file = temp_path
            
            # Write file contents to temporary file
            with os.fdopen(temp_fd, 'wb') as f:
                f.write(file_contents)
            
            # Open the slide using OpenSlide
            try:
                slide = openslide.OpenSlide(temp_path)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not open WSI file with OpenSlide: {str(e)}. File may be corrupted or unsupported."
                )
            
            try:
                # Get thumbnail (downscale to 1024x1024 max)
                # This gives a good balance between quality and file size
                thumbnail_size = (1024, 1024)
                thumbnail = slide.get_thumbnail(thumbnail_size)
                
                # Convert thumbnail to RGB if needed (some WSI formats are grayscale)
                if thumbnail.mode != 'RGB':
                    thumbnail = thumbnail.convert('RGB')
                
                # Convert PIL Image to PNG bytes
                img_buffer = io.BytesIO()
                thumbnail.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                # Return streaming response
                return StreamingResponse(
                    io.BytesIO(img_buffer.read()),
                    media_type="image/png"
                ), "image/png"
                
            finally:
                # Always close the slide
                slide.close()
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing WSI file: {str(e)}"
            )
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    # Log but don't fail if cleanup fails
                    print(f"Warning: Could not delete temporary file {temp_file}: {str(e)}")

