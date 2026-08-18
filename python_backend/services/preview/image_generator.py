"""
Image preview generator for standard image formats (JPG, PNG, JPEG).
Returns images directly without modification (bypass behavior).
"""
import io
from typing import Tuple
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from .base import PreviewGenerator


class ImagePreviewGenerator(PreviewGenerator):
    """
    Generator for standard image formats.
    Simply returns the image bytes directly (bypass behavior).
    """
    
    # Supported image extensions
    SUPPORTED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    
    # Supported MIME types
    SUPPORTED_MIME_TYPES = {
        'image/jpeg',
        'image/jpg',
        'image/png'
    }
    
    def can_handle(self, filename: str = None, content_type: str = None) -> bool:
        """
        Check if this is a supported image file.
        """
        # Check by content type first (more reliable)
        if content_type:
            if content_type.lower() in self.SUPPORTED_MIME_TYPES:
                return True
        
        # Fallback to filename extension
        if filename:
            ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if ext in self.SUPPORTED_EXTENSIONS:
                return True
        
        return False
    
    def generate_preview(self, file_contents: bytes, filename: str = None, content_type: str = None) -> Tuple[StreamingResponse, str]:
        """
        Generate preview by returning the image bytes directly (bypass).
        Maintains the existing simple_preview endpoint behavior.
        """
        # Determine media type - default to image/jpeg if not provided
        media_type = content_type
        if not media_type:
            # Try to determine from filename extension
            if filename:
                ext = filename.lower().split('.')[-1]
                if ext in ['jpg', 'jpeg']:
                    media_type = 'image/jpeg'
                elif ext == 'png':
                    media_type = 'image/png'
                else:
                    media_type = 'image/jpeg'  # Default fallback
            else:
                media_type = 'image/jpeg'  # Default fallback
        
        # Return the file as a streaming response (bypass behavior)
        return StreamingResponse(
            io.BytesIO(file_contents),
            media_type=media_type
        ), media_type



