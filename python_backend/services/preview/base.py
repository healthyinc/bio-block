"""
Abstract base class for preview generators.
"""
from abc import ABC, abstractmethod
from typing import Tuple, BinaryIO
from fastapi.responses import StreamingResponse


class PreviewGenerator(ABC):
    """
    Abstract base class for generating previews of different file types.
    
    All preview generators must implement the generate_preview method
    which takes file bytes and returns a StreamingResponse.
    """
    
    @abstractmethod
    def generate_preview(self, file_contents: bytes, filename: str = None, content_type: str = None) -> Tuple[StreamingResponse, str]:
        """
        Generate a preview of the given file.
        
        Args:
            file_contents: The raw file bytes
            filename: Optional filename for type detection
            content_type: Optional MIME content type
            
        Returns:
            Tuple of (StreamingResponse, media_type) where media_type is the MIME type
            of the preview (e.g., "image/png", "image/jpeg")
            
        Raises:
            HTTPException: If the file cannot be processed
        """
        pass
    
    @abstractmethod
    def can_handle(self, filename: str = None, content_type: str = None) -> bool:
        """
        Check if this generator can handle the given file.
        
        Args:
            filename: Optional filename
            content_type: Optional MIME content type
            
        Returns:
            True if this generator can handle the file, False otherwise
        """
        pass



