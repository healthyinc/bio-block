"""
Factory for creating appropriate preview generators based on file type.
"""
from typing import Optional, List, Type
from fastapi import HTTPException

from .base import PreviewGenerator
from .image_generator import ImagePreviewGenerator
from .dicom_generator import DicomPreviewGenerator

# Try to import WSI generator (may fail if OpenSlide not available)
try:
    from .wsi_generator import WsiPreviewGenerator
    wsi_available = True
except ImportError:
    wsi_available = False
    WsiPreviewGenerator = None
except Exception:
    # Handle other import errors (e.g., DLL loading issues)
    wsi_available = False
    WsiPreviewGenerator = None


class PreviewFactory:
    """
    Factory class for creating preview generators based on file type.
    
    Determines the appropriate generator based on filename extension
    or content-type, and returns an instance of the correct generator class.
    """
    
    # Registry of available generators (ordered by priority)
    # WSI generator checked first for specific extensions (.svs, .ndpi, etc.)
    # Then DICOM, then standard images
    _generators: List[Type[PreviewGenerator]] = [
        DicomPreviewGenerator,
        ImagePreviewGenerator,
    ]
    
    # Initialize generators list with WSI if available
    if wsi_available and WsiPreviewGenerator is not None:
        # Insert WSI generator at the beginning (highest priority)
        # This ensures .svs, .ndpi, etc. are handled by WSI generator
        # before falling back to other generators
        _generators.insert(0, WsiPreviewGenerator)
    
    @classmethod
    def create_generator(cls, filename: str = None, content_type: str = None) -> PreviewGenerator:
        """
        Create an appropriate preview generator for the given file.
        
        Args:
            filename: Optional filename for type detection
            content_type: Optional MIME content type
            
        Returns:
            An instance of the appropriate PreviewGenerator subclass
            
        Raises:
            HTTPException: If no generator can handle the file type
        """
        # Try each generator to see if it can handle the file
        for generator_class in cls._generators:
            generator = generator_class()
            if generator.can_handle(filename=filename, content_type=content_type):
                return generator
        
        # No generator found
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Filename: {filename}, Content-Type: {content_type}"
        )
    
    @classmethod
    def register_generator(cls, generator_class: Type[PreviewGenerator], priority: int = None):
        """
        Register a new generator class (for future extensibility).
        
        Args:
            generator_class: The generator class to register
            priority: Optional priority (lower = higher priority). If None, appends to end.
        """
        if priority is not None:
            cls._generators.insert(priority, generator_class)
        else:
            cls._generators.append(generator_class)

