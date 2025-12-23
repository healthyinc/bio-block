# 🔧 Refactor File Preview Logic with Factory Pattern (Phase 1)

## 📋 Overview

This PR refactors the file preview logic in the Python backend to use a **Factory Pattern** architecture. This is Phase 1 of a larger plan to support complex medical data types in future phases.

The refactoring maintains **100% backward compatibility** - all existing API endpoints continue to work exactly as before, so the frontend remains unaffected.

---

## 🎯 Goals

- ✅ **Modularize preview logic** - Move preview code out of `main.py` into a service layer
- ✅ **Implement Factory Pattern** - Make it easy to add new file type handlers in the future
- ✅ **Maintain backward compatibility** - All existing endpoints (`/simple_preview`, `/preview_dicom`) work unchanged
- ✅ **No breaking changes** - Frontend doesn't need any modifications
- ✅ **No new dependencies** - Only uses existing packages from `requirements.txt`

---

## 📁 What Changed

### New Directory Structure

Created `python_backend/services/preview/` with the following files:

```
python_backend/services/
├── __init__.py
└── preview/
    ├── __init__.py
    ├── base.py              # Abstract base class (PreviewGenerator)
    ├── image_generator.py   # Handles JPG/PNG/JPEG files
    ├── dicom_generator.py   # Handles DICOM files
    └── factory.py           # Factory class to select the right generator
```

### Key Files

1. **`base.py`** - Abstract base class defining the interface:
   - `generate_preview()` - Method to generate preview from file bytes
   - `can_handle()` - Method to check if a generator can handle a file type

2. **`image_generator.py`** - `ImagePreviewGenerator` class:
   - Handles standard image formats (JPG, JPEG, PNG)
   - Maintains the original "bypass" behavior (returns image bytes directly)
   - Moved logic from the original `/simple_preview` endpoint

3. **`dicom_generator.py`** - `DicomPreviewGenerator` class:
   - Handles DICOM medical imaging files (.dcm, .dicom)
   - Converts DICOM pixel data to PNG images for preview
   - Moved logic from the original `/preview_dicom` endpoint
   - Gracefully handles missing `pydicom` library

4. **`factory.py`** - `PreviewFactory` class:
   - Automatically detects file type based on filename extension or content-type
   - Returns the appropriate generator instance
   - Throws clear error messages for unsupported file types
   - Easy to extend with new generators in Phase 2

### Refactored Files

- **`main.py`** - Updated to use `PreviewFactory`:
  - `/simple_preview` endpoint now uses the factory
  - `/preview_dicom` endpoint now uses the factory
  - Both endpoints maintain identical behavior for backward compatibility

---

## 🔄 How It Works

### Before (Monolithic)
```
main.py
├── /simple_preview endpoint (inline image logic)
└── /preview_dicom endpoint (inline DICOM logic)
```

### After (Factory Pattern)
```
main.py
├── /simple_preview endpoint → PreviewFactory → ImagePreviewGenerator
└── /preview_dicom endpoint → PreviewFactory → DicomPreviewGenerator
```

### Flow Diagram

```
1. File Upload → FastAPI Endpoint
2. Endpoint calls PreviewFactory.create_generator(filename, content_type)
3. Factory checks each generator's can_handle() method
4. Factory returns appropriate generator (ImagePreviewGenerator or DicomPreviewGenerator)
5. Generator.generate_preview() processes the file
6. Returns StreamingResponse to client
```

---

## ✅ Testing

### Manual Testing Checklist

- [x] ✅ `/simple_preview` endpoint works with JPG files
- [x] ✅ `/simple_preview` endpoint works with PNG files
- [x] ✅ `/simple_preview` endpoint works with JPEG files
- [x] ✅ `/preview_dicom` endpoint works with .dcm files
- [x] ✅ `/preview_dicom` endpoint works with .dicom files
- [x] ✅ Error handling works for unsupported file types
- [x] ✅ All imports work correctly
- [x] ✅ No breaking changes to frontend

### API Response Examples

**Image Preview (JPG/PNG):**
```bash
POST /simple_preview
Content-Type: multipart/form-data
File: image.jpg

→ Returns: image/jpeg stream (direct bypass)
```

**DICOM Preview:**
```bash
POST /preview_dicom
Content-Type: multipart/form-data
File: medical_image.dcm

→ Returns: image/png stream (converted from DICOM)
```

---

## 🚀 Benefits

### Immediate Benefits
- **Cleaner code** - Preview logic separated from API routes
- **Better organization** - Related code grouped in one place
- **Easier testing** - Each generator can be tested independently

### Future Benefits (Phase 2)
- **Easy extensibility** - Add new file types by creating a new generator class
- **Support for complex formats** - Ready for OpenSlide (WSI), NiBabel (NIfTI), etc.
- **Better error handling** - Centralized error management
- **Type safety** - Abstract base class ensures consistent interface

---

## 🔮 Phase 2 Preview

In future phases, adding new file types will be as simple as:

```python
# services/preview/wsi_generator.py
class WsiPreviewGenerator(PreviewGenerator):
    def can_handle(self, filename, content_type):
        return filename.endswith('.svs')  # Whole Slide Images
    
    def generate_preview(self, file_contents, ...):
        # Use OpenSlide to generate thumbnail
        ...
```

Then register it in the factory - no changes needed to `main.py`!

---

## 📝 Code Quality

- ✅ Type hints using standard `typing` module (Python 3.7+ compatible)
- ✅ Comprehensive docstrings
- ✅ Error handling with clear messages
- ✅ No linter errors
- ✅ Follows existing code style

---

## 🔒 Safety

- ✅ **No breaking changes** - All existing endpoints work identically
- ✅ **No new dependencies** - Uses only existing packages
- ✅ **Graceful degradation** - Handles missing optional libraries (pydicom)
- ✅ **Backward compatible** - Frontend requires no changes

---

## 📚 Related

- Part of the larger plan to support complex medical data types
- Prepares architecture for Phase 2 (OpenSlide, NiBabel, etc.)
- Maintains compatibility with existing frontend

---

## 👀 Review Checklist

- [ ] Verify `/simple_preview` endpoint behavior is unchanged
- [ ] Verify `/preview_dicom` endpoint behavior is unchanged
- [ ] Check that error handling works for unsupported types
- [ ] Confirm no new dependencies were added
- [ ] Review code organization and structure

---

## 🎉 Summary

This refactoring sets up a solid foundation for future medical data type support while maintaining 100% backward compatibility. The factory pattern makes it trivial to add new file type handlers in Phase 2 without touching the main API code.



