# WSI (Whole Slide Image) Support - Setup Instructions

## Overview

Phase 2 implementation adds support for Whole Slide Images (WSI) used in pathology, including formats like `.svs`, `.ndpi`, `.scn`, etc.

## Recommended Installation Method (Cross-Platform)

### Using Conda (Recommended)

The easiest and most reliable way to install OpenSlide with cross-platform support:

```bash
conda install -c conda-forge openslide-python
```

This command installs:
- The native libopenslide library (`.dll` on Windows, `.so` on Linux, `.dylib` on macOS)
- The Python bindings (openslide-python)
- All necessary dependencies

**Works on:** Windows, macOS, and Linux ✅

## Alternative Installation Methods

### Using pip (Linux/macOS)

On Linux or macOS, you can use pip with system packages:

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install openslide-tools python3-openslide
pip install openslide-python
```

**macOS:**
```bash
brew install openslide
pip install openslide-python
```

### Windows Fallback (Manual Setup)

If you're on Windows and can't use conda, you can use the PowerShell setup script:

```powershell
cd python_backend
.\setup_openslide.ps1
pip install openslide-python
```

**Note:** The setup script is Windows-only and downloads binaries manually. The conda method is preferred.

## How It Works

The WSI generator automatically:

1. **First tries to import openslide directly** - Works if installed via conda or pip with system libraries
2. **Falls back to Windows DLL loading** - Only on Windows, only if direct import failed
3. **Gracefully degrades** - If OpenSlide isn't available, WSI preview is disabled but everything else works

## Supported File Formats

The WsiPreviewGenerator supports these formats:
- `.svs` - Aperio ScanScope
- `.ndpi` - Hamamatsu NanoZoomer
- `.scn` - Leica ScanScope
- `.vms`, `.vmu` - Ventana
- `.mrxs` - MIRAX
- `.tiff`, `.tif` - TIFF-based WSI formats
- `.bif` - Bio-Imaging Format

## Testing

### Test WSI Preview

You can test the WSI preview functionality through the existing `/simple_preview` endpoint:

```bash
POST /simple_preview
Content-Type: multipart/form-data
File: pathology_image.svs

→ Returns: image/png thumbnail (1024x1024)
```

### Verify Existing Functionality Still Works

Make sure existing generators still work:

- **Image Preview (JPG/PNG)**: Should still work exactly as before
- **DICOM Preview**: Should still work exactly as before

## Backward Compatibility

✅ **All existing functionality is preserved:**
- Image preview (JPG/PNG) works exactly as before
- DICOM preview works exactly as before
- API endpoints remain unchanged
- No breaking changes to frontend
- Windows PowerShell script still works as fallback

The WSI support is purely additive and does not affect existing features.
