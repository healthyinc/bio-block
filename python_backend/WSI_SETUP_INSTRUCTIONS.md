# WSI (Whole Slide Image) Support - Setup Instructions

## Overview

Phase 2 implementation adds support for Whole Slide Images (WSI) used in pathology, including formats like `.svs`, `.ndpi`, `.scn`, etc.

## Setup Steps

### Step 1: Download OpenSlide Binaries

Run the PowerShell setup script to automatically download and extract OpenSlide binaries:

```powershell
cd python_backend
.\setup_openslide.ps1
```

This script will:
- Download OpenSlide Windows binaries from GitHub releases
- Extract them to `python_backend/openslide_binaries/`
- Clean up temporary files

**Note:** If the script asks about overwriting existing binaries, choose `y` to re-download or `n` to use existing.

### Step 2: Install Python Package

Install the openslide-python package:

```powershell
# Make sure your virtual environment is activated
.\venv\Scripts\Activate.ps1

# Install openslide-python
pip install openslide-python
```

Or install all requirements (including openslide-python):

```powershell
pip install -r requirements.txt
```

### Step 3: Verify Setup

The WSI generator will automatically:
- Detect the OpenSlide binaries in `python_backend/openslide_binaries/bin/`
- Load the DLLs dynamically
- Register them for use with the openslide-python library

## Supported File Formats

The WsiPreviewGenerator supports the following WSI formats:

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

## File Structure

After setup, your directory structure should look like:

```
python_backend/
├── openslide_binaries/          # OpenSlide binaries (created by setup script)
│   ├── bin/                     # DLLs (automatically loaded)
│   ├── lib/
│   └── ...
├── services/
│   └── preview/
│       ├── base.py
│       ├── image_generator.py   # Existing - unchanged
│       ├── dicom_generator.py   # Existing - unchanged
│       ├── wsi_generator.py     # New - WSI support
│       └── factory.py           # Updated - includes WSI generator
└── setup_openslide.ps1          # Setup script
```

## How It Works

1. **Factory Pattern**: The `PreviewFactory` automatically detects WSI files by extension
2. **Dynamic DLL Loading**: `WsiPreviewGenerator` automatically finds and loads OpenSlide DLLs
3. **Thumbnail Generation**: Creates 1024x1024 PNG thumbnails from large WSI files
4. **Temporary File Handling**: Safely handles temporary files for OpenSlide processing

## Troubleshooting

### "OpenSlide not available" error

- Make sure you ran `setup_openslide.ps1`
- Verify `python_backend/openslide_binaries/bin/` exists
- Make sure `openslide-python` is installed: `pip install openslide-python`

### DLL loading errors

- Check that OpenSlide binaries are in the correct location
- Try re-running the setup script to re-download binaries
- Verify you're using a compatible Python version (3.7+)

### "Could not open WSI file" error

- File may be corrupted
- File format may not be supported by OpenSlide
- Check that the file is a valid WSI format

## Backward Compatibility

✅ **All existing functionality is preserved:**
- Image preview (JPG/PNG) works exactly as before
- DICOM preview works exactly as before
- API endpoints remain unchanged
- No breaking changes to frontend

The WSI support is purely additive and does not affect existing features.

