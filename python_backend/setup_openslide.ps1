# PowerShell script to download and set up OpenSlide binaries for Windows
# This script downloads OpenSlide binaries and extracts them locally

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "OpenSlide Binary Setup Script" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$openslideUrl = "https://github.com/openslide/openslide-winbuild/releases/download/v20171122/openslide-win64-20171122.zip"
$zipPath = Join-Path $scriptDir "openslide-win64-20171122.zip"
$extractPath = Join-Path $scriptDir "openslide_extracted"
$targetPath = Join-Path $scriptDir "openslide_binaries"

# Check if binaries already exist
if (Test-Path $targetPath) {
    Write-Host "OpenSlide binaries already exist at: $targetPath" -ForegroundColor Yellow
    $overwrite = Read-Host "Do you want to re-download and overwrite? (y/n)"
    if ($overwrite -ne "y" -and $overwrite -ne "Y") {
        Write-Host "Skipping download. Using existing binaries." -ForegroundColor Green
        exit 0
    }
    Remove-Item -Path $targetPath -Recurse -Force
    Write-Host "Removed existing binaries." -ForegroundColor Yellow
}

Write-Host "Step 1: Downloading OpenSlide binaries..." -ForegroundColor Green
Write-Host "URL: $openslideUrl" -ForegroundColor Gray
Write-Host "Destination: $zipPath" -ForegroundColor Gray

try {
    # Download the zip file
    Invoke-WebRequest -Uri $openslideUrl -OutFile $zipPath -UseBasicParsing
    Write-Host "✓ Download completed successfully!" -ForegroundColor Green
} catch {
    Write-Host "✗ Error downloading OpenSlide binaries:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 2: Extracting zip file..." -ForegroundColor Green

try {
    # Extract to temporary location
    if (Test-Path $extractPath) {
        Remove-Item -Path $extractPath -Recurse -Force
    }
    Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
    Write-Host "✓ Extraction completed!" -ForegroundColor Green
} catch {
    Write-Host "✗ Error extracting zip file:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Remove-Item -Path $zipPath -ErrorAction SilentlyContinue
    exit 1
}

Write-Host ""
Write-Host "Step 3: Moving binaries to target location..." -ForegroundColor Green

try {
    # Find the inner folder (should be openslide-win64-20171122)
    $innerFolders = Get-ChildItem -Path $extractPath -Directory
    if ($innerFolders.Count -eq 0) {
        throw "No inner folder found in extracted archive"
    }
    
    $sourceFolder = $innerFolders[0].FullName
    Write-Host "Source folder: $sourceFolder" -ForegroundColor Gray
    
    # Move to target location
    Move-Item -Path $sourceFolder -Destination $targetPath -Force
    Write-Host "✓ Binaries moved to: $targetPath" -ForegroundColor Green
} catch {
    Write-Host "✗ Error moving binaries:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Remove-Item -Path $zipPath -ErrorAction SilentlyContinue
    Remove-Item -Path $extractPath -Recurse -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host ""
Write-Host "Step 4: Cleaning up temporary files..." -ForegroundColor Green

try {
    # Delete zip file
    Remove-Item -Path $zipPath -Force
    Write-Host "✓ Zip file deleted" -ForegroundColor Green
    
    # Delete extraction folder (should be empty now)
    if (Test-Path $extractPath) {
        Remove-Item -Path $extractPath -Recurse -Force
        Write-Host "✓ Temporary extraction folder deleted" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠ Warning: Could not clean up some temporary files:" -ForegroundColor Yellow
    Write-Host $_.Exception.Message -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "✓ Setup completed successfully!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "OpenSlide binaries are now available at:" -ForegroundColor White
Write-Host "  $targetPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Install Python package: pip install openslide-python" -ForegroundColor Gray
Write-Host "  2. The WsiPreviewGenerator will automatically find these binaries" -ForegroundColor Gray
Write-Host ""

