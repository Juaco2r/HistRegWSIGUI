$ErrorActionPreference = "Stop"

Write-Host "Cleaning previous builds..."
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

Write-Host "Building HistRegGUI (onedir mode)..."

pyinstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name HistRegGUI `
  --add-data "deeperhistreg;deeperhistreg" `
  --add-data "external;external" `
  src\histreggui\app.py

Write-Host ""
Write-Host "Build complete!"
Write-Host "Executable folder: dist\HistRegGUI\"
Write-Host ""
Write-Host "Zip the entire dist\HistRegGUI folder for GitHub Release."