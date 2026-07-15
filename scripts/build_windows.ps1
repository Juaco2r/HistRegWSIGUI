$ErrorActionPreference = "Stop"
$env:PIP_NO_CACHE_DIR = "1"

Write-Host "Installing the CPU build dependencies..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-build.txt

Write-Host "Running tests..."
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_hardware.py

Write-Host "Building HistRegGUI..."
python scripts/build_app.py --variant cpu --platform-label windows --architecture x64
python scripts/package_release.py --platform Windows --architecture x64 --variant cpu

Write-Host ""
Write-Host "Build complete. Archive created under release-assets/."
