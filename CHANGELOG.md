# Changelog

## [1.0] - 2026-07-15

### Added
- First public HistRegGUI release for Windows, Linux, macOS Intel, and macOS Apple Silicon.
- One-fixed-target batch registration for selecting and processing multiple moving images.
- Per-image queue, reader, running, completion, and failure status in the moving-image table.
- Sequential batch execution with GPU-memory cleanup between images and continuation after individual failures.
- Collision-safe numbered batch outputs plus CSV and JSON registration manifests.
- GitHub Actions builds for four CPU packages and Windows/Linux CUDA packages.
- Automatic GitHub Release publication for `v*` tags.
- CPU-default execution with optional CUDA detection and explicit user activation.
- Automatic and manual DeeperHistReg reader selection for TIFF/OME-TIFF, whole-slide, raster, and mixed-format pairs.
- Memory-conscious previews through tifffile, OpenSlide, libvips, Pillow, and SimpleITK.
- Support in the file picker for TIFF, OME-TIFF, SVS, NDPI, MRXS, SCN, VMS, VMU, BIF, SVSLIDE, DICOM, JPEG, PNG, BMP, WebP, and JPEG 2000 extensions.
- Packaged executable smoke tests on every target platform.
- `CITATION.cff`, `.zenodo.json`, and release metadata validation for Zenodo DOI ingestion.

### Fixed
- Bundled `PIL._tkinter_finder`, `PIL.ImageTk`, Pillow plugins, and Tk binary modules in PyInstaller builds.
- Added a `PIL.tkinter_finder` compatibility alias for diagnostics using the legacy non-underscore name.
- Kept setuptools 81.0.0 for PyInstaller `pkg_resources.NullProvider` compatibility.
- Used the Intel-compatible PyTorch 2.2.2/torchvision 0.17.2 pair on macOS Intel.
- Applied the selected reader consistently to registration and final deformation.
