# Changelog

## [2.0] - 2026-07-16

### Added
- Multichannel IF-to-H&E registration, channel-preserving scientific warping, cascading serial-section registration, and streamed 3-D OME-TIFF merge output.

### Fixed
- Prevented large TIFF/OME-TIFF registration guides from falling back to full-image NumPy allocations.
- Added sparse guide-intensity sampling and single-channel tile reads for large IF images.
- Added a libvips single-page scientific TIFF fallback with the documented `unlimited` option for trusted local files.
- Pinned mutually compatible tifffile, Zarr, numcodecs, and imagecodecs versions in packaged applications.
- Warped multichannel TIFF pages independently instead of opening all pages as one tall libvips image.

## [1.0] - 2026-07-15

### Added
- First public HistRegGUI release for Windows, Linux, macOS Intel, and macOS Apple Silicon.
- One-fixed-target batch registration for selecting and processing multiple moving images.
- Ordered cascading registration for consecutive sections: each slice is registered to the previous successfully warped slice.
- Streamed 1×–32× registration downsampling with retained fixed-reference OME-TIFF and temporary per-slice working images.
- No application-level slice-count limit; sequential processing keeps memory independent of the number of slices.
- Cascade dependency handling that stops after a failed step and records later slices as `skipped_dependency`.
- Per-image queue, reader, running, completion, and failure status in the moving-image table.
- Sequential batch execution with GPU-memory cleanup between images and continuation after individual failures.
- Collision-safe numbered batch outputs plus CSV and JSON registration manifests.
- Optional merged, Imaris-friendly BigTIFF OME-TIFF volume after registration.
- IF↔H&E and IF↔IF registration through separate RGB registration guides and channel-preserving scientific payloads.
- Automatic DAPI/Hoechst guide selection plus selected-channel, maximum, and mean IF guide modes.
- Optional fluorescence-guide inversion and target-grid physical calibration for cross-modality H&E/IF outputs.
- Application of each calculated displacement field to every original IF channel while preserving channel names and integer dtype.
- Mixed H&E/IF scientific OME-TIFF export with `ZCYX` axes, union channel schema, and zero filling for channels absent from a Z slice.
- Optional simultaneous RGB `ZYXS` quality-control stack and scientific `ZCYX` analysis stack.
- Streaming 256 × 256 tile writer that avoids loading the complete Z-stack or complete slides into RAM.
- Fixed-target inclusion, editable Z spacing, automatic/editable XY calibration, and 1×–32× merge downsampling.
- Moving-image Move up/Move down controls so Z-slice order is explicit before stack creation.
- JSON sidecar recording Z order, source images, calibration, axes, downsample, compression, and reader backend.
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
