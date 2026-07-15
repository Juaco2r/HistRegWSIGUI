# HistRegGUI Protocol

Version: 1.0

Author: Jose Rodriguez-Rojas

Last updated: 2026-07-15

## 1. Purpose

HistRegGUI registers one or more moving histology images into the coordinate system of one fixed target image using DeeperHistReg. It provides a desktop interface for Windows, macOS, and Linux.

## 2. Release editions

- **CPU**: default and recommended for maximum portability. No NVIDIA GPU is required.
- **CUDA**: available for Windows and Linux. It contains a CUDA-enabled PyTorch runtime but still starts in CPU mode unless the user explicitly enables CUDA.
- **macOS Intel / Apple Silicon**: CPU execution. NVIDIA CUDA is not supported on macOS.

## 3. Input procedure

1. Open HistRegGUI.
2. Select the single **Target (Fixed)** image.
3. Click **Add images...** and select one or several moving images.
4. Review the moving-image table. Selecting a row displays that image in the moving preview.
5. Keep **Input reader** on automatic or choose a reader manually.
6. Choose a registration preset.
7. Optionally enable **Save intermediate results**.
8. Optionally open **Hardware → Check CUDA availability...**.
9. Enable **Use CUDA acceleration (NVIDIA)** only when the check succeeds.
10. Click **Run registration** or **Run registration batch**.

Accepted picker formats include TIFF/OME-TIFF, SVS, NDPI, MRXS, SCN, VMS, VMU, BIF, SVSLIDE, DICOM, JPG/JPEG, PNG, BMP, WebP, and JPEG 2000. Previews use tifffile, OpenSlide, libvips, Pillow, and SimpleITK fallbacks and are downsampled only for display.

With automatic reader selection, the reader is resolved independently for every moving/fixed pair. A batch can therefore use different readers for different source formats.

## 4. Batch behavior

- All moving images are registered to the same fixed target.
- Duplicate selections are ignored.
- Images are processed sequentially to limit CPU, RAM, and GPU-memory growth.
- CUDA cache and Python objects are released between registrations.
- One failed moving image is logged and does not prevent the remaining images from running.
- The table reports queued, running, completed, or failed status for each image.

## 5. Hardware selection

CPU is selected by default on every build.

The CUDA check verifies:

- whether the bundled PyTorch runtime was compiled with CUDA;
- whether a compatible NVIDIA driver and GPU are visible;
- whether a small allocation on `cuda:0` succeeds.

When CUDA is unavailable, the checkbox remains disabled. Before each registration, all nested DeeperHistReg `device` values and `cuda` flags are normalized to the selected mode.

## 6. Output

For one moving image, the final warped image is written next to the fixed image:

```text
<moving>_warped_to_<fixed>.tif
```

For several moving images, a timestamped directory is created next to the fixed image:

```text
HistRegGUI_batch_<fixed>_<timestamp>/
├── warped/
│   ├── 001_<moving>_warped_to_<fixed>.tif
│   └── 002_<moving>_warped_to_<fixed>.tif
├── intermediate/
├── registration_manifest.csv
├── registration_manifest.json
└── HistRegGUI_error.log
```

Numbered output filenames prevent collisions when moving images from different folders have the same basename.

When intermediate saving is disabled, successful intermediate directories are removed. A failed item's intermediate directory is retained when available to support troubleshooting.

## 7. Manifests and error handling

Every run writes CSV and JSON manifests containing:

- fixed and moving input paths;
- output and intermediate paths;
- selected reader, device, and preset;
- start and finish timestamps;
- success or failure status;
- error text when applicable.

Failures are also appended to `HistRegGUI_error.log` with the Python traceback.

## 8. Validation

Registration results should be reviewed visually before quantitative analysis. Confirm that corresponding tissue structures, orientation, and specimen region agree between fixed and warped images.

## 9. Distribution

GitHub Actions builds the application independently on Windows, Ubuntu, and macOS runners. Version tags matching `v*` publish the generated archives in a GitHub Release.

The application is built with PyInstaller and includes the installed DeeperHistReg package, its resources, and binary image backends needed by the release.

## 10. Licensing

HistRegGUI wrapper code is distributed under the MIT License. DeeperHistReg and all bundled dependencies remain under their respective licenses. Review `THIRD_PARTY_NOTICES.md` before redistribution.
