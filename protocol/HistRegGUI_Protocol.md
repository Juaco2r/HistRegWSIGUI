# HistRegGUI Protocol

Version: 1.1.0

Author: Jose Rodriguez-Rojas

Last updated: 2026-07-15

## 1. Purpose

HistRegGUI registers a moving histology image into the coordinate system of a fixed target image using DeeperHistReg. It provides a desktop interface for Windows, macOS and Linux.

## 2. Release editions

- **CPU**: default and recommended for maximum portability. No NVIDIA GPU is required.
- **CUDA**: available for Windows and Linux. It contains a CUDA-enabled PyTorch runtime but still starts and runs in CPU mode unless the user explicitly enables CUDA.
- **macOS Intel / Apple Silicon**: CPU execution. NVIDIA CUDA is not supported on macOS.

## 3. Input procedure

1. Open HistRegGUI.
2. Select the **Target (Fixed)** image.
3. Select the **Moving (Warp)** image.
4. Choose a registration preset.
5. Optionally enable **Save intermediate results**.
6. Optionally open **Hardware → Check CUDA availability...**.
7. Enable **Use CUDA acceleration (NVIDIA)** only when the check succeeds.
8. Click **Run registration**.

Accepted picker formats are TIFF/TIF, JPG/JPEG, PNG and BMP. Previews are downsampled only for display.

## 4. Hardware selection

CPU is selected by default on every build.

The CUDA check verifies:

- whether the bundled PyTorch runtime was compiled with CUDA;
- whether a compatible NVIDIA driver and GPU are visible;
- whether a small allocation on `cuda:0` succeeds.

When CUDA is unavailable, the checkbox remains disabled. Before each registration, all nested DeeperHistReg `device` values and `cuda` flags are normalized to the selected mode.

## 5. Output

The final warped image is written next to the fixed image:

```text
<moving>_warped_to_<fixed>.tif
```

When intermediate saving is enabled, the application retains:

```text
Run_<timestamp>/
```

Otherwise, that temporary folder is removed after a successful result.

## 6. Error handling

Failures are shown in a dialog and appended to:

```text
HistRegGUI_error.log
```

The log records the fixed and moving paths, whether CUDA was requested, and the Python traceback.

## 7. Validation

Registration results should be reviewed visually before quantitative analysis. Confirm that corresponding tissue structures, orientation and specimen region agree between fixed and warped images.

## 8. Distribution

GitHub Actions builds the application independently on Windows, Ubuntu and macOS runners. Version tags matching `v*` publish the generated archives in a GitHub Release.

The application is built with PyInstaller and includes the installed DeeperHistReg package, its resources, and binary image backends needed by the release.

## 9. Licensing

HistRegGUI wrapper code is distributed under the MIT License. DeeperHistReg and all bundled dependencies remain under their respective licenses. Review `THIRD_PARTY_NOTICES.md` before redistribution.
