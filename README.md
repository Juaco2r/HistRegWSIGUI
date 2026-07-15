# HistRegGUI

Desktop GUI for histological image registration with **DeeperHistReg**.

The application runs on **Windows, macOS and Linux**. CPU remains the default execution mode. Windows and Linux releases also provide an optional CUDA edition that can use a compatible NVIDIA GPU when the user explicitly enables it.

![Hist Reg App Concept](assets/screenshots/DeeperHistReg_concept.png)

## Downloads

GitHub Actions creates these release assets when a tag such as `v1.1.0` is pushed:

- `HistRegGUI-Windows-x64-CPU.zip`
- `HistRegGUI-Windows-x64-CUDA.zip`
- `HistRegGUI-Linux-x64-CPU.tar.gz`
- `HistRegGUI-Linux-x64-CUDA.tar.gz`
- `HistRegGUI-macOS-Intel-x64-CPU.zip`
- `HistRegGUI-macOS-Apple-Silicon-CPU.zip`

The CPU downloads do not require an NVIDIA GPU or CUDA. The CUDA downloads contain a CUDA-enabled PyTorch runtime, but still run in CPU mode by default and can also be opened on a system without an NVIDIA GPU.

## Quick start

1. Select the **Target (Fixed)** image.
2. Select the **Moving (Warp)** image.
3. Choose a registration preset.
4. Optionally keep intermediate results.
5. Optionally enable **Use CUDA acceleration (NVIDIA)** when the hardware check reports CUDA as available.
6. Click **Run registration**.

The warped image is saved next to the fixed image as:

```text
<moving>_warped_to_<fixed>.tif
```

![Hist Reg App Screenshot](assets/screenshots/DeeperHistReg.png)

## CUDA behavior

CPU is always the default. The application no longer globally disables CUDA or monkey-patches PyTorch.

The **Hardware** menu provides:

- **Check CUDA availability...**: checks the PyTorch build, NVIDIA driver and GPU, then performs a small CUDA allocation.
- **Use CUDA acceleration when available**: enables `cuda:0` only after a successful check.
- **Build information...**: shows whether the downloaded application is a CPU or CUDA build.

When CUDA is not available, the checkbox is disabled and registration remains on CPU. DeeperHistReg's nested `device` and `cuda` parameters are normalized before each run so the chosen execution mode is applied consistently.

CUDA is not available on macOS. The application reports this normally and continues to work on CPU.

## GitHub Actions builds

The workflow is located at:

```text
.github/workflows/build-release.yml
```

It can be started in two ways:

- **Actions → Build desktop releases → Run workflow** for test artifacts.
- Push a version tag such as `v1.1.0` to build every platform and publish the archives in a GitHub Release.

Example:

```bash
git add .
git commit -m "Add multiplatform releases and optional CUDA"
git tag v1.1.0
git push origin main
git push origin v1.1.0
```

Manual workflow runs include a switch for skipping the large CUDA packages. Tag builds create CPU and CUDA editions automatically. Every job launches the packaged application in a non-GUI self-test mode before uploading it.

## Why DeeperHistReg is installed during the build

The previous Windows workflow attempted to bundle local `deeperhistreg/` and `external/` folders that were not tracked by Git. A clean GitHub runner therefore did not have the required content.

The new build installs the published `deeperhistreg` package, locates its installed source and model files, and includes them automatically in the PyInstaller bundle. It also installs self-contained OpenSlide and libvips Python binary packages for portability.

## Local Windows CPU build

From PowerShell:

```powershell
./scripts/build_windows.ps1
```

The archive is created under `release-assets/`.

## Supported inputs

The picker accepts TIFF/TIF, JPG/JPEG, PNG and BMP. Previews are downsampled only for display and do not modify the registration input.

## Troubleshooting

Registration failures are appended to:

```text
HistRegGUI_error.log
```

macOS applications produced by public GitHub Actions are unsigned. On first launch, macOS may require the usual **Open** confirmation from Finder or Privacy & Security.

CUDA packages are considerably larger than CPU packages because they include the NVIDIA CUDA runtime libraries used by PyTorch. The workflow uses the CUDA 11.8 wheel to maximize compatibility with supported NVIDIA drivers.

## Licensing

- HistRegGUI wrapper: MIT License.
- DeeperHistReg: see its upstream license and attribution requirements.
- PyTorch, Pillow, OpenSlide, libvips and other components remain under their respective licenses.

See `THIRD_PARTY_NOTICES.md`.
