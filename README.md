# HistRegGUI v1.0

Desktop GUI for histological image registration with **DeeperHistReg**.

HistRegGUI runs on Windows, Linux, macOS Intel, and macOS Apple Silicon. CPU is
the default execution mode. Separate Windows and Linux CUDA packages can use a
compatible NVIDIA GPU when CUDA is explicitly enabled inside the application.

![Hist Reg App Concept](assets/screenshots/DeeperHistReg_concept.png)

## Release downloads

A `v1.0` tag builds and publishes all six packages:

- `HistRegGUI-Windows-x64-CPU.zip`
- `HistRegGUI-Windows-x64-CUDA.zip`
- `HistRegGUI-Linux-x64-CPU.tar.gz`
- `HistRegGUI-Linux-x64-CUDA.tar.gz`
- `HistRegGUI-macOS-Intel-x64-CPU.zip`
- `HistRegGUI-macOS-Apple-Silicon-CPU.zip`

The CUDA archives are large because they include the CUDA-enabled PyTorch
runtime and NVIDIA libraries. The manual workflow input still allows `none`,
`windows`, `linux`, or `both`, with `both` selected by default.

## Quick start

1. Select the single **Target (Fixed)** image.
2. Use **Add images...** to select one or several **Moving** images.
3. Leave **Input reader** on **Auto (recommended)** or select a reader manually.
4. Choose a registration preset.
5. Use **Move up** and **Move down** to place moving images in the intended Z order.
6. Optionally keep intermediate outputs.
7. Optionally enable **Create merged OME-TIFF stack after registration**. Configure whether the fixed target is the first Z slice, the output downsample, XY calibration, and Z spacing.
8. Optionally enable **Use CUDA acceleration (NVIDIA)** when the hardware check succeeds.
9. Select **Run registration** or **Run registration batch**.

With one moving image, the warped image is saved next to the fixed image as:

```text
<moving>_warped_to_<fixed>.tif
```

With several moving images, HistRegGUI processes them sequentially against the
same fixed target and creates a timestamped folder:

```text
HistRegGUI_batch_<fixed>_<timestamp>/
├── warped/
│   ├── 001_<moving>_warped_to_<fixed>.tif
│   └── 002_<moving>_warped_to_<fixed>.tif
├── intermediate/                 # only retained when requested or after a failure
├── merged/                       # when merged-volume export is enabled
│   ├── HistRegGUI_registered_stack_<fixed>.ome.tif
│   └── HistRegGUI_registered_stack_<fixed>_stack.json
├── registration_manifest.csv
├── registration_manifest.json
└── HistRegGUI_error.log          # when registration or stack creation fails
```

Batch processing is sequential by design so GPU and system memory are released
between large registrations. A failed moving image is recorded and the remaining
images continue processing.

![Hist Reg App Screenshot](assets/screenshots/DeeperHistReg.png)


## One target, multiple moving images

The moving-image table supports multi-file selection, duplicate prevention,
removal/clearing, per-image reader display, queue/running/success/failure status,
and click-to-preview. Automatic reader selection is resolved independently for
each moving/fixed pair, so one batch may use different readers for TIFF, WSI,
raster, or mixed-format inputs. The latest successful warped image is displayed
in the result preview.

Each run writes CSV and JSON manifests containing the source path, output path,
reader, device, preset, timestamps, and error status for every moving image.

## Merged 3-D OME-TIFF output

The optional merged-volume export creates a tiled, compressed **BigTIFF OME-TIFF** with axes `ZYXS` (Z, Y, X, RGB samples). This organization follows the successful stack pattern used in the accompanying research notebook, while replacing its all-in-memory `numpy.stack`/`tifffile.imread` approach with streaming output.

The stack order is explicit:

1. The fixed target is written as Z=0 when **Include fixed target as first Z slice** is enabled.
2. Successful warped images follow in the order shown in the moving-image table.
3. Failed registrations are omitted and recorded in the run manifest.

The writer uses 256 × 256 tiles, Deflate compression, and BigTIFF. It opens one source at a time and yields one small tile at a time to `tifffile`; it does not create a complete in-memory volume. A partial file is removed if writing fails, and the completed file is structurally checked for BigTIFF, OME-XML, axes, and shape before it replaces the final path.

For large WSI datasets, the default merge downsample is 4×. Available values are 1×, 2×, 4×, 8×, 16×, and 32×. XY pixel size is read from OME-TIFF, TIFF resolution tags, OpenSlide metadata, or libvips when possible; it can also be entered manually. Output XY calibration is multiplied by the selected downsample. Z spacing is user-editable and defaults to 4 µm.

A JSON sidecar beside the OME-TIFF records every Z index, fixed/warped role, original source path, reader backend, physical calibration, output shape, tile size, compression, and downsample. The registration JSON manifest also records whether stack creation succeeded.

## Supported inputs

The file picker and preview system support:

- TIFF and OME-TIFF: `.tif`, `.tiff`, `.ome.tif`, `.ome.tiff`
- Whole-slide/pathology: `.svs`, `.ndpi`, `.mrxs`, `.scn`, `.vms`, `.vmu`, `.bif`, `.svslide`, `.dcm`
- Standard raster: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.jp2`, `.j2k`

Automatic registration-reader selection uses:

- **TIFF loader** for TIFF/OME-TIFF pairs
- **OpenSlide loader** for whole-slide pairs
- **Pillow loader** for ordinary raster pairs
- **libvips loader** for mixed-format pairs

The preview path is independent from registration and tries multiple backends:
tifffile, OpenSlide, libvips, Pillow, and SimpleITK. OME-TIFF prefers tifffile
so scientific axes and pyramid information are retained for preview selection.
Large pyramidal images use reduced levels or sampled access rather than a full
image decode whenever the backend permits it.

A failed preview does not clear the chosen file. The user can still select a
manual registration reader for troubleshooting.

## Pillow/Tk packaging compatibility

The PyInstaller build explicitly bundles:

- `PIL.ImageTk`
- `PIL._tkinter_finder`
- `PIL._imagingtk`
- `tkinter` and `_tkinter`
- Pillow image plugins and binary components

A compatibility alias also exposes `PIL.tkinter_finder` for older diagnostics
that omit the underscore. Every platform job imports both names before building
and repeats the check inside the packaged executable smoke test.

## CUDA behavior

CPU remains the default. HistRegGUI does not hide CUDA or monkey-patch PyTorch.

The **Hardware** menu provides:

- **Check CUDA availability...**
- **Use CUDA acceleration when available**
- **Build information...**

The CUDA probe checks the PyTorch build, driver visibility, device names, and a
small allocation. If the check fails, CUDA is disabled and registration remains
on CPU. macOS builds are CPU-only.

## Build with GitHub Actions

The workflow is located at:

```text
.github/workflows/build-release.yml
```

For a test build:

1. Open **Actions → Build desktop releases**.
2. Select **Run workflow**.
3. Leave `cuda_target` as `both` to generate all six packages.

For the v1.0 release:

```bash
python scripts/validate_release_metadata.py --tag v1.0
git add .
git commit -m "Release HistRegGUI v1.0"
git push origin main
git tag v1.0
git push origin v1.0
```

The tag run tests, packages, smoke-tests, and publishes all platform archives in
a GitHub Release. macOS applications are ad-hoc signed and verified, but they
are not notarized with an Apple Developer ID.

## Zenodo DOI

The repository includes:

- `CITATION.cff` for GitHub citation rendering and interoperable software metadata
- `.zenodo.json` for Zenodo-specific release metadata
- `ZENODO_RELEASE.md` with the one-time connection and release instructions

Before creating the first release, sign in to Zenodo with GitHub, synchronize
repositories, and enable `Juaco2r/HistRegWSIGUI`. Once enabled, the GitHub
Release created from tag `v1.0` is automatically ingested by Zenodo and receives
a DOI. Repository files cannot perform that one-time account authorization.

## Local Windows CPU build

From PowerShell:

```powershell
./scripts/build_windows.ps1
```

The archive is created under `release-assets/`.

## Troubleshooting

Registration failures are appended to:

```text
HistRegGUI_error.log
```

The application records the chosen loader and whether CUDA was requested.

## Licensing

- HistRegGUI wrapper: MIT License
- DeeperHistReg and bundled dependencies: their respective upstream licenses

See `THIRD_PARTY_NOTICES.md`.
