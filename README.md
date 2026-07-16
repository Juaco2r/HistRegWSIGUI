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

1. Select the **Target (Fixed)** image. In cascading mode this is **Slice 1**.
2. Use **Add images...** to select the following moving images. Their table order is the slice order.
3. Choose a registration mode:
   - **Independent**: every moving image is registered directly to the same fixed target.
   - **Cascading**: Slice 2 is warped to Slice 1, Slice 3 is warped to the already warped Slice 2, and this continues through the ordered series.
4. Choose **Registration downsample** from 1× to 32×. Values above 1 create streamed tiled OME-TIFF working images and produce warped outputs at that reduced resolution.
5. For IF or other scientific multichannel OME-TIFF files, leave **Preserve all IF channels** enabled. The registration guide defaults to DAPI/Hoechst when channel names are available; a channel number, maximum projection, or mean projection can also be selected.
6. Leave **Input reader** on **Auto (recommended)** or select a reader manually. Downsampled or multichannel guide images use the TIFF reader automatically.
7. Choose a registration preset.
8. Use **Move up** and **Move down** to correct the consecutive Z order.
9. Optionally retain intermediate/temporary working images.
10. Optionally enable **Create merged OME-TIFF stack after registration**. Choose RGB display, scientific multichannel, or both; then configure fixed-slice inclusion, additional merge downsample, original XY calibration, and Z spacing.
11. Optionally enable **Use CUDA acceleration (NVIDIA)** when the hardware check succeeds.
12. Start the independent batch or cascading registration.

There is no application-level slice-count limit. Registrations are performed one
at a time and the final volume is written one tile at a time, so the practical
limits are processing time, disk capacity, and the operating system's file
selection limits rather than RAM proportional to the number of slices.

The original single-image, full-resolution independent output remains:

```text
<moving>_warped_to_<fixed>.tif
```

Independent multi-image runs create:

```text
HistRegGUI_batch_<fixed>_<timestamp>/
├── warped/
│   ├── 001_<moving>_warped_to_<fixed>.tif
│   └── 002_<moving>_warped_to_<fixed>.tif
├── intermediate/
├── merged/
├── registration_manifest.csv
├── registration_manifest.json
└── HistRegGUI_error.log
```

Cascading runs always receive a dedicated folder because every step depends on
the previous warped output:

```text
HistRegGUI_cascade_<slice1>_<timestamp>/
├── reference/
│   └── 000_fixed_<slice1>_regds4.ome.tif   # only when registration downsample > 1
├── warped/                              # RGB registration guides/results
│   ├── 001_<slice2>_cascaded_to_000_<slice1>_regds4.tif
│   ├── 002_<slice3>_cascaded_to_001_<slice2>_regds4.tif
│   └── ...
├── warped_scientific/                   # CYX OME-TIFF, all original channels
│   ├── 001_<slice2>_cascaded_scientific_regds4.ome.tif
│   └── ...
├── working/                               # retained only when requested
├── intermediate/                          # retained when requested or after failure
├── merged/
│   ├── HistRegGUI_cascade_stack_<slice1>.ome.tif
│   └── HistRegGUI_cascade_stack_<slice1>_stack.json
├── registration_manifest.csv
├── registration_manifest.json
└── HistRegGUI_error.log
```

A cascading failure stops the chain. Later slices are marked
`skipped_dependency`, because registering them to a different target would no
longer represent the requested consecutive sequence. A successful prefix can
still be exported as a clearly documented partial merged stack.

![Hist Reg App Screenshot](assets/screenshots/DeeperHistReg.png)


## Cascading consecutive-slice registration

Cascading mode follows the ordered dependency chain `Slice 2 → Slice 1`,
`Slice 3 → warped Slice 2`, `Slice 4 → warped Slice 3`, and so on. Each output
therefore remains in the coordinate system propagated from the first slice,
while every local registration compares anatomically adjacent sections. This is
useful for long histological series where direct registration of distant slices
to one reference may be less stable. As with any cascade, local errors and
resampling effects can accumulate, so the CSV/JSON manifest records the exact
source and actual target used at every step.

Registration downsampling is performed before DeeperHistReg runs. HistRegGUI
creates each moving working image only when its turn begins, registers it, then
removes the temporary copy unless retention was requested. The downsampled first
reference is retained because it defines the output geometry and can be included
as Z=0. This keeps disk and memory usage bounded for very long sequences.

## One target, multiple moving images

The moving-image table supports multi-file selection, duplicate prevention,
removal/clearing, per-image reader display, queue/running/success/failure status,
and click-to-preview. In independent mode, automatic reader selection is resolved
for each moving/fixed pair. In cascading mode it is resolved for the actual
source/previous-warped-target pair at each step. A run may therefore use
different readers for TIFF, WSI, raster, or mixed-format inputs. The latest
successful warped image is displayed in the result preview.

Each run writes CSV and JSON manifests containing the source path, output path,
reader, device, preset, timestamps, and error status for every moving image.


## Registering four-channel IF with H&E

HistRegGUI does not send a four-channel fluorescence array directly through the normal RGB registration path. Instead it separates alignment from scientific data preservation:

1. An RGB `uint8` **registration guide** is created from each image. H&E keeps its RGB appearance. IF uses DAPI/Hoechst automatically when recognized, or the selected channel/composite. The optional inversion makes bright fluorescence nuclei dark on a white background, which can improve structural similarity to H&E.
2. DeeperHistReg calculates one displacement field from the guide pair.
3. That same field is applied to every original IF channel independently with zero-valued fluorescence background. Channel names, channel count, and integer dtype are retained in a `CYX` OME-TIFF. The warped file is calibrated on the fixed/target guide grid, so H&E and IF scanner pixel sizes are not incorrectly mixed.
4. In a cascade, the previous **warped guide** is the next registration target, while the separate channel-preserving warped payload is retained for analysis and volume export.

The expected IF input is a 2-D TIFF/OME-TIFF with axes such as `CYX` or `YXC`. The paired H&E image may be TIFF/OME-TIFF, a standard raster image, or a libvips-supported whole-slide format such as SVS, NDPI, MRXS, or SCN. For files containing additional Z or T dimensions, HistRegGUI consistently uses the first Z/T plane; a true 3-D or time-series registration should first be split into the intended 2-D sections.

A mixed H&E/IF scientific merge uses OME-TIFF axes `ZCYX`. It creates a union channel schema such as `H&E Red`, `H&E Green`, `H&E Blue`, `DAPI`, `FITC`, `TRITC`, and `Cy5`. Channels absent from a particular Z section are zero-filled. If any IF source is `uint16`, RGB `uint8` values are expanded to `uint16` while fluorescence intensities are preserved. An optional RGB `ZYXS` guide stack can be written alongside it for rapid visual quality control.

## Merged 3-D OME-TIFF output

The optional merged-volume export can create an RGB guide stack with axes `ZYXS`, a channel-preserving scientific stack with axes `ZCYX`, or both. The RGB output is a tiled, compressed **BigTIFF OME-TIFF** (Z, Y, X, RGB samples). This organization follows the successful stack pattern used in the accompanying research notebook, while replacing its all-in-memory `numpy.stack`/`tifffile.imread` approach with streaming output.

The stack order is explicit:

1. The fixed target is written as Z=0 when **Include fixed target as first Z slice** is enabled.
2. Successful warped images follow in the order shown in the moving-image table.
3. Failed registrations are omitted and recorded in the run manifest.

The writer uses 256 × 256 tiles, Deflate compression, and BigTIFF. It opens one source at a time and yields one small tile at a time to `tifffile`; it does not create a complete in-memory volume. A partial file is removed if writing fails, and the completed file is structurally checked for BigTIFF, OME-XML, axes, and shape before it replaces the final path.

For large WSI datasets, the default **additional merge downsample** is 4×. Available values are 1×, 2×, 4×, 8×, 16×, and 32×. When registration itself used a downsample, the nominal total reduction is `registration downsample × merge downsample`. XY pixel size is read from OME-TIFF, TIFF resolution tags, OpenSlide metadata, or libvips when possible; it can also be entered manually. The downsampled reference records the actual X/Y scale after integer dimension rounding, and the merged output applies the additional merge factor. Z spacing is user-editable and defaults to 4 µm.

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
