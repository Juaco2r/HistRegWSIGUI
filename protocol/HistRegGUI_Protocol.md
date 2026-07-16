# HistRegGUI Protocol

Version: 1.0

Author: Jose Rodriguez-Rojas

Last updated: 2026-07-16

## 1. Purpose

HistRegGUI registers histological images with DeeperHistReg using either:

- **Independent registration**: one or more moving images are each registered directly to one fixed target.
- **Cascading consecutive-slice registration**: Slice 2 is warped to Slice 1, Slice 3 is warped to the already warped Slice 2, and the dependency chain continues through the ordered series.

The application also provides optional streamed registration downsampling and memory-efficient 3-D OME-TIFF stack creation on Windows, macOS, and Linux.

## 2. Release editions

- **CPU**: default and recommended for maximum portability. No NVIDIA GPU is required.
- **CUDA**: available for Windows and Linux. It contains a CUDA-enabled PyTorch runtime but still starts in CPU mode unless the user explicitly enables CUDA.
- **macOS Intel / Apple Silicon**: CPU execution. NVIDIA CUDA is not supported on macOS.

## 3. Input procedure

1. Open HistRegGUI.
2. Select the **Target (Fixed)** image. In cascading mode this is Slice 1.
3. Click **Add images...** and select the following moving/consecutive slices.
4. Use **Move up** and **Move down** to establish the exact order.
5. Choose **Independent** or **Cascading** registration.
6. Choose **Registration downsample**: 1×, 2×, 4×, 8×, 16×, or 32×.
7. Keep **Input reader** on automatic or choose a reader manually. Registration downsample values above 1 create TIFF working images and therefore use the TIFF reader.
8. Choose a registration preset.
9. For IF/OME-TIFF, keep **Preserve all IF channels** enabled and select the registration guide. Auto prefers a channel named DAPI, Hoechst, nuclei, or nuclear.
10. Optionally enable **Save intermediate and temporary working images**.
11. Optionally enable **Create merged OME-TIFF stack after registration**, choose RGB display, scientific multichannel, or both, and set first-slice inclusion, additional merge downsample, original XY pixel size, and Z spacing.
11. Optionally open **Hardware → Check CUDA availability...**.
12. Enable **Use CUDA acceleration (NVIDIA)** only when the check succeeds.
13. Start the registration.

Accepted picker formats include TIFF/OME-TIFF, SVS, NDPI, MRXS, SCN, VMS, VMU, BIF, SVSLIDE, DICOM, JPG/JPEG, PNG, BMP, WebP, and JPEG 2000. Previews use tifffile, OpenSlide, libvips, Pillow, and SimpleITK fallbacks and are reduced only for display.

## 4. Registration behavior

### 4.1 Independent mode

- Every moving image is registered to the same fixed target.
- One failed image is logged and the remaining images continue.
- Automatic reader selection is resolved for every moving/fixed pair.

### 4.2 Cascading mode

- The fixed image is Slice 1.
- The first moving image is registered and warped to Slice 1.
- Each later image is registered to the immediately previous **warped** output, not to the original previous input.
- The ordered chain is therefore `2 → 1`, `3 → warped 2`, `4 → warped 3`, and so forth.
- A failed step stops the chain. Later slices are marked `skipped_dependency`, because their required previous warped target does not exist.
- The manifest records the actual source and target used for every step.
- Local errors and interpolation effects can accumulate along a long cascade, so every result must be reviewed visually.

There is no application-level slice-count limit. Images are processed sequentially and the list of paths is the only memory that grows with the number of slices.

### 4.3 Registration downsampling

For factors above 1, HistRegGUI creates tiled OME-BigTIFF working images by streaming one small region at a time. It does not load the complete slide into RAM.

- The downsampled first/fixed reference is retained because it defines the output geometry.
- Each moving working image is created immediately before its registration step.
- Temporary moving working images are removed after use unless intermediate retention is enabled.
- Warped outputs are produced at the selected reduced resolution; this is not a full-resolution warp.
- Physical X/Y calibration is scaled using the actual output dimensions, including odd-size rounding.


### 4.4 Multichannel IF and H&E

- Each selected file represents one 2-D section. Multichannel IF should be TIFF/OME-TIFF `CYX` or `YXC`; H&E may also be a libvips-supported whole-slide format such as SVS, NDPI, MRXS, or SCN. The first plane is used for additional Z/T dimensions.
- H&E is converted to an RGB guide. IF is converted to an RGB guide from DAPI/Hoechst, a selected channel, a maximum composite, or a mean composite. Optional guide inversion maps bright fluorescence nuclei to dark structures on a light background.
- DeeperHistReg estimates the displacement field from the RGB guide pair.
- The field is then applied to all original source channels with zero fluorescence padding. The scientific warped output is OME-TIFF `CYX`, retains channel names, channel count, and dtype, and adopts the fixed/target registration-grid calibration.
- In cascading mode, the previous warped RGB guide is the next target. Scientific payloads remain separate and do not lose channels as the cascade proceeds.
- The scientific merged output uses `ZCYX`. H&E RGB and IF marker channels form a union schema; missing channels in each Z plane are written as zeros.

## 5. Hardware selection

CPU is selected by default on every build.

The CUDA check verifies:

- whether the bundled PyTorch runtime was compiled with CUDA;
- whether a compatible NVIDIA driver and GPU are visible;
- whether a small allocation on `cuda:0` succeeds.

When CUDA is unavailable, the checkbox remains disabled. Before each registration, all nested DeeperHistReg `device` values and `cuda` flags are normalized to the selected mode.

## 6. Output

The original one-moving-image, full-resolution independent output is written next to the fixed image:

```text
<moving>_warped_to_<fixed>.tif
```

Independent batches create `HistRegGUI_batch_<fixed>_<timestamp>/`. Cascading runs create `HistRegGUI_cascade_<slice1>_<timestamp>/` with ordered warped files, manifests, logs, and optional reference/working/merged folders.

Numbered filenames prevent collisions and preserve the requested section order.

### Optional merged volume

HistRegGUI can write a tiled, Deflate-compressed RGB guide BigTIFF OME-TIFF with axes `ZYXS`, a channel-preserving scientific OME-TIFF with axes `ZCYX`, or both. The first fixed/reference image can be Z=0, followed by successful warped images in table order. The writer opens one image at a time and yields one 256 × 256 tile at a time; it never constructs the complete volume in RAM.

The merge downsample is additional to the registration downsample. For example, a 4× registration downsample plus a 2× merge downsample gives a nominal total reduction of 8×. The retained downsampled reference stores the actual physical X/Y calibration, and Z spacing is supplied in micrometres.

If a cascade fails after a successful prefix, the application may export that prefix as a clearly documented partial stack. The JSON sidecar records every included Z index and original source path.

## 7. Manifests and error handling

Every run writes CSV and JSON manifests containing:

- registration mode and registration downsample;
- original first fixed image and moving inputs;
- actual registration source and target for every step;
- output and intermediate paths;
- selected reader, device, and preset;
- start and finish timestamps;
- success, failure, or cascade-dependency skip status;
- merged-volume settings and output metadata;
- error text when applicable.

Failures are also appended to `HistRegGUI_error.log` with the Python traceback.

## 8. Validation

Registration results should be reviewed visually before quantitative analysis. Confirm that corresponding tissue structures, orientation, specimen region, and cumulative cascade behavior remain plausible. For long series, inspect several checkpoints rather than only the first and final slices.

## 9. Distribution

GitHub Actions builds the application independently on Windows, Ubuntu, and macOS runners. Version tags matching `v*` publish the generated archives in a GitHub Release. When the public repository is enabled in Zenodo, the GitHub Release for `v1.0` is ingested and receives a DOI.

The application is built with PyInstaller and includes the installed DeeperHistReg package, its resources, and binary image backends needed by the release.

## 10. Licensing

HistRegGUI wrapper code is distributed under the MIT License. DeeperHistReg and all bundled dependencies remain under their respective licenses. Review `THIRD_PARTY_NOTICES.md` before redistribution.
