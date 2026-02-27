# HistRegGUI Protocol
Version: 1.0
Author: Jose Rodriguez-Rojas  
Last updated: 2026  
License (wrapper): MIT  
Registration engine: DeeperHistReg (CC BY-SA 4.0)

---

## 1. General Description

HistRegGUI is a desktop application designed to perform histological image registration using the **DeeperHistReg** engine.

It allows the user to:
- Select a **fixed (target)** image and a **moving (warp)** image
- Choose a registration **preset** (rigid, nonrigid, initial, etc.)
- Run registration on **CPU**
- Save the **warped moving image aligned to the fixed image**

---

## 2. System Requirements

- Operating system: Windows 10 or later
- Architecture: 64-bit
- CPU compatible with PyTorch (GPU not required)
- Sufficient disk space for intermediate files and TIFF outputs
- The Windows release is portable and includes required runtime components

---

## 3. Supported Formats

The file picker accepts:

- TIFF (`.tif`, `.tiff`)
- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- BMP (`.bmp`)

Previews shown in the GUI are downsampled for speed and do not affect registration.

---

## 4. Usage Procedure

1. Run `HistRegGUI.exe`.
2. Select the **Target (Fixed)** image using the corresponding **Select...** button.
3. Select the **Moving (Warp)** image (the image to be deformed).
4. Choose a **registration preset** from the dropdown menu.
5. (Optional) Enable **Save intermediate results** to keep intermediate outputs.
6. Click **Run registration** and wait for completion.
7. Review the output image saved in the same folder as the fixed image.

---

## 5. Registration Presets

Presets are detected dynamically from `deeperhistreg.configs`:

- Only callable config factories with **no arguments** that return a **dict** are included.
- Presets are forced to run with `device="cpu"` even if the original config uses CUDA.

Common presets include (names may vary by DeeperHistReg version):

- **Initial + Nonrigid (default)**: initial alignment followed by non-rigid deformation
- **Nonrigid**: non-rigid deformation only
- **Rigid**: rigid transformation (rotation + translation)
- **Initial**: initial alignment only

---

## 6. Outputs

Generated artifacts include:

- **Warped image (final output)**:  
  `<moving>_warped_to_<fixed>.tif`

- **Temporary run folder** (if intermediate saving is enabled):  
  `Run_<timestamp>`

- **Error log** (only if a failure occurs):  
  `HistRegGUI_error.log`

---

## 7. Interpretation of Results

The saved output corresponds to the **moving image warped into the coordinate system of the fixed image**.

Recommended validation:
- visually inspect alignment of key histological structures
- verify orientation and content correspond to the same specimen/region

---

## 8. Error Handling

If an error occurs during registration:

- A popup window will show error details.
- A full traceback is written to `HistRegGUI_error.log` in the output folder.
- Recommended checks:
  - confirm both images are readable and not corrupted
  - confirm images correspond to the same specimen / comparable region
  - try a different preset (e.g., rigid only) to diagnose issues

---

## 9. Technical Considerations

- Registration is executed on **CPU only**.
- CUDA is explicitly disabled to improve portability and avoid CUDA initialization failures.
- Large images can require substantial RAM and time.
- For best results, use images with comparable scale and orientation.

---

## 10. Licensing Notes

- HistRegGUI wrapper code is provided under the MIT License.
- DeeperHistReg is licensed under **CC BY-SA 4.0**. If you redistribute DeeperHistReg (e.g., bundled in the release ZIP), you must comply with its license, including attribution and ShareAlike requirements where applicable.
