from __future__ import annotations

"""Image-format detection, previews, and DeeperHistReg loader selection.

The registration library uses one loader class for both input images.  This
module therefore selects the most specific common loader for a pair, while the
preview code can try several backends independently and fall back safely.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image


# Keep multi-suffix formats before their shorter suffixes when presenting them.
TIFF_EXTENSIONS: tuple[str, ...] = (
    ".ome.tiff",
    ".ome.tif",
    ".tiff",
    ".tif",
)

WSI_EXTENSIONS: tuple[str, ...] = (
    ".svs",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".vms",
    ".vmu",
    ".bif",
    ".svslide",
    ".dcm",
)

RASTER_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".jp2",
    ".j2k",
)

SUPPORTED_EXTENSIONS: tuple[str, ...] = TIFF_EXTENSIONS + WSI_EXTENSIONS + RASTER_EXTENSIONS

# Labels shown in the GUI. The values are DeeperHistReg's loader keys.
LOADER_CHOICES: dict[str, str] = {
    "Auto (recommended)": "auto",
    "TIFF / OME-TIFF": "tiff",
    "OpenSlide whole-slide image": "openslide",
    "libvips generic / mixed formats": "vips",
    "Pillow standard image": "pil",
}


@dataclass(frozen=True)
class PreviewInfo:
    reader: str
    width: int
    height: int
    source_width: int | None = None
    source_height: int | None = None
    levels: int | None = None
    axes: str | None = None

    def summary(self) -> str:
        dimensions = f"{self.source_width or self.width} × {self.source_height or self.height}"
        details = [self.reader, dimensions]
        if self.levels and self.levels > 1:
            details.append(f"{self.levels} levels")
        if self.axes:
            details.append(f"axes {self.axes}")
        return " | ".join(details)


def has_extension(path_or_name: str | Path, extensions: Iterable[str]) -> bool:
    name = str(path_or_name).lower()
    return any(name.endswith(extension) for extension in extensions)


def is_ome_tiff(path_or_name: str | Path) -> bool:
    return has_extension(path_or_name, (".ome.tif", ".ome.tiff"))


def tkinter_image_filetypes() -> list[tuple[str, str]]:
    supported = " ".join(f"*{extension}" for extension in SUPPORTED_EXTENSIONS)
    wsi = " ".join(f"*{extension}" for extension in WSI_EXTENSIONS)
    tiff = " ".join(f"*{extension}" for extension in TIFF_EXTENSIONS)
    raster = " ".join(f"*{extension}" for extension in RASTER_EXTENSIONS)
    return [
        ("Supported microscopy images", supported),
        ("Whole-slide images", wsi),
        ("TIFF / OME-TIFF", tiff),
        ("Standard raster images", raster),
        ("All files", "*.*"),
    ]


def preferred_loader_for_path(path: str | Path) -> str:
    """Return the most specific DeeperHistReg loader key for one path."""

    if has_extension(path, WSI_EXTENSIONS):
        return "openslide"
    if has_extension(path, TIFF_EXTENSIONS):
        return "tiff"
    if has_extension(path, RASTER_EXTENSIONS):
        return "pil"
    return "vips"


def choose_registration_loader(source: str | Path, target: str | Path) -> str:
    """Select one DeeperHistReg loader that can open both inputs.

    DeeperHistReg's full-resolution pipeline accepts one loader for the image
    pair.  Matching pairs use the most specific loader; mixed pairs use libvips,
    which is the broad generic backend bundled with the application.
    """

    source_loader = preferred_loader_for_path(source)
    target_loader = preferred_loader_for_path(target)

    if source_loader == target_loader:
        return source_loader

    # TIFF and OME-TIFF are both handled by the TIFF loader even when one name
    # has a compound suffix and the other has a normal TIFF suffix.
    if has_extension(source, TIFF_EXTENSIONS) and has_extension(target, TIFF_EXTENSIONS):
        return "tiff"

    return "vips"


def resolve_loader_choice(choice_label_or_key: str, source: str | Path, target: str | Path) -> str:
    key = LOADER_CHOICES.get(choice_label_or_key, choice_label_or_key).strip().lower()
    if key == "auto":
        return choose_registration_loader(source, target)
    if key not in {"tiff", "openslide", "vips", "pil"}:
        raise ValueError(f"Unsupported image loader: {choice_label_or_key}")
    return key


def deeperhistreg_loader_class(deeperhistreg_module: Any, loader_key: str):
    mapping = {
        "tiff": deeperhistreg_module.loaders.TIFFLoader,
        "openslide": deeperhistreg_module.loaders.OpenSlideLoader,
        "vips": deeperhistreg_module.loaders.VIPSLoader,
        "pil": deeperhistreg_module.loaders.PILLoader,
    }
    try:
        return mapping[loader_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported DeeperHistReg loader key: {loader_key}") from exc


def configure_registration_loader(parameters: dict[str, Any], loader_key: str) -> dict[str, Any]:
    loading = parameters.setdefault("loading_params", {})
    if not isinstance(loading, dict):
        raise TypeError("registration_parameters['loading_params'] must be a dictionary")
    loading["loader"] = loader_key
    return parameters


def _normalize_plane_to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255

    values = array.astype(np.float32, copy=False)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8)

    valid = values[finite]
    low = float(np.percentile(valid, 1.0))
    high = float(np.percentile(valid, 99.8))
    if high <= low:
        low = float(valid.min())
        high = float(valid.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)

    values = np.clip((values - low) / (high - low), 0.0, 1.0)
    return (values * 255.0).astype(np.uint8)


def _representative_array(array: np.ndarray, axes: str | None = None) -> np.ndarray:
    """Reduce T/Z/etc. to the first plane while preserving Y/X and C/S."""

    array = np.asarray(array)
    axes = (axes or "").strip().upper()

    if axes and len(axes) == array.ndim and "Y" in axes and "X" in axes:
        slicer: list[int | slice] = []
        kept_axes: list[str] = []
        for axis in axes:
            if axis in {"Y", "X", "C", "S"}:
                slicer.append(slice(None))
                kept_axes.append(axis)
            else:
                slicer.append(0)
        array = np.asarray(array[tuple(slicer)])
        reduced_axes = "".join(kept_axes)
        order = [reduced_axes.index("Y"), reduced_axes.index("X")]
        if "C" in reduced_axes:
            order.append(reduced_axes.index("C"))
        elif "S" in reduced_axes:
            order.append(reduced_axes.index("S"))
        return np.transpose(array, order)

    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[0] <= 16 and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    return array


def _array_to_rgb(array: np.ndarray, axes: str | None = None) -> np.ndarray:
    array = _representative_array(array, axes)

    if array.ndim == 2:
        gray = _normalize_plane_to_uint8(array)
        return np.stack((gray, gray, gray), axis=-1)

    if array.ndim != 3:
        raise ValueError(f"Unsupported preview array shape: {array.shape}")

    if array.shape[-1] in (3, 4):
        rgb = array[..., :3]
        if rgb.dtype == np.uint8:
            return np.ascontiguousarray(rgb)
        # Preserve RGB channel balance when data are integer-valued.
        if np.issubdtype(rgb.dtype, np.integer):
            info = np.iinfo(rgb.dtype)
            if info.max > 0:
                return np.clip(rgb.astype(np.float32) / info.max * 255.0, 0, 255).astype(np.uint8)
        return _normalize_plane_to_uint8(rgb)

    # Scientific multichannel image: create a simple additive RGB composite
    # from the first three channels. A single/two-channel image becomes gray.
    channels = int(array.shape[-1])
    if channels <= 2:
        gray = _normalize_plane_to_uint8(array[..., 0])
        return np.stack((gray, gray, gray), axis=-1)

    output = np.zeros((*array.shape[:2], 3), dtype=np.uint8)
    for channel in range(3):
        output[..., channel] = _normalize_plane_to_uint8(array[..., channel])
    return output


def _fit_rgb(rgb: np.ndarray, max_side: int) -> Image.Image:
    image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _preview_with_pillow(path: Path, max_side: int) -> tuple[Image.Image, PreviewInfo]:
    with Image.open(path) as source:
        source_width, source_height = source.size
        image = source.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        preview = image.copy()
    return preview, PreviewInfo(
        reader="Pillow",
        width=preview.width,
        height=preview.height,
        source_width=source_width,
        source_height=source_height,
        levels=1,
    )


def _preview_with_openslide(path: Path, max_side: int) -> tuple[Image.Image, PreviewInfo]:
    import openslide

    slide = openslide.OpenSlide(str(path))
    try:
        source_width, source_height = slide.dimensions
        preview = slide.get_thumbnail((max_side, max_side)).convert("RGB")
        return preview, PreviewInfo(
            reader="OpenSlide",
            width=preview.width,
            height=preview.height,
            source_width=int(source_width),
            source_height=int(source_height),
            levels=int(slide.level_count),
        )
    finally:
        slide.close()


def _vips_to_numpy(image: Any) -> np.ndarray:
    if hasattr(image, "numpy"):
        return np.asarray(image.numpy())

    format_to_dtype = {
        "uchar": np.uint8,
        "char": np.int8,
        "ushort": np.uint16,
        "short": np.int16,
        "uint": np.uint32,
        "int": np.int32,
        "float": np.float32,
        "double": np.float64,
    }
    dtype = format_to_dtype.get(str(image.format), np.uint8)
    memory = image.write_to_memory()
    return np.frombuffer(memory, dtype=dtype).reshape(image.height, image.width, image.bands)


def _preview_with_pyvips(path: Path, max_side: int) -> tuple[Image.Image, PreviewInfo]:
    import pyvips

    # thumbnail() asks libvips/OpenSlide for a reduced representation where the
    # source format supports it, avoiding a full WSI decode merely for display.
    try:
        source = pyvips.Image.new_from_file(str(path), access="sequential")
        source_width, source_height = int(source.width), int(source.height)
        levels = int(source.get_n_pages()) if hasattr(source, "get_n_pages") else 1
    except Exception:
        source = None
        source_width = source_height = 0
        levels = 1

    preview_vips = pyvips.Image.thumbnail(
        str(path),
        int(max_side),
        height=int(max_side),
        size="down",
        auto_rotate=True,
    )
    if preview_vips.bands == 1:
        preview_vips = preview_vips.bandjoin([preview_vips, preview_vips])
    elif preview_vips.bands == 2:
        first_band = preview_vips.extract_band(0)
        preview_vips = first_band.bandjoin([first_band, first_band])
    elif preview_vips.bands > 3:
        preview_vips = preview_vips.extract_band(0, n=3)

    rgb = _array_to_rgb(_vips_to_numpy(preview_vips), "YXS")
    preview = _fit_rgb(rgb, max_side)
    return preview, PreviewInfo(
        reader="libvips",
        width=preview.width,
        height=preview.height,
        source_width=source_width or None,
        source_height=source_height or None,
        levels=levels,
    )


def _tifffile_spatial_shape(shape: Sequence[int], axes: str) -> tuple[int, int]:
    if axes and len(axes) == len(shape) and "Y" in axes and "X" in axes:
        return int(shape[axes.index("X")]), int(shape[axes.index("Y")])
    if len(shape) >= 2:
        return int(shape[-1]), int(shape[-2])
    raise ValueError(f"TIFF series has no spatial dimensions: {shape}")


def _sample_tiff_zarr(series: Any, axes: str, max_side: int) -> np.ndarray:
    import zarr

    store = series.aszarr()
    array = zarr.open(store, mode="r")
    axes = axes if axes and len(axes) == array.ndim else ""

    if axes and "Y" in axes and "X" in axes:
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
    else:
        y_axis = array.ndim - 2
        x_axis = array.ndim - 1

    source_height = int(array.shape[y_axis])
    source_width = int(array.shape[x_axis])
    step = max(1, int(np.ceil(max(source_width, source_height) / float(max_side))))

    slicer: list[int | slice] = []
    kept_axes: list[str] = []
    for index in range(array.ndim):
        axis = axes[index] if axes else ""
        if index == y_axis:
            slicer.append(slice(0, source_height, step))
            kept_axes.append("Y")
        elif index == x_axis:
            slicer.append(slice(0, source_width, step))
            kept_axes.append("X")
        elif axis in {"C", "S"}:
            slicer.append(slice(None))
            kept_axes.append(axis)
        else:
            slicer.append(0)

    sampled = np.asarray(array[tuple(slicer)])
    return _array_to_rgb(sampled, "".join(kept_axes))


def _preview_with_tifffile(path: Path, max_side: int) -> tuple[Image.Image, PreviewInfo]:
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        axes = str(getattr(series, "axes", "") or "")
        shape = tuple(int(value) for value in series.shape)
        source_width, source_height = _tifffile_spatial_shape(shape, axes)
        levels = list(getattr(series, "levels", ()) or ())

        if len(levels) > 1:
            # The smallest pyramid image is ideal for a GUI preview.
            selected = levels[-1]
            selected_axes = str(getattr(selected, "axes", axes) or axes)
            rgb = _array_to_rgb(selected.asarray(), selected_axes)
        else:
            spatial_pixels = source_width * source_height
            if spatial_pixels <= 25_000_000:
                rgb = _array_to_rgb(series.asarray(), axes)
            else:
                rgb = _sample_tiff_zarr(series, axes, max_side)

    preview = _fit_rgb(rgb, max_side)
    return preview, PreviewInfo(
        reader="tifffile",
        width=preview.width,
        height=preview.height,
        source_width=source_width,
        source_height=source_height,
        levels=max(1, len(levels)),
        axes=axes or None,
    )


def _preview_with_simpleitk(path: Path, max_side: int) -> tuple[Image.Image, PreviewInfo]:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    source_size = image.GetSize()
    array = sitk.GetArrayFromImage(image)
    rgb = _array_to_rgb(array)
    preview = _fit_rgb(rgb, max_side)
    source_width = int(source_size[0]) if len(source_size) >= 1 else preview.width
    source_height = int(source_size[1]) if len(source_size) >= 2 else preview.height
    return preview, PreviewInfo(
        reader="SimpleITK",
        width=preview.width,
        height=preview.height,
        source_width=source_width,
        source_height=source_height,
        levels=1,
    )


def _preview_attempts(path: Path) -> list[tuple[str, Callable[[Path, int], tuple[Image.Image, PreviewInfo]]]]:
    if is_ome_tiff(path):
        return [
            ("tifffile", _preview_with_tifffile),
            ("libvips", _preview_with_pyvips),
            ("OpenSlide", _preview_with_openslide),
            ("Pillow", _preview_with_pillow),
        ]
    if has_extension(path, WSI_EXTENSIONS):
        return [
            ("OpenSlide", _preview_with_openslide),
            ("libvips", _preview_with_pyvips),
            ("tifffile", _preview_with_tifffile),
            ("SimpleITK", _preview_with_simpleitk),
            ("Pillow", _preview_with_pillow),
        ]
    if has_extension(path, TIFF_EXTENSIONS):
        return [
            ("tifffile", _preview_with_tifffile),
            ("libvips", _preview_with_pyvips),
            ("OpenSlide", _preview_with_openslide),
            ("Pillow", _preview_with_pillow),
        ]
    if has_extension(path, RASTER_EXTENSIONS):
        return [
            ("Pillow", _preview_with_pillow),
            ("libvips", _preview_with_pyvips),
        ]
    return [
        ("libvips", _preview_with_pyvips),
        ("Pillow", _preview_with_pillow),
        ("tifffile", _preview_with_tifffile),
        ("OpenSlide", _preview_with_openslide),
        ("SimpleITK", _preview_with_simpleitk),
    ]


def load_image_preview(path: str | Path, max_side: int = 600) -> tuple[Image.Image, PreviewInfo]:
    """Load a memory-conscious preview using several optional backends."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    errors: list[str] = []
    for reader_name, reader in _preview_attempts(path):
        try:
            return reader(path, max_side)
        except Exception as exc:
            errors.append(f"{reader_name}: {type(exc).__name__}: {exc}")

    formatted = "\n".join(f"- {error}" for error in errors)
    raise RuntimeError(
        f"Could not preview image: {path}\n\nReaders tried:\n{formatted}\n\n"
        "The image may still be supported by a manually selected registration loader, "
        "but its preview could not be decoded."
    )


def supported_formats_text() -> str:
    return ", ".join(SUPPORTED_EXTENSIONS)
