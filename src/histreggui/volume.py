from __future__ import annotations

"""Memory-conscious creation of registered 3-D OME-TIFF stacks.

The writer deliberately streams one 256 x 256 tile at a time. It never builds
``numpy.stack([...])`` for all registered slides and never needs to keep more
than a small source region in memory. BigTIFF is always enabled so output is not
limited by the classic TIFF 4 GiB boundary.
"""

import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image

from histreggui.image_io import RASTER_EXTENSIONS, TIFF_EXTENSIONS, WSI_EXTENSIONS, has_extension


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class VolumeSlice:
    path: Path
    role: str = "warped"
    source_path: Path | None = None
    label: str | None = None


@dataclass(frozen=True)
class MergedVolumeResult:
    path: Path
    sidecar_json: Path
    z_slices: int
    width: int
    height: int
    axes: str
    voxel_xy_um: float | None
    voxel_z_um: float
    downsample: int
    tile_size: int
    compression: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["sidecar_json"] = str(self.sidecar_json)
        return data


@dataclass(frozen=True)
class WorkingImageResult:
    source_path: Path
    path: Path
    original_width: int
    original_height: int
    width: int
    height: int
    downsample: int
    scale_x: float
    scale_y: float
    pixel_size_x_um: float | None
    pixel_size_y_um: float | None
    reader: str
    tile_size: int
    compression: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["source_path"] = str(self.source_path)
        data["path"] = str(self.path)
        return data


@dataclass(frozen=True)
class _ReaderInfo:
    width: int
    height: int
    output_width: int
    output_height: int
    backend: str


class _BaseSliceReader:
    info: _ReaderInfo

    def read_tile(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _spatial_shape(shape: Sequence[int], axes: str) -> tuple[int, int]:
    axes = (axes or "").upper()
    if axes and len(axes) == len(shape) and "Y" in axes and "X" in axes:
        return int(shape[axes.index("X")]), int(shape[axes.index("Y")])
    if len(shape) >= 2:
        return int(shape[-1]), int(shape[-2])
    raise ValueError(f"Image has no Y/X dimensions: shape={tuple(shape)}, axes={axes!r}")


def _representative_yx(array: np.ndarray, axes: str = "") -> np.ndarray:
    array = np.asarray(array)
    axes = (axes or "").upper()
    if axes and len(axes) == array.ndim and "Y" in axes and "X" in axes:
        slicer: list[int | slice] = []
        kept: list[str] = []
        for axis in axes:
            if axis in {"Y", "X", "C", "S"}:
                slicer.append(slice(None))
                kept.append(axis)
            else:
                slicer.append(0)
        array = np.asarray(array[tuple(slicer)])
        kept_axes = "".join(kept)
        order = [kept_axes.index("Y"), kept_axes.index("X")]
        if "S" in kept_axes:
            order.append(kept_axes.index("S"))
        elif "C" in kept_axes:
            order.append(kept_axes.index("C"))
        array = np.transpose(array, order)
    else:
        array = np.squeeze(array)
        if array.ndim == 3 and array.shape[0] <= 16 and array.shape[-1] not in (1, 2, 3, 4):
            array = np.moveaxis(array, 0, -1)
    return np.asarray(array)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        if info.max <= 255:
            return np.clip(array, 0, 255).astype(np.uint8)
        return np.clip(array.astype(np.float32) * (255.0 / float(info.max)), 0, 255).astype(np.uint8)
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8)
    maximum = float(np.nanmax(values))
    minimum = float(np.nanmin(values))
    if 0.0 <= minimum and maximum <= 1.0:
        values = values * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def _to_rgb_uint8(array: np.ndarray, axes: str = "") -> np.ndarray:
    array = _representative_yx(array, axes)
    if array.ndim == 2:
        gray = _to_uint8(array)
        return np.ascontiguousarray(np.stack((gray, gray, gray), axis=-1))
    if array.ndim != 3:
        raise ValueError(f"Cannot convert image region to RGB: shape={array.shape}")
    if array.shape[-1] == 1:
        gray = _to_uint8(array[..., 0])
        return np.ascontiguousarray(np.stack((gray, gray, gray), axis=-1))
    if array.shape[-1] == 2:
        gray = _to_uint8(array[..., 0])
        return np.ascontiguousarray(np.stack((gray, gray, gray), axis=-1))
    return np.ascontiguousarray(_to_uint8(array[..., :3]))


def _resize_exact(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    rgb = _to_rgb_uint8(rgb)
    width = max(1, int(width))
    height = max(1, int(height))
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return rgb
    image = Image.fromarray(rgb, mode="RGB")
    return np.asarray(image.resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8)


class _TiffSliceReader(_BaseSliceReader):
    def __init__(self, path: Path, downsample: int) -> None:
        import tifffile
        import zarr

        self.path = Path(path)
        self.tif = tifffile.TiffFile(str(path))
        base = self.tif.series[0]
        base_axes = str(getattr(base, "axes", "") or "")
        base_width, base_height = _spatial_shape(tuple(base.shape), base_axes)
        output_width = max(1, int(math.ceil(base_width / float(downsample))))
        output_height = max(1, int(math.ceil(base_height / float(downsample))))

        levels = list(getattr(base, "levels", ()) or ())
        candidates = levels or [base]
        suitable: list[tuple[int, int, Any, str]] = []
        for level in candidates:
            axes = str(getattr(level, "axes", base_axes) or base_axes)
            width, height = _spatial_shape(tuple(level.shape), axes)
            # Do not choose a level smaller than the requested output, because
            # that would irreversibly upsample a reduced pyramid image.
            if width >= output_width and height >= output_height:
                suitable.append((width * height, width, level, axes))
        if suitable:
            _area, _width_key, selected, selected_axes = min(suitable, key=lambda item: item[0])
        else:
            selected, selected_axes = base, base_axes

        self.axes = selected_axes
        self.level_width, self.level_height = _spatial_shape(tuple(selected.shape), self.axes)
        try:
            self.array = zarr.open(selected.aszarr(), mode="r")
        except Exception:
            # Memory mapping is safe for uncompressed/non-tiled TIFFs and keeps
            # this fallback useful when zarr is unavailable.
            self.array = tifffile.memmap(str(path), series=0)
            self.axes = base_axes
            self.level_width, self.level_height = base_width, base_height

        self.info = _ReaderInfo(
            width=base_width,
            height=base_height,
            output_width=output_width,
            output_height=output_height,
            backend="tifffile",
        )

    def _read_level_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        axes = self.axes if self.axes and len(self.axes) == self.array.ndim else ""
        if axes and "Y" in axes and "X" in axes:
            y_axis, x_axis = axes.index("Y"), axes.index("X")
        else:
            y_axis, x_axis = self.array.ndim - 2, self.array.ndim - 1
        slicer: list[int | slice] = []
        kept: list[str] = []
        for index in range(self.array.ndim):
            axis = axes[index] if axes else ""
            if index == y_axis:
                slicer.append(slice(y, y + height))
                kept.append("Y")
            elif index == x_axis:
                slicer.append(slice(x, x + width))
                kept.append("X")
            elif axis in {"C", "S"}:
                slicer.append(slice(None))
                kept.append(axis)
            else:
                slicer.append(0)
        return _to_rgb_uint8(np.asarray(self.array[tuple(slicer)]), "".join(kept))

    def read_tile(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        out_w, out_h = self.info.output_width, self.info.output_height
        sx0 = int(math.floor(x * self.level_width / float(out_w)))
        sy0 = int(math.floor(y * self.level_height / float(out_h)))
        sx1 = int(math.ceil((x + width) * self.level_width / float(out_w)))
        sy1 = int(math.ceil((y + height) * self.level_height / float(out_h)))
        sx0 = min(max(0, sx0), self.level_width - 1)
        sy0 = min(max(0, sy0), self.level_height - 1)
        sx1 = min(max(sx0 + 1, sx1), self.level_width)
        sy1 = min(max(sy0 + 1, sy1), self.level_height)
        region = self._read_level_region(sx0, sy0, sx1 - sx0, sy1 - sy0)
        return _resize_exact(region, width, height)

    def close(self) -> None:
        try:
            self.tif.close()
        except Exception:
            pass


class _OpenSlideSliceReader(_BaseSliceReader):
    def __init__(self, path: Path, downsample: int) -> None:
        import openslide

        self.slide = openslide.OpenSlide(str(path))
        base_width, base_height = [int(v) for v in self.slide.dimensions]
        output_width = max(1, int(math.ceil(base_width / float(downsample))))
        output_height = max(1, int(math.ceil(base_height / float(downsample))))
        suitable = [
            index
            for index, (width, height) in enumerate(self.slide.level_dimensions)
            if int(width) >= output_width and int(height) >= output_height
        ]
        self.level = suitable[-1] if suitable else 0
        self.level_downsample = float(self.slide.level_downsamples[self.level])
        self.info = _ReaderInfo(
            width=base_width,
            height=base_height,
            output_width=output_width,
            output_height=output_height,
            backend="OpenSlide",
        )

    def read_tile(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        full_x0 = int(math.floor(x * self.info.width / float(self.info.output_width)))
        full_y0 = int(math.floor(y * self.info.height / float(self.info.output_height)))
        full_x1 = int(math.ceil((x + width) * self.info.width / float(self.info.output_width)))
        full_y1 = int(math.ceil((y + height) * self.info.height / float(self.info.output_height)))
        level_width = max(1, int(math.ceil((full_x1 - full_x0) / self.level_downsample)))
        level_height = max(1, int(math.ceil((full_y1 - full_y0) / self.level_downsample)))
        region = self.slide.read_region(
            (full_x0, full_y0), self.level, (level_width, level_height)
        ).convert("RGB")
        return _resize_exact(np.asarray(region), width, height)

    def close(self) -> None:
        try:
            self.slide.close()
        except Exception:
            pass


class _PillowSliceReader(_BaseSliceReader):
    def __init__(self, path: Path, downsample: int) -> None:
        self.image = Image.open(path)
        width, height = self.image.size
        self.info = _ReaderInfo(
            width=int(width),
            height=int(height),
            output_width=max(1, int(math.ceil(width / float(downsample)))),
            output_height=max(1, int(math.ceil(height / float(downsample)))),
            backend="Pillow",
        )

    def read_tile(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        sx0 = int(math.floor(x * self.info.width / float(self.info.output_width)))
        sy0 = int(math.floor(y * self.info.height / float(self.info.output_height)))
        sx1 = int(math.ceil((x + width) * self.info.width / float(self.info.output_width)))
        sy1 = int(math.ceil((y + height) * self.info.height / float(self.info.output_height)))
        region = self.image.crop((sx0, sy0, sx1, sy1)).convert("RGB")
        return _resize_exact(np.asarray(region), width, height)

    def close(self) -> None:
        try:
            self.image.close()
        except Exception:
            pass


class _VipsSliceReader(_BaseSliceReader):
    def __init__(self, path: Path, downsample: int) -> None:
        import pyvips

        self.pyvips = pyvips
        image = pyvips.Image.new_from_file(str(path), access="random", page=0)
        base_width, base_height = int(image.width), int(image.height)
        if image.bands == 1:
            image = image.bandjoin([image, image])
        elif image.bands == 2:
            first = image.extract_band(0)
            image = first.bandjoin([first, first])
        elif image.bands > 3:
            image = image.extract_band(0, n=3)
        if str(image.format) != "uchar":
            image = image.cast("uchar")
        if downsample > 1:
            image = image.resize(1.0 / float(downsample), kernel="lanczos3")
        self.image = image
        self.info = _ReaderInfo(
            width=base_width,
            height=base_height,
            output_width=int(image.width),
            output_height=int(image.height),
            backend="libvips",
        )

    @staticmethod
    def _numpy(image: Any) -> np.ndarray:
        if hasattr(image, "numpy"):
            return np.asarray(image.numpy())
        memory = image.write_to_memory()
        return np.frombuffer(memory, dtype=np.uint8).reshape(image.height, image.width, image.bands)

    def read_tile(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        tile = self.image.crop(int(x), int(y), int(width), int(height))
        return _to_rgb_uint8(self._numpy(tile), "YXS")


def _reader_attempts(path: Path):
    if has_extension(path, TIFF_EXTENSIONS):
        # libvips is preferred in packaged builds because it provides robust,
        # lazy decoding for compressed/tiled BigTIFF without materializing the
        # full image. tifffile remains the deterministic pure-Python fallback.
        return (_VipsSliceReader, _TiffSliceReader, _OpenSlideSliceReader, _PillowSliceReader)
    if has_extension(path, WSI_EXTENSIONS):
        return (_VipsSliceReader, _OpenSlideSliceReader, _TiffSliceReader)
    if has_extension(path, RASTER_EXTENSIONS):
        return (_PillowSliceReader, _VipsSliceReader)
    return (_VipsSliceReader, _TiffSliceReader, _OpenSlideSliceReader, _PillowSliceReader)


def open_slice_reader(path: str | Path, downsample: int = 1) -> _BaseSliceReader:
    path = Path(path)
    errors: list[str] = []
    for reader_class in _reader_attempts(path):
        try:
            return reader_class(path, downsample)
        except Exception as exc:
            errors.append(f"{reader_class.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"Could not open merge slice {path}. Readers tried:\n- " + "\n- ".join(errors)
    )


def _unit_to_um(value: float, unit: str | None) -> float | None:
    unit_norm = str(unit or "um").strip().lower().replace("µ", "u")
    factors = {
        "um": 1.0,
        "micrometer": 1.0,
        "micrometre": 1.0,
        "nm": 0.001,
        "mm": 1000.0,
        "cm": 10000.0,
        "m": 1_000_000.0,
    }
    factor = factors.get(unit_norm)
    return float(value) * factor if factor is not None else None


def infer_pixel_size_um(path: str | Path) -> tuple[float, float] | None:
    """Read target X/Y calibration from OME-TIFF, TIFF tags, OpenSlide, or libvips."""

    path = Path(path)
    if has_extension(path, TIFF_EXTENSIONS):
        try:
            import tifffile

            with tifffile.TiffFile(str(path)) as tif:
                ome_xml = tif.ome_metadata
                if ome_xml:
                    root = ET.fromstring(ome_xml)
                    pixels = next((node for node in root.iter() if node.tag.endswith("Pixels")), None)
                    if pixels is not None:
                        px = pixels.attrib.get("PhysicalSizeX")
                        py = pixels.attrib.get("PhysicalSizeY")
                        if px and py:
                            x_um = _unit_to_um(float(px), pixels.attrib.get("PhysicalSizeXUnit"))
                            y_um = _unit_to_um(float(py), pixels.attrib.get("PhysicalSizeYUnit"))
                            if x_um and y_um and x_um > 0 and y_um > 0:
                                return x_um, y_um
                page = tif.pages[0]
                x_tag = page.tags.get("XResolution")
                y_tag = page.tags.get("YResolution")
                unit_tag = page.tags.get("ResolutionUnit")
                if x_tag and y_tag and unit_tag:
                    x_res = float(x_tag.value[0]) / float(x_tag.value[1]) if isinstance(x_tag.value, tuple) else float(x_tag.value)
                    y_res = float(y_tag.value[0]) / float(y_tag.value[1]) if isinstance(y_tag.value, tuple) else float(y_tag.value)
                    unit_name = getattr(unit_tag.value, "name", str(unit_tag.value)).upper()
                    if x_res > 0 and y_res > 0:
                        if "INCH" in unit_name:
                            return 25400.0 / x_res, 25400.0 / y_res
                        if "CENTIMETER" in unit_name:
                            return 10000.0 / x_res, 10000.0 / y_res
        except Exception:
            pass

    try:
        import openslide

        slide = openslide.OpenSlide(str(path))
        try:
            px = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
            py = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
            if px and py and float(px) > 0 and float(py) > 0:
                return float(px), float(py)
        finally:
            slide.close()
    except Exception:
        pass

    try:
        import pyvips

        image = pyvips.Image.new_from_file(str(path), access="sequential", page=0)
        # libvips xres/yres are pixels per millimetre.
        if float(image.xres) > 0 and float(image.yres) > 0:
            return 1000.0 / float(image.xres), 1000.0 / float(image.yres)
    except Exception:
        pass

    return None


def estimate_uncompressed_size_bytes(
    slice_count: int, width: int, height: int, downsample: int = 1
) -> int:
    out_width = max(1, int(math.ceil(int(width) / float(downsample))))
    out_height = max(1, int(math.ceil(int(height) / float(downsample))))
    return int(slice_count) * out_width * out_height * 3


def _validate_options(
    slices: Sequence[VolumeSlice],
    downsample: int,
    voxel_xy_um: float | None,
    voxel_z_um: float,
    tile_size: int,
) -> None:
    if not slices:
        raise ValueError("At least one fixed/warped image is required for a merged volume.")
    if int(downsample) < 1:
        raise ValueError("Merge downsample must be at least 1.")
    if voxel_xy_um is not None and float(voxel_xy_um) <= 0:
        raise ValueError("XY pixel size must be greater than zero or left on Auto.")
    if float(voxel_z_um) <= 0:
        raise ValueError("Z spacing must be greater than zero.")
    if int(tile_size) < 16 or int(tile_size) % 16 != 0:
        raise ValueError("TIFF tile size must be a multiple of 16.")


def _stack_sidecar_path(output_path: Path) -> Path:
    name = output_path.name
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if name.lower().endswith(suffix):
            return output_path.with_name(name[: -len(suffix)] + "_stack.json")
    return output_path.with_suffix(output_path.suffix + ".json")


def create_downsampled_registration_tiff(
    source_path: str | Path,
    output_path: str | Path,
    *,
    downsample: int,
    source_pixel_size_um: tuple[float, float] | None = None,
    tile_size: int = 256,
    compression: str = "deflate",
    progress_callback: ProgressCallback | None = None,
) -> WorkingImageResult:
    """Create a tiled OME-BigTIFF registration copy without full-image RAM use.

    The output is RGB uint8 because DeeperHistReg's histology pipeline expects a
    visual image. The reader selects an existing pyramid level when possible and
    streams one small output tile at a time. Physical X/Y calibration is scaled
    using the *actual* output dimensions, avoiding rounding drift for odd image
    sizes.
    """

    import tifffile

    source_path = Path(source_path)
    output_path = Path(output_path)
    downsample = int(downsample)
    tile_size = int(tile_size)
    if downsample < 1:
        raise ValueError("Registration downsample must be at least 1.")
    if tile_size < 16 or tile_size % 16 != 0:
        raise ValueError("TIFF tile size must be a multiple of 16.")
    if not source_path.is_file():
        raise FileNotFoundError(f"Registration source does not exist: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".partial")
    partial_path.unlink(missing_ok=True)

    with open_slice_reader(source_path, downsample) as probe_reader:
        info = probe_reader.info
    width = int(info.output_width)
    height = int(info.output_height)
    scale_x = float(info.width) / float(width)
    scale_y = float(info.height) / float(height)

    if source_pixel_size_um is None:
        source_pixel_size_um = infer_pixel_size_um(source_path)
    pixel_size_x_um = None
    pixel_size_y_um = None
    if source_pixel_size_um is not None:
        pixel_size_x_um = float(source_pixel_size_um[0]) * scale_x
        pixel_size_y_um = float(source_pixel_size_um[1]) * scale_y

    tiles_x = int(math.ceil(width / tile_size))
    tiles_y = int(math.ceil(height / tile_size))
    total_tiles = tiles_x * tiles_y
    completed_tiles = 0

    def tile_iterator() -> Iterable[np.ndarray]:
        nonlocal completed_tiles
        with open_slice_reader(source_path, downsample) as reader:
            for y in range(0, height, tile_size):
                tile_height = min(tile_size, height - y)
                for x in range(0, width, tile_size):
                    tile_width = min(tile_size, width - x)
                    tile = reader.read_tile(x, y, tile_width, tile_height)
                    completed_tiles += 1
                    if progress_callback and (
                        completed_tiles == total_tiles
                        or completed_tiles % max(1, total_tiles // 100) == 0
                    ):
                        progress_callback(
                            completed_tiles,
                            total_tiles,
                            f"Preparing downsampled registration image: "
                            f"{completed_tiles}/{total_tiles} tiles",
                        )
                    yield np.ascontiguousarray(tile, dtype=np.uint8)

    metadata: dict[str, object] = {
        "axes": "YXS",
        "Name": output_path.stem,
    }
    if pixel_size_x_um is not None and pixel_size_y_um is not None:
        metadata.update(
            {
                "PhysicalSizeX": pixel_size_x_um,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size_y_um,
                "PhysicalSizeYUnit": "µm",
            }
        )

    if progress_callback:
        progress_callback(0, total_tiles, f"Preparing registration image: {source_path.name}")

    try:
        with tifffile.TiffWriter(str(partial_path), bigtiff=True, ome=True) as writer:
            writer.write(
                data=tile_iterator(),
                shape=(height, width, 3),
                dtype=np.uint8,
                photometric="rgb",
                planarconfig="contig",
                tile=(tile_size, tile_size),
                compression=compression,
                metadata=metadata,
                software="HistRegGUI v1.0",
            )

        with tifffile.TiffFile(str(partial_path)) as tif:
            series = tif.series[0]
            if not tif.is_bigtiff:
                raise RuntimeError("Registration working image was not written as BigTIFF.")
            if tif.ome_metadata is None:
                raise RuntimeError("Registration working image is missing OME metadata.")
            if tuple(series.shape) != (height, width, 3):
                raise RuntimeError(
                    f"Registration working image shape mismatch: {series.shape}; "
                    f"expected {(height, width, 3)}"
                )

        os.replace(partial_path, output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    result = WorkingImageResult(
        source_path=source_path,
        path=output_path,
        original_width=int(info.width),
        original_height=int(info.height),
        width=width,
        height=height,
        downsample=downsample,
        scale_x=scale_x,
        scale_y=scale_y,
        pixel_size_x_um=pixel_size_x_um,
        pixel_size_y_um=pixel_size_y_um,
        reader=info.backend,
        tile_size=tile_size,
        compression=compression,
        size_bytes=output_path.stat().st_size,
    )
    if progress_callback:
        progress_callback(total_tiles, total_tiles, f"Registration image ready: {output_path.name}")
    return result


def create_merged_ome_tiff(
    output_path: str | Path,
    slices: Sequence[VolumeSlice],
    *,
    downsample: int = 1,
    voxel_xy_um: float | None = None,
    voxel_z_um: float = 4.0,
    tile_size: int = 256,
    compression: str = "deflate",
    progress_callback: ProgressCallback | None = None,
) -> MergedVolumeResult:
    """Create a tiled BigTIFF OME volume without loading all slices into RAM.

    Output pixels are RGB uint8 because that is the most interoperable form for
    brightfield histology viewers. Slice order is exactly the order supplied.
    """

    import tifffile

    normalized = tuple(
        VolumeSlice(
            path=Path(item.path),
            role=str(item.role),
            source_path=Path(item.source_path) if item.source_path else None,
            label=item.label,
        )
        for item in slices
    )
    downsample = int(downsample)
    tile_size = int(tile_size)
    voxel_xy_um = float(voxel_xy_um) if voxel_xy_um is not None else None
    voxel_z_um = float(voxel_z_um)
    _validate_options(normalized, downsample, voxel_xy_um, voxel_z_um, tile_size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".partial")
    partial_path.unlink(missing_ok=True)

    # Probe each slice using the same reader/downsample path that will be used
    # during writing. Readers are closed immediately after the inexpensive probe.
    probes: list[_ReaderInfo] = []
    for index, item in enumerate(normalized, start=1):
        if progress_callback:
            progress_callback(index - 1, len(normalized), f"Checking merge slice {index}/{len(normalized)}: {item.path.name}")
        with open_slice_reader(item.path, downsample) as reader:
            probes.append(reader.info)

    width = probes[0].output_width
    height = probes[0].output_height
    mismatches = [
        (item.path, probe.output_width, probe.output_height)
        for item, probe in zip(normalized, probes)
        if (probe.output_width, probe.output_height) != (width, height)
    ]
    if mismatches:
        details = "\n".join(f"- {path}: {w} x {h}" for path, w, h in mismatches[:10])
        raise ValueError(
            "All merged slices must have the same registered target dimensions. "
            f"Expected {width} x {height}; mismatches:\n{details}"
        )

    tiles_x = int(math.ceil(width / tile_size))
    tiles_y = int(math.ceil(height / tile_size))
    total_tiles = len(normalized) * tiles_x * tiles_y
    completed_tiles = 0

    def tile_iterator() -> Iterable[np.ndarray]:
        nonlocal completed_tiles
        for z_index, item in enumerate(normalized, start=1):
            if progress_callback:
                progress_callback(
                    completed_tiles,
                    total_tiles,
                    f"Writing merged slice {z_index}/{len(normalized)}: {item.path.name}",
                )
            with open_slice_reader(item.path, downsample) as reader:
                for y in range(0, height, tile_size):
                    tile_height = min(tile_size, height - y)
                    for x in range(0, width, tile_size):
                        tile_width = min(tile_size, width - x)
                        tile = reader.read_tile(x, y, tile_width, tile_height)
                        completed_tiles += 1
                        if progress_callback and (
                            completed_tiles == total_tiles
                            or completed_tiles % max(1, total_tiles // 100) == 0
                        ):
                            progress_callback(
                                completed_tiles,
                                total_tiles,
                                f"Writing merged volume: {completed_tiles}/{total_tiles} tiles",
                            )
                        yield np.ascontiguousarray(tile, dtype=np.uint8)

    metadata: dict[str, object] = {
        "axes": "ZYXS",
        "Name": output_path.stem,
        "PhysicalSizeZ": voxel_z_um,
        "PhysicalSizeZUnit": "µm",
    }
    if voxel_xy_um is not None:
        output_xy = voxel_xy_um * downsample
        metadata.update(
            {
                "PhysicalSizeX": output_xy,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": output_xy,
                "PhysicalSizeYUnit": "µm",
            }
        )

    try:
        with tifffile.TiffWriter(str(partial_path), bigtiff=True, ome=True) as writer:
            writer.write(
                data=tile_iterator(),
                shape=(len(normalized), height, width, 3),
                dtype=np.uint8,
                photometric="rgb",
                planarconfig="contig",
                tile=(tile_size, tile_size),
                compression=compression,
                metadata=metadata,
                software="HistRegGUI v1.0",
            )

        # Structural validation does not load the volume. It confirms that the
        # OME series has the intended dimensionality and BigTIFF container.
        with tifffile.TiffFile(str(partial_path)) as tif:
            series = tif.series[0]
            if not tif.is_bigtiff:
                raise RuntimeError("Merged output was not written as BigTIFF.")
            if tif.ome_metadata is None:
                raise RuntimeError("Merged output is missing OME-XML metadata.")
            if tuple(series.shape) != (len(normalized), height, width, 3):
                raise RuntimeError(
                    f"Merged OME-TIFF shape mismatch: {series.shape}; "
                    f"expected {(len(normalized), height, width, 3)}"
                )

        os.replace(partial_path, output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    sidecar_path = _stack_sidecar_path(output_path)
    payload = {
        "format": "OME-TIFF BigTIFF",
        "axes": "ZYXS",
        "shape": [len(normalized), height, width, 3],
        "dtype": "uint8",
        "tile_size": [tile_size, tile_size],
        "compression": compression,
        "downsample": downsample,
        "voxel_xy_um_at_source": voxel_xy_um,
        "voxel_xy_um_at_output": voxel_xy_um * downsample if voxel_xy_um is not None else None,
        "voxel_z_um": voxel_z_um,
        "slices": [
            {
                "z_index": index,
                "role": item.role,
                "label": item.label or item.path.stem,
                "image_path": str(item.path),
                "source_path": str(item.source_path) if item.source_path else None,
                "reader": probe.backend,
            }
            for index, (item, probe) in enumerate(zip(normalized, probes))
        ],
    }
    sidecar_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    result = MergedVolumeResult(
        path=output_path,
        sidecar_json=sidecar_path,
        z_slices=len(normalized),
        width=width,
        height=height,
        axes="ZYXS",
        voxel_xy_um=voxel_xy_um * downsample if voxel_xy_um is not None else None,
        voxel_z_um=voxel_z_um,
        downsample=downsample,
        tile_size=tile_size,
        compression=compression,
        size_bytes=output_path.stat().st_size,
    )
    if progress_callback:
        progress_callback(total_tiles, total_tiles, f"Merged volume saved: {output_path.name}")
    return result
