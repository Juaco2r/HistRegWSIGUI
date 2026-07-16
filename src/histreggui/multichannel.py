from __future__ import annotations

"""Multichannel fluorescence helpers for HistRegGUI.

The central design rule is to separate the *registration guide* from the
scientific image payload:

* DeeperHistReg receives an RGB uint8 guide (for example a DAPI channel
  replicated to RGB) so its normal histology preprocessing remains valid.
* The resulting displacement field is then applied to the original scientific
  image while preserving every channel, channel name, and integer dtype.
* Scientific merged volumes use OME-TIFF ``ZCYX`` rather than RGB ``ZYXS``.

All TIFF conversion and merge writers are tile-streamed.  They do not create a
full in-memory image stack.
"""

import json
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image

from histreggui.image_io import RASTER_EXTENSIONS, TIFF_EXTENSIONS, has_extension


ProgressCallback = Callable[[int, int, str], None]

GUIDE_MODE_AUTO = "auto"
GUIDE_MODE_CHANNEL = "channel"
GUIDE_MODE_MAX = "max"
GUIDE_MODE_MEAN = "mean"
GUIDE_MODES = {
    "Auto: DAPI/Hoechst, otherwise first channel": GUIDE_MODE_AUTO,
    "Selected channel": GUIDE_MODE_CHANNEL,
    "Maximum of all IF channels": GUIDE_MODE_MAX,
    "Mean of all IF channels": GUIDE_MODE_MEAN,
}

MERGE_MODE_AUTO = "auto"
MERGE_MODE_DISPLAY = "display"
MERGE_MODE_SCIENTIFIC = "scientific"
MERGE_MODE_BOTH = "both"
MERGE_MODES = {
    "Auto: scientific + RGB display when multichannel": MERGE_MODE_AUTO,
    "RGB display stack only": MERGE_MODE_DISPLAY,
    "Scientific multichannel stack only": MERGE_MODE_SCIENTIFIC,
    "Both RGB display and scientific multichannel": MERGE_MODE_BOTH,
}


@dataclass(frozen=True)
class GuideSettings:
    mode: str = GUIDE_MODE_AUTO
    channel_index: int | None = None  # zero-based
    invert: bool = False


@dataclass(frozen=True)
class ImageDataInfo:
    path: Path
    width: int
    height: int
    axes: str
    shape: tuple[int, ...]
    dtype: str
    channel_count: int
    channel_names: tuple[str, ...]
    is_rgb: bool
    is_multichannel: bool
    ome: bool

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["shape"] = list(self.shape)
        data["channel_names"] = list(self.channel_names)
        return data


@dataclass(frozen=True)
class ScientificImageResult:
    source_path: Path
    path: Path
    width: int
    height: int
    channel_count: int
    channel_names: tuple[str, ...]
    dtype: str
    axes: str
    pixel_size_x_um: float | None
    pixel_size_y_um: float | None
    downsample: int
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["source_path"] = str(self.source_path)
        data["path"] = str(self.path)
        data["channel_names"] = list(self.channel_names)
        return data


@dataclass(frozen=True)
class ScientificVolumeResult:
    path: Path
    sidecar_json: Path
    z_slices: int
    channels: int
    width: int
    height: int
    axes: str
    dtype: str
    channel_names: tuple[str, ...]
    voxel_xy_um: float | None
    voxel_z_um: float
    downsample: int
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["sidecar_json"] = str(self.sidecar_json)
        data["channel_names"] = list(self.channel_names)
        return data


def normalize_guide_mode(value: str) -> str:
    mode = GUIDE_MODES.get(str(value), str(value)).strip().lower()
    if mode not in {GUIDE_MODE_AUTO, GUIDE_MODE_CHANNEL, GUIDE_MODE_MAX, GUIDE_MODE_MEAN}:
        raise ValueError(f"Unsupported IF guide mode: {value}")
    return mode


def normalize_merge_mode(value: str) -> str:
    mode = MERGE_MODES.get(str(value), str(value)).strip().lower()
    if mode not in {MERGE_MODE_AUTO, MERGE_MODE_DISPLAY, MERGE_MODE_SCIENTIFIC, MERGE_MODE_BOTH}:
        raise ValueError(f"Unsupported merge mode: {value}")
    return mode


def _ome_channel_names(ome_xml: str | None, count: int) -> tuple[str, ...]:
    if not ome_xml:
        return tuple(f"Channel {index + 1}" for index in range(count))
    try:
        root = ET.fromstring(ome_xml)
        names: list[str] = []
        for elem in root.iter():
            if elem.tag.endswith("Channel"):
                names.append(str(elem.attrib.get("Name") or f"Channel {len(names) + 1}"))
        if len(names) >= count:
            return tuple(names[:count])
    except Exception:
        pass
    return tuple(f"Channel {index + 1}" for index in range(count))


def _tiff_channel_axis(shape: Sequence[int], axes: str, is_rgb: bool) -> tuple[int | None, int]:
    axes = (axes or "").upper()
    if axes and len(axes) == len(shape):
        if "C" in axes:
            index = axes.index("C")
            return index, int(shape[index])
        if "S" in axes:
            index = axes.index("S")
            count = int(shape[index])
            return index, min(count, 3) if is_rgb and count >= 3 else count
    if len(shape) >= 3 and int(shape[-1]) in (3, 4):
        return len(shape) - 1, 3 if is_rgb else int(shape[-1])
    if len(shape) >= 3 and int(shape[0]) <= 64:
        return 0, int(shape[0])
    return None, 1


def inspect_image_data(path: str | Path) -> ImageDataInfo:
    """Inspect a 2-D microscopy image without loading full-resolution pixels."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if has_extension(path, TIFF_EXTENSIONS):
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            series = tif.series[0]
            shape = tuple(int(v) for v in series.shape)
            axes = str(getattr(series, "axes", "") or "").upper()
            if axes and len(axes) == len(shape) and "Y" in axes and "X" in axes:
                height = int(shape[axes.index("Y")])
                width = int(shape[axes.index("X")])
            elif len(shape) >= 2:
                height, width = int(shape[-2]), int(shape[-1])
            else:
                raise ValueError(f"TIFF has no Y/X dimensions: {path} shape={shape}")

            photometric = ""
            try:
                photometric = str(tif.pages[0].photometric.name).upper()
            except Exception:
                pass
            is_rgb = photometric == "RGB" or ("S" in axes and shape[axes.index("S")] in (3, 4))
            _axis, channel_count = _tiff_channel_axis(shape, axes, is_rgb)
            channel_names = (
                ("Red", "Green", "Blue")
                if is_rgb and channel_count == 3
                else _ome_channel_names(tif.ome_metadata, channel_count)
            )
            return ImageDataInfo(
                path=path,
                width=width,
                height=height,
                axes=axes,
                shape=shape,
                dtype=str(np.dtype(series.dtype)),
                channel_count=channel_count,
                channel_names=tuple(channel_names),
                is_rgb=is_rgb,
                is_multichannel=(not is_rgb and channel_count > 1),
                ome=tif.ome_metadata is not None,
            )

    try:
        import pyvips

        image = pyvips.Image.new_from_file(str(path), access="sequential", page=0)
        bands = int(image.bands)
        is_rgb = bands >= 3
        count = min(bands, 3) if is_rgb else max(1, bands)
        names = ("Red", "Green", "Blue") if is_rgb else tuple(
            f"Channel {index + 1}" for index in range(count)
        )
        return ImageDataInfo(
            path=path,
            width=int(image.width),
            height=int(image.height),
            axes="YXS" if is_rgb else ("YX" if count == 1 else "YXC"),
            shape=(int(image.height), int(image.width), count) if count > 1 else (int(image.height), int(image.width)),
            dtype=str(_vips_numpy_dtype(str(image.format))),
            channel_count=count,
            channel_names=tuple(names),
            is_rgb=is_rgb,
            is_multichannel=(not is_rgb and count > 1),
            ome=False,
        )
    except Exception:
        with Image.open(path) as image:
            bands = len(image.getbands())
            is_rgb = image.mode in {"RGB", "RGBA", "CMYK"} or bands >= 3
            count = 3 if is_rgb else 1
            return ImageDataInfo(
                path=path,
                width=int(image.width),
                height=int(image.height),
                axes="YXS" if is_rgb else "YX",
                shape=(int(image.height), int(image.width), count) if count > 1 else (int(image.height), int(image.width)),
                dtype="uint8",
                channel_count=count,
                channel_names=("Red", "Green", "Blue") if is_rgb else ("Channel 1",),
                is_rgb=is_rgb,
                is_multichannel=False,
                ome=False,
            )


def series_requires_scientific_preservation(paths: Sequence[str | Path]) -> bool:
    for path in paths:
        try:
            info = inspect_image_data(path)
            if info.is_multichannel or info.channel_count > 3:
                return True
            if info.axes and "C" in info.axes and info.channel_count > 1 and not info.is_rgb:
                return True
        except Exception:
            continue
    return False


def _resize_plane(array: np.ndarray, width: int, height: int) -> np.ndarray:
    array = np.asarray(array)
    width, height = max(1, int(width)), max(1, int(height))
    if array.shape == (height, width):
        return np.ascontiguousarray(array)
    try:
        import cv2

        interpolation = cv2.INTER_AREA if width < array.shape[1] or height < array.shape[0] else cv2.INTER_LINEAR
        resized = cv2.resize(array, (width, height), interpolation=interpolation)
        return np.asarray(resized, dtype=array.dtype)
    except Exception:
        # Pillow reliably supports uint8 and uint16; float arrays use mode F.
        image = Image.fromarray(array)
        resized = image.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(resized).astype(array.dtype, copy=False)


class _ChannelReader:
    info: ImageDataInfo
    output_width: int
    output_height: int
    dtype: np.dtype

    def read_channels_tile(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class _TiffChannelReader(_ChannelReader):
    def __init__(self, path: Path, downsample: int) -> None:
        import tifffile

        self.path = Path(path)
        self.info = inspect_image_data(path)
        self.tif = tifffile.TiffFile(str(path))
        base = self.tif.series[0]
        base_axes = str(getattr(base, "axes", "") or "").upper()
        output_width = max(1, int(math.ceil(self.info.width / float(downsample))))
        output_height = max(1, int(math.ceil(self.info.height / float(downsample))))

        candidates = list(getattr(base, "levels", ()) or ()) or [base]
        suitable: list[tuple[int, Any, str, int, int]] = []
        for level in candidates:
            axes = str(getattr(level, "axes", base_axes) or base_axes).upper()
            shape = tuple(int(v) for v in level.shape)
            if axes and len(axes) == len(shape) and "Y" in axes and "X" in axes:
                height = int(shape[axes.index("Y")])
                width = int(shape[axes.index("X")])
            else:
                height, width = int(shape[-2]), int(shape[-1])
            if width >= output_width and height >= output_height:
                suitable.append((width * height, level, axes, width, height))
        if suitable:
            _area, selected, axes, level_width, level_height = min(suitable, key=lambda item: item[0])
        else:
            selected, axes = base, base_axes
            level_width, level_height = self.info.width, self.info.height

        self.axes = axes
        self.level_width = int(level_width)
        self.level_height = int(level_height)
        try:
            import zarr
            self.array = zarr.open(selected.aszarr(), mode="r")
        except Exception:
            try:
                self.array = tifffile.memmap(str(path), series=0)
            except Exception:
                # Test/minimal environments may not have zarr.  The packaged app
                # installs zarr and therefore keeps compressed TIFF access lazy;
                # this final fallback preserves correctness for small files.
                self.array = base.asarray()
            self.axes = base_axes
            self.level_width = self.info.width
            self.level_height = self.info.height
        self.output_width = output_width
        self.output_height = output_height
        self.dtype = np.dtype(getattr(self.array, "dtype", base.dtype))

    def _slice_region(self, sx0: int, sy0: int, sx1: int, sy1: int) -> np.ndarray:
        axes = self.axes if self.axes and len(self.axes) == self.array.ndim else ""
        y_axis = axes.index("Y") if axes and "Y" in axes else self.array.ndim - 2
        x_axis = axes.index("X") if axes and "X" in axes else self.array.ndim - 1
        channel_axis = None
        if axes and "C" in axes:
            channel_axis = axes.index("C")
        elif axes and "S" in axes:
            channel_axis = axes.index("S")
        elif (
            self.array.ndim >= 3
            and self.info.channel_count > 1
            and int(self.array.shape[0]) == int(self.info.channel_count)
        ):
            # Generic multi-page TIFFs may be reported as QYX/IYX instead of CYX.
            channel_axis = 0

        slicer: list[int | slice] = []
        kept_axes: list[str] = []
        for index in range(self.array.ndim):
            axis = axes[index] if axes else ""
            if index == y_axis:
                slicer.append(slice(sy0, sy1)); kept_axes.append("Y")
            elif index == x_axis:
                slicer.append(slice(sx0, sx1)); kept_axes.append("X")
            elif index == channel_axis:
                slicer.append(slice(None)); kept_axes.append("C")
            else:
                # A single 2-D registration payload is expected. For T/Z/etc.,
                # preserve the first plane consistently.
                slicer.append(0)

        array = np.asarray(self.array[tuple(slicer)])
        kept = "".join(kept_axes)
        if "C" in kept:
            order = [kept.index("C"), kept.index("Y"), kept.index("X")]
            array = np.transpose(array, order)
        else:
            order = [kept.index("Y"), kept.index("X")]
            array = np.transpose(array, order)[None, ...]
        if self.info.is_rgb and array.shape[0] > 3:
            array = array[:3]
        return np.asarray(array)

    def read_channels_tile(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        sx0 = int(math.floor(x * self.level_width / float(self.output_width)))
        sy0 = int(math.floor(y * self.level_height / float(self.output_height)))
        sx1 = int(math.ceil((x + width) * self.level_width / float(self.output_width)))
        sy1 = int(math.ceil((y + height) * self.level_height / float(self.output_height)))
        sx0 = min(max(0, sx0), self.level_width - 1)
        sy0 = min(max(0, sy0), self.level_height - 1)
        sx1 = min(max(sx0 + 1, sx1), self.level_width)
        sy1 = min(max(sy0 + 1, sy1), self.level_height)
        channels = self._slice_region(sx0, sy0, sx1, sy1)
        if channels.shape[1:] != (height, width):
            channels = np.stack(
                [_resize_plane(channel, width, height) for channel in channels], axis=0
            )
        return np.ascontiguousarray(channels)

    def close(self) -> None:
        try:
            self.tif.close()
        except Exception:
            pass


class _PillowChannelReader(_ChannelReader):
    def __init__(self, path: Path, downsample: int) -> None:
        self.path = Path(path)
        self.info = inspect_image_data(path)
        self.image = Image.open(path).convert("RGB" if self.info.is_rgb else "L")
        self.output_width = max(1, int(math.ceil(self.info.width / float(downsample))))
        self.output_height = max(1, int(math.ceil(self.info.height / float(downsample))))
        self.dtype = np.dtype(np.uint8)

    def read_channels_tile(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        sx0 = int(math.floor(x * self.info.width / float(self.output_width)))
        sy0 = int(math.floor(y * self.info.height / float(self.output_height)))
        sx1 = int(math.ceil((x + width) * self.info.width / float(self.output_width)))
        sy1 = int(math.ceil((y + height) * self.info.height / float(self.output_height)))
        region = self.image.crop((sx0, sy0, sx1, sy1))
        region = region.resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(region)
        if array.ndim == 2:
            return array[None, ...]
        return np.moveaxis(array[..., :3], -1, 0)

    def close(self) -> None:
        try:
            self.image.close()
        except Exception:
            pass



class _VipsChannelReader(_ChannelReader):
    """Lazy region reader for H&E WSI and other libvips-supported images."""

    def __init__(self, path: Path, downsample: int) -> None:
        import pyvips

        self.path = Path(path)
        self.info = inspect_image_data(path)
        self.image = pyvips.Image.new_from_file(str(path), access="random", page=0)
        if self.image.bands > 3 and self.info.is_rgb:
            self.image = self.image.extract_band(0, n=3)
        self.output_width = max(1, int(math.ceil(self.info.width / float(downsample))))
        self.output_height = max(1, int(math.ceil(self.info.height / float(downsample))))
        self.dtype = _vips_numpy_dtype(str(self.image.format))

    def _to_numpy(self, image: Any) -> np.ndarray:
        if hasattr(image, "numpy"):
            return np.asarray(image.numpy())
        memory = image.write_to_memory()
        return np.frombuffer(memory, dtype=self.dtype).reshape(
            int(image.height), int(image.width), int(image.bands)
        )

    def read_channels_tile(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        sx0 = int(math.floor(x * self.info.width / float(self.output_width)))
        sy0 = int(math.floor(y * self.info.height / float(self.output_height)))
        sx1 = int(math.ceil((x + width) * self.info.width / float(self.output_width)))
        sy1 = int(math.ceil((y + height) * self.info.height / float(self.output_height)))
        sx0 = min(max(0, sx0), self.info.width - 1)
        sy0 = min(max(0, sy0), self.info.height - 1)
        sx1 = min(max(sx0 + 1, sx1), self.info.width)
        sy1 = min(max(sy0 + 1, sy1), self.info.height)
        region = self.image.crop(sx0, sy0, sx1 - sx0, sy1 - sy0)
        if int(region.width) != int(width) or int(region.height) != int(height):
            region = region.resize(
                float(width) / float(region.width),
                vscale=float(height) / float(region.height),
                kernel="lanczos3",
            )
        # Normalise occasional one-pixel rounding differences from libvips.
        if int(region.width) > int(width) or int(region.height) > int(height):
            region = region.crop(0, 0, min(int(region.width), int(width)), min(int(region.height), int(height)))
        if int(region.width) != int(width) or int(region.height) != int(height):
            region = region.gravity(
                "centre",
                int(width),
                int(height),
                extend="copy",
            )
        array = self._to_numpy(region)
        if array.ndim == 2:
            return np.ascontiguousarray(array[None, ...])
        if array.shape[-1] > 3 and self.info.is_rgb:
            array = array[..., :3]
        return np.ascontiguousarray(np.moveaxis(array, -1, 0))


def open_channel_reader(path: str | Path, downsample: int = 1) -> _ChannelReader:
    path = Path(path)
    errors: list[str] = []
    if has_extension(path, TIFF_EXTENSIONS):
        try:
            return _TiffChannelReader(path, downsample)
        except Exception as exc:
            errors.append(f"tifffile: {type(exc).__name__}: {exc}")
    if has_extension(path, RASTER_EXTENSIONS):
        try:
            return _PillowChannelReader(path, downsample)
        except Exception as exc:
            errors.append(f"Pillow: {type(exc).__name__}: {exc}")
    # Whole-slide H&E formats (SVS, NDPI, MRXS, SCN, etc.) and mixed image
    # inputs are read lazily through libvips. Scientific IF channel preservation
    # still requires a TIFF/OME-TIFF that exposes its C axis/pages.
    try:
        return _VipsChannelReader(path, downsample)
    except Exception as exc:
        errors.append(f"libvips: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"Could not open {path} for channel-preserving guide/merge processing. "
        + "; ".join(errors)
    )


def _channel_index_for_guide(info: ImageDataInfo, settings: GuideSettings) -> int:
    if settings.channel_index is not None:
        index = int(settings.channel_index)
        if index < 0 or index >= info.channel_count:
            raise ValueError(
                f"Guide channel {index + 1} is outside the available range "
                f"1–{info.channel_count} for {info.path.name}."
            )
        return index
    for index, name in enumerate(info.channel_names):
        normalized = re.sub(r"[^a-z0-9]+", "", name.lower())
        if any(token in normalized for token in ("dapi", "hoechst", "nuclei", "nuclear")):
            return index
    return 0


def _sample_guide_window(path: Path, downsample: int, settings: GuideSettings) -> tuple[float, float]:
    info = inspect_image_data(path)
    sample_ds = max(int(downsample), int(math.ceil(max(info.width, info.height) / 1536.0)))
    with open_channel_reader(path, sample_ds) as reader:
        channels = reader.read_channels_tile(0, 0, reader.output_width, reader.output_height)
    mode = normalize_guide_mode(settings.mode)
    if mode in {GUIDE_MODE_AUTO, GUIDE_MODE_CHANNEL}:
        values = channels[_channel_index_for_guide(info, settings)]
    elif mode == GUIDE_MODE_MAX:
        values = np.max(channels.astype(np.float32), axis=0)
    else:
        values = np.mean(channels.astype(np.float32), axis=0)
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0.0, 1.0
    full_data = values[finite]
    data = full_data
    nonzero = full_data[full_data > 0]
    if nonzero.size >= max(100, full_data.size // 1000):
        data = nonzero
    low = float(np.percentile(data, 0.5))
    high = float(np.percentile(data, 99.8))
    if high <= low:
        # Uniform positive fluorescence is common after masks/thresholding.  In
        # that case excluding zeros would collapse both signal and background
        # to black, so use the complete sample to retain contrast.
        low, high = float(np.min(full_data)), float(np.max(full_data))
    if high <= low:
        high = low + 1.0
    return low, high


def _scale_to_uint8(array: np.ndarray, low: float, high: float, invert: bool = False) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    values = np.clip((values - float(low)) / max(float(high) - float(low), 1e-12), 0.0, 1.0)
    if invert:
        values = 1.0 - values
    return np.asarray(np.rint(values * 255.0), dtype=np.uint8)


def create_registration_guide_tiff(
    source_path: str | Path,
    output_path: str | Path,
    *,
    downsample: int = 1,
    settings: GuideSettings | None = None,
    source_pixel_size_um: tuple[float, float] | None = None,
    tile_size: int = 256,
    compression: str = "deflate",
    progress_callback: ProgressCallback | None = None,
) -> ScientificImageResult:
    """Create a streamed RGB guide suitable for DeeperHistReg.

    Multichannel fluorescence data are converted using the selected registration
    channel/composite only. The scientific channels are *not* modified here and
    are warped separately after the displacement field is calculated.
    """

    import tifffile

    source_path, output_path = Path(source_path), Path(output_path)
    settings = settings or GuideSettings()
    downsample, tile_size = int(downsample), int(tile_size)
    if downsample < 1:
        raise ValueError("Registration downsample must be at least 1.")
    info = inspect_image_data(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    partial.unlink(missing_ok=True)

    with open_channel_reader(source_path, downsample) as probe:
        width, height = probe.output_width, probe.output_height
    scale_x = info.width / float(width)
    scale_y = info.height / float(height)
    px = source_pixel_size_um[0] * scale_x if source_pixel_size_um else None
    py = source_pixel_size_um[1] * scale_y if source_pixel_size_um else None

    mode = normalize_guide_mode(settings.mode)
    low, high = (0.0, 255.0)
    if not info.is_rgb:
        low, high = _sample_guide_window(source_path, downsample, settings)

    tiles_x = int(math.ceil(width / tile_size))
    tiles_y = int(math.ceil(height / tile_size))
    total = tiles_x * tiles_y
    completed = 0

    def tiles() -> Iterable[np.ndarray]:
        nonlocal completed
        with open_channel_reader(source_path, downsample) as reader:
            for y in range(0, height, tile_size):
                th = min(tile_size, height - y)
                for x in range(0, width, tile_size):
                    tw = min(tile_size, width - x)
                    channels = reader.read_channels_tile(x, y, tw, th)
                    if info.is_rgb:
                        rgb = np.moveaxis(channels[:3], 0, -1)
                        if rgb.dtype == np.uint8:
                            guide = rgb
                        elif np.issubdtype(rgb.dtype, np.integer):
                            max_value = float(np.iinfo(rgb.dtype).max)
                            guide = np.clip(rgb.astype(np.float32) * 255.0 / max_value, 0, 255).astype(np.uint8)
                        else:
                            guide = _scale_to_uint8(rgb, float(np.nanmin(rgb)), float(np.nanmax(rgb)))
                    else:
                        if mode in {GUIDE_MODE_AUTO, GUIDE_MODE_CHANNEL}:
                            plane = channels[_channel_index_for_guide(info, settings)]
                        elif mode == GUIDE_MODE_MAX:
                            plane = np.max(channels.astype(np.float32), axis=0)
                        else:
                            plane = np.mean(channels.astype(np.float32), axis=0)
                        gray = _scale_to_uint8(plane, low, high, settings.invert)
                        guide = np.stack((gray, gray, gray), axis=-1)
                    completed += 1
                    if progress_callback and (completed == total or completed % max(1, total // 100) == 0):
                        progress_callback(completed, total, f"Preparing registration guide: {completed}/{total} tiles")
                    yield np.ascontiguousarray(guide, dtype=np.uint8)

    metadata: dict[str, object] = {
        "axes": "YXS",
        "Name": output_path.stem,
        "Description": (
            f"HistRegGUI registration guide from {source_path.name}; mode={mode}; "
            f"channel={_channel_index_for_guide(info, settings) + 1 if mode in {GUIDE_MODE_AUTO, GUIDE_MODE_CHANNEL} and not info.is_rgb else 'RGB/composite'}"
        ),
    }
    if px is not None and py is not None:
        metadata.update({
            "PhysicalSizeX": px,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": py,
            "PhysicalSizeYUnit": "µm",
        })

    if progress_callback:
        progress_callback(0, total, f"Preparing registration guide: {source_path.name}")
    try:
        with tifffile.TiffWriter(str(partial), bigtiff=True, ome=True) as writer:
            writer.write(
                data=tiles(),
                shape=(height, width, 3),
                dtype=np.uint8,
                photometric="rgb",
                planarconfig="contig",
                tile=(tile_size, tile_size),
                compression=compression,
                metadata=metadata,
                software="HistRegGUI v1.0",
            )
        with tifffile.TiffFile(str(partial)) as tif:
            if tuple(tif.series[0].shape) != (height, width, 3) or tif.ome_metadata is None:
                raise RuntimeError("Registration guide validation failed.")
        os.replace(partial, output_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    return ScientificImageResult(
        source_path=source_path,
        path=output_path,
        width=width,
        height=height,
        channel_count=3,
        channel_names=("Red", "Green", "Blue"),
        dtype="uint8",
        axes="YXS",
        pixel_size_x_um=px,
        pixel_size_y_um=py,
        downsample=downsample,
        size_bytes=output_path.stat().st_size,
    )


def create_scientific_payload_copy(
    source_path: str | Path,
    output_path: str | Path,
    *,
    downsample: int = 1,
    source_pixel_size_um: tuple[float, float] | None = None,
    tile_size: int = 256,
    compression: str = "deflate",
    progress_callback: ProgressCallback | None = None,
) -> ScientificImageResult:
    """Create a streamed CYX OME-TIFF copy preserving all scientific channels."""

    import tifffile

    source_path, output_path = Path(source_path), Path(output_path)
    info = inspect_image_data(source_path)
    downsample, tile_size = int(downsample), int(tile_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    partial.unlink(missing_ok=True)

    with open_channel_reader(source_path, downsample) as reader:
        width, height = reader.output_width, reader.output_height
        dtype = np.dtype(reader.dtype)
    channel_count = 3 if info.is_rgb else info.channel_count
    names = ("Red", "Green", "Blue") if info.is_rgb else info.channel_names
    scale_x, scale_y = info.width / float(width), info.height / float(height)
    px = source_pixel_size_um[0] * scale_x if source_pixel_size_um else None
    py = source_pixel_size_um[1] * scale_y if source_pixel_size_um else None
    tiles_x, tiles_y = math.ceil(width / tile_size), math.ceil(height / tile_size)
    total = channel_count * tiles_x * tiles_y
    completed = 0

    def tiles() -> Iterable[np.ndarray]:
        nonlocal completed
        with open_channel_reader(source_path, downsample) as reader:
            for channel in range(channel_count):
                for y in range(0, height, tile_size):
                    th = min(tile_size, height - y)
                    for x in range(0, width, tile_size):
                        tw = min(tile_size, width - x)
                        data = reader.read_channels_tile(x, y, tw, th)
                        if channel >= data.shape[0]:
                            tile = np.zeros((th, tw), dtype=dtype)
                        else:
                            tile = np.asarray(data[channel], dtype=dtype)
                        completed += 1
                        if progress_callback and (completed == total or completed % max(1, total // 100) == 0):
                            progress_callback(completed, total, f"Writing scientific payload: {completed}/{total} tiles")
                        yield np.ascontiguousarray(tile)

    metadata: dict[str, object] = {
        "axes": "CYX",
        "Name": output_path.stem,
        "Channel": {"Name": list(names)},
    }
    if px is not None and py is not None:
        metadata.update({
            "PhysicalSizeX": px,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": py,
            "PhysicalSizeYUnit": "µm",
        })
    try:
        with tifffile.TiffWriter(str(partial), bigtiff=True, ome=True) as writer:
            writer.write(
                data=tiles(),
                shape=(channel_count, height, width),
                dtype=dtype,
                photometric="minisblack",
                tile=(tile_size, tile_size),
                compression=compression,
                metadata=metadata,
                software="HistRegGUI v1.0",
            )
        with tifffile.TiffFile(str(partial)) as tif:
            if tuple(tif.series[0].shape) != (channel_count, height, width):
                raise RuntimeError("Scientific payload validation failed.")
        os.replace(partial, output_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    return ScientificImageResult(
        source_path=source_path,
        path=output_path,
        width=width,
        height=height,
        channel_count=channel_count,
        channel_names=tuple(names),
        dtype=str(dtype),
        axes="CYX",
        pixel_size_x_um=px,
        pixel_size_y_um=py,
        downsample=downsample,
        size_bytes=output_path.stat().st_size,
    )


def _vips_numpy_dtype(format_name: str) -> np.dtype:
    mapping = {
        "uchar": np.uint8,
        "char": np.int8,
        "ushort": np.uint16,
        "short": np.int16,
        "uint": np.uint32,
        "int": np.int32,
        "float": np.float32,
        "double": np.float64,
    }
    if format_name not in mapping:
        raise ValueError(f"Unsupported libvips pixel format: {format_name}")
    return np.dtype(mapping[format_name])


def _load_pyvips_channels(path: Path, expected_channels: int) -> Any:
    """Load bands/pages lazily as one libvips multiband image."""

    import pyvips

    image = pyvips.Image.new_from_file(str(path), access="random", n=-1)
    page_height = 0
    try:
        if image.get_typeof("page-height"):
            page_height = int(image.get("page-height"))
    except Exception:
        page_height = 0

    if page_height > 0 and image.height > page_height and expected_channels > image.bands:
        page_count = image.height // page_height
        pages = [image.crop(0, index * page_height, image.width, page_height) for index in range(page_count)]
        if pages and all(page.bands == 1 for page in pages[:expected_channels]):
            joined = pages[0]
            for page in pages[1:expected_channels]:
                joined = joined.bandjoin(page)
            image = joined
    if image.bands > expected_channels:
        image = image.extract_band(0, n=expected_channels)
    if image.bands < expected_channels:
        raise RuntimeError(
            f"libvips exposed {image.bands} bands for {path.name}, but {expected_channels} are required. "
            "Convert the source to a planar OME-TIFF before multichannel warping."
        )
    return image


def _calculate_center_padding(source_width: int, source_height: int, target_width: int, target_height: int):
    canvas_width, canvas_height = max(source_width, target_width), max(source_height, target_height)
    source_left = (canvas_width - source_width) // 2
    source_top = (canvas_height - source_height) // 2
    target_left = (canvas_width - target_width) // 2
    target_top = (canvas_height - target_height) // 2
    return {
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "source_left": source_left,
        "source_top": source_top,
        "target_left": target_left,
        "target_top": target_top,
    }


def _write_pyvips_as_ome_cyx(
    image: Any,
    output_path: Path,
    *,
    channel_names: Sequence[str],
    pixel_size_um: tuple[float, float] | None,
    tile_size: int,
    compression: str,
    source_path: Path,
    downsample: int,
    progress_callback: ProgressCallback | None,
) -> ScientificImageResult:
    import tifffile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    partial.unlink(missing_ok=True)
    width, height, channels = int(image.width), int(image.height), int(image.bands)
    dtype = _vips_numpy_dtype(str(image.format))
    tile_count = channels * math.ceil(width / tile_size) * math.ceil(height / tile_size)
    completed = 0

    def numpy_tile(vips_tile: Any) -> np.ndarray:
        if hasattr(vips_tile, "numpy"):
            return np.asarray(vips_tile.numpy())
        memory = vips_tile.write_to_memory()
        return np.frombuffer(memory, dtype=dtype).reshape(vips_tile.height, vips_tile.width, vips_tile.bands)

    def tiles() -> Iterable[np.ndarray]:
        nonlocal completed
        for channel in range(channels):
            band = image.extract_band(channel)
            for y in range(0, height, tile_size):
                th = min(tile_size, height - y)
                for x in range(0, width, tile_size):
                    tw = min(tile_size, width - x)
                    tile = numpy_tile(band.crop(x, y, tw, th))
                    if tile.ndim == 3:
                        tile = tile[..., 0]
                    completed += 1
                    if progress_callback and (completed == tile_count or completed % max(1, tile_count // 100) == 0):
                        progress_callback(completed, tile_count, f"Warping scientific channels: {completed}/{tile_count} tiles")
                    yield np.ascontiguousarray(tile, dtype=dtype)

    metadata: dict[str, object] = {
        "axes": "CYX",
        "Name": output_path.stem,
        "Channel": {"Name": list(channel_names)},
    }
    if pixel_size_um:
        metadata.update({
            "PhysicalSizeX": float(pixel_size_um[0]),
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": float(pixel_size_um[1]),
            "PhysicalSizeYUnit": "µm",
        })
    try:
        with tifffile.TiffWriter(str(partial), bigtiff=True, ome=True) as writer:
            writer.write(
                data=tiles(),
                shape=(channels, height, width),
                dtype=dtype,
                photometric="minisblack",
                tile=(tile_size, tile_size),
                compression=compression,
                metadata=metadata,
                software="HistRegGUI v1.0",
            )
        with tifffile.TiffFile(str(partial)) as tif:
            if tuple(tif.series[0].shape) != (channels, height, width):
                raise RuntimeError(f"Warped scientific OME-TIFF shape mismatch: {tif.series[0].shape}")
        os.replace(partial, output_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    return ScientificImageResult(
        source_path=source_path,
        path=output_path,
        width=width,
        height=height,
        channel_count=channels,
        channel_names=tuple(channel_names),
        dtype=str(dtype),
        axes="CYX",
        pixel_size_x_um=pixel_size_um[0] if pixel_size_um else None,
        pixel_size_y_um=pixel_size_um[1] if pixel_size_um else None,
        downsample=downsample,
        size_bytes=output_path.stat().st_size,
    )


def warp_scientific_payload(
    source_path: str | Path,
    source_guide_path: str | Path,
    target_guide_path: str | Path,
    displacement_field_path: str | Path,
    output_path: str | Path,
    *,
    source_pixel_size_um: tuple[float, float] | None = None,
    target_pixel_size_um: tuple[float, float] | None = None,
    downsample: int = 1,
    tile_size: int = 256,
    compression: str = "deflate",
    progress_callback: ProgressCallback | None = None,
) -> ScientificImageResult:
    """Apply a DeeperHistReg field to every original source channel.

    The function uses libvips lazily and therefore does not materialize the full
    multichannel slide. The displacement field is the one calculated from the
    RGB guide pair. Fluorescence background is zero-filled.
    """

    import pyvips
    from deeperhistreg.dhr_input_output.dhr_loaders.displacement_loader import DisplacementFieldLoader
    from deeperhistreg.dhr_utils import warping as dhr_warping

    source_path = Path(source_path)
    source_guide_path = Path(source_guide_path)
    target_guide_path = Path(target_guide_path)
    displacement_field_path = Path(displacement_field_path)
    output_path = Path(output_path)
    source_info = inspect_image_data(source_path)
    source_guide = inspect_image_data(source_guide_path)
    target_guide = inspect_image_data(target_guide_path)
    expected_channels = 3 if source_info.is_rgb else source_info.channel_count
    names = ("Red", "Green", "Blue") if source_info.is_rgb else source_info.channel_names

    image = _load_pyvips_channels(source_path, expected_channels)
    scale_x = source_guide.width / float(image.width)
    scale_y = source_guide.height / float(image.height)
    if abs(scale_x - 1.0) > 1e-9 or abs(scale_y - 1.0) > 1e-9:
        image = image.resize(scale_x, vscale=scale_y, kernel="lanczos3")
    # libvips rounds dimensions after resampling. Normalise rare one-pixel
    # differences so scientific payload geometry exactly matches the guide.
    if image.width != source_guide.width or image.height != source_guide.height:
        if image.width > source_guide.width or image.height > source_guide.height:
            image = image.crop(
                0,
                0,
                min(int(image.width), int(source_guide.width)),
                min(int(image.height), int(source_guide.height)),
            )
        if image.width != source_guide.width or image.height != source_guide.height:
            image = image.gravity(
                "centre",
                int(source_guide.width),
                int(source_guide.height),
                extend="background",
                background=[0.0] * int(image.bands),
            )

    padding = _calculate_center_padding(
        source_guide.width, source_guide.height, target_guide.width, target_guide.height
    )
    background = [0.0] * int(image.bands)
    padded = image.gravity(
        "centre",
        int(padding["canvas_width"]),
        int(padding["canvas_height"]),
        extend="background",
        background=background,
    )
    displacement = DisplacementFieldLoader().load(str(displacement_field_path))
    warped = dhr_warping.warp_pyvips_with_tc_df(padded, displacement, pad_value=0.0)
    warped = warped.crop(
        int(padding["target_left"]),
        int(padding["target_top"]),
        int(target_guide.width),
        int(target_guide.height),
    )

    # The warped scientific payload is sampled on the target guide grid.  Its
    # physical spacing must therefore follow the target/reference image, not
    # the source IF scanner.  Falling back to the source-derived spacing keeps
    # backward compatibility when target calibration is unavailable.
    output_pixel_size = None
    if target_pixel_size_um:
        output_pixel_size = (
            float(target_pixel_size_um[0]),
            float(target_pixel_size_um[1]),
        )
    elif source_pixel_size_um:
        output_pixel_size = (
            float(source_pixel_size_um[0]) * (source_info.width / float(source_guide.width)),
            float(source_pixel_size_um[1]) * (source_info.height / float(source_guide.height)),
        )
    return _write_pyvips_as_ome_cyx(
        warped,
        output_path,
        channel_names=names,
        pixel_size_um=output_pixel_size,
        tile_size=tile_size,
        compression=compression,
        source_path=source_path,
        downsample=downsample,
        progress_callback=progress_callback,
    )


def _schema_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _slice_channel_labels(info: ImageDataInfo, role: str) -> tuple[str, ...]:
    role = str(role).lower()
    rgb_named = (
        info.channel_count == 3
        and tuple(_schema_key(name) for name in info.channel_names)
        == ("red", "green", "blue")
    )
    if info.is_rgb or rgb_named:
        prefix = "H&E" if role in {"fixed", "brightfield", "he"} else "RGB"
        return (f"{prefix} Red", f"{prefix} Green", f"{prefix} Blue")
    return tuple(info.channel_names)


def _scientific_stack_sidecar(output_path: Path) -> Path:
    lower = output_path.name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if lower.endswith(suffix):
            return output_path.with_name(output_path.name[: -len(suffix)] + "_scientific_stack.json")
    return output_path.with_suffix(output_path.suffix + ".json")


def create_merged_scientific_ome_tiff(
    output_path: str | Path,
    slices: Sequence[Any],
    *,
    downsample: int = 1,
    voxel_xy_um: float | None = None,
    voxel_z_um: float = 4.0,
    tile_size: int = 256,
    compression: str = "deflate",
    progress_callback: ProgressCallback | None = None,
) -> ScientificVolumeResult:
    """Create one mixed H&E/IF OME-TIFF with axes ``ZCYX``.

    A union channel schema is created. H&E RGB planes occupy three named
    channels, IF planes occupy their original named channels, and channels not
    present in a given Z slice are zero-filled. If any source is uint16, uint8
    sources are expanded to uint16 (0–255 becomes 0–65535) while IF uint16
    intensities remain unchanged.
    """

    import tifffile

    if not slices:
        raise ValueError("At least one slice is required for scientific merge.")
    downsample = int(downsample)
    output_path = Path(output_path)
    normalized: list[dict[str, Any]] = []
    schema_names: list[str] = []
    schema_keys: dict[str, int] = {}
    output_dtype = np.dtype(np.uint8)

    for item in slices:
        path = Path(item.path)
        role = str(getattr(item, "role", "warped"))
        info = inspect_image_data(path)
        labels = _slice_channel_labels(info, role)
        mapping: list[int] = []
        for label in labels:
            key = _schema_key(label)
            if key not in schema_keys:
                schema_keys[key] = len(schema_names)
                schema_names.append(label)
            mapping.append(schema_keys[key])
        dtype = np.dtype(info.dtype)
        if dtype.kind not in "uif":
            raise ValueError(f"Unsupported scientific dtype {dtype} in {path}")
        if dtype.itemsize > output_dtype.itemsize or dtype.kind == "f":
            output_dtype = np.dtype(np.float32) if dtype.kind == "f" else dtype
        normalized.append({"item": item, "path": path, "info": info, "mapping": mapping, "labels": labels})

    probes: list[tuple[int, int]] = []
    for record in normalized:
        with open_channel_reader(record["path"], downsample) as reader:
            probes.append((reader.output_width, reader.output_height))
    width, height = probes[0]
    for record, dims in zip(normalized, probes):
        if dims != (width, height):
            raise ValueError(
                "All scientific slices must have the same registered dimensions. "
                f"Expected {width}×{height}, got {dims[0]}×{dims[1]} for {record['path']}."
            )

    channels = len(schema_names)
    tiles_x, tiles_y = math.ceil(width / tile_size), math.ceil(height / tile_size)
    total = len(normalized) * channels * tiles_x * tiles_y
    completed = 0

    def cast_plane(plane: np.ndarray) -> np.ndarray:
        plane = np.asarray(plane)
        if plane.dtype == output_dtype:
            return plane
        if output_dtype == np.dtype(np.uint16) and plane.dtype == np.uint8:
            return plane.astype(np.uint16) * 257
        return plane.astype(output_dtype)

    def tiles() -> Iterable[np.ndarray]:
        nonlocal completed
        for z_index, record in enumerate(normalized):
            with open_channel_reader(record["path"], downsample) as reader:
                reverse = {schema_index: local_index for local_index, schema_index in enumerate(record["mapping"])}
                for schema_channel in range(channels):
                    local_channel = reverse.get(schema_channel)
                    for y in range(0, height, tile_size):
                        th = min(tile_size, height - y)
                        for x in range(0, width, tile_size):
                            tw = min(tile_size, width - x)
                            if local_channel is None:
                                plane = np.zeros((th, tw), dtype=output_dtype)
                            else:
                                data = reader.read_channels_tile(x, y, tw, th)
                                if local_channel >= data.shape[0]:
                                    plane = np.zeros((th, tw), dtype=output_dtype)
                                else:
                                    plane = cast_plane(data[local_channel])
                            completed += 1
                            if progress_callback and (completed == total or completed % max(1, total // 100) == 0):
                                progress_callback(completed, total, f"Writing scientific stack: {completed}/{total} tiles")
                            yield np.ascontiguousarray(plane, dtype=output_dtype)

    metadata: dict[str, object] = {
        "axes": "ZCYX",
        "Name": output_path.stem,
        "Channel": {"Name": schema_names},
        "PhysicalSizeZ": float(voxel_z_um),
        "PhysicalSizeZUnit": "µm",
    }
    if voxel_xy_um is not None:
        output_xy = float(voxel_xy_um) * downsample
        metadata.update({
            "PhysicalSizeX": output_xy,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": output_xy,
            "PhysicalSizeYUnit": "µm",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        with tifffile.TiffWriter(str(partial), bigtiff=True, ome=True) as writer:
            writer.write(
                data=tiles(),
                shape=(len(normalized), channels, height, width),
                dtype=output_dtype,
                photometric="minisblack",
                tile=(tile_size, tile_size),
                compression=compression,
                metadata=metadata,
                software="HistRegGUI v1.0",
            )
        with tifffile.TiffFile(str(partial)) as tif:
            expected = (len(normalized), channels, height, width)
            if tuple(tif.series[0].shape) != expected or tif.series[0].axes != "ZCYX":
                raise RuntimeError(
                    f"Scientific stack validation failed: {tif.series[0].shape} {tif.series[0].axes}; expected {expected} ZCYX"
                )
        os.replace(partial, output_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    sidecar = _scientific_stack_sidecar(output_path)
    payload = {
        "format": "OME-TIFF BigTIFF",
        "axes": "ZCYX",
        "shape": [len(normalized), channels, height, width],
        "dtype": str(output_dtype),
        "channel_names": schema_names,
        "voxel_xy_um_at_source": voxel_xy_um,
        "voxel_xy_um_at_output": float(voxel_xy_um) * downsample if voxel_xy_um is not None else None,
        "voxel_z_um": float(voxel_z_um),
        "slices": [
            {
                "z_index": index,
                "path": str(record["path"]),
                "role": str(getattr(record["item"], "role", "warped")),
                "source_path": str(getattr(record["item"], "source_path", "") or "") or None,
                "local_channel_names": list(record["labels"]),
                "schema_indices": list(record["mapping"]),
            }
            for index, record in enumerate(normalized)
        ],
    }
    sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return ScientificVolumeResult(
        path=output_path,
        sidecar_json=sidecar,
        z_slices=len(normalized),
        channels=channels,
        width=width,
        height=height,
        axes="ZCYX",
        dtype=str(output_dtype),
        channel_names=tuple(schema_names),
        voxel_xy_um=float(voxel_xy_um) * downsample if voxel_xy_um is not None else None,
        voxel_z_um=float(voxel_z_um),
        downsample=downsample,
        size_bytes=output_path.stat().st_size,
    )
