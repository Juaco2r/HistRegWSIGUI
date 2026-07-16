from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from histreggui.multichannel import (
    GuideSettings,
    create_merged_scientific_ome_tiff,
    create_registration_guide_tiff,
    create_scientific_payload_copy,
    inspect_image_data,
    series_requires_scientific_preservation,
)
from histreggui.volume import VolumeSlice


def _write_if(path: Path) -> np.ndarray:
    array = np.zeros((4, 33, 35), dtype=np.uint16)
    for channel in range(4):
        array[channel] = (channel + 1) * 1000
    tifffile.imwrite(
        path,
        array,
        ome=True,
        metadata={
            "axes": "CYX",
            "Channel": {"Name": ["DAPI", "FITC", "TRITC", "Cy5"]},
            "PhysicalSizeX": 0.5,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.6,
            "PhysicalSizeYUnit": "µm",
        },
    )
    return array


def _write_he(path: Path) -> np.ndarray:
    array = np.zeros((33, 35, 3), dtype=np.uint8)
    array[..., 0] = 100
    array[..., 1] = 50
    tifffile.imwrite(path, array, ome=True, photometric="rgb", metadata={"axes": "YXS"})
    return array


def test_inspection_detects_four_channel_if_and_channel_names(tmp_path: Path) -> None:
    path = tmp_path / "if.ome.tif"
    _write_if(path)
    info = inspect_image_data(path)
    assert info.axes == "CYX"
    assert info.channel_count == 4
    assert info.channel_names == ("DAPI", "FITC", "TRITC", "Cy5")
    assert info.dtype == "uint16"
    assert info.is_multichannel
    assert series_requires_scientific_preservation([path])


def test_registration_guide_uses_dapi_and_is_rgb_uint8(tmp_path: Path) -> None:
    source = tmp_path / "if.ome.tif"
    _write_if(source)
    output = tmp_path / "guide.ome.tif"
    result = create_registration_guide_tiff(
        source,
        output,
        downsample=2,
        settings=GuideSettings(mode="auto"),
        tile_size=16,
    )
    with tifffile.TiffFile(output) as tif:
        assert tif.series[0].axes == "YXS"
        assert tif.series[0].shape == (17, 18, 3)
        assert tif.series[0].dtype == np.dtype(np.uint8)
    guide = tifffile.imread(output)
    assert np.array_equal(guide[..., 0], guide[..., 1])
    assert np.array_equal(guide[..., 1], guide[..., 2])
    assert result.channel_count == 3



def test_auto_guide_finds_named_dapi_and_optional_inversion(tmp_path: Path) -> None:
    source = tmp_path / "if_named_dapi.ome.tif"
    array = np.zeros((4, 16, 18), dtype=np.uint16)
    array[0, :, :] = 100
    array[1, 3:13, 4:14] = 5000  # DAPI is intentionally channel 2.
    array[2, :, :] = 200
    array[3, :, :] = 300
    tifffile.imwrite(
        source,
        array,
        ome=True,
        metadata={
            "axes": "CYX",
            "Channel": {"Name": ["FITC", "DAPI", "TRITC", "Cy5"]},
        },
    )
    normal = tmp_path / "normal.ome.tif"
    inverted = tmp_path / "inverted.ome.tif"
    create_registration_guide_tiff(
        source, normal, settings=GuideSettings(mode="auto", invert=False), tile_size=16
    )
    create_registration_guide_tiff(
        source, inverted, settings=GuideSettings(mode="auto", invert=True), tile_size=16
    )
    normal_pixels = tifffile.imread(normal)[..., 0]
    inverted_pixels = tifffile.imread(inverted)[..., 0]
    assert normal_pixels[8, 8] > normal_pixels[0, 0]
    assert inverted_pixels[8, 8] < inverted_pixels[0, 0]
    assert np.max(np.abs(normal_pixels.astype(np.int16) + inverted_pixels.astype(np.int16) - 255)) <= 1

def test_scientific_payload_preserves_four_channels_dtype_and_names(tmp_path: Path) -> None:
    source = tmp_path / "if.ome.tif"
    _write_if(source)
    output = tmp_path / "scientific.ome.tif"
    result = create_scientific_payload_copy(source, output, downsample=2, tile_size=16)
    with tifffile.TiffFile(output) as tif:
        assert tif.series[0].axes == "CYX"
        assert tif.series[0].shape == (4, 17, 18)
        assert tif.series[0].dtype == np.dtype(np.uint16)
        assert "DAPI" in (tif.ome_metadata or "")
        assert "Cy5" in (tif.ome_metadata or "")
    assert result.channel_names == ("DAPI", "FITC", "TRITC", "Cy5")
    assert result.dtype == "uint16"


def test_mixed_he_if_scientific_merge_uses_union_zcyx_schema(tmp_path: Path) -> None:
    if_source = tmp_path / "if.ome.tif"
    he_source = tmp_path / "he.ome.tif"
    _write_if(if_source)
    _write_he(he_source)

    if_copy = create_scientific_payload_copy(
        if_source, tmp_path / "if_registered.ome.tif", downsample=2, tile_size=16
    )
    he_copy = create_scientific_payload_copy(
        he_source, tmp_path / "he_registered.ome.tif", downsample=2, tile_size=16
    )
    result = create_merged_scientific_ome_tiff(
        tmp_path / "mixed_stack.ome.tif",
        [
            VolumeSlice(he_copy.path, role="he", label="H&E"),
            VolumeSlice(if_copy.path, role="if", label="IF"),
        ],
        downsample=1,
        voxel_xy_um=1.0,
        voxel_z_um=4.0,
        tile_size=16,
    )
    with tifffile.TiffFile(result.path) as tif:
        assert tif.series[0].axes == "ZCYX"
        assert tif.series[0].shape == (2, 7, 17, 18)
        assert tif.series[0].dtype == np.dtype(np.uint16)
        ome = tif.ome_metadata or ""
        for name in ("H&amp;E Red", "DAPI", "FITC", "TRITC", "Cy5"):
            assert name in ome

    volume = tifffile.imread(result.path)
    # H&E uint8 is expanded to the full uint16 range, while IF values are unchanged.
    assert volume[0, 0, 0, 0] == 100 * 257
    assert volume[0, 1, 0, 0] == 50 * 257
    assert np.array_equal(volume[0, 3:, 0, 0], np.zeros(4, dtype=np.uint16))
    assert np.array_equal(volume[1, :3, 0, 0], np.zeros(3, dtype=np.uint16))
    assert tuple(volume[1, 3:, 0, 0]) == (1000, 2000, 3000, 4000)

    sidecar = json.loads(result.sidecar_json.read_text(encoding="utf-8"))
    assert sidecar["axes"] == "ZCYX"
    assert sidecar["channel_names"] == [
        "H&E Red",
        "H&E Green",
        "H&E Blue",
        "DAPI",
        "FITC",
        "TRITC",
        "Cy5",
    ]



def test_wsi_like_he_can_be_streamed_through_libvips(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("pyvips")
    # A TIFF payload with an SVS suffix exercises the same libvips fallback used
    # for H&E whole-slide formats without requiring a proprietary test slide.
    source = tmp_path / "he_like.svs"
    array = np.zeros((31, 37, 3), dtype=np.uint8)
    array[..., 0] = 120
    array[..., 1] = 80
    array[5:25, 7:28, 2] = 200
    tifffile.imwrite(source, array, photometric="rgb")

    guide = create_registration_guide_tiff(
        source, tmp_path / "he_guide.ome.tif", downsample=2, tile_size=16, compression=None
    )
    scientific = create_scientific_payload_copy(
        source, tmp_path / "he_scientific.ome.tif", downsample=2, tile_size=16, compression=None
    )
    assert (guide.width, guide.height, guide.channel_count) == (19, 16, 3)
    assert (scientific.width, scientific.height, scientific.channel_count) == (19, 16, 3)
    with tifffile.TiffFile(scientific.path) as tif:
        assert tif.series[0].axes == "CYX"
        assert tif.series[0].shape == (3, 16, 19)

def test_identity_deformation_is_applied_to_every_if_channel_when_backends_available(
    tmp_path: Path,
) -> None:
    # This integration check runs in GitHub Actions, where DeeperHistReg,
    # SimpleITK and pyvips are installed. Minimal source-only environments may
    # skip it without weakening the pure OME-TIFF tests above.
    import pytest

    pytest.importorskip("pyvips")
    sitk = pytest.importorskip("SimpleITK")
    pytest.importorskip("deeperhistreg")
    from histreggui.multichannel import warp_scientific_payload

    source = tmp_path / "if_identity.ome.tif"
    target = tmp_path / "he_identity.tif"
    array = np.zeros((4, 19, 21), dtype=np.uint16)
    ramp = np.arange(19 * 21, dtype=np.uint16).reshape(19, 21)
    for channel in range(4):
        array[channel] = ramp + (channel + 1) * 1000
    tifffile.imwrite(
        source,
        array,
        ome=True,
        metadata={
            "axes": "CYX",
            "Channel": {"Name": ["DAPI", "FITC", "TRITC", "Cy5"]},
        },
    )
    tifffile.imwrite(target, np.zeros((19, 21, 3), dtype=np.uint8), photometric="rgb")
    source_guide = create_registration_guide_tiff(
        source,
        tmp_path / "source_guide.ome.tif",
        source_pixel_size_um=(0.5, 0.6),
        tile_size=16,
        compression=None,
    )
    target_guide = create_registration_guide_tiff(
        target,
        tmp_path / "target_guide.ome.tif",
        source_pixel_size_um=(1.5, 1.6),
        tile_size=16,
        compression=None,
    )

    # DeeperHistReg stores a 2-D field as two scalar MHA planes (2,Y,X).
    field = np.zeros((2, 19, 21), dtype=np.float32)
    field_path = tmp_path / "identity.mha"
    sitk.WriteImage(sitk.GetImageFromArray(field), str(field_path))
    result = warp_scientific_payload(
        source,
        source_guide.path,
        target_guide.path,
        field_path,
        tmp_path / "warped_if.ome.tif",
        source_pixel_size_um=(0.5, 0.6),
        target_pixel_size_um=(1.5, 1.6),
        tile_size=16,
        compression=None,
    )
    warped = tifffile.imread(result.path)
    assert warped.shape == array.shape
    assert warped.dtype == np.uint16
    # libvips mapim may round an edge interpolant by one intensity unit.
    assert np.max(np.abs(warped.astype(np.int32) - array.astype(np.int32))) <= 1
    assert result.channel_names == ("DAPI", "FITC", "TRITC", "Cy5")
    assert result.pixel_size_x_um == 1.5
    assert result.pixel_size_y_um == 1.6



def test_selected_if_guide_reads_only_one_channel_per_tile(tmp_path: Path, monkeypatch) -> None:
    """The selected-channel guide must not decode all IF channels for every tile."""
    import histreggui.multichannel as mc

    source = tmp_path / "if.ome.tif"
    _write_if(source)
    calls = {"single": 0, "all": 0}
    original_open = mc.open_channel_reader

    class Proxy:
        def __init__(self, reader):
            self._reader = reader
            self.info = reader.info
            self.output_width = reader.output_width
            self.output_height = reader.output_height
            self.dtype = reader.dtype

        def read_channel_tile(self, *args, **kwargs):
            calls["single"] += 1
            return self._reader.read_channel_tile(*args, **kwargs)

        def read_channels_tile(self, *args, **kwargs):
            calls["all"] += 1
            return self._reader.read_channels_tile(*args, **kwargs)

        def __enter__(self):
            self._reader.__enter__()
            return self

        def __exit__(self, *args):
            return self._reader.__exit__(*args)

    monkeypatch.setattr(mc, "open_channel_reader", lambda *a, **k: Proxy(original_open(*a, **k)))
    mc.create_registration_guide_tiff(
        source,
        tmp_path / "guide.ome.tif",
        settings=mc.GuideSettings(mode="channel", channel_index=1),
        tile_size=16,
        compression=None,
    )
    assert calls["single"] > 0
    assert calls["all"] == 0


def test_large_tiff_reader_blocks_unsafe_full_array_fallback(tmp_path: Path, monkeypatch) -> None:
    """Source code must retain the explicit large-full-read guard."""
    import inspect
    import histreggui.multichannel as mc

    source = inspect.getsource(mc._TiffChannelReader)
    assert "_MAX_SMALL_FULL_READ_BYTES" in source
    assert "full-image fallback was blocked" in source
