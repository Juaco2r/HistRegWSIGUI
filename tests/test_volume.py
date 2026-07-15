from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from histreggui.volume import (
    VolumeSlice,
    create_merged_ome_tiff,
    estimate_uncompressed_size_bytes,
    infer_pixel_size_um,
)


def test_streamed_ome_bigtiff_stack_preserves_slice_order(tmp_path: Path) -> None:
    fixed = np.zeros((30, 35, 3), dtype=np.uint8)
    fixed[..., 0] = 25
    warped = np.zeros((30, 35, 3), dtype=np.uint8)
    warped[..., 1] = 125

    fixed_path = tmp_path / "fixed.ome.tif"
    warped_path = tmp_path / "warped.tif"
    tifffile.imwrite(
        fixed_path,
        fixed,
        ome=True,
        photometric="rgb",
        metadata={
            "axes": "YXS",
            "PhysicalSizeX": 0.5,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.5,
            "PhysicalSizeYUnit": "µm",
        },
    )
    tifffile.imwrite(warped_path, warped, photometric="rgb")

    output = tmp_path / "registered_stack.ome.tif"
    result = create_merged_ome_tiff(
        output,
        [
            VolumeSlice(fixed_path, role="fixed"),
            VolumeSlice(warped_path, role="warped", source_path=tmp_path / "moving.svs"),
        ],
        downsample=1,
        voxel_xy_um=0.5,
        voxel_z_um=4.0,
        tile_size=16,
    )

    with tifffile.TiffFile(output) as tif:
        assert tif.is_bigtiff
        assert tif.ome_metadata is not None
        assert tif.series[0].shape == (2, 30, 35, 3)
        assert tif.series[0].axes == "ZYXS"

    volume = tifffile.imread(output)
    assert tuple(volume[0, 0, 0]) == (25, 0, 0)
    assert tuple(volume[1, 0, 0]) == (0, 125, 0)
    assert result.voxel_xy_um == 0.5

    sidecar = json.loads(result.sidecar_json.read_text(encoding="utf-8"))
    assert sidecar["shape"] == [2, 30, 35, 3]
    assert [item["role"] for item in sidecar["slices"]] == ["fixed", "warped"]


def test_downsample_is_applied_without_loading_stack(tmp_path: Path) -> None:
    first = np.full((31, 35, 3), 10, dtype=np.uint8)
    second = np.full((31, 35, 3), 20, dtype=np.uint8)
    paths = []
    for index, array in enumerate((first, second), start=1):
        path = tmp_path / f"slice_{index}.tif"
        tifffile.imwrite(path, array, photometric="rgb")
        paths.append(path)

    result = create_merged_ome_tiff(
        tmp_path / "small.ome.tif",
        [VolumeSlice(path) for path in paths],
        downsample=4,
        voxel_xy_um=0.25,
        voxel_z_um=5.0,
        tile_size=16,
    )
    assert (result.height, result.width) == (8, 9)
    assert result.voxel_xy_um == 1.0
    assert tifffile.imread(result.path).shape == (2, 8, 9, 3)


def test_infer_ome_pixel_size_and_size_estimate(tmp_path: Path) -> None:
    path = tmp_path / "calibrated.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((8, 9), dtype=np.uint8),
        ome=True,
        metadata={
            "axes": "YX",
            "PhysicalSizeX": 0.24,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.25,
            "PhysicalSizeYUnit": "µm",
        },
    )
    assert infer_pixel_size_um(path) == (0.24, 0.25)
    assert estimate_uncompressed_size_bytes(4, 400, 200, downsample=2) == 4 * 200 * 100 * 3
