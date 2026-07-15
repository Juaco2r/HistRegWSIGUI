from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from histreggui.batch import (
    REGISTRATION_MODE_CASCADE,
    REGISTRATION_MODE_SAME_TARGET,
    build_registration_batch_plan,
    default_merged_volume_path,
    default_reference_image_path,
    registration_target_for_step,
    safe_stem,
    unique_paths,
    write_registration_manifest,
)


def test_unique_paths_preserves_order_and_removes_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "a.tif"
    second = tmp_path / "b.svs"
    assert unique_paths([first, second, first]) == [first, second]


def test_single_registration_preserves_original_output_location(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed.tif"
    moving = tmp_path / "moving.tif"
    plan = build_registration_batch_plan(fixed, [moving], "20260715_220000")

    assert plan.batch_root is None
    assert plan.registration_mode == REGISTRATION_MODE_SAME_TARGET
    assert plan.items[0].warped_output == tmp_path / "moving_warped_to_fixed.tif"
    assert plan.items[0].run_directory == tmp_path / "Run_20260715_220000"


def test_batch_registration_uses_numbered_collision_safe_outputs(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed image.ome.tif"
    first = tmp_path / "folder-a" / "sample.svs"
    second = tmp_path / "folder-b" / "sample.svs"
    plan = build_registration_batch_plan(fixed, [first, second], "20260715_220000")

    assert plan.batch_root == tmp_path / "HistRegGUI_batch_fixed_image.ome_20260715_220000"
    assert plan.items[0].warped_output.name.startswith("001_sample_")
    assert plan.items[1].warped_output.name.startswith("002_sample_")
    assert plan.items[0].warped_output != plan.items[1].warped_output


def test_cascade_plan_is_ordered_and_always_has_run_folder(tmp_path: Path) -> None:
    fixed = tmp_path / "slice 1.tif"
    moving = [tmp_path / "slice 2.tif", tmp_path / "slice 3.tif"]
    plan = build_registration_batch_plan(
        fixed,
        moving,
        "stamp",
        registration_mode=REGISTRATION_MODE_CASCADE,
    )

    assert plan.is_cascade
    assert plan.batch_root == tmp_path / "HistRegGUI_cascade_slice_1_stamp"
    assert plan.items[0].warped_output.name == "001_slice_2_cascaded_to_000_slice_1.tif"
    assert plan.items[1].warped_output.name == "002_slice_3_cascaded_to_001_slice_2.tif"
    assert default_merged_volume_path(plan).name == "HistRegGUI_cascade_stack_slice_1.ome.tif"


def test_registration_target_for_cascade_uses_previous_warp(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed.tif"
    previous = tmp_path / "001_warped.tif"
    previous.write_bytes(b"ready")

    assert registration_target_for_step(REGISTRATION_MODE_CASCADE, fixed, None) == fixed
    assert registration_target_for_step(REGISTRATION_MODE_CASCADE, fixed, previous) == previous
    assert registration_target_for_step(REGISTRATION_MODE_SAME_TARGET, fixed, previous) == fixed

    previous.unlink()
    with pytest.raises(FileNotFoundError):
        registration_target_for_step(REGISTRATION_MODE_CASCADE, fixed, previous)


def test_downsampled_single_registration_gets_reference_and_working_paths(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed.tif"
    moving = tmp_path / "moving.svs"
    plan = build_registration_batch_plan(
        fixed,
        [moving],
        "stamp",
        registration_downsample=4,
    )

    assert plan.batch_root == tmp_path / "HistRegGUI_downsampled_batch_fixed_stamp"
    assert plan.items[0].working_source is not None
    assert plan.items[0].working_source.name.endswith("_regds4.ome.tif")
    assert plan.items[0].warped_output.name.endswith("_regds4.tif")
    assert default_reference_image_path(plan).name == "000_fixed_fixed_regds4.ome.tif"


def test_manifest_is_written_as_csv_and_json(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed.tif"
    moving = tmp_path / "moving.tif"
    plan = build_registration_batch_plan(fixed, [moving], "run")
    results = [
        {
            "index": 1,
            "status": "success",
            "registration_mode": REGISTRATION_MODE_SAME_TARGET,
            "registration_downsample": 1,
            "fixed_image": str(fixed),
            "moving_image": str(moving),
            "registration_source": str(moving),
            "registration_target": str(fixed),
            "warped_output": str(plan.items[0].warped_output),
            "loader": "tiff",
            "device": "cpu",
            "preset": "default",
        }
    ]

    write_registration_manifest(plan, results)

    with plan.manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads(plan.manifest_json.read_text(encoding="utf-8"))

    assert rows[0]["status"] == "success"
    assert rows[0]["registration_target"] == str(fixed)
    assert payload["registration_mode"] == REGISTRATION_MODE_SAME_TARGET
    assert payload["items"][0]["loader"] == "tiff"
    assert safe_stem("sample image #1") == "sample_image_1"


def test_default_merged_volume_paths(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed image.tif"
    one = build_registration_batch_plan(fixed, [tmp_path / "moving.tif"], "stamp")
    assert default_merged_volume_path(one).name == "HistRegGUI_registered_stack_fixed_image_stamp.ome.tif"

    batch = build_registration_batch_plan(
        fixed, [tmp_path / "a.tif", tmp_path / "b.tif"], "stamp"
    )
    assert default_merged_volume_path(batch) == (
        batch.batch_root / "merged" / "HistRegGUI_registered_stack_fixed_image.ome.tif"
    )


def test_cascade_planning_has_no_fixed_slice_count_limit(tmp_path: Path) -> None:
    fixed = tmp_path / "slice_0000.tif"
    moving = [tmp_path / f"slice_{index:04d}.tif" for index in range(1, 1001)]
    plan = build_registration_batch_plan(
        fixed,
        moving,
        "long_series",
        registration_mode=REGISTRATION_MODE_CASCADE,
        registration_downsample=8,
    )
    assert len(plan.items) == 1000
    assert plan.items[-1].index == 1000
    assert plan.items[-1].working_source is not None
