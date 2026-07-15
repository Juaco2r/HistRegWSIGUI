from __future__ import annotations

import csv
import json
from pathlib import Path

from histreggui.batch import (
    build_registration_batch_plan,
    default_merged_volume_path,
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


def test_manifest_is_written_as_csv_and_json(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed.tif"
    moving = tmp_path / "moving.tif"
    plan = build_registration_batch_plan(fixed, [moving], "run")
    results = [
        {
            "index": 1,
            "status": "success",
            "fixed_image": str(fixed),
            "moving_image": str(moving),
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
