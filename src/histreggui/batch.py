from __future__ import annotations

"""Pure helpers for independent and cascading registration batches."""

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REGISTRATION_MODE_SAME_TARGET = "same_target"
REGISTRATION_MODE_CASCADE = "cascade"
REGISTRATION_MODES = {
    "Independent: every moving image → fixed target": REGISTRATION_MODE_SAME_TARGET,
    "Cascading: each slice → previous warped slice": REGISTRATION_MODE_CASCADE,
}


@dataclass(frozen=True)
class RegistrationPlanItem:
    index: int
    moving_path: Path
    warped_output: Path
    run_directory: Path
    working_source: Path | None = None


@dataclass(frozen=True)
class RegistrationBatchPlan:
    fixed_path: Path
    registration_mode: str
    registration_downsample: int
    batch_root: Path | None
    manifest_csv: Path
    manifest_json: Path
    error_log: Path
    items: tuple[RegistrationPlanItem, ...]

    @property
    def is_batch(self) -> bool:
        return len(self.items) > 1

    @property
    def is_cascade(self) -> bool:
        return self.registration_mode == REGISTRATION_MODE_CASCADE

    @property
    def output_directory(self) -> Path:
        if self.batch_root is not None:
            return self.batch_root / "warped"
        return self.fixed_path.parent

    @property
    def reference_directory(self) -> Path:
        if self.batch_root is not None:
            return self.batch_root / "reference"
        return self.fixed_path.parent

    @property
    def working_directory(self) -> Path:
        if self.batch_root is not None:
            return self.batch_root / "working"
        return self.fixed_path.parent / f"HistRegGUI_work_{self.manifest_csv.stem}"

    @property
    def intermediate_directory(self) -> Path:
        if self.batch_root is not None:
            return self.batch_root / "intermediate"
        return self.items[0].run_directory.parent if self.items else self.fixed_path.parent


def normalize_registration_mode(value: str) -> str:
    mode = REGISTRATION_MODES.get(str(value), str(value)).strip().lower()
    if mode not in {REGISTRATION_MODE_SAME_TARGET, REGISTRATION_MODE_CASCADE}:
        raise ValueError(f"Unsupported registration mode: {value}")
    return mode


def registration_target_for_step(
    registration_mode: str,
    fixed_registration_path: str | Path,
    previous_warped_output: str | Path | None,
) -> Path:
    """Return the target for one ordered step.

    Independent mode always uses the original fixed target. Cascading mode uses
    the previously warped output after the first step. A missing previous output
    is an error because continuing would break the consecutive-slice chain.
    """

    mode = normalize_registration_mode(registration_mode)
    fixed = Path(fixed_registration_path)
    if mode == REGISTRATION_MODE_SAME_TARGET or previous_warped_output is None:
        return fixed
    previous = Path(previous_warped_output)
    if not previous.exists():
        raise FileNotFoundError(
            "The previous warped slice required by the cascade does not exist: "
            f"{previous}"
        )
    return previous


def _path_identity(path: Path) -> str:
    """Return a stable identity for duplicate removal without requiring existence."""

    try:
        normalized = path.expanduser().resolve(strict=False)
    except Exception:
        normalized = path.expanduser().absolute()
    # normcase applies case normalization only on platforms that need it
    # (notably Windows), while preserving distinct case-sensitive paths on Linux.
    return os.path.normcase(str(normalized))


def unique_paths(paths: Iterable[str | Path]) -> list[Path]:
    """Preserve order while removing duplicate file selections."""

    output: list[Path] = []
    seen: set[str] = set()
    for value in paths:
        path = Path(value).expanduser()
        identity = _path_identity(path)
        if identity in seen:
            continue
        seen.add(identity)
        output.append(path)
    return output


def safe_stem(value: str, fallback: str = "image") -> str:
    """Return a filesystem-friendly stem while keeping names recognizable."""

    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return text or fallback


def _downsample_suffix(registration_downsample: int) -> str:
    factor = int(registration_downsample)
    return "" if factor == 1 else f"_regds{factor}"


def build_registration_batch_plan(
    fixed_path: str | Path,
    moving_paths: Sequence[str | Path],
    run_stamp: str,
    *,
    registration_mode: str = REGISTRATION_MODE_SAME_TARGET,
    registration_downsample: int = 1,
) -> RegistrationBatchPlan:
    """Plan outputs for independent or consecutive cascading registration.

    The original single-image behavior is retained only for independent,
    full-resolution registration. Cascading runs and pre-downsampled runs always
    receive a dedicated run folder because they create a reference image,
    ordered intermediate dependencies, or both.
    """

    fixed = Path(fixed_path).expanduser()
    moving = unique_paths(moving_paths)
    if not moving:
        raise ValueError("At least one moving image is required.")

    mode = normalize_registration_mode(registration_mode)
    factor = int(registration_downsample)
    if factor < 1:
        raise ValueError("Registration downsample must be at least 1.")

    fixed_stem = safe_stem(fixed.stem, "fixed")
    stamp = safe_stem(run_stamp, "run")
    suffix = _downsample_suffix(factor)
    items: list[RegistrationPlanItem] = []

    legacy_single = len(moving) == 1 and mode == REGISTRATION_MODE_SAME_TARGET and factor == 1
    if legacy_single:
        source = moving[0]
        source_stem = safe_stem(source.stem, "moving")
        items.append(
            RegistrationPlanItem(
                index=1,
                moving_path=source,
                warped_output=fixed.parent / f"{source_stem}_warped_to_{fixed_stem}.tif",
                run_directory=fixed.parent / f"Run_{stamp}",
                working_source=None,
            )
        )
        return RegistrationBatchPlan(
            fixed_path=fixed,
            registration_mode=mode,
            registration_downsample=factor,
            batch_root=None,
            manifest_csv=fixed.parent / f"HistRegGUI_registration_{stamp}.csv",
            manifest_json=fixed.parent / f"HistRegGUI_registration_{stamp}.json",
            error_log=fixed.parent / "HistRegGUI_error.log",
            items=tuple(items),
        )

    if mode == REGISTRATION_MODE_CASCADE:
        folder_prefix = "HistRegGUI_cascade"
    elif factor > 1:
        folder_prefix = "HistRegGUI_downsampled_batch"
    else:
        folder_prefix = "HistRegGUI_batch"

    batch_root = fixed.parent / f"{folder_prefix}_{fixed_stem}_{stamp}"
    warped_root = batch_root / "warped"
    intermediate_root = batch_root / "intermediate"
    working_root = batch_root / "working"

    previous_label = f"000_{fixed_stem}"
    for index, source in enumerate(moving, start=1):
        source_stem = safe_stem(source.stem, "moving")
        prefix = f"{index:03d}_{source_stem}"
        if mode == REGISTRATION_MODE_CASCADE:
            output_name = f"{prefix}_cascaded_to_{previous_label}{suffix}.tif"
            previous_label = f"{index:03d}_{source_stem}"
        else:
            output_name = f"{prefix}_warped_to_{fixed_stem}{suffix}.tif"
        working_source = (
            working_root / f"{prefix}_working{suffix}.ome.tif" if factor > 1 else None
        )
        items.append(
            RegistrationPlanItem(
                index=index,
                moving_path=source,
                warped_output=warped_root / output_name,
                run_directory=intermediate_root / f"{prefix}_Run",
                working_source=working_source,
            )
        )

    return RegistrationBatchPlan(
        fixed_path=fixed,
        registration_mode=mode,
        registration_downsample=factor,
        batch_root=batch_root,
        manifest_csv=batch_root / "registration_manifest.csv",
        manifest_json=batch_root / "registration_manifest.json",
        error_log=batch_root / "HistRegGUI_error.log",
        items=tuple(items),
    )


def default_reference_image_path(plan: RegistrationBatchPlan) -> Path:
    """Return the retained downsampled fixed/reference image path."""

    fixed_stem = safe_stem(plan.fixed_path.stem, "fixed")
    factor = int(plan.registration_downsample)
    return plan.reference_directory / f"000_fixed_{fixed_stem}_regds{factor}.ome.tif"



def default_fixed_guide_path(plan: RegistrationBatchPlan) -> Path:
    """Return the RGB guide used as the fixed registration image."""

    fixed_stem = safe_stem(plan.fixed_path.stem, "fixed")
    factor = int(plan.registration_downsample)
    return plan.reference_directory / f"000_fixed_{fixed_stem}_guide_regds{factor}.ome.tif"


def default_fixed_scientific_path(plan: RegistrationBatchPlan) -> Path:
    """Return the channel-preserving fixed/reference payload path."""

    fixed_stem = safe_stem(plan.fixed_path.stem, "fixed")
    factor = int(plan.registration_downsample)
    return plan.reference_directory / f"000_fixed_{fixed_stem}_scientific_regds{factor}.ome.tif"


def default_moving_guide_path(plan: RegistrationBatchPlan, item: RegistrationPlanItem) -> Path:
    """Return a transient RGB registration guide for one moving image."""

    source_stem = safe_stem(item.moving_path.stem, "moving")
    factor = int(plan.registration_downsample)
    return plan.working_directory / f"{item.index:03d}_{source_stem}_guide_regds{factor}.ome.tif"


def default_scientific_warped_path(
    plan: RegistrationBatchPlan, item: RegistrationPlanItem
) -> Path:
    """Return the OME-TIFF that preserves all warped source channels."""

    source_stem = safe_stem(item.moving_path.stem, "moving")
    fixed_stem = safe_stem(plan.fixed_path.stem, "fixed")
    suffix = _downsample_suffix(plan.registration_downsample)
    if plan.batch_root is not None:
        root = plan.batch_root / "warped_scientific"
    else:
        root = plan.fixed_path.parent
    if plan.is_cascade:
        name = f"{item.index:03d}_{source_stem}_cascaded_scientific{suffix}.ome.tif"
    else:
        name = f"{item.index:03d}_{source_stem}_warped_scientific_to_{fixed_stem}{suffix}.ome.tif"
    return root / name


def default_scientific_merged_volume_path(plan: RegistrationBatchPlan) -> Path:
    """Return the mixed H&E/IF channel-preserving ZCYX OME-TIFF path."""

    fixed_stem = safe_stem(plan.fixed_path.stem, "fixed")
    if plan.batch_root is not None:
        prefix = (
            "HistRegGUI_cascade_scientific_stack"
            if plan.is_cascade
            else "HistRegGUI_registered_scientific_stack"
        )
        return plan.batch_root / "merged" / f"{prefix}_{fixed_stem}.ome.tif"
    manifest_stem = plan.manifest_csv.stem
    stamp = manifest_stem.removeprefix("HistRegGUI_registration_")
    return (
        plan.fixed_path.parent
        / f"HistRegGUI_registered_scientific_stack_{fixed_stem}_{stamp}.ome.tif"
    )

def default_merged_volume_path(plan: RegistrationBatchPlan) -> Path:
    """Return a collision-safe OME-TIFF path for an optional merged stack."""

    fixed_stem = safe_stem(plan.fixed_path.stem, "fixed")
    if plan.batch_root is not None:
        prefix = "HistRegGUI_cascade_stack" if plan.is_cascade else "HistRegGUI_registered_stack"
        return plan.batch_root / "merged" / f"{prefix}_{fixed_stem}.ome.tif"

    manifest_stem = plan.manifest_csv.stem
    stamp = manifest_stem.removeprefix("HistRegGUI_registration_")
    return plan.fixed_path.parent / f"HistRegGUI_registered_stack_{fixed_stem}_{stamp}.ome.tif"


def write_registration_manifest(
    plan: RegistrationBatchPlan,
    results: Sequence[dict[str, object]],
    run_summary: dict[str, object] | None = None,
) -> None:
    """Write machine-readable CSV and JSON summaries for a completed run."""

    plan.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(result) for result in results]
    fieldnames = [
        "index",
        "status",
        "registration_mode",
        "registration_downsample",
        "fixed_image",
        "moving_image",
        "registration_source",
        "registration_target",
        "registration_guide_source",
        "registration_guide_target",
        "warped_output",
        "scientific_warped_output",
        "source_channel_count",
        "source_channel_names",
        "source_dtype",
        "intermediate_directory",
        "loader",
        "device",
        "preset",
        "started_at",
        "finished_at",
        "error",
    ]

    with plan.manifest_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    payload = {
        "fixed_image": str(plan.fixed_path),
        "registration_mode": plan.registration_mode,
        "registration_downsample": plan.registration_downsample,
        "batch_root": str(plan.batch_root) if plan.batch_root else None,
        "items": rows,
        "run_summary": dict(run_summary or {}),
    }
    plan.manifest_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
