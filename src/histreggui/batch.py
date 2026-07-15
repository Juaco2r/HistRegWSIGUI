from __future__ import annotations

"""Pure helpers for one-target/many-moving-image registration batches."""

import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RegistrationPlanItem:
    index: int
    moving_path: Path
    warped_output: Path
    run_directory: Path


@dataclass(frozen=True)
class RegistrationBatchPlan:
    fixed_path: Path
    batch_root: Path | None
    manifest_csv: Path
    manifest_json: Path
    error_log: Path
    items: tuple[RegistrationPlanItem, ...]

    @property
    def is_batch(self) -> bool:
        return len(self.items) > 1

    @property
    def output_directory(self) -> Path:
        if self.batch_root is not None:
            return self.batch_root / "warped"
        return self.fixed_path.parent


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


def build_registration_batch_plan(
    fixed_path: str | Path,
    moving_paths: Sequence[str | Path],
    run_stamp: str,
) -> RegistrationBatchPlan:
    """Plan output paths for either a single registration or a safe batch.

    Single-image output intentionally preserves the original v1.0 behavior:
    the warped TIFF is written next to the fixed image. Multiple moving images
    are grouped under one timestamped batch directory with numbered filenames,
    preventing collisions when different folders contain equal basenames.
    """

    fixed = Path(fixed_path).expanduser()
    moving = unique_paths(moving_paths)
    if not moving:
        raise ValueError("At least one moving image is required.")

    fixed_stem = safe_stem(fixed.stem, "fixed")
    stamp = safe_stem(run_stamp, "run")
    items: list[RegistrationPlanItem] = []

    if len(moving) == 1:
        source = moving[0]
        source_stem = safe_stem(source.stem, "moving")
        items.append(
            RegistrationPlanItem(
                index=1,
                moving_path=source,
                warped_output=fixed.parent / f"{source_stem}_warped_to_{fixed_stem}.tif",
                run_directory=fixed.parent / f"Run_{stamp}",
            )
        )
        return RegistrationBatchPlan(
            fixed_path=fixed,
            batch_root=None,
            manifest_csv=fixed.parent / f"HistRegGUI_registration_{stamp}.csv",
            manifest_json=fixed.parent / f"HistRegGUI_registration_{stamp}.json",
            error_log=fixed.parent / "HistRegGUI_error.log",
            items=tuple(items),
        )

    batch_root = fixed.parent / f"HistRegGUI_batch_{fixed_stem}_{stamp}"
    warped_root = batch_root / "warped"
    intermediate_root = batch_root / "intermediate"

    for index, source in enumerate(moving, start=1):
        source_stem = safe_stem(source.stem, "moving")
        prefix = f"{index:03d}_{source_stem}"
        items.append(
            RegistrationPlanItem(
                index=index,
                moving_path=source,
                warped_output=warped_root / f"{prefix}_warped_to_{fixed_stem}.tif",
                run_directory=intermediate_root / f"{prefix}_Run",
            )
        )

    return RegistrationBatchPlan(
        fixed_path=fixed,
        batch_root=batch_root,
        manifest_csv=batch_root / "registration_manifest.csv",
        manifest_json=batch_root / "registration_manifest.json",
        error_log=batch_root / "HistRegGUI_error.log",
        items=tuple(items),
    )


def write_registration_manifest(
    plan: RegistrationBatchPlan,
    results: Sequence[dict[str, object]],
) -> None:
    """Write machine-readable CSV and JSON summaries for a completed run."""

    plan.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(result) for result in results]
    fieldnames = [
        "index",
        "status",
        "fixed_image",
        "moving_image",
        "warped_output",
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
        "batch_root": str(plan.batch_root) if plan.batch_root else None,
        "items": rows,
    }
    plan.manifest_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
