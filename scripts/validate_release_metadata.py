#!/usr/bin/env python3
"""Validate version metadata used by GitHub, Zenodo, and the application."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def application_version() -> str:
    namespace: dict[str, object] = {}
    version_file = ROOT / "src" / "histreggui" / "__init__.py"
    exec(version_file.read_text(encoding="utf-8"), namespace)
    version = str(namespace.get("__version__", "")).strip()
    if not version:
        raise RuntimeError(f"Missing __version__ in {version_file}")
    return version


def citation_version() -> str:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"(?m)^version:\s*[\"']?([^\"'\n]+)", text)
    if not match:
        raise RuntimeError("CITATION.cff does not contain a version field")
    return match.group(1).strip()


def zenodo_version() -> str:
    metadata = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    required = {"title", "version", "creators", "upload_type", "license", "access_right"}
    missing = sorted(required.difference(metadata))
    if missing:
        raise RuntimeError(f".zenodo.json is missing required project fields: {', '.join(missing)}")
    if metadata["upload_type"] != "software":
        raise RuntimeError(".zenodo.json upload_type must be 'software'")
    if not isinstance(metadata["creators"], list) or not metadata["creators"]:
        raise RuntimeError(".zenodo.json must contain at least one creator")
    return str(metadata["version"]).strip()


def normalize_tag(tag: str) -> str:
    tag = tag.strip()
    return tag[1:] if tag.lower().startswith("v") else tag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="Optional Git tag, for example v1.0")
    args = parser.parse_args()

    versions = {
        "application": application_version(),
        "CITATION.cff": citation_version(),
        ".zenodo.json": zenodo_version(),
    }
    unique = set(versions.values())
    if len(unique) != 1:
        details = ", ".join(f"{name}={version}" for name, version in versions.items())
        raise RuntimeError(f"Release metadata versions do not match: {details}")

    version = next(iter(unique))
    if args.tag and normalize_tag(args.tag) != version:
        raise RuntimeError(
            f"Git tag {args.tag!r} does not match project version {version!r}. "
            f"Use tag v{version} or update all version metadata first."
        )

    print(f"Release metadata valid for HistRegGUI v{version}.")


if __name__ == "__main__":
    main()
