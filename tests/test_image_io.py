from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from histreggui.image_io import (
    SUPPORTED_EXTENSIONS,
    choose_registration_loader,
    configure_registration_loader,
    deeperhistreg_loader_class,
    has_extension,
    load_image_preview,
    resolve_loader_choice,
)


def test_compound_and_wsi_extensions_are_recognized() -> None:
    assert has_extension("case.OME.TIFF", SUPPORTED_EXTENSIONS)
    assert has_extension("slide.svs", SUPPORTED_EXTENSIONS)
    assert has_extension("scan.ndpi", SUPPORTED_EXTENSIONS)
    assert has_extension("image.png", SUPPORTED_EXTENSIONS)


def test_auto_loader_selection_for_common_pairs() -> None:
    assert choose_registration_loader("a.ome.tif", "b.tiff") == "tiff"
    assert choose_registration_loader("a.svs", "b.ndpi") == "openslide"
    assert choose_registration_loader("a.png", "b.jpg") == "pil"
    assert choose_registration_loader("a.svs", "b.png") == "vips"
    assert resolve_loader_choice("Auto (recommended)", "a.svs", "b.png") == "vips"


def test_manual_loader_choice_overrides_auto() -> None:
    assert resolve_loader_choice("libvips generic / mixed formats", "a.tif", "b.tif") == "vips"
    assert resolve_loader_choice("pil", "a.png", "b.jpg") == "pil"


def test_registration_parameter_loader_is_updated() -> None:
    parameters = {"loading_params": {"loader": "tiff", "pad_value": 255}}
    returned = configure_registration_loader(parameters, "vips")
    assert returned is parameters
    assert parameters["loading_params"]["loader"] == "vips"
    assert parameters["loading_params"]["pad_value"] == 255


def test_loader_class_mapping() -> None:
    loaders = SimpleNamespace(
        TIFFLoader=object(),
        OpenSlideLoader=object(),
        VIPSLoader=object(),
        PILLoader=object(),
    )
    module = SimpleNamespace(loaders=loaders)
    assert deeperhistreg_loader_class(module, "tiff") is loaders.TIFFLoader
    assert deeperhistreg_loader_class(module, "openslide") is loaders.OpenSlideLoader
    assert deeperhistreg_loader_class(module, "vips") is loaders.VIPSLoader
    assert deeperhistreg_loader_class(module, "pil") is loaders.PILLoader


def test_png_preview_uses_a_supported_backend(tmp_path) -> None:
    path = tmp_path / "preview.png"
    array = np.zeros((80, 120, 3), dtype=np.uint8)
    array[..., 0] = 180
    Image.fromarray(array).save(path)

    preview, info = load_image_preview(path, max_side=64)

    assert preview.mode == "RGB"
    assert max(preview.size) <= 64
    assert info.source_width == 120
    assert info.source_height == 80
    assert info.reader in {"Pillow", "libvips"}


def test_extended_pathology_and_raster_extensions_are_exposed() -> None:
    for filename in (
        "case.ome.tiff",
        "case.svs",
        "case.ndpi",
        "case.mrxs",
        "case.scn",
        "case.dcm",
        "case.webp",
        "case.jp2",
    ):
        assert has_extension(filename, SUPPORTED_EXTENSIONS), filename
