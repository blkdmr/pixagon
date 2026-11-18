"""Tests for the model catalog loader."""
import json
import pathlib

from server import catalog


def test_catalog_loads_and_has_models():
    models = catalog.all_models()
    assert len(models) >= 1
    for m in models:
        assert "id" in m
        assert "name" in m
        assert "status" in m
        assert m["status"] in ("supported", "planned")


def test_default_model_exists():
    d = catalog.default_model()
    assert d is not None
    assert d["default"] is True
    assert d["status"] == "supported"


def test_supported_models_excludes_planned():
    for m in catalog.supported_models():
        assert m["status"] == "supported"


def test_find_known_model():
    m = catalog.find("black-forest-labs/FLUX.2-klein-4B")
    assert m is not None
    assert m["name"] == "FLUX.2 Klein"


def test_find_unknown_returns_none():
    assert catalog.find("nonexistent/model") is None


def test_fits_vram_filters_correctly():
    # 6 GB should exclude models needing more than 6 GB bf16
    fits = catalog.fits_vram(6.0, "bf16")
    for m in fits:
        assert m["vram"]["bf16"] <= 6.0


def test_catalog_for_ui_has_expected_fields():
    items = catalog.catalog_for_ui()
    assert len(items) >= 1
    for item in items:
        assert "id" in item
        assert "name" in item
        assert "params" in item
        assert "status" in item
        assert "vram_bf16" in item
        assert "description" in item
        assert "capabilities" in item


def test_catalog_json_is_valid(tmp_path):
    """The catalog file parses and every model has required fields."""
    cat_file = pathlib.Path(__file__).resolve().parent.parent / "model_catalog.json"
    data = json.loads(cat_file.read_text(encoding="utf-8"))
    assert "models" in data
    required = {"id", "name", "params", "architecture", "pipeline",
                "status", "steps", "guidance", "max_side", "vram",
                "capabilities", "description"}
    for m in data["models"]:
        missing = required - m.keys()
        assert not missing, f"{m.get('id', '?')}: missing {missing}"
        assert isinstance(m["vram"], dict)
        assert "bf16" in m["vram"]


def test_reload_clears_cache():
    catalog.reload()
    models = catalog.all_models()
    assert len(models) >= 1
