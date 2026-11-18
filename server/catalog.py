"""Model catalog: loads the curated model list and filters by hardware."""
import json
import pathlib

_CATALOG_FILE = pathlib.Path(__file__).resolve().parent / "model_catalog.json"
_cache: list[dict] | None = None


def _load() -> list[dict]:
    global _cache
    if _cache is None:
        with open(_CATALOG_FILE, encoding="utf-8") as f:
            data = json.load(f)
        _cache = data["models"]
    return _cache


def all_models() -> list[dict]:
    """Every model in the catalog, in catalog order."""
    return list(_load())


def supported_models() -> list[dict]:
    """Only models whose pipeline is implemented (status == 'supported')."""
    return [m for m in _load() if m["status"] == "supported"]


def default_model() -> dict | None:
    """The catalog entry marked as default, or None."""
    for m in _load():
        if m.get("default"):
            return m
    return None


def find(model_id: str) -> dict | None:
    """Look up one model by its HuggingFace repo id."""
    for m in _load():
        if m["id"] == model_id:
            return m
    return None


def fits_vram(vram_gb: float, mode: str = "bf16") -> list[dict]:
    """Models whose VRAM requirement for *mode* is at or below *vram_gb*."""
    return [m for m in _load() if m["vram"].get(mode, 999) <= vram_gb]


def catalog_for_ui() -> list[dict]:
    """Slim view for the launcher UI: everything the picker needs."""
    out = []
    for m in _load():
        out.append({
            "id": m["id"],
            "name": m["name"],
            "params": m["params"],
            "status": m["status"],
            "default": m.get("default", False),
            "vram_bf16": m["vram"].get("bf16"),
            "vram_offload": m["vram"].get("offload"),
            "vram_mps": m["vram"].get("mps"),
            "capabilities": m["capabilities"],
            "description": m["description"],
        })
    return out


def reload():
    """Force a re-read on next access (for tests)."""
    global _cache
    _cache = None
