"""Emit the canonical models as JSON Schema.

Partner sites and the modelling team do not run this Python package, so the
contract has to be publishable in a language-neutral form. ``cxr-harmony schema
--out docs/schema`` writes one file per entity plus a bundle, which is what a
site's engineering team validates their export against before shipping anything.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import EXPORTED_MODELS

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def build_schemas() -> dict[str, dict]:
    """Return ``{entity_name: json_schema}`` for every canonical model."""
    schemas: dict[str, dict] = {}
    for model in EXPORTED_MODELS:
        schema = model.model_json_schema(mode="serialization")
        schema["$schema"] = SCHEMA_DIALECT
        schema["$id"] = f"https://github.com/SamarjeetMalik/cxr-harmony/schema/{model.__name__}.json"
        schemas[model.__name__] = schema
    return schemas


def write_schemas(out_dir: Path) -> list[Path]:
    """Write one schema file per entity plus ``bundle.json``. Returns paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schemas = build_schemas()

    written: list[Path] = []
    for name, schema in sorted(schemas.items()):
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)

    bundle = out_dir / "bundle.json"
    bundle.write_text(
        json.dumps(
            {"$schema": SCHEMA_DIALECT, "entities": dict(sorted(schemas.items()))},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(bundle)
    return written
