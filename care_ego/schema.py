"""Architecture-neutral inference result contract.

The contract models inputs, model provenance, and one or more typed outputs.
Prediction ``value`` is deliberately polymorphic so the same envelope can
carry spatial geometry, generated text, classifications, embeddings, or
structured tool output without depending on a particular neural architecture.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

RESULT_SCHEMA_NAME = "wake-ai/inference-result"
RESULT_SCHEMA_VERSION = "1.0.0"


RESULT_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:wake-ai:schema:inference-result:1.0.0",
    "title": "WAKE AI inference result",
    "description": (
        "Architecture-neutral result envelope for CNN, VLM, LLM, transformer, "
        "and hybrid inference workers."
    ),
    "type": "object",
    "required": ["schema", "request", "created_at", "model", "inputs", "outputs"],
    "additionalProperties": False,
    "properties": {
        "schema": {
            "type": "object",
            "required": ["name", "version"],
            "additionalProperties": False,
            "properties": {
                "name": {"const": RESULT_SCHEMA_NAME},
                "version": {"const": RESULT_SCHEMA_VERSION},
            },
        },
        "request": {
            "type": "object",
            "required": ["id"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "config": {"type": "object", "additionalProperties": True},
            },
        },
        "created_at": {"type": "string", "format": "date-time"},
        "model": {"$ref": "#/$defs/model"},
        "inputs": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/input"},
        },
        "outputs": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/output"},
        },
        "runtime": {"type": "object", "additionalProperties": True},
    },
    "$defs": {
        "model": {
            "type": "object",
            "required": ["id", "families"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "version": {"type": "string"},
                "architecture": {"type": "string"},
                "families": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "framework": {"type": "string"},
                "checkpoint": {"type": "string"},
                "attributes": {"type": "object", "additionalProperties": True},
            },
        },
        "input": {
            "type": "object",
            "required": ["id", "modality", "source"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "modality": {"type": "string", "minLength": 1},
                "source": {
                    "type": "object",
                    "required": ["type", "value"],
                    "additionalProperties": False,
                    "properties": {
                        "type": {"type": "string", "minLength": 1},
                        "value": {},
                    },
                },
                "attributes": {"type": "object", "additionalProperties": True},
            },
        },
        "output": {
            "type": "object",
            "required": ["id", "modality", "unit", "items"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "modality": {"type": "string", "minLength": 1},
                "unit": {"type": "string", "minLength": 1},
                "coordinate_system": {
                    "type": "object",
                    "additionalProperties": True,
                },
                "items": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/prediction"},
                    },
                },
                "attributes": {"type": "object", "additionalProperties": True},
            },
        },
        "prediction": {
            "type": "object",
            "required": ["id", "task", "type", "value"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "task": {"type": "string", "minLength": 1},
                "type": {"type": "string", "minLength": 1},
                "value": {},
                "label": {"type": "string"},
                "score": {"type": ["number", "null"]},
                "track_id": {"type": ["integer", "string"]},
                "source_id": {"type": "string"},
                "span": {
                    "type": "object",
                    "required": ["start", "end", "unit"],
                    "additionalProperties": False,
                    "properties": {
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "unit": {"type": "string"},
                    },
                },
                "attributes": {"type": "object", "additionalProperties": True},
            },
        },
    },
}


def json_schema() -> dict[str, Any]:
    """Return an isolated copy safe for callers to mutate."""
    return deepcopy(RESULT_JSON_SCHEMA)


def build_result_document(
    *,
    request_id: str,
    model: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any] | None = None,
    request_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-compatible result and assign stable prediction IDs."""
    normalized_outputs: list[dict[str, Any]] = []
    for output in outputs:
        normalized = dict(output)
        output_id = str(normalized["id"])
        normalized_items: dict[int | str, list[dict[str, Any]]] = {}
        for unit_id, predictions in normalized["items"].items():
            records = []
            for ordinal, prediction in enumerate(predictions):
                record = dict(prediction)
                for required in ("task", "type", "value"):
                    if required not in record:
                        raise ValueError(f"Prediction is missing required field: {required}")
                record.setdefault("id", f"{output_id}:{unit_id}:{ordinal}")
                records.append(record)
            normalized_items[unit_id] = records
        normalized["items"] = normalized_items
        normalized_outputs.append(normalized)

    document: dict[str, Any] = {
        "schema": {"name": RESULT_SCHEMA_NAME, "version": RESULT_SCHEMA_VERSION},
        "request": {
            "id": request_id,
            **({"config": dict(request_config)} if request_config is not None else {}),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": dict(model),
        "inputs": [dict(item) for item in inputs],
        "outputs": normalized_outputs,
    }
    if runtime is not None:
        document["runtime"] = dict(runtime)
    return document
