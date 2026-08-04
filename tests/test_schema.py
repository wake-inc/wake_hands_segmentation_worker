import json

from jsonschema import Draft202012Validator

from care_ego.schema import RESULT_JSON_SCHEMA, build_result_document


def _validate_after_json_round_trip(document: dict) -> None:
    Draft202012Validator.check_schema(RESULT_JSON_SCHEMA)
    encoded = json.dumps(document)
    Draft202012Validator(RESULT_JSON_SCHEMA).validate(json.loads(encoded))


def test_schema_carries_frame_geometry_from_hybrid_vision_model() -> None:
    document = build_result_document(
        request_id="video-job",
        model={
            "id": "care-ego",
            "families": ["cnn", "transformer"],
            "framework": "pytorch",
        },
        inputs=[
            {
                "id": "video",
                "modality": "video",
                "source": {"type": "uri", "value": "file:///input.mp4"},
            }
        ],
        outputs=[
            {
                "id": "frame_predictions",
                "modality": "vision",
                "unit": "frame",
                "items": {
                    0: [
                        {
                            "task": "semantic_segmentation",
                            "type": "polygon",
                            "track_id": 1,
                            "label": "left_hand",
                            "value": [[1, 2], [3, 4], [1, 2]],
                        }
                    ]
                },
            }
        ],
    )

    assert document["outputs"][0]["items"][0][0]["id"] == "frame_predictions:0:0"
    _validate_after_json_round_trip(document)


def test_schema_carries_multimodal_llm_generation() -> None:
    document = build_result_document(
        request_id="vlm-job",
        model={
            "id": "example-vlm",
            "families": ["vlm", "llm", "transformer"],
            "framework": "pytorch",
        },
        inputs=[
            {
                "id": "prompt",
                "modality": "text",
                "source": {"type": "inline", "value": "Describe this image"},
            },
            {
                "id": "image",
                "modality": "image",
                "source": {"type": "uri", "value": "file:///image.jpg"},
            },
        ],
        outputs=[
            {
                "id": "completion",
                "modality": "text",
                "unit": "message",
                "items": {
                    "assistant": [
                        {
                            "task": "generation",
                            "type": "text",
                            "value": "A hand is holding an object.",
                            "score": None,
                        }
                    ]
                },
            }
        ],
        runtime={"input_tokens": 8, "output_tokens": 7},
        request_config={"pipeline": {"next_task": "summarize"}},
    )

    assert document["outputs"][0]["items"]["assistant"][0]["type"] == "text"
    assert document["request"]["config"]["pipeline"]["next_task"] == "summarize"
    _validate_after_json_round_trip(document)
