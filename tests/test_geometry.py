import numpy as np

from care_ego.geometry import mask_annotations


def test_mask_annotations_emit_polygon_bbox_and_point() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:50, 20:45] = 1

    annotations = mask_annotations(
        mask,
        label="left_hand",
        track_id=1,
        simplify_tolerance=1.0,
        min_area=10,
    )

    assert {item["type"] for item in annotations} == {"polygon", "bbox", "point"}
    assert {item["task"] for item in annotations} == {"semantic_segmentation"}
    assert {item["source_id"] for item in annotations} == {"cascadepsp"}
    assert {item["track_id"] for item in annotations} == {1}
    bbox = next(item for item in annotations if item["type"] == "bbox")
    assert bbox["value"] == [[20, 10], [44, 49]]
    point = next(item for item in annotations if item["type"] == "point")
    assert len(point["value"]) == 2


def test_small_components_are_removed() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[1:3, 1:3] = 1

    assert (
        mask_annotations(
            mask,
            label="noise",
            track_id=99,
            simplify_tolerance=1.0,
            min_area=10,
        )
        == []
    )
