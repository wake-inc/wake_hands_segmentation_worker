"""Convert segmentation masks into compact JSON annotation geometry."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np
from shapely import GeometryCollection, MultiPolygon, Polygon, make_valid

from .inference import Prediction

TRACK_IDS = {
    "left_hand": 1,
    "right_hand": 2,
    "left_object": 3,
    "right_object": 4,
    "shared_object": 5,
    "hand_object_contact": 6,
}


def _polygons(geometry: Any) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        if not geometry.is_empty:
            yield geometry
    elif isinstance(geometry, (MultiPolygon, GeometryCollection)):
        for child in geometry.geoms:
            yield from _polygons(child)


def _round_point(x: float, y: float) -> list[float | int]:
    def clean(value: float) -> float | int:
        rounded = round(float(value), 2)
        return int(rounded) if rounded.is_integer() else rounded

    return [clean(x), clean(y)]


def mask_annotations(
    mask: np.ndarray,
    *,
    label: str,
    track_id: int,
    simplify_tolerance: float,
    min_area: int,
) -> list[dict]:
    """Create polygon, bbox, and point annotations for mask components."""
    binary = (mask > 0).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    annotations: list[dict] = []

    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) < min_area:
            continue
        component_mask = (components == component).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            coordinates = contour[:, 0, :]
            if len(coordinates) < 3:
                continue
            geometry = Polygon(coordinates)
            if not geometry.is_valid:
                geometry = make_valid(geometry)
            geometry = geometry.simplify(simplify_tolerance, preserve_topology=True)
            for polygon in _polygons(geometry):
                if polygon.area < min_area:
                    continue
                exterior = [_round_point(x, y) for x, y in polygon.exterior.coords]
                min_x, min_y, max_x, max_y = polygon.bounds
                point = polygon.representative_point()
                common = {
                    "task": "semantic_segmentation",
                    "source_id": "cascadepsp",
                    "track_id": track_id,
                    "label": label,
                }
                annotations.extend(
                    (
                        {**common, "type": "polygon", "value": exterior},
                        {
                            **common,
                            "type": "bbox",
                            "value": [
                                _round_point(min_x, min_y),
                                _round_point(max_x, max_y),
                            ],
                        },
                        {
                            **common,
                            "type": "point",
                            "value": _round_point(point.x, point.y),
                        },
                    )
                )
    return annotations


def _contact_annotations(mask: np.ndarray, min_area: int) -> list[dict]:
    count, _components, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    return [
        {
            "task": "contact_detection",
            "source_id": "cascadepsp",
            "track_id": TRACK_IDS["hand_object_contact"],
            "label": "hand_object_contact",
            "type": "point",
            "value": _round_point(*centroids[index]),
        }
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_area
    ]


def prediction_annotations(
    prediction: Prediction,
    *,
    simplify_tolerance: float = 2.0,
    min_area: int = 64,
) -> list[dict]:
    """Convert one prediction to the worker's annotation list."""
    shared = prediction.shared_object > 0
    masks = {
        "left_hand": prediction.hands == 1,
        "right_hand": prediction.hands == 2,
        "left_object": (prediction.left_object > 0) & ~shared,
        "right_object": (prediction.right_object > 0) & ~shared,
        "shared_object": shared,
    }
    annotations = []
    for label, mask in masks.items():
        annotations.extend(
            mask_annotations(
                mask,
                label=label,
                track_id=TRACK_IDS[label],
                simplify_tolerance=simplify_tolerance,
                min_area=min_area,
            )
        )
    annotations.extend(_contact_annotations(prediction.derived_contact, max(4, min_area // 8)))
    return annotations
