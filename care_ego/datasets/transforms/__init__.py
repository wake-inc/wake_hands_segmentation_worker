"""CaRe-Ego transforms used by the Python training configuration."""

from .add_transform_with_ORD import (
    LabelResizeSeperateTwoObj,
    LoadSeperateTwoObjAnnotation,
    PackSeperateTwoObjLabelSegInputs,
    RandomSeperateObjectCrop,
    ThreeLabelResizeSeperateTwoobj,
)
from .add_transforms_egohos import LoadMultiLabelImageFromFile

__all__ = [
    "LabelResizeSeperateTwoObj",
    "LoadMultiLabelImageFromFile",
    "LoadSeperateTwoObjAnnotation",
    "PackSeperateTwoObjLabelSegInputs",
    "RandomSeperateObjectCrop",
    "ThreeLabelResizeSeperateTwoobj",
]
