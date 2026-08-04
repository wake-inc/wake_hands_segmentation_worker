"""CaRe-Ego hand-object segmentation package."""

from .schema import RESULT_SCHEMA_NAME, RESULT_SCHEMA_VERSION, json_schema

__version__ = "0.1.0"


def register_all_modules() -> None:
    """Register CaRe-Ego components with MMSegmentation."""
    from .registry import register_all_modules as register

    register()


__all__ = [
    "RESULT_SCHEMA_NAME",
    "RESULT_SCHEMA_VERSION",
    "__version__",
    "json_schema",
    "register_all_modules",
]
