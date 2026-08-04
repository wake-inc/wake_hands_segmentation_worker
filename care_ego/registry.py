"""MMSegmentation registry integration."""

from __future__ import annotations

from .compat import install_mmcv_lite_extension_stub

_REGISTERED = False


def register_all_modules() -> None:
    """Register MMSegmentation and all CaRe-Ego extensions once."""
    global _REGISTERED
    if _REGISTERED:
        return

    install_mmcv_lite_extension_stub()
    from mmseg.utils import register_all_modules as register_mmseg

    register_mmseg(init_default_scope=True)

    # Imports execute the registry decorators in the research implementation.
    from . import datasets, metrics, models  # noqa: F401

    _REGISTERED = True


# Enables ``custom_imports = dict(imports=['care_ego.registry'])`` in configs.
register_all_modules()
