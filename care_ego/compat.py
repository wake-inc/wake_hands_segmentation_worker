"""Compatibility helpers for optional MMCV native operators."""

from __future__ import annotations

import importlib
import sys
import types


def install_mmcv_lite_extension_stub() -> None:
    """Allow inference-only use with ``mmcv-lite``.

    MMSegmentation imports wrappers for all MMCV native operators eagerly,
    although CaRe-Ego inference does not use them. Full MMCV installations are
    left unchanged. If a native operator is actually requested from mmcv-lite,
    the stub fails with a clear runtime error.
    """
    if "mmcv._ext" in sys.modules:
        return
    try:
        importlib.import_module("mmcv._ext")
        return
    except ModuleNotFoundError:
        pass

    extension = types.ModuleType("mmcv._ext")
    extension.__file__ = "<mmcv-lite-no-native-extension>"

    def unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "This operation requires full mmcv with native extensions; "
            "mmcv-lite supports CaRe-Ego inference only."
        )

    extension.__getattr__ = lambda _name: unavailable
    sys.modules["mmcv._ext"] = extension
