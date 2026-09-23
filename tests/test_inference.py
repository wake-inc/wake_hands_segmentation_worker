from __future__ import annotations

import pickle

import torch

from care_ego.inference import extract_state_dict


def test_extract_state_dict_falls_back_for_legacy_checkpoint(monkeypatch) -> None:
    calls: list[tuple[bool, bool]] = []

    def load(_path, *, map_location, mmap, weights_only):
        assert map_location == "cpu"
        calls.append((mmap, weights_only))
        if mmap:
            raise RuntimeError(
                "mmap can only be used with files saved with torch.save(..., "
                "_use_new_zipfile_serialization=True)"
            )
        if weights_only:
            raise pickle.UnpicklingError("Unsupported operand 118")
        return {"state_dict": {"weight": torch.ones(1)}}

    monkeypatch.setattr(torch, "load", load)

    state = extract_state_dict("legacy-checkpoint.pth")

    assert list(state) == ["weight"]
    assert calls == [(True, True), (False, True), (False, False)]
