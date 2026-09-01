import numpy as np
import pytest

from care_ego.inference import Prediction
from care_ego import refinement
from care_ego.refinement import CascadePspRefiner, ensure_cascadepsp_model, refinement_mode


class IdentityBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, int]] = []

    def refine(self, _image, mask, *, fast: bool, L: int):
        self.calls.append((fast, L))
        return mask


def _prediction() -> Prediction:
    hands = np.zeros((16, 16), dtype=np.uint8)
    hands[2:8, 2:7] = 1
    hands[8:14, 9:14] = 2
    left_object = np.zeros_like(hands)
    left_object[3:10, 6:11] = 1
    right_object = np.zeros_like(hands)
    right_object[7:13, 5:10] = 1
    return Prediction(
        hands=hands,
        left_object=left_object,
        right_object=right_object,
        shared_object=((left_object > 0) & (right_object > 0)).astype(np.uint8),
        raw_contact=np.zeros_like(hands),
        derived_contact=np.zeros_like(hands),
    )


@pytest.mark.parametrize(
    ("mode", "expected_call"),
    [("low", (True, 600)), ("medium", (False, 900))],
)
def test_cascadepsp_profiles_refine_all_semantic_masks(mode, expected_call) -> None:
    backend = IdentityBackend()
    refiner = CascadePspRefiner("cpu", backend=backend)
    original = _prediction()

    refined = refiner.refine_prediction(
        np.zeros((16, 16, 3), dtype=np.uint8), original, refinement_mode(mode)
    )

    assert backend.calls == [expected_call] * 4
    np.testing.assert_array_equal(refined.hands, original.hands)
    np.testing.assert_array_equal(refined.left_object, original.left_object)
    np.testing.assert_array_equal(refined.right_object, original.right_object)
    assert refined.derived_contact.any()


def test_refinement_mode_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="low, medium"):
        refinement_mode("high")


def test_cascadepsp_uses_cpu_for_unsupported_mps_path() -> None:
    refiner = CascadePspRefiner("mps", backend=IdentityBackend())

    assert refiner.device == "cpu"


def test_existing_cascadepsp_model_must_match_sha256(tmp_path, monkeypatch) -> None:
    import hashlib

    expected = b"verified model"
    model_path = tmp_path / "cascadepsp_v1_0.pth"
    model_path.write_bytes(expected)
    monkeypatch.setattr(refinement, "CASCADEPSP_MODEL_SHA256", hashlib.sha256(expected).hexdigest())

    assert ensure_cascadepsp_model(tmp_path, allow_download=False) == model_path

    model_path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="SHA-256"):
        ensure_cascadepsp_model(tmp_path, allow_download=False)


def test_cascadepsp_rejects_misaligned_batches() -> None:
    refiner = CascadePspRefiner("cpu", backend=IdentityBackend())

    with pytest.raises(ValueError, match="same length"):
        refiner.refine_predictions(
            [np.zeros((16, 16, 3), dtype=np.uint8)],
            [_prediction(), _prediction()],
            "low",
        )


def test_cascadepsp_oom_batch_falls_back_to_individual_frames(monkeypatch) -> None:
    refiner = CascadePspRefiner("cpu", backend=IdentityBackend())
    images = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]
    masks = [prediction.hands > 0 for prediction in (_prediction(), _prediction())]
    calls = 0

    def fail_batch(*_args):
        raise RuntimeError("CUDA out of memory")

    def score(_image, mask, _profile):
        nonlocal calls
        calls += 1
        return mask.astype(np.uint8) * 255

    monkeypatch.setattr(refiner, "_score_batch_impl", fail_batch)
    monkeypatch.setattr(refiner, "_score", score)

    result = refiner._score_batch(images, masks, refinement.REFINEMENT_PROFILES["low"])

    assert len(result) == 2
    assert calls == 2
