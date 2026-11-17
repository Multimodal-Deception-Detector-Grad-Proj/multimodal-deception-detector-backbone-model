"""Contract tests for the audio stream (src/common/contracts.py): check_audio_sample / verify_audio_tensors."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.common.constants import AUDIO_CHANNEL_INDEX, D_AUD, TARGET_T  # noqa: E402
from src.common.contracts import check_audio_sample, verify_audio_tensors  # noqa: E402

V = AUDIO_CHANNEL_INDEX["valid"]


def _good(n_valid=100):
    x = np.zeros((TARGET_T, D_AUD), np.float32)
    x[:n_valid, V] = 1
    return x


def _sample(x, label=None):
    return {"audio_feats": torch.from_numpy(x), "label": torch.tensor(1) if label is None else label}


def test_good_sample_passes():
    verify_audio_tensors(torch.from_numpy(_good()))
    verify_audio_tensors(torch.from_numpy(_good(TARGET_T)))        # a full 8 s clip has no padding
    check_audio_sample(_sample(_good()))


def _bad_tensors():
    nan, gap, dirty, nonbin = _good(), _good(), _good(), _good()
    nan[0, 0] = np.nan
    gap[50, V] = 0                                    # hole inside the valid prefix
    dirty[150, 3] = 1.0                               # non-zero padding
    nonbin[:100, V] = 0.5                             # mask is not binary
    return {
        "nan": nan,
        "inf": np.full((TARGET_T, D_AUD), np.inf, np.float32),
        "too_short": np.zeros((TARGET_T - 1, D_AUD), np.float32),
        "too_wide": np.zeros((TARGET_T, D_AUD + 1), np.float32),
        "float64": _good().astype(np.float64),
        "valid_gap": gap,
        "dirty_padding": dirty,
        "empty_mask": np.zeros((TARGET_T, D_AUD), np.float32),
        "non_binary_mask": nonbin,
        "batched": np.stack([_good(), _good()]),
    }


@pytest.mark.parametrize("name", list(_bad_tensors()))
def test_bad_tensors_are_rejected(name):
    with pytest.raises(AssertionError):
        verify_audio_tensors(torch.from_numpy(_bad_tensors()[name]))


def test_bad_labels_and_missing_keys_are_rejected():
    for bad in (torch.tensor([1]), torch.tensor(1.0)):
        with pytest.raises(AssertionError):
            check_audio_sample(_sample(_good(), bad))
    with pytest.raises(AssertionError):
        check_audio_sample({"audio_feats": torch.from_numpy(_good())})            # no label
    with pytest.raises(AssertionError):
        check_audio_sample({"label": torch.tensor(1)})                           # no features