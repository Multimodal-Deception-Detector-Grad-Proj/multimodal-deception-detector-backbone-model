"""AudioCNN.from_config: the architecture comes from configs/base_config.yaml (audio.model)."""
import pytest

torch = pytest.importorskip("torch")

from src.extractors.audio.audio_cnn import AudioCNN, AudioClassifier, audio_model_kwargs  # noqa: E402

GOOD = "audio:\n  model:\n    prosody_only: true\n    d_hidden: 48\n    dropout: 0.4\n    d_latent: 256\n"


def _cfg(tmp_path, body):
    p = tmp_path / "c.yaml"
    p.write_text(body)
    return p


def test_kwargs_follow_the_yaml(tmp_path):
    assert audio_model_kwargs(_cfg(tmp_path, GOOD)) == {"d_hidden": 48, "p_drop": 0.4, "d_latent": 256, "prosody_only": True}


def test_from_config_builds_the_configured_network(tmp_path):
    cfg = _cfg(tmp_path, GOOD)
    m = AudioCNN.from_config(cfg)
    assert m.d_hidden == 48 and m.proj[0].out_features == 256 and m.stem.in_channels == 4     # 4 prosody channels
    c = AudioClassifier.from_config(cfg, d_latent=64)                                          # keyword arguments override
    assert c.head.in_features == 64
    assert AudioCNN.from_config(_cfg(tmp_path, GOOD.replace("true", "false"))).stem.in_channels == 63   # mel bins + 4


def test_incomplete_unknown_and_invalid_settings_are_rejected(tmp_path):
    with pytest.raises(KeyError, match="d_latent"):
        audio_model_kwargs(_cfg(tmp_path, GOOD.replace("    d_latent: 256\n", "")))
    with pytest.raises(ValueError, match="unknown"):
        audio_model_kwargs(_cfg(tmp_path, GOOD + "    hidden_size: 3\n"))
    with pytest.raises(KeyError, match="audio.model"):
        audio_model_kwargs(_cfg(tmp_path, "audio:\n  train: {}\n"))
    with pytest.raises(ValueError, match="multiple of 8"):
        audio_model_kwargs(_cfg(tmp_path, GOOD.replace("48", "50")))
    with pytest.raises(ValueError, match="dropout"):
        audio_model_kwargs(_cfg(tmp_path, GOOD.replace("0.4", "1.5")))


def test_the_repository_config_builds_a_model():
    AudioCNN.from_config()                                  # configs/base_config.yaml must describe a valid model
