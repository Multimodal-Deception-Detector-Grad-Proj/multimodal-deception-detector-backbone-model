import torch
from src.common.constants import D_AUD, TARGET_T, AUDIO_CHANNEL_INDEX
from src.models.audio.audio_cnn import AudioCNN, AudioClassifier, D_LATENT


def _batch(n_valid=(200, 120, 40)):
    x = torch.randn(len(n_valid), TARGET_T, D_AUD)
    x[..., AUDIO_CHANNEL_INDEX["valid"]] = 0
    for i, n in enumerate(n_valid):
        x[i, :n, AUDIO_CHANNEL_INDEX["valid"]] = 1
        x[i, n:] = 0
    return x


def test_shapes_and_finite():
    z, seq = AudioCNN().eval()(_batch())
    assert z.shape == (3, D_LATENT) and seq.shape[:2] == (3, TARGET_T)
    assert torch.isfinite(z).all() and torch.isfinite(seq).all()


def test_padding_does_not_change_output():
    m = AudioCNN().eval()
    x = _batch((100,))
    y = x.clone(); y[0, 100:, :AUDIO_CHANNEL_INDEX["valid"]] = 5.0   # garbage in padded frames
    assert torch.allclose(m(x)[0], m(y)[0], atol=1e-5)


def test_backward_gives_gradients():
    m = AudioClassifier()
    m(_batch()).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_prosody_only_variant():
    m = AudioCNN(prosody_only=True).eval()
    z, _ = m(_batch())
    assert z.shape == (3, D_LATENT) and torch.isfinite(z).all()
    x = _batch((100,)); y = x.clone(); y[0, :, :59] = 7.0           # mel bins must be ignored
    assert torch.allclose(m(x)[0], m(y)[0], atol=1e-5)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("PASS", k)