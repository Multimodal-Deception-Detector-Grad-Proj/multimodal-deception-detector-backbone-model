import torch
from src.extractors.visual.visual_normalizer import VisualNormalizer


def test_normalizer_fits_train_only_and_reuses_stats():
    train = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    val = torch.full((1, 3, 2), 1000.0)
    normalizer = VisualNormalizer()
    train_normalized = normalizer.fit_transform(train)
    mean_before = normalizer.mean.clone()
    val_normalized = normalizer.transform(val)

    assert train_normalized.shape == train.shape
    assert val_normalized.shape == val.shape
    assert torch.allclose(normalizer.mean, mean_before)
    assert torch.isfinite(train_normalized).all()
    assert torch.isfinite(val_normalized).all()


def test_normalizer_requires_fit_before_transform():
    normalizer = VisualNormalizer()
    try:
        normalizer.transform(torch.zeros(1, 2, 3))
    except RuntimeError:
        return
    raise AssertionError("Expected transform before fit to fail")
