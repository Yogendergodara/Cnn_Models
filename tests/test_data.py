import numpy as np
import torch
from PIL import Image

from cifar_cnn.config import DataConfig
from cifar_cnn.data import build_transforms


def test_transforms_shape_and_range():
    train_tf, eval_tf = build_transforms(DataConfig())
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    for tf in (train_tf, eval_tf):
        t = tf(img)
        assert t.shape == (3, 32, 32) and t.dtype == torch.float32
        assert -3.5 < t.min() and t.max() < 3.5


def test_eval_transform_is_deterministic():
    _, eval_tf = build_transforms(DataConfig())
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    assert torch.equal(eval_tf(img), eval_tf(img))