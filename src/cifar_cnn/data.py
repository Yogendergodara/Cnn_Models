from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms as T

from .config import DataConfig
from .utils import seed_worker

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
CLASSES = (
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle",
    "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon",
    "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail",
    "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
    "willow_tree", "wolf", "woman", "worm",
)


def build_transforms(cfg: DataConfig):
    train_tf = [T.RandomCrop(32, padding=4, padding_mode="reflect"),
                T.RandomHorizontalFlip()]
    if cfg.augment == "trivial":
        train_tf.append(T.TrivialAugmentWide())
    train_tf += [T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    if cfg.random_erasing > 0:
        train_tf.append(T.RandomErasing(p=cfg.random_erasing, value="random"))

    eval_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    return T.Compose(train_tf), eval_tf


def build_loaders(cfg: DataConfig, seed: int):
    train_tf, eval_tf = build_transforms(cfg)

    # Use the existing local copy only; a wrong root fails here instead of re-downloading 170 MB
    # Use the local copy if present; otherwise download it into data.root (needs internet and a writable root)
    download = not (Path(cfg.root) / "cifar-100-python").is_dir()
    full_train_aug = datasets.CIFAR100(cfg.root, train=True, download=download, transform=train_tf)
    full_train_eval = datasets.CIFAR100(cfg.root, train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR100(cfg.root, train=False, download=download, transform=eval_tf)

    # Fixed, seeded split so every run uses the same validation images
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full_train_aug), generator=g).tolist()
    val_idx, train_idx = perm[:cfg.val_size], perm[cfg.val_size:]

    train_set = Subset(full_train_aug, train_idx)
    val_set = Subset(full_train_eval, val_idx)

    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    persistent = cfg.num_workers > 0
    # Only the train loader keeps its workers alive between epochs; val/test would just hold idle processes
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, persistent_workers=persistent,
                              generator=torch.Generator().manual_seed(seed), **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def mixup_cutmix(x, y, alpha: float, num_classes: int):
    """Returns mixed images and soft labels. 50% MixUp, 50% CutMix."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    y1 = torch.nn.functional.one_hot(y, num_classes).float()
    y2 = y1[idx]
    if np.random.rand() < 0.5:                       # MixUp
        x = lam * x + (1 - lam) * x[idx]
    else:                                            # CutMix
        H, W = x.shape[2:]
        rh, rw = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
        cy, cx = np.random.randint(H), np.random.randint(W)
        y0, y1b = np.clip(cy - rh // 2, 0, H), np.clip(cy + rh // 2, 0, H)
        x0, x1b = np.clip(cx - rw // 2, 0, W), np.clip(cx + rw // 2, 0, W)
        x = x.clone()
        x[:, :, y0:y1b, x0:x1b] = x[idx, :, y0:y1b, x0:x1b]
        lam = 1 - ((y1b - y0) * (x1b - x0) / (H * W))
    return x, lam * y1 + (1 - lam) * y2