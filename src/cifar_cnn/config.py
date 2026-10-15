from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    root: str = "./data"
    val_size: int = 5000
    batch_size: int = 128
    num_workers: int = 4
    augment: str = "trivial"
    random_erasing: float = 0.25
    mixup_alpha: float = 0.0


@dataclass
class ModelConfig:
    name: str = "resnet18"
    num_classes: int = 10
    dropout: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 60
    optimizer: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    grad_clip: float = 2.0
    amp: bool = True
    early_stop_patience: int = 15
    compile: bool = False


@dataclass
class Config:
    seed: int = 42
    experiment: str = "resnet18_cifar10"
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        return cls.from_dict(yaml.safe_load(path.read_text()), base_dir=path.parent)

    @classmethod
    def from_dict(cls, raw: dict, base_dir: Path | None = None) -> "Config":
        data = DataConfig(**raw.get("data", {}))
        # A relative data root is resolved against the config file, so it works from any folder
        if base_dir is not None and not Path(data.root).is_absolute():
            data.root = str((base_dir / data.root).resolve())
        return cls(
            seed=raw.get("seed", 42),
            experiment=raw.get("experiment", "run"),
            output_dir=raw.get("output_dir", "runs"),
            data=data,
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)