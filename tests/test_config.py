from pathlib import Path

from cifar_cnn.config import Config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_data_root_is_resolved_from_the_config_file_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                       # run from an unrelated folder
    cfg = Config.from_yaml(CONFIGS / "resnet_cifar10.yaml")
    assert Path(cfg.data.root).is_absolute()
    assert Path(cfg.data.root) == (CONFIGS / "../../data").resolve()


def test_all_shipped_configs_load_and_name_a_known_model():
    from cifar_cnn.models import MODELS
    for path in CONFIGS.glob("*.yaml"):
        cfg = Config.from_yaml(path)
        assert cfg.model.name in MODELS, path.name
        assert cfg.train.optimizer in ("sgd", "adamw"), path.name
