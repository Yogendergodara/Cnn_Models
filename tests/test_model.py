import pytest
import torch

from cifar_cnn.models import MODELS, build_model


@pytest.mark.parametrize("name", list(MODELS))
def test_output_shape(name):
    model = build_model(name).eval()
    out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)
    assert torch.isfinite(out).all()


def test_all_params_get_gradients():
    model = build_model("resnet_small")
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"No gradient for: {missing}"


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    model = build_model("resnet_small")
    x, y = torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_eval_is_deterministic():
    model = build_model("resnet18", dropout=0.5).eval()
    x = torch.randn(2, 3, 32, 32)
    assert torch.equal(model(x), model(x))


def test_checkpoint_roundtrip(tmp_path):
    a, b = build_model("resnet_small"), build_model("resnet_small")
    torch.save(a.state_dict(), tmp_path / "m.pt")
    b.load_state_dict(torch.load(tmp_path / "m.pt"))
    x = torch.randn(1, 3, 32, 32)
    assert torch.allclose(a.eval()(x), b.eval()(x))