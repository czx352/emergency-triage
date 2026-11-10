import torch

from er_triage.models.fusion import FusionModel, probs_from_logits
from er_triage.models.losses import CoralLoss, CostSensitiveCE, coral_targets
from er_triage.models.text import HashTextEncoder

COST = [[0, 1, 2, 4, 8],
        [1, 0, 1, 3, 6],
        [2, 1, 0, 1, 3],
        [4, 3, 1, 0, 1],
        [8, 6, 3, 1, 0]]


def test_coral_targets_encoding():
    y = torch.tensor([0, 2, 4])
    targets = coral_targets(y, 5)
    assert targets.shape == (3, 4)
    assert targets[0].tolist() == [0.0, 0.0, 0.0, 0.0]      # ESI1 不超过任何阈值
    assert targets[1].tolist() == [1.0, 1.0, 0.0, 0.0]      # ESI3 超过 1、2
    assert targets[2].tolist() == [1.0, 1.0, 1.0, 1.0]      # ESI5 超过全部


def test_coral_loss_is_finite_and_backward_works():
    logits = torch.randn(4, 4, requires_grad=True)
    y = torch.tensor([0, 1, 3, 4])
    loss = CoralLoss()(logits, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_cost_sensitive_ce_penalizes_critical_downgrade_harder():
    criterion = CostSensitiveCE(COST)
    # 同一组强烈指向 ESI5 的 logits：真实为 ESI1 时代价 8，真实为 ESI4 时代价 1
    logits = torch.tensor([[-8.0, -8.0, -8.0, -8.0, 8.0]])
    loss_critical = criterion(logits, torch.tensor([0]))
    loss_mild = criterion(logits, torch.tensor([3]))
    assert loss_critical > loss_mild


def test_coral_probs_are_valid_distribution():
    logits = torch.randn(16, 4)
    probs = probs_from_logits(logits, "coral")
    assert probs.shape == (16, 5)
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=1), torch.ones(16), atol=1e-5)


def test_fusion_forward_shapes():
    encoder = HashTextEncoder(vocab_size=1024, out_dim=32)
    model = FusionModel(encoder, n_numeric=10, text_dim=32,
                        vitals_hidden=(16, 8), fusion="gated", head="coral")
    logits = model(torch.randn(6, 10), torch.randint(0, 1024, (6, 24)))
    assert logits.shape == (6, 4)
