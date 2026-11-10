"""损失函数。

- coral：序数回归损失，把「分级距离」纳入优化，比普通交叉熵更契合 ESI 的有序结构；
- ce：普通交叉熵，作为对照；
- cost_ce：代价敏感交叉熵，按临床代价矩阵最小化期望风险（危重降级惩罚最重）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import N_CLASSES


def coral_targets(y: torch.Tensor, n_classes: int = N_CLASSES) -> torch.Tensor:
    """0-based 标签 -> K-1 维累积目标：target[k] = 1 if y > k。"""
    thresholds = torch.arange(n_classes - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > thresholds).float()


class CoralLoss(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, coral_targets(y, self.n_classes))


class CostSensitiveCE(nn.Module):
    """L = NLL + lam * E[cost]

    期望代价项用模型自己的概率分布计算，可导；lam 控制风险最小化强度。
    """

    def __init__(self, cost_matrix, lam: float = 1.0, n_classes: int = N_CLASSES):
        super().__init__()
        cost = torch.as_tensor(cost_matrix, dtype=torch.float32)
        if cost.shape != (n_classes, n_classes):
            raise ValueError(f"代价矩阵形状应为 {(n_classes, n_classes)}，实际 {tuple(cost.shape)}")
        self.register_buffer("cost", cost / cost.max().clamp(min=1.0))
        self.lam = lam

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        nll = -logp.gather(1, y.view(-1, 1)).squeeze(1)
        expected_cost = (logp.exp() * self.cost[y]).sum(dim=1)
        return nll.mean() + self.lam * expected_cost.mean()


def build_loss(name: str, cost_matrix=None, lam: float = 1.0) -> nn.Module:
    if name == "coral":
        return CoralLoss()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "cost_ce":
        if cost_matrix is None:
            raise ValueError("cost_ce 需要提供 cost_matrix")
        return CostSensitiveCE(cost_matrix, lam=lam)
    raise ValueError(f"未知损失函数: {name}")
