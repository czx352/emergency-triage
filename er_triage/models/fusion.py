"""双塔融合模型。

结构化塔：生命体征 MLP
文本塔：主诉编码器（hash / hf）
融合：门控加权（gated）或直接拼接（concat）
输出头：coral（K-1 个累积 logit）或 ce（K 类 logit）
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..schema import N_CLASSES


class VitalsEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        return self.net(x_num)


class FusionModel(nn.Module):
    def __init__(self, text_encoder: nn.Module, n_numeric: int, text_dim: int = 128,
                 vitals_hidden: tuple[int, ...] = (128, 64), fusion: str = "gated",
                 head: str = "coral", dropout: float = 0.2):
        super().__init__()
        if fusion not in {"gated", "concat"}:
            raise ValueError(f"未知 fusion: {fusion}")
        if head not in {"coral", "ce"}:
            raise ValueError(f"未知 head: {head}")
        self.text_encoder = text_encoder
        self.vitals_encoder = VitalsEncoder(n_numeric, vitals_hidden, dropout)
        self.fusion = fusion
        self.head = head
        v_dim = self.vitals_encoder.out_dim
        if fusion == "gated":
            self.gate = nn.Sequential(nn.Linear(text_dim + v_dim, text_dim), nn.Sigmoid())
            fused_dim = text_dim + v_dim
        else:
            self.gate = None
            fused_dim = text_dim + v_dim
        out_dim = N_CLASSES - 1 if head == "coral" else N_CLASSES
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x_num: torch.Tensor, x_text: torch.Tensor) -> torch.Tensor:
        text_repr = self.text_encoder(x_text)
        vitals_repr = self.vitals_encoder(x_num)
        if self.gate is not None:
            g = self.gate(torch.cat([text_repr, vitals_repr], dim=-1))
            text_repr = g * text_repr
        return self.classifier(torch.cat([text_repr, vitals_repr], dim=-1))


def probs_from_logits(logits: torch.Tensor, head: str) -> torch.Tensor:
    """把输出头转成 5 类概率。

    coral 输出 K-1 个累积概率 P(y > k)，相邻相减得到各类概率，
    天然满足单调性，避免出现「1 级概率低于 2 级」这种非法分布。
    """
    if head == "ce":
        return torch.softmax(logits, dim=-1)
    cumulative = torch.sigmoid(logits)                      # [B, K-1]
    ones = torch.ones_like(cumulative[:, :1])
    zeros = torch.zeros_like(cumulative[:, :1])
    upper = torch.cat([ones, cumulative], dim=1)            # P(y > k-1)
    lower = torch.cat([cumulative, zeros], dim=1)           # P(y > k)
    probs = (upper - lower).clamp(min=1e-6)
    return probs / probs.sum(dim=1, keepdim=True)


def build_model(cfg: dict, n_numeric: int, text_encoder: nn.Module) -> FusionModel:
    m = cfg["model"]
    return FusionModel(
        text_encoder=text_encoder,
        n_numeric=n_numeric,
        text_dim=int(m["text_dim"]),
        vitals_hidden=tuple(m["vitals_hidden"]),
        fusion=m["fusion"],
        head=m["head"],
        dropout=float(m["dropout"]),
    )
