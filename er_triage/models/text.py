"""文本塔：两种后端。

- hash：字符 n-gram 哈希 + 掩码均值池化。无需下载预训练权重，CPU 可跑，
  用于离线冒烟与「TextCNN 级」基线；
- hf：HuggingFace 临床领域预训练模型（默认 Bio_ClinicalBERT），用于正式实验。
"""
from __future__ import annotations

import zlib
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


def _tokens(text: str) -> list[str]:
    text = (text or "").lower().strip()
    if not text:
        return ["<empty>"]
    unigram = list(text)
    bigram = [text[i:i + 2] for i in range(len(text) - 1)]
    return unigram + bigram + text.split()


def _stable_hash(token: str) -> int:
    # 用 crc32 而不是内置 hash，保证跨进程、跨机器可复现
    return zlib.crc32(token.encode("utf-8"))


class TextProcessor:
    """文本 -> id 序列的统一入口，训练与服务共用，保证一致性。"""

    def __init__(self, backend: str = "hash", max_len: int = 24,
                 vocab_size: int = 32768, model_name: str | None = None):
        if backend not in {"hash", "hf"}:
            raise ValueError(f"未知 text_backend: {backend}")
        self.backend = backend
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.model_name = model_name
        self._tokenizer = None

    def _hf_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name or "emilyalsentzer/Bio_ClinicalBERT")
        return self._tokenizer

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(t) if t is not None else "" for t in texts]
        if self.backend == "hf":
            enc = self._hf_tokenizer()(
                texts, padding="max_length", truncation=True,
                max_length=self.max_len, return_tensors="np",
            )
            return enc["input_ids"].astype(np.int64)
        out = np.zeros((len(texts), self.max_len), dtype=np.int64)
        for i, text in enumerate(texts):
            ids = [_stable_hash(tok) % (self.vocab_size - 1) + 1 for tok in _tokens(text)]
            ids = ids[: self.max_len]
            out[i, : len(ids)] = ids
        return out

    def build_encoder(self, out_dim: int, dropout: float = 0.2) -> nn.Module:
        if self.backend == "hf":
            return HFTextEncoder(self.model_name or "emilyalsentzer/Bio_ClinicalBERT", out_dim, dropout)
        return HashTextEncoder(self.vocab_size, out_dim, dropout)

    def state(self) -> dict:
        return {
            "backend": self.backend,
            "max_len": self.max_len,
            "vocab_size": self.vocab_size,
            "model_name": self.model_name,
        }

    @classmethod
    def from_state(cls, state: dict) -> "TextProcessor":
        return cls(
            backend=state["backend"],
            max_len=state["max_len"],
            vocab_size=state["vocab_size"],
            model_name=state.get("model_name"),
        )


class HashTextEncoder(nn.Module):
    """哈希词嵌入 + 掩码均值 + 线性投影。"""

    def __init__(self, vocab_size: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, out_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_text: torch.Tensor) -> torch.Tensor:
        mask = (x_text != 0).unsqueeze(-1).float()
        emb = self.embedding(x_text)
        summed = (emb * mask).sum(1)
        denom = mask.sum(1).clamp(min=1.0)
        return self.proj(summed / denom)


class HFTextEncoder(nn.Module):
    """临床预训练语言模型 + 掩码均值池化 + 线性投影。"""

    def __init__(self, model_name: str, out_dim: int, dropout: float = 0.2):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_text: torch.Tensor) -> torch.Tensor:
        attn = (x_text != 0).long()
        out = self.backbone(input_ids=x_text, attention_mask=attn).last_hidden_state
        mask = attn.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.proj(pooled)
