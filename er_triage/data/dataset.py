"""PyTorch Dataset：数值特征 + 已编码的文本 id + 标签。"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TriageDataset(Dataset):
    def __init__(self, x_num: np.ndarray, x_text: np.ndarray, y: np.ndarray | None = None):
        self.x_num = torch.as_tensor(x_num, dtype=torch.float32)
        self.x_text = torch.as_tensor(x_text, dtype=torch.long)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.x_num.shape[0]

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.x_num[idx], self.x_text[idx]
        return self.x_num[idx], self.x_text[idx], self.y[idx]
