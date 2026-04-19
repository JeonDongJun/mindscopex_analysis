from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """ReLU 또는 TopK 희소 오토인코더.

    Parameters
    ----------
    d_input : int
        입력 차원 (= 모델 hidden_size).
    d_hidden : int
        사전 크기 (보통 4×~8× d_input).
    k : int | None
        TopK 모드에서 유지할 활성 feature 수. None이면 ReLU + L1 모드.
    """

    def __init__(self, d_input: int, d_hidden: int, k: int | None = None) -> None:
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.k = k

        self.W_enc = nn.Linear(d_input, d_hidden)
        self.W_dec = nn.Linear(d_hidden, d_input, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_input))

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc.weight)
        nn.init.zeros_(self.W_enc.bias)
        nn.init.kaiming_uniform_(self.W_dec.weight)
        with torch.no_grad():
            self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, dim=0)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.W_enc(x - self.b_dec)
        if self.k is not None:
            vals, idx = torch.topk(z, self.k, dim=-1)
            sparse = torch.zeros_like(z)
            sparse.scatter_(-1, idx, F.relu(vals))
            return sparse
        return F.relu(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.W_dec(z) + self.b_dec

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        z = self.encode(x)
        x_hat = self.decode(z)

        recon = (x - x_hat).pow(2).mean()
        l1 = z.abs().mean()
        l0 = (z > 0).float().sum(dim=-1).mean()

        return x_hat, z, {"recon_loss": recon, "l1_loss": l1, "l0": l0}

    # ------------------------------------------------------------------
    def normalize_decoder_(self) -> None:
        """학습 스텝 후 decoder 열을 단위 벡터로 정규화."""
        with torch.no_grad():
            self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, dim=0)
