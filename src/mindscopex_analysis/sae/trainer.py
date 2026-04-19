from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .model import SparseAutoencoder


@dataclass
class SAETrainConfig:
    d_input: int = 896
    d_hidden: int = 896 * 4
    k: int | None = None
    lr: float = 3e-4
    l1_coeff: float = 5e-3
    batch_size: int = 256
    epochs: int = 5
    normalize_decoder: bool = True
    device: str = "cpu"
    log_every: int = 50


class SAETrainer:
    """Activation 텐서 → SAE 학습 루프."""

    def __init__(self, cfg: SAETrainConfig) -> None:
        self.cfg = cfg
        self.sae = SparseAutoencoder(cfg.d_input, cfg.d_hidden, k=cfg.k).to(cfg.device)
        self.optimizer = torch.optim.Adam(self.sae.parameters(), lr=cfg.lr)
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    def _make_loader(self, activations: torch.Tensor) -> DataLoader:
        ds = TensorDataset(activations.to(self.cfg.device))
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)

    # ------------------------------------------------------------------
    def train(self, activations: torch.Tensor) -> list[dict[str, float]]:
        """activations: (N, d_input)."""
        loader = self._make_loader(activations)
        self.sae.train()

        step = 0
        for epoch in range(self.cfg.epochs):
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs}")
            for (batch,) in pbar:
                self.optimizer.zero_grad()
                _x_hat, _z, losses = self.sae(batch)
                loss = losses["recon_loss"] + self.cfg.l1_coeff * losses["l1_loss"]
                loss.backward()
                self.optimizer.step()

                if self.cfg.normalize_decoder:
                    self.sae.normalize_decoder_()

                row = {
                    "step": step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "recon": losses["recon_loss"].item(),
                    "l1": losses["l1_loss"].item(),
                    "l0": losses["l0"].item(),
                }
                self.history.append(row)
                if step % self.cfg.log_every == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}", l0=f"{losses['l0'].item():.0f}")
                step += 1

        return self.history

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.sae.state_dict(), "config": self.cfg.__dict__},
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> SAETrainer:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = SAETrainConfig(**ckpt["config"])
        cfg.device = device
        trainer = cls(cfg)
        trainer.sae.load_state_dict(ckpt["state_dict"])
        return trainer
