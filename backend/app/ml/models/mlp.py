"""PyTorch MLP wrapped in an sklearn-compatible estimator.

Implements the minimal `fit`/`predict`/`predict_proba` surface the trainer
expects so the same `Pipeline(features, model)` pattern works for every
candidate. Training uses Adam + early stopping on validation loss.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], n_classes: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-style PyTorch classifier.

    Hyperparameters mirror `params.yaml`. Uses CPU by default; if CUDA is
    available it auto-selects the GPU.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 60,
        patience: int = 8,
        seed: int = 42,
    ) -> None:
        self.hidden_dims = hidden_dims or [128, 64]
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

    def _device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchMLPClassifier":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        device = self._device()

        # Holdout 10% for early stopping.
        n_val = max(1, int(0.1 * len(x)))
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(x))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        x_tr = torch.from_numpy(x[tr_idx]).to(device)
        y_tr = torch.from_numpy(y[tr_idx]).to(device)
        x_va = torch.from_numpy(x[val_idx]).to(device)
        y_va = torch.from_numpy(y[val_idx]).to(device)

        loader = DataLoader(
            TensorDataset(x_tr, y_tr),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Class-balanced cross entropy (Dropout is the rare class we care about).
        counts = np.bincount(y_tr.cpu().numpy(), minlength=n_classes).astype(np.float32)
        weights = torch.from_numpy(counts.sum() / (counts * n_classes + 1e-6)).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        self.model_ = _MLP(x.shape[1], self.hidden_dims, n_classes, self.dropout).to(device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        bad_epochs = 0
        for _epoch in range(self.max_epochs):
            self.model_.train()
            for xb, yb in loader:
                opt.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(x_va)
                val_loss = criterion(val_logits, y_va).item()
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_", "classes_"])
        device = self._device()
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes_[proba.argmax(axis=1)]


def build_mlp(params: dict[str, Any], seed: int) -> TorchMLPClassifier:
    return TorchMLPClassifier(seed=seed, **params)
