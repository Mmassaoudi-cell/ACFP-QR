"""
Registry of regression models for the price-recovery benchmark: classical, boosting, and
compact deep-learning baselines (Sec. 14 benchmark hierarchy). All models share a common
sklearn-like interface: fit(X, y) / predict(X). Deep models get a thin wrapper.
"""
import numpy as np
import torch
import torch.nn as nn

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Classical / boosting factories
# ---------------------------------------------------------------------------

def make_classical_models(seed: int) -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=seed),
        "kNN": KNeighborsRegressor(n_neighbors=15, weights="distance"),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.5, gamma="scale"),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=seed),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, max_depth=14, n_jobs=-1, random_state=seed),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_depth=8, learning_rate=0.05, max_iter=400, random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, n_jobs=-1, random_state=seed, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=500, max_depth=-1, num_leaves=63, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=-1),
        "CatBoost": CatBoostRegressor(iterations=500, depth=8, learning_rate=0.05, random_seed=seed,
                                       verbose=False),
    }


SOURCE_METHOD_MODELS = ["SVR", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM"]


# ---------------------------------------------------------------------------
# Deep learning baselines
# ---------------------------------------------------------------------------

class _TorchRegressorBase:
    def __init__(self, epochs=60, lr=1e-3, batch_size=256, seed=0, patience=8, weight_decay=1e-5):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.patience = patience
        self.weight_decay = weight_decay
        self.model = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def _build(self, n_features):
        raise NotImplementedError

    def _forward(self, xb):
        return self.model(xb)

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.y_mean, self.y_std = float(np.mean(y)), float(np.std(y) + 1e-8)
        self._build(X.shape[1])
        self.model.to(DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor((y - self.y_mean) / self.y_std, dtype=torch.float32).view(-1, 1)
        n = len(Xt)
        best_val = np.inf
        best_state = None
        patience_left = self.patience
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb, yb = Xt[idx].to(DEVICE), yt[idx].to(DEVICE)
                opt.zero_grad()
                pred = self._forward(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    pv = self.predict(X_val)
                vloss = float(np.mean((pv - y_val) ** 2))
                if vloss < best_val - 1e-6:
                    best_val = vloss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_left = self.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            pred = self._forward(Xt).cpu().numpy().flatten()
        return pred * self.y_std + self.y_mean


class MLPRegressorTorch(_TorchRegressorBase):
    def _build(self, n_features):
        self.model = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )


class TabularGRURegressor(_TorchRegressorBase):
    """Treats the 6 chronologically-ordered price/load lag pairs as a short sequence,
    concatenates the GRU's final hidden state with the remaining static features."""
    LAG_ORDER = ["price_lag168", "price_lag48", "price_lag24", "price_lag3", "price_lag2", "price_lag1"]
    LOAD_LAG_ORDER = ["zload_lag168", "zload_lag48", "zload_lag24", "zload_lag3", "zload_lag2", "zload_lag1"]

    def __init__(self, feature_names, hidden=32, **kw):
        super().__init__(**kw)
        self.feature_names = feature_names
        self.hidden = hidden
        self.seq_idx = [feature_names.index(c) for c in self.LAG_ORDER]
        self.load_idx = [feature_names.index(c) for c in self.LOAD_LAG_ORDER]
        self.static_idx = [i for i in range(len(feature_names))
                            if i not in self.seq_idx and i not in self.load_idx]

    def _build(self, n_features):
        self.gru = nn.GRU(input_size=2, hidden_size=self.hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.hidden + len(self.static_idx), 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.model = nn.ModuleDict({"gru": self.gru, "head": self.head})

    def _forward(self, xb):
        seq = torch.stack([xb[:, self.seq_idx], xb[:, self.load_idx]], dim=-1)  # (B, 6, 2)
        static = xb[:, self.static_idx]
        _, h = self.model["gru"](seq)
        h = h.squeeze(0)
        return self.model["head"](torch.cat([h, static], dim=-1))


class TabularTCNRegressor(TabularGRURegressor):
    def _build(self, n_features):
        ch = 16
        self.tcn = nn.Sequential(
            nn.Conv1d(2, ch, kernel_size=3, padding=1, dilation=1), nn.ReLU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=2, dilation=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(ch + len(self.static_idx), 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.model = nn.ModuleDict({"tcn": self.tcn, "head": self.head})

    def _forward(self, xb):
        seq = torch.stack([xb[:, self.seq_idx], xb[:, self.load_idx]], dim=1)  # (B, 2, 6)
        static = xb[:, self.static_idx]
        h = self.model["tcn"](seq).squeeze(-1)
        return self.model["head"](torch.cat([h, static], dim=-1))


class SmallTransformerRegressor(TabularGRURegressor):
    """FT-Transformer-style compact encoder: tokenizes each of the 6 lag-pairs as a token,
    self-attention, mean-pool, concat static features."""
    def _build(self, n_features):
        d = 16
        self.embed = nn.Linear(2, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=2, dim_feedforward=32,
                                            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(nn.Linear(d + len(self.static_idx), 64), nn.ReLU(), nn.Linear(64, 1))
        self.model = nn.ModuleDict({"embed": self.embed, "encoder": self.encoder, "head": self.head})

    def _forward(self, xb):
        seq = torch.stack([xb[:, self.seq_idx], xb[:, self.load_idx]], dim=-1)
        static = xb[:, self.static_idx]
        tok = self.model["embed"](seq)
        enc = self.model["encoder"](tok).mean(dim=1)
        return self.model["head"](torch.cat([enc, static], dim=-1))


def make_deep_models(seed: int, feature_names: list) -> dict:
    return {
        "MLP": MLPRegressorTorch(seed=seed, epochs=80),
        "GRU": TabularGRURegressor(feature_names, seed=seed, epochs=80),
        "TCN": TabularTCNRegressor(feature_names, seed=seed, epochs=80),
        "Transformer": SmallTransformerRegressor(feature_names, seed=seed, epochs=80),
    }
