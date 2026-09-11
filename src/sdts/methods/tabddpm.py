"""TabDDPM (Kotelnikov et al., ICML 2023) via the vendored official code in
``third_party/tab_ddpm`` (MIT, commit b476257).

The diffusion model and denoiser are upstream verbatim. This module
replaces the upstream on-disk data pipeline with an in-memory one that
mirrors it exactly:

* numeric columns: sklearn QuantileTransformer to a normal output with
  ``n_quantiles = max(min(n // 30, 1000), 10)``, fitted on the training
  rows (upstream ``normalization = "quantile"``);
* categorical columns: ordinal codes from the schema's fixed level list,
  so the multinomial diffusion has ``K_j = len(levels_j)`` classes;
* the target conditions the denoiser (``is_y_cond = True``,
  ``num_classes = 2``) and is sampled from the empirical class
  distribution, as in upstream ``scripts/sample.py``;
* training is AdamW with linear lr annealing and an EMA copy kept for
  parity, sampling uses the raw (non-EMA) weights like upstream
  ``pipeline.py``;
* after sampling, numeric columns are inverse-transformed and integer
  valued columns with at most 32 distinct training values are snapped to
  the nearest training value (upstream ``round_columns``).

Default hyper-parameters are the upstream ``train()`` function defaults
plus the MLP from CONFIG_DESCRIPTION.md (lr 2e-3, weight decay 1e-4,
batch 1024, 1000 timesteps, cosine schedule, MLP [256, 256], dropout 0)
with one change: **10,000 training steps instead of 1,000**. Measured
2026-09-07 on adult n=5000: at 1,000 steps the samples are unusable
(TSTR AUROC 0.30, marginals collapsed to extremes); at 10,000 steps TSTR
AUROC is 0.89, equal to real data. The upstream released per-dataset
configs use 5k to 30k steps; 1,000 is their smoke value. Runs on CUDA when
available, otherwise CPU. MPS is supported via ``device: mps`` but is not
the default: measured non-deterministic for a fixed seed and not faster.
"""

from __future__ import annotations

import contextlib
import io
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer

from sdts.methods.base import Method, register, seed_everything

IMPL = "vendored yandex-research/tab-ddpm@b476257 (third_party/tab_ddpm)"


def _device(name: str = "auto") -> torch.device:
    """auto -> cuda if available else cpu. MPS only when asked for explicitly:
    it is not deterministic for a fixed seed (measured 2026-09-07)."""
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@register("tabddpm")
class TabDDPM(Method):
    @classmethod
    def default_hparams(cls) -> dict[str, Any]:
        return {
            "steps": 10000, "lr": 2e-3, "weight_decay": 1e-4, "batch_size": 1024,
            "num_timesteps": 1000, "scheduler": "cosine", "gaussian_loss_type": "mse",
            "d_layers": [256, 256], "dropout": 0.0, "dim_t": 128,
            "sample_batch_size": 10000, "device": "auto",
        }

    # ------------------------------------------------------------ preprocessing
    def _encode(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = self.schema
        x_num = df[s.numeric].to_numpy(dtype=np.float64) if s.numeric else np.zeros((len(df), 0))
        x_cat = np.stack([df[c].cat.codes.to_numpy() for c in s.categorical], axis=1) \
            if s.categorical else np.zeros((len(df), 0), dtype=np.int64)
        if (x_cat < 0).any():
            raise ValueError("categorical value outside schema levels")
        y = df[s.target].to_numpy(dtype=np.int64)
        return x_num, x_cat, y

    def fit(self, df: pd.DataFrame) -> None:
        from tab_ddpm import GaussianMultinomialDiffusion, MLPDiffusion

        seed_everything(self.seed)
        hp = self.hparams
        s = self.schema
        n = len(df)
        x_num, x_cat, y = self._encode(df)

        self._num_transform = None
        if x_num.shape[1]:
            self._num_transform = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=max(min(n // 30, 1000), 10),
                subsample=int(1e9), random_state=self.seed,
            ).fit(x_num)
            x_num_t = self._num_transform.transform(x_num)
        else:
            x_num_t = x_num
        self._x_num_train = x_num
        self._disc_cols = [
            j for j in range(x_num.shape[1])
            if len(u := np.unique(x_num[:, j])) <= 32 and np.all(u == np.round(u))
        ]

        self._K = np.array([len(s.levels[c]) for c in s.categorical], dtype=np.int64)
        K_for_model = self._K if len(self._K) else np.array([0])
        d_in = int(x_num.shape[1] + K_for_model.sum())
        self._num_numerical = int(x_num.shape[1])

        classes, counts = np.unique(y, return_counts=True)
        self._y_dist = torch.zeros(2, dtype=torch.float32)
        for c, k in zip(classes, counts):
            self._y_dist[int(c)] = float(k)

        dev = _device(str(hp["device"]))
        self.device = str(dev)
        model = MLPDiffusion(
            d_in=d_in, num_classes=2, is_y_cond=True,
            rtdl_params={"d_layers": list(hp["d_layers"]), "dropout": float(hp["dropout"])},
            dim_t=int(hp["dim_t"]),
        ).to(dev)
        self._diffusion = GaussianMultinomialDiffusion(
            num_classes=K_for_model, num_numerical_features=self._num_numerical,
            denoise_fn=model, gaussian_loss_type=hp["gaussian_loss_type"],
            num_timesteps=int(hp["num_timesteps"]), scheduler=hp["scheduler"], device=dev,
        ).to(dev)
        if dev.type == "mps":  # MPS has no float64; cast the upstream schedule constants
            for k, v in list(vars(self._diffusion).items()):
                if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
                    setattr(self._diffusion, k, v.float().to(dev))
        self._diffusion.train()

        X = torch.from_numpy(np.concatenate([x_num_t, x_cat.astype(np.float64)], axis=1)).float()
        Y = torch.from_numpy(y).long()
        steps, lr = int(hp["steps"]), float(hp["lr"])
        bs = min(int(hp["batch_size"]), n)
        opt = torch.optim.AdamW(self._diffusion.parameters(), lr=lr,
                                weight_decay=float(hp["weight_decay"]))
        ema = deepcopy(model)
        for p in ema.parameters():
            p.detach_()

        gen = torch.Generator().manual_seed(self.seed)
        perm, ptr = torch.randperm(n, generator=gen), 0
        for step in range(steps):
            if ptr + bs > n:
                perm, ptr = torch.randperm(n, generator=gen), 0
            idx = perm[ptr:ptr + bs]
            ptr += bs
            xb, yb = X[idx].to(dev), Y[idx].to(dev)
            opt.zero_grad()
            loss_multi, loss_gauss = self._diffusion.mixed_loss(xb, {"y": yb})
            loss = loss_multi + loss_gauss
            loss.backward()
            opt.step()
            for g in opt.param_groups:  # linear anneal, as upstream
                g["lr"] = lr * (1 - step / steps)
            with torch.no_grad():
                for t_, s_ in zip(ema.parameters(), model.parameters()):
                    t_.mul_(0.999).add_(s_.detach(), alpha=0.001)
            if not torch.isfinite(loss):
                raise RuntimeError(f"tabddpm loss diverged at step {step}: {loss.item()}")
        self._ema = ema
        self._diffusion.eval()
        self._fitted = True

    # ------------------------------------------------------------ sampling
    def sample(self, n: int) -> pd.DataFrame:
        seed_everything(self.seed)
        s = self.schema
        bs = min(int(self.hparams["sample_batch_size"]), n)
        with contextlib.redirect_stdout(io.StringIO()):  # upstream prints per timestep
            x_gen, y_gen = self._diffusion.sample_all(n, bs, self._y_dist, ddim=False)
        x_gen, y_gen = x_gen.numpy(), y_gen.numpy()
        out = pd.DataFrame(index=range(n))
        k = self._num_numerical
        if k:
            x_num = self._num_transform.inverse_transform(x_gen[:, :k])
            for j in self._disc_cols:
                u = np.unique(self._x_num_train[:, j])
                x_num[:, j] = u[np.abs(x_num[:, j][:, None] - u[None, :]).argmin(axis=1)]
            out[s.numeric] = x_num
        if len(self._K):
            codes = x_gen[:, k:].astype(np.int64)
            for j, c in enumerate(s.categorical):
                out[c] = np.asarray(s.levels[c], dtype=object)[np.clip(codes[:, j], 0, self._K[j] - 1)]
        out[s.target] = y_gen.astype(np.int64)
        return self.finalize(out)
