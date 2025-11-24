import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import copy
import os
import hashlib

import pandas as pd
import numpy as np
import math
import random
import time
from tqdm.notebook import tqdm

from typing import Mapping, Optional, Tuple, Literal, List, Union, Any, Dict
from dataclasses import dataclass, field, asdict

import logging

import optuna

# ─────────────────────────────────────────────────────────────────────────────
# Debugging options
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DebugConfig:
    """
    Lightweight switches for extra runtime checks and visibility during training.
    """
    enabled: bool = True
    detect_anomaly: bool = False         # torch.autograd anomaly detection
    check_nan_every_step: bool = True
    log_grad_norm_every: int = 10        # batches

def set_global_seed(seed: int):
    """
    Set RNG seeds for Python, NumPy, and PyTorch (CPU & CUDA) for reproducibility.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass

def derive_seed(master: int, tag: str) -> int:
    """
    Deterministically derive a new seed from a master seed and a string tag.
    """
    h = hashlib.sha256(f"{master}:{tag}".encode()).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


# For Data Normalization

@dataclass
class NormalizeConfig:
    """
    Configuration for feature/target normalization.

    Modes
    -----
    - "none": no normalization.
    - "global": normalize using statistics computed once on the *training slice*.
    - "per_sample": normalize each batch item using its encoder window statistics.

    Attributes
    ----------
    mode : {"none","global","per_sample"}
        Normalization strategy.
    norm_type : {"zscore","minmax"}
        Type of normalization to apply (currently only "zscore" is implemented).
    eps : float
        Small constant added to standard deviations to avoid division by zero.
    std_ddof : int
        Delta degrees of freedom for std (0 = population, 1 = sample).
    exog_skip_cols : tuple[str, ...]
        Exogenous columns to *exclude* from normalization (e.g., sin/cos).
    endog_skip_cols : tuple[str, ...]
        Endogenous columns to *exclude* from normalization.
    min_range : float
        Minimum of target range for min-max normalization (if used).
    max_range : float
        Maximum of target range for min-max normalization (if used).
    """
    mode: Literal["none", "global", "per_sample"] = "global"
    norm_type: Literal["zscore", "minmax"] = "zscore"
    eps: float = 1e-6
    std_ddof: int = 0  # 0 (population, sklearn-style) or 1 (sample)
    exog_skip_cols: Tuple[str, ...] = (
        "hour_sin","hour_cos",
        "dow_sin","dow_cos",
        "month_sin","month_cos",
        "wss_sin","wss_cos",
        "is_cold_season",
    )
    endog_skip_cols: Tuple[str, ...] = ()
    min_range: float = 0.0   
    max_range: float = 1.0

    def __post_init__(self):
        if self.mode not in {"none", "global", "per_sample"}:
            raise ValueError(f"Invalid norm mode: {self.mode}")
        if self.norm_type not in {"zscore", "minmax"}:
            raise ValueError(f"Invalid norm type: {self.norm_type}")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")
        if self.std_ddof not in {0, 1}:
            raise ValueError("std_ddof must be 0 or 1.")
        if self.norm_type == "minmax":
            if self.min_range >= self.max_range:
                raise ValueError("min_range must be less than max_range.")

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "norm_type": self.norm_type,
            "eps": self.eps,
            "std_ddof": self.std_ddof,
            "exog_skip_cols": self.exog_skip_cols,
            "endog_skip_cols": self.endog_skip_cols,
            "min_range": self.min_range,
            "max_range": self.max_range,
        }

@dataclass
class GlobalNormStats:
    """
    Precomputed global (training-slice) statistics for normalization.

    Attributes
    ----------
    endog_mean : np.ndarray
        Mean per endogenous column (shape: [C_endog], float32).
    endog_std : np.ndarray
        Std per endogenous column (shape: [C_endog], float32).
    exog_mean : np.ndarray
        Mean per exogenous column (shape: [D_exog], float32; empty if none).
    exog_std : np.ndarray
        Std per exogenous column (shape: [D_exog], float32; empty if none).
    y_mu : float
        Mean of the target series on the training slice.
    y_sigma : float
        Std of the target series on the training slice (>= eps).
    """
    endog_mean: np.ndarray
    endog_std:  np.ndarray
    exog_mean:  np.ndarray
    exog_std:   np.ndarray
    y_mu:       float
    y_sigma:    float

@dataclass
class GlobalMinMaxStats:
    """
    Precomputed global (training-slice) min/max statistics for min-max normalization.
    """
    endog_min: np.ndarray
    endog_max: np.ndarray
    exog_min: np.ndarray
    exog_max: np.ndarray
    y_min: float
    y_max: float

GlobalStats = Union[GlobalNormStats, GlobalMinMaxStats]

def compute_global_stats(
    norm_type: str,
    endog_tr: pd.DataFrame,
    exog_tr: pd.DataFrame,
    y_tr: pd.Series,
    *,
    ddof: int = 0,
) -> GlobalStats:
    """
    Compute training-slice statistics used by global normalization.

    Parameters
    ----------
    norm_type : {"zscore","minmax"}
        Type of normalization to compute stats for.
    endog_tr : pd.DataFrame
        Endogenous features on the *training* slice (shape: [T, C_endog]).
    exog_tr : pd.DataFrame
        Exogenous features on the *training* slice (shape: [T, D_exog]).
    y_tr : pd.Series
        Target series on the *training* slice (length T).
    ddof : int, default 0
        Delta degrees of freedom used in std computation.

    Returns
    -------
    GlobalStats
        Container with means/stds (or min/max) for endog/exog and mean/std (or min/max) for the target.
    """
    if norm_type == "zscore":
        em = endog_tr.mean(axis=0).to_numpy(dtype=np.float32)
        es = endog_tr.std(axis=0, ddof=ddof).to_numpy(dtype=np.float32)
        if exog_tr.shape[1] > 0:
            xm = exog_tr.mean(axis=0).to_numpy(dtype=np.float32)
            xs = exog_tr.std(axis=0, ddof=ddof).to_numpy(dtype=np.float32)
        else:
            xm = np.zeros((0,), np.float32)
            xs = np.ones((0,), np.float32)
        ym = float(y_tr.mean())
        ys = float(y_tr.std(ddof=ddof))
        # avoid zeros
        es[es == 0] = 1.0
        xs[xs == 0] = 1.0
        ys = ys if ys != 0.0 else 1.0
        return GlobalNormStats(em, es, xm, xs, ym, ys)
    elif norm_type == "minmax":
        emin = endog_tr.min(axis=0).to_numpy(dtype=np.float32)
        emax = endog_tr.max(axis=0).to_numpy(dtype=np.float32)
        if exog_tr.shape[1] > 0:
            xmin = exog_tr.min(axis=0).to_numpy(dtype=np.float32)
            xmax = exog_tr.max(axis=0).to_numpy(dtype=np.float32)
        else:
            xmin = np.zeros((0,), np.float32)
            xmax = np.ones((0,), np.float32)
        ymin = float(y_tr.min())
        ymax = float(y_tr.max())
        return GlobalMinMaxStats(emin, emax, xmin, xmax, ymin, ymax)

# Dataset

def is_for_endog_fut(col_name: str, output_len: int) -> bool:
    """
    Heuristic to decide if an endogenous lag feature is safe for the decoder
    (i.e., available in the future without leakage).
    """
    # check if it's a lag column with lag >= output_len
    if "_lag" in col_name:
        after = col_name.split("_lag")[-1]
        try:
            lag = int(after)
        except ValueError:
            lag = int(after.split("_")[0])
        return lag >= output_len
    return False
    
class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for encoder–decoder LSTM forecasting.

    Each item contains:
      - past endogenous/exogenous (encoder input)
      - future exogenous (+ subset of safe endogenous lags) (decoder input)
      - last 24h of target and the future target (for AR / teacher forcing)
      - optional normalization stats and anchors for Δ24 reconstruction

    Normalization
    -------------
    - mode="global": arrays are normalized once using provided `global_stats`.
    - mode="per_sample": each item recomputes stats from its encoder window.
    - mode="none": raw values are returned.
    """
    def __init__(
        self,
        y: pd.Series | pd.DataFrame,
        endog_df: pd.DataFrame,
        exog_df: pd.DataFrame,
        input_len: int,
        output_len: int,
        *,
        stride: int = 1,
        norm_cfg: NormalizeConfig = NormalizeConfig(),
        global_stats: Optional[GlobalStats] = None,
        y_levels: Optional[pd.Series | pd.DataFrame] = None,
        target_is_diff24: bool = False,
        norm_idx_endog: Optional[np.ndarray] = None,
        norm_idx_exog: Optional[np.ndarray] = None,
    ):
        """
        Build the dataset from aligned target/endog/exog frames.

        Parameters
        ----------
        y : pd.Series | pd.DataFrame
            Target series (single column). Will be normalized per `norm_cfg`.
            If predicting Δ24 (`target_is_diff24=True`), this is the differenced target.
        endog_df : pd.DataFrame
            Endogenous features used by encoder; a subset of safe lagged cols is also
            available to the decoder (see leakage rules).
        exog_df : pd.DataFrame
            Exogenous features (used by both encoder and decoder).
        input_len : int
            Encoder length (hours).
        output_len : int
            Decoder/forecast length (hours).
        stride : int, default=1
            Step between consecutive sliding windows.
        norm_cfg : NormalizeConfig, default=NormalizeConfig()
            Normalization policy and skip lists.
        global_stats : GlobalStats, optional
            Required when `norm_cfg.mode == "global"`.
        y_levels : pd.Series | pd.DataFrame, optional
            Level target for Δ24 reconstruction/evaluation. If None, `y` is used.
        target_is_diff24 : bool, default=False
            If True, the dataset expects Δ24 targets and returns anchors for undiff.
        norm_idx_endog : np.ndarray, optional
            Indices of endogenous columns to normalize (others are left as-is).
        norm_idx_exog : np.ndarray, optional
            Indices of exogenous columns to normalize (others are left as-is).
        """
        self.raise_on_bad_window = True

        # Validate y
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("y DataFrame must have exactly one column.")
            y = y.iloc[:, 0]
        if y_levels is not None and isinstance(y_levels, pd.DataFrame):
            if y_levels.shape[1] != 1:
                raise ValueError("y_levels DataFrame must have exactly one column.")
            y_levels = y_levels.iloc[:, 0]

        if not (len(y) == len(endog_df) == len(exog_df)):
            raise ValueError("y, endog_df, exog_df must have the same length.")
        if input_len <= 0 or output_len <= 0 or stride <= 0:
            raise ValueError("input_len, output_len, stride must be positive integers.")
        if input_len < 24:
            raise ValueError("input_len must be at least 24.")

        # Convert to numpy float32
        self.y = y.reset_index(drop=True).to_numpy(dtype=np.float32)                # target series (level or diff)
        self.endog = endog_df.reset_index(drop=True).to_numpy(dtype=np.float32)     # (T, C_endog)
        self.exog = exog_df.reset_index(drop=True).to_numpy(dtype=np.float32)       # (T, D_exog)
        self.index = endog_df.index.to_series().reset_index(drop=True) 

        # remember which endog columns are available in the future without leakage
        cols = list(endog_df.columns)
        self._fut_idx = np.array(
            [i for i, c in enumerate(cols) if is_for_endog_fut(c, output_len)],
            dtype=np.int64
        )

        # levels for reconstruction (if None, use target as levels)
        if y_levels is None:
            self.y_levels = y.reset_index(drop=True).to_numpy(dtype=np.float32)
        else:
            self.y_levels = y_levels.reset_index(drop=True).to_numpy(dtype=np.float32)

        self.target_is_diff24 = bool(target_is_diff24)

        self.T_in, self.T_out = int(input_len), int(output_len)
        total_len = self.T_in + self.T_out

        if len(self.y) < total_len:
            raise ValueError(f"Not enough rows ({len(self.y)}) for T_in+T_out={total_len}.")

        # Precompute valid window starts
        self.starts = []
        for s in range(0, len(self.y) - total_len + 1, stride):
            e, k = s + self.T_in, s + total_len
            enc_ok = np.isfinite(self.endog[s:e]).all() and np.isfinite(self.exog[s:e]).all() and np.isfinite(self.y[s:e]).all()
            dec_ok = np.isfinite(self.exog[e:k]).all() and np.isfinite(self.y[e:k]).all()
            if self._fut_idx.size:
                dec_ok = dec_ok and np.isfinite(self.endog[e:k][:, self._fut_idx]).all()
            # need 24h of level history if predicting diffs
            hist_ok = (not self.target_is_diff24) or (e - 24 >= 0)
            if enc_ok and dec_ok and hist_ok:
                self.starts.append(s)
            else:
                if self.raise_on_bad_window:
                    raise ValueError(f"NaN/Inf within window starting at {s} "
                                    f"(enc_ok={enc_ok}, dec_ok={dec_ok}, hist_ok={hist_ok}).")

        # Cached dims
        self.C_endog = self.endog.shape[1]
        self.D_exog = self.exog.shape[1] if self.exog.ndim == 2 else 0

        self.norm_cfg = norm_cfg
        self.global_stats = global_stats

        # Normalization indices: column indices of variables to be normalized
        self.norm_idx_endog = (
            np.asarray(norm_idx_endog, dtype=np.int64)
            if norm_idx_endog is not None else
            np.arange(self.endog.shape[1], dtype=np.int64)
        )
        self.norm_idx_exog = (
            np.asarray(norm_idx_exog, dtype=np.int64)
            if (norm_idx_exog is not None and self.D_exog > 0) else
            np.arange(self.exog.shape[1], dtype=np.int64) if self.D_exog > 0 else
            np.array([], dtype=np.int64)
        )
        self._fut_norm_mask = np.isin(self._fut_idx, self.norm_idx_endog)

        # Optional GLOBAL normalization (do it once for speed)
        if self.norm_cfg.mode == "global":
            if self.global_stats is None:
                raise ValueError("global_stats must be provided when mode='global'.")
            gs, eps = self.global_stats, self.norm_cfg.eps
            if norm_cfg.norm_type == "zscore":
                # endog
                if self.norm_idx_endog.size:
                    idx = self.norm_idx_endog
                    self.endog[:, idx] = (self.endog[:, idx] - gs.endog_mean[idx][None, :]) / (gs.endog_std[idx][None, :] + eps)
                # exog
                if self.D_exog > 0 and self.norm_idx_exog.size:
                    idx = self.norm_idx_exog
                    self.exog[:, idx] = (self.exog[:, idx] - gs.exog_mean[idx][None, :]) / (gs.exog_std[idx][None, :] + eps)
                # y target: (T,)
                self.y = (self.y - gs.y_mu) / (gs.y_sigma + eps)
            elif norm_cfg.norm_type == "minmax":
                # endog
                if self.norm_idx_endog.size:
                    idx = self.norm_idx_endog
                    self.endog[:, idx] = (self.endog[:, idx] - gs.endog_min[idx][None, :]) / (gs.endog_max[idx][None, :] - gs.endog_min[idx][None, :] + eps)
                    self.endog[:, idx] = self.endog[:, idx] * (norm_cfg.max_range - norm_cfg.min_range) + norm_cfg.min_range
                # exog
                if self.D_exog > 0 and self.norm_idx_exog.size:
                    idx = self.norm_idx_exog
                    self.exog[:, idx] = (self.exog[:, idx] - gs.exog_min[idx][None, :]) / (gs.exog_max[idx][None, :] - gs.exog_min[idx][None, :] + eps)
                    self.exog[:, idx] = self.exog[:, idx] * (norm_cfg.max_range - norm_cfg.min_range) + norm_cfg.min_range
                # y target: (T,)
                self.y = (self.y - gs.y_min) / (gs.y_max - gs.y_min + eps)
                self.y = self.y * (norm_cfg.max_range - norm_cfg.min_range) + norm_cfg.min_range
            else:
                raise ValueError(f"Unsupported norm_type: {norm_cfg.norm_type}")
    
    def __repr__(self) -> str:
        """Human-friendly summary with sizes, normalization mode, and Δ24 flag."""
        return (f"TimeSeriesDataset(n={len(self)}, T_in={self.T_in}, T_out={self.T_out}, "
                f"C_endog={self.C_endog}, D_exog={self.D_exog}, norm={self.norm_cfg.mode}, "
                f"diff24={self.target_is_diff24})")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        """
        Return one training/eval example.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys and shapes:
              - endog_past : (T_in, C_endog)
              - endog_fut  : (T_out, C_endog_fut)    # only safe lags
              - exog_past  : (T_in, D_exog)          # empty (T_in, 0) if none
              - exog_fut   : (T_out, D_exog)         # empty (T_out, 0) if none
              - y_last_24h : (24, 1)
              - y_fut      : (T_out, 1)
              - y_mu       : () or (1,)  [optional; per-sample with norm type "zscore"]
              - y_sigma    : () or (1,)  [optional; per-sample with norm type "zscore"]
              - y_min      : () or (1,)  [optional; per-sample with norm type "minmax"]
              - y_max      : () or (1,)  [optional; per-sample with norm type "minmax"]
              - y_anchor24 : (24, 1)     [Δ24 only; levels, unnormalized]
              - y_fut_level: (T_out, 1)  [Δ24 only; levels, unnormalized]
        """
        s = self.starts[idx]
        e, k = s + self.T_in, s + self.T_in + self.T_out

        endog_past = self.endog[s:e]  # (T_in, C_endog)
        endog_fut = self.endog[e:k]
        endog_fut = endog_fut[:, self._fut_idx] if self._fut_idx.size else np.zeros((self.T_out, 0), np.float32) # (T_out, C_endog_fut)
        exog_past  = self.exog[s:e] if self.D_exog > 0 else np.zeros((self.T_in, 0), np.float32)
        exog_fut   = self.exog[e:k] if self.D_exog > 0 else np.zeros((self.T_out, 0), np.float32)
        y_hist     = self.y[s:e].copy()
        y_last_24h = self.y[e-24:e].reshape(24, 1)         # last 24 TARGET values (level or diff)
        y_fut       = self.y[e:k].reshape(self.T_out, 1)    # TARGET sequence (level or diff)

        # only copy if we’re going to mutate (per-sample mode)
        if self.norm_cfg.mode == "per_sample":
            endog_past = endog_past.copy()
            endog_fut  = endog_fut.copy()
            if self.D_exog > 0:
                exog_past = exog_past.copy()
                exog_fut  = exog_fut.copy()

        # PER-SAMPLE normalization (compute from encoder window only)
        ym, ys = None, None
        if self.norm_cfg.mode == "per_sample":
            eps = self.norm_cfg.eps
            # endog
            if self.norm_idx_endog.size:
                if self.norm_cfg.norm_type == "zscore":
                    em = endog_past.mean(axis=0, keepdims=True)
                    es = endog_past.std(axis=0, keepdims=True); es[es==0] = 1.0
                    idx = self.norm_idx_endog
                    endog_past[:, idx] = (endog_past[:, idx] - em[:, idx]) / (es[:, idx] + eps)
                    em_f = em[:, self._fut_idx]         # (1, C_endog_fut)
                    es_f = es[:, self._fut_idx]
                    m = self._fut_norm_mask             # (C_endog_fut,)
                    if m.any():
                        endog_fut[:, m] = (endog_fut[:, m] - em_f[:, m]) / (es_f[:, m] + eps)
                if self.norm_cfg.norm_type == "minmax":
                    em = endog_past.min(axis=0, keepdims=True)
                    eM = endog_past.max(axis=0, keepdims=True)
                    idx = self.norm_idx_endog
                    endog_past[:, idx] = (endog_past[:, idx] - em[:, idx]) / (eM[:, idx] - em[:, idx] + eps)
                    endog_past[:, idx] = endog_past[:, idx] * (self.norm_cfg.max_range - self.norm_cfg.min_range) + self.norm_cfg.min_range
                    em_f = em[:, self._fut_idx]
                    eM_f = eM[:, self._fut_idx]
                    m = self._fut_norm_mask
                    if m.any():
                        endog_fut[:, m] = (endog_fut[:, m] - em_f[:, m]) / (eM_f[:, m] - em_f[:, m] + eps)
                        endog_fut[:, m] = endog_fut[:, m] * (self.norm_cfg.max_range - self.norm_cfg.min_range) + self.norm_cfg.min_range
            # exog
            if self.D_exog > 0 and self.norm_idx_exog.size:
                if self.norm_cfg.norm_type == "zscore":
                    xm = exog_past.mean(axis=0, keepdims=True)
                    xs = exog_past.std(axis=0, keepdims=True); xs[xs==0] = 1.0
                    idx = self.norm_idx_exog
                    exog_past[:, idx] = (exog_past[:, idx] - xm[:, idx]) / (xs[:, idx] + eps)
                    exog_fut[:, idx]  = (exog_fut[:, idx]  - xm[:, idx]) / (xs[:, idx] + eps)
                elif self.norm_cfg.norm_type == "minmax":
                    xm = exog_past.min(axis=0, keepdims=True)
                    xM = exog_past.max(axis=0, keepdims=True)
                    idx = self.norm_idx_exog
                    exog_past[:, idx] = (exog_past[:, idx] - xm[:, idx]) / (xM[:, idx] - xm[:, idx] + eps)
                    exog_past[:, idx] = exog_past[:, idx] * (self.norm_cfg.max_range - self.norm_cfg.min_range) + self.norm_cfg.min_range
                    exog_fut[:, idx]  = (exog_fut[:, idx]  - xm[:, idx]) / (xM[:, idx] - xm[:, idx] + eps)
                    exog_fut[:, idx]  = exog_fut[:, idx] * (self.norm_cfg.max_range - self.norm_cfg.min_range) + self.norm_cfg.min_range

            # y target stats from past target window
            if self.norm_cfg.norm_type == "zscore":
                ym = y_hist.mean()
                ys = y_hist.std();  ys = ys if ys != 0 else 1.0
                y_last_24h = (y_last_24h - ym) / (ys + eps)
                y_fut      = (y_fut       - ym) / (ys + eps)
            elif self.norm_cfg.norm_type == "minmax":
                ymin = y_hist.min()
                ymax = y_hist.max()
                y_last_24h = (y_last_24h - ymin) / (ymax - ymin + eps)
                y_last_24h = y_last_24h * (self.norm_cfg.max_range - self.norm_cfg.min_range) + self.norm_cfg.min_range
                y_fut      = (y_fut       - ymin) / (ymax - ymin + eps)
                y_fut      = y_fut * (self.norm_cfg.max_range - self.norm_cfg.min_range) + self.norm_cfg.min_range

        out = {
            "endog_past":  torch.from_numpy(endog_past),
            "endog_fut":   torch.from_numpy(endog_fut),
            "exog_past":   torch.from_numpy(exog_past),
            "exog_fut":    torch.from_numpy(exog_fut),
            "y_last_24h":  torch.from_numpy(y_last_24h),  # last 24h TARGET values
            "y_fut":       torch.from_numpy(y_fut),       # future TARGET values
        }
        if self.norm_cfg.mode == "per_sample":
            if self.norm_cfg.norm_type == "zscore":
                out["y_mu"] = torch.tensor(ym, dtype=torch.float32)
                out["y_sigma"] = torch.tensor(ys, dtype=torch.float32)
            elif self.norm_cfg.norm_type == "minmax":
                out["y_min"] = torch.tensor(ymin, dtype=torch.float32)
                out["y_max"] = torch.tensor(ymax, dtype=torch.float32)

        # When predicting 24h diffs, also return anchors/levels for reconstruction & original-scale eval
        if self.target_is_diff24:
            # last 24 levels from encoder window
            y_anchor24 = self.y_levels[e-24:e].reshape(24, 1)
            y_fut_level = self.y_levels[e:k].reshape(self.T_out, 1)
            out["y_anchor24"]  = torch.from_numpy(y_anchor24)    # last 24 levels from encoder window (always unnormalized)
            out["y_fut_level"] = torch.from_numpy(y_fut_level)   # future levels (always unnormalized)

        return out

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class EncDecLSTM(nn.Module):
    """
    Encoder-Decoder LSTM with optional autoregressive (AR) inputs and two head options.
    """
    def __init__(
            self, 
            *,
            input_size_enc: int, 
            input_size_dec: int, 
            hidden_size: int, 
            n_layers: int = 1, 
            head: Literal["linear", "mlp"] = "linear", 
            dropout: float = 0.0,
            use_ar: Literal["prev", "none", "24h"] = "prev"
        ):
        """
        Encoder-Decoder LSTM for time series forecasting.
        """
        super().__init__()
        inter_drop = dropout if n_layers > 1 else 0.0  # LSTM between-layer dropout
        self.encoder = nn.LSTM(input_size_enc, hidden_size, n_layers, batch_first=True, dropout=inter_drop)
        self.decoder = nn.LSTM(input_size_dec, hidden_size, n_layers, batch_first=True, dropout=inter_drop)
        self.input_size_enc = input_size_enc
        self.input_size_dec = input_size_dec
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        if use_ar not in ("prev", "none", "24h"):
            raise ValueError(f"Unsupported use_ar '{use_ar}'. Supported values are: 'prev', 'none', '24h'.")
        self.use_ar = use_ar
        self.use_ar_prev = bool(use_ar=="prev")
        self.use_ar_24h  = bool(use_ar=="24h")

        if head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_size, 1)
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_size, 1)
            )

    def assert_batch_shapes(self, batch):
        """
        Validate tensor dimensions and assembled input sizes for encoder/decoder.
        """
        B = batch["y_fut"].shape[0]
        T_in = batch["endog_past"].shape[1]
        T_out = batch["y_fut"].shape[1]
        assert batch["exog_past"].shape[0] == B and batch["endog_fut"].shape[0] == B and batch["exog_fut"].shape[0] == B
        assert batch["exog_past"].shape[1] == T_in
        assert batch["endog_fut"].shape[1] == T_out and batch["exog_fut"].shape[1] == T_out
        if self.use_ar_24h:
            assert T_out > 24, f"T_out must be > 24 when use_ar='24h', got T_out={T_out}"
        assert batch["y_last_24h"].shape == (B, 24, 1), f"y_last_24h bad shape: {batch['y_last_24h'].shape}"
        assert batch["y_fut"].shape == (B, T_out, 1), f"y_fut bad shape: {batch['y_fut'].shape}"
        assert self.input_size_enc == batch["endog_past"].shape[2] + batch["exog_past"].shape[2], f"input_size_enc mismatch: {self.input_size_enc} != {batch['endog_past'].shape[2] + batch['exog_past'].shape[2]}"
        expected_dec = batch["endog_fut"].shape[2] + batch["exog_fut"].shape[2] + (1 if self.use_ar_prev or self.use_ar_24h else 0)
        assert self.input_size_dec == expected_dec, f"input_size_dec mismatch: {self.input_size_dec} != {expected_dec}"

    def forward(
            self, 
            batch: dict, 
            teacher_forcing: float = 0.5,
            tf_generator: Optional[torch.Generator] = None
        ) -> torch.Tensor:
        """
        Run an encoder-decoder pass and return stepwise predictions.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            A sample from `TimeSeriesDataset.__getitem__` (already on device).
        teacher_forcing : float, default 0.5
            Probability of using ground-truth prev target(s) during decoding.
        tf_generator : torch.Generator, optional
            RNG for scheduled sampling to make behavior reproducible.
        """
        self.assert_batch_shapes(batch)
        endog_past = batch["endog_past"]    # (B, T_in, C_endog)
        endog_fut  = batch["endog_fut"]     # (B, T_out, C_endog_fut) with C_endog_fut < C_endog
        exog_past  = batch["exog_past"]     # (B, T_in, D_exog)
        exog_fut   = batch["exog_fut"]      # (B, T_out, D_exog)
        y_last_24h = batch["y_last_24h"]    # (B, 24, 1)
        y_fut      = batch["y_fut"]         # (B, T_out, 1)

        _, T_out, _ = exog_fut.shape

        # Encoder input
        enc_input = torch.cat([endog_past, exog_past], dim=-1)  # (B, T_in, input_size_enc)

        # Encoder output
        _, (h, c) = self.encoder(enc_input)

        # Decoder 
        # ===== Non-autoregressive case (no previous-pred input at all) =====
        if self.use_ar=="none":
            # Ignore teacher forcing and y_last_24h entirely
            dec_input = torch.cat([endog_fut, exog_fut], dim=-1)     # (B, T_out, C_endog_fut + D_exog)
            out, _ = self.decoder(dec_input, (h, c))                 # (B, T_out, H)
            return self.head(out)                                    # (B, T_out, 1)
        
        # ===== 24h autoregressive case (use last 24h preds of target as input) =====
        if self.use_ar=="24h":

            # ---- Fast path: full teacher forcing -> single decoder call ----
            if teacher_forcing == 1.0:
                # previous target sequence = [y_{-24..-1}] + y_{0..T_out-1-24}
                y_prev_seq = torch.cat([y_last_24h, y_fut[:, :-24, :]], dim=1)   # (B,T_out,1)
                dec_input = torch.cat([y_prev_seq, endog_fut, exog_fut], dim=-1) # (B,T_out,1+D_exog+C_endog_fut)
                out, _ = self.decoder(dec_input, (h, c))                         # (B,T_out,H)
                return self.head(out)                                            # (B,T_out,1)

            # ---- Fallback: scheduled sampling / no TF -> step loop ----
            y_prev = y_last_24h
            outputs = []

            steps = math.ceil(T_out / 24)
            res = T_out % 24
            chunk_size = 24
            for s in range(steps):
                t = s * 24
                chunk_size = 24 if (s < steps - 1 or res == 0) else res
                dec_input = torch.cat([y_prev[:, :chunk_size, :], endog_fut[:, t:t+chunk_size, :], exog_fut[:, t:t+chunk_size, :]], dim=-1)  # (B, chunk_size, input_size_dec)
                out, (h, c) = self.decoder(dec_input, (h, c))
                y_hat = self.head(out)  # (B, chunk_size, 1)
                outputs.append(y_hat)

                # per-step GLOBAL coin (one decision for the whole batch at step t)
                if teacher_forcing in (0.0, 1.0):
                    use_tf = teacher_forcing == 1.0
                else:
                    use_tf = (torch.rand((), device=y_hat.device, generator=tf_generator) < teacher_forcing)

                y_prev = y_fut[:, t:t+24, :] if use_tf else y_hat
            
            return torch.cat(outputs, dim=1)  # (B, T_out, 1)

        # ===== fully autoregressive case (use last pred of target as input) =====
        if self.use_ar=="prev":
            y_last_seen = y_last_24h[:, -1:, :]  # (B, 1, 1) last seen target value

            # ---- Fast path: full teacher forcing -> single decoder call ----
            if teacher_forcing == 1.0:
                # previous target sequence = [y_{-1}] + y_{0..T_out-1}
                y_prev_seq = torch.cat([y_last_seen, y_fut[:, :-1, :]], dim=1)   # (B,T_out,1)
                dec_input = torch.cat([y_prev_seq, endog_fut, exog_fut], dim=-1) # (B,T_out,1+D_exog+C_endog_fut)
                out, _ = self.decoder(dec_input, (h, c))                         # (B,T_out,H)
                return self.head(out)                                            # (B,T_out,1)

            # ---- Fallback: scheduled sampling / no TF -> step loop ----
            y_prev = y_last_seen
            outputs = []

            for t in range(T_out):
                dec_input = torch.cat([y_prev, endog_fut[:, t:t+1, :], exog_fut[:, t:t+1, :]], dim=-1)  # (B, 1, input_size_dec)
                out, (h, c) = self.decoder(dec_input, (h, c))
                y_hat = self.head(out)  # (B, 1, 1)
                outputs.append(y_hat)

                # per-step GLOBAL coin (one decision for the whole batch at step t)
                if teacher_forcing in (0.0, 1.0):
                    use_tf = teacher_forcing == 1.0
                else:
                    use_tf = (torch.rand((), device=y_hat.device, generator=tf_generator) < teacher_forcing)

                y_prev = y_fut[:, t:t+1, :] if use_tf else y_hat

            return torch.cat(outputs, dim=1)  # (B, T_out, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TFScheduler:
    """
    Teacher-forcing (TF) scheduler that maps epoch -> TF ratio in [0, 1].

    Behavior
    --------
    Counting epochs starting from `start_epoch` (zero-based), the schedule has three phases:
    - Warmup [start_epoch .. warmup_end]: fixed `initial_tf`.
    - Transition (warmup_end .. zero_start): linearly (or stepwise) moves toward `late_tf`.
    - Late [zero_start .. end): uses `late_tf` with optional periodic spikes to `spike_level`.

    Attributes
    ----------
    drop_epochs : int
        Horizon (in epochs) that defines where warmup/zero boundaries land.
    mode : {"linear","step"}, default="linear"
        Interpolation between `initial_tf` and `late_tf`.
    warmup_frac : float, default=0.20
        Fraction of `drop_epochs` for the warmup phase.
    zero_frac : float, default=0.90
        Fraction of `drop_epochs` after which we consider TF to have "reached late".
    initial_tf : float, default=1.0
        TF ratio used during warmup.
    late_tf : float, default=0.0
        TF ratio used in the late phase.
    spike_level : float, default=0.30
        Temporary TF value injected every `spike_period` epochs during late phase.
    spike_period : int, default=3
        Spike frequency in epochs (0 disables spikes).
    start_epoch : int, default=0
        Zero-based epoch offset (useful when training resumes mid-run).
    logger : logging.Logger | None
        Logger for info messages.
    """
    drop_epochs: int               # horizon (in epochs) used to compute warmup/zero boundaries
    mode: Literal["linear", "step"] = "linear"
    warmup_frac: float = 0.20
    zero_frac: float = 0.90
    initial_tf: float = 1.0
    late_tf: float = 0.0
    spike_level: float = 0.30      # TF during late spikes
    spike_period: int = 3          # every N epochs after zero_start
    start_epoch: int = 0           # assume trainer epochs are 0-based
    logger: Optional[logging.Logger] = None

    def __post_init__(self):
        if self.drop_epochs <= 0:
            raise ValueError("drop_epochs must be > 0")
        if not (0.0 <= self.warmup_frac <= self.zero_frac <= 1.0):
            raise ValueError("Require 0 <= warmup_frac <= zero_frac <= 1")
        if not (0.0 <= self.initial_tf <= 1.0 and 0.0 <= self.late_tf <= 1.0 and 0.0 <= self.spike_level <= 1.0):
            raise ValueError("TF ratios must be in [0,1]")
        if self.mode not in ("linear", "step"):
            raise ValueError(f"Unsupported mode '{self.mode}'. Supported values are: 'linear', 'step'.")
        # precompute epoch boundaries
        H = self.drop_epochs
        w = math.floor(self.warmup_frac * H)
        z = math.floor(self.zero_frac  * H)
        # Clamp to valid epoch indices [0, H-1] and enforce order
        w = max(0, min(H - 1, w))
        z = max(w, min(H - 1, z))
        self._warmup_ep     = w
        self._zero_start_ep = z

        self.logger = self.logger or logging.getLogger(__name__)
        self.logger.info(f"[tf] mode={self.mode}, warmup_end_ep={self._warmup_ep+1}, late_tf_start_ep={self._zero_start_ep+1} (both inclusive and 1-based), initial_tf={self.initial_tf}, late_tf={self.late_tf}, spike_level={self.spike_level}, spike_period={self.spike_period}")

    def get_tf_ratio(self, current_epoch) -> float:
        """
        Return the TF ratio for a given (zero-based) epoch.
        """
        e = current_epoch - self.start_epoch # NB: zero-based epoch

        if self.mode == "linear":
            if e < self._warmup_ep:
                r = self.initial_tf
            elif e >= self._zero_start_ep:
                # optional late spikes anchored to zero_start
                if self.spike_period and ((e - self._zero_start_ep) % self.spike_period == 0 and e != self._zero_start_ep):
                    r = self.spike_level
                else:
                    r = self.late_tf
            else:
                # linear interp from initial_tf -> late_tf between warmup and zero_start
                denom = max(1, self._zero_start_ep - self._warmup_ep)
                t = (e - self._warmup_ep) / denom
                r = self.initial_tf * (1.0 - t) + self.late_tf * t

        elif self.mode == "step":
            if e < self._warmup_ep:
                r = self.initial_tf
            elif e < self._zero_start_ep:
                r = (self.initial_tf + self.late_tf) / 2
            else:
                # optional late spikes anchored to zero_start
                if self.spike_period and ((e - self._zero_start_ep) % self.spike_period == 0):
                    r = self.spike_level
                else:
                    r = self.late_tf

        # clamp for safety
        return float(min(1.0, max(0.0, r)))

    def is_zero_ep(self, current_epoch) -> bool:
        """Whether `current_epoch` is at or beyond the zero-start boundary."""
        e = current_epoch - self.start_epoch
        return e >= self._zero_start_ep

class LSTMTrainer:
    """
    High-level training loop for `EncDecLSTM` with normalization-aware loss,
    teacher forcing scheduling, gradient clipping, LR drops, early stopping,
    and Optuna pruning/reporting.
    """

    def __init__(
        self,
        model: EncDecLSTM,
        device: torch.device,
        norm_cfg: Optional[NormalizeConfig] = None,
        global_stats: Optional[GlobalNormStats] = None,
        logger: Optional[logging.Logger] = None,
        loss_kind: str = "mae",
        target_is_diff24: bool = False, # are we predicting 24h diffs?
        tf_seed: Optional[int] = None, # seed for teacher forcing
        debug: Optional[DebugConfig] = None,
    ):
        """
        Parameters
        ----------
        model : EncDecLSTM
            The model to train.
        device : torch.device
            CPU or CUDA device.
        norm_cfg : NormalizeConfig | None, default=None
            Normalization config used to denormalize predictions for loss.
        global_stats : GlobalStats | None, default=None
            Required if `norm_cfg.mode == "global"`.
        logger : logging.Logger | None, default=None
            Logger for progress and diagnostics.
        loss_kind : {"mae","mse"}, default="mae"
            Loss computed on original target scale.
        target_is_diff24 : bool, default=False
            If True, apply Δ24 reconstruction before computing loss.
        tf_seed : int | None, default=None
            Seed controlling TF stochasticity; derived from pipeline seed if provided.
        debug : DebugConfig | None, default=None
            Extra checks (NaN/Inf, grad-norm logs, anomaly detection).
        """
        self.model = model.to(device)
        self.device = device
        self._logger = logger or logging.getLogger(__name__)
        self.seed = tf_seed

        self.debug = debug or DebugConfig(enabled=False)
        self.history = []  

        self.norm_cfg  = norm_cfg
        self.norm_mode = getattr(norm_cfg, "mode", "none") if norm_cfg is not None else "none"
        self.eps       = getattr(norm_cfg, "eps", 1e-8) if norm_cfg is not None else 1e-8
        self.global_stats = global_stats
        self.target_is_diff24 = bool(target_is_diff24)

        if self.norm_mode == "global" and self.global_stats is None:
            raise ValueError("norm_mode='global' requires global_stats (train-slice stats).")

        self.loss_kind = loss_kind.lower()
        if self.loss_kind not in ("mae", "mse"):
            raise ValueError(f"Unsupported loss_kind {self.loss_kind}. Supported values are: 'mae', 'mse'.")

        # built-in losses 
        self._l1   = nn.L1Loss()
        self._mse  = nn.MSELoss()

        # Used during training
        self.train_losses_orig = []  # on original scale
        self.val_losses_orig = []

        self.best_val_loss_orig = float("inf")
        self.best_epoch_zb = None

    # ---------- helpers ----------
    def _get_mu_sigma(self, batch, *, device, dtype) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Retrieve (mu, sigma) broadcastable to (B, T_out, 1) for denormalization.
        """
        if self.norm_cfg.norm_type != "zscore":
            return None, None
        if self.norm_mode == "none":
            return None, None
        if self.norm_mode == "global":
            mu = torch.tensor(self.global_stats.y_mu, device=device, dtype=dtype).view(1,1,1)
            sg = torch.tensor(self.global_stats.y_sigma, device=device, dtype=dtype).view(1,1,1) + self.eps
            return mu, sg

        mu = batch.get("y_mu", None)
        sg = batch.get("y_sigma", None)
        if (mu is None) or (sg is None):
            raise ValueError("Missing normalization statistics.")

        mu = mu.to(device=device, dtype=dtype)
        sg = sg.to(device=device, dtype=dtype) + self.eps

        # Ensure (B,1,1) for broadcasting with (B,T_out,1)
        if mu.dim() == 1:  # (B,)
            mu = mu.view(-1, 1, 1)
        elif mu.dim() == 2:  # (B,1)
            mu = mu.unsqueeze(-1)
        if sg.dim() == 1:
            sg = sg.view(-1, 1, 1)
        elif sg.dim() == 2:
            sg = sg.unsqueeze(-1)

        return mu, sg
    
    def _get_min_max(self, batch, *, device, dtype) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Retrieve (y_min, y_max) broadcastable to (B, T_out, 1) for denormalization.
        """
        if self.norm_cfg.norm_type != "minmax":
            return None, None
        if self.norm_mode == "none":
            return None, None
        if self.norm_mode == "global":
            y_min = torch.tensor(self.global_stats.y_min, device=device, dtype=dtype).view(1,1,1)
            y_max = torch.tensor(self.global_stats.y_max, device=device, dtype=dtype).view(1,1,1)
            return y_min, y_max

        y_min = batch.get("y_min", None)
        y_max = batch.get("y_max", None)
        if (y_min is None) or (y_max is None):
            raise ValueError("Missing normalization statistics.")

        y_min = y_min.to(device=device, dtype=dtype)
        y_max = y_max.to(device=device, dtype=dtype)

        # Ensure (B,1,1) for broadcasting with (B,T_out,1)
        if y_min.dim() == 1:  # (B,)
            y_min = y_min.view(-1, 1, 1)
        elif y_min.dim() == 2:  # (B,1)
            y_min = y_min.unsqueeze(-1)
        if y_max.dim() == 1:
            y_max = y_max.view(-1, 1, 1)
        elif y_max.dim() == 2:
            y_max = y_max.unsqueeze(-1)

        return y_min, y_max

    def _denorm_pair(self, y_pred, y_true, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert (y_pred, y_true), of shapes (B, T_out, 1), from normalized to original scale.
        """
        if self.norm_mode == "none":
            return y_pred, y_true
        if self.norm_cfg.norm_type == "minmax":
            y_min, y_max = self._get_min_max(batch, device=y_pred.device, dtype=y_pred.dtype)
            if (y_min is None) or (y_max is None):
                raise ValueError("._get_min_max returned None but .norm_mode is not 'none'.")
            return y_pred * (y_max - y_min) + y_min, y_true * (y_max - y_min) + y_min
        if self.norm_cfg.norm_type == "zscore":
            mu, sigma = self._get_mu_sigma(batch, device=y_pred.device, dtype=y_pred.dtype)
            if (mu is None) or (sigma is None):
                raise ValueError("._get_mu_sigma returned None but .norm_mode is not 'none'.")
            return y_pred * sigma + mu, y_true * sigma + mu

    def _loss(self, y_pred, y_true) -> torch.Tensor:
        """
        Compute elementwise loss (MAE or MSE) on original scale.
        """
        if self.loss_kind == "mae":
            return self._l1(y_pred, y_true)  # |ŷ - y|
        elif self.loss_kind == "mse":
            return self._mse(y_pred, y_true) # (ŷ - y)^2
        else:
            raise ValueError("loss_kind must be one of: 'mae', 'mse'.")

    def _pair_to_orig_levels(self, y_pred, y_true, batch: dict) -> torch.Tensor:
        """
        Denormalize and optionally (that is, only if target_is_diff24) undiff y_pred and y_true.
        """
        if (self.target_is_diff24 and batch.get("y_fut_level", None) is None):
            raise ValueError("y_fut_level must be provided when use_differences is True.")
        
        # Denorm
        y_pred, y_true = self._denorm_pair(y_pred, y_true, batch)
        
        # Optional undiff
        if self.target_is_diff24:
            y_pred = self._undiff24(y_pred, batch["y_anchor24"])
            y_true = batch["y_fut_level"]

        return y_pred, y_true

    def _build_param_groups(self, weight_decay: float, log_names: bool = True):
        """
        Build Adam/AdamW parameter groups with a sane weight-decay policy.

        Parameters
        ----------
        weight_decay : float
            Weight decay value to apply to the "decay" group (e.g., 1e-2).
            The "no-decay" group always uses 0.0.
        log_names : bool, default True
            If True, logs per-group parameter names and the chosen weight_decay.
            Useful for auditing, but can be verbose in large models.

        Returns
        -------
        list[dict]
            A list of two optimizer param group dicts:
            [
            {"params": <iterable of decay tensors>,    "weight_decay": float(weight_decay)},
            {"params": <iterable of no-decay tensors>, "weight_decay": 0.0},
            ]
        """
        no_decay_mods = (torch.nn.LayerNorm,
                        torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                        torch.nn.Embedding)
        decay_mods = (torch.nn.Linear,
                    torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                    torch.nn.LSTM)   # include LSTM: its 2D weights should decay, biases should not

        decay, no_decay = [], []
        if log_names: decay_names, no_decay_names = [], []

        # Walk modules and their direct parameters (no recursion on params)
        for mod_name, module in self.model.named_modules():
            for p_name, p in module.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if p_name == "bias" or p.ndim < 2:
                    no_decay.append(p)
                    if log_names: no_decay_names.append(f"{mod_name}.{p_name}" if mod_name else p_name)
                    continue
                if isinstance(module, no_decay_mods):
                    no_decay.append(p)
                    if log_names: no_decay_names.append(f"{mod_name}.{p_name}" if mod_name else p_name)
                    continue
                if isinstance(module, decay_mods):
                    decay.append(p)
                    if log_names: decay_names.append(f"{mod_name}.{p_name}" if mod_name else p_name)
                    continue
                if p.ndim >= 2:
                    decay.append(p)
                    if log_names: decay_names.append(f"{mod_name}.{p_name}" if mod_name else p_name)
                else:
                    no_decay.append(p)
                    if log_names: no_decay_names.append(f"{mod_name}.{p_name}" if mod_name else p_name)

        param_groups = [
            {"params": decay,     "weight_decay": float(weight_decay)},
            {"params": no_decay,  "weight_decay": 0.0},
        ]

        if log_names:
            self._logger.info(f"[wd] parameter groups for optimizer:")
            self._logger.info(f"  Decay ({len(decay)} params): {decay_names}")
            self._logger.info(f"  No decay ({len(no_decay)} params): {no_decay_names}")
            self._logger.info(f"  Weight decay: {weight_decay}")

        return param_groups

    @staticmethod
    def _undiff24(y_diff_pred: torch.Tensor, y_anchor24: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct levels from Δ24 predictions using a rolling 24h anchor.

        Parameters
        ----------
        y_diff_pred : torch.Tensor
            Predicted differences, shape (B, T_out, 1), on original scale.
        y_anchor24 : torch.Tensor
            Last 24 observed *levels* from encoder, shape (B, 24, 1).

        Returns
        -------
        torch.Tensor
            Reconstructed levels, shape (B, T_out, 1).
        """
        B, T_out, _ = y_diff_pred.shape
        hist = y_anchor24.clone()  # (B, 24, 1)
        levels = torch.zeros_like(y_diff_pred)
        for t in range(T_out):
            anchor = hist[:, -24, :]             # y_{t-24}
            levels[:, t, :] = anchor + y_diff_pred[:, t, :]
            hist = torch.cat([hist[:, 1:, :], levels[:, t:t+1, :]], dim=1)
        return levels

    # ---------- evaluation ----------
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Run validation on `val_loader` with TF=0.0 and return average loss
        on original scale.
        """
        self.model.eval()
        total_loss_orig_sum, total_loss_orig_cnt = 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                B, T, _ = batch["y_fut"].shape
                nb = (self.device.type == "cuda")
                batch = {k: (v.to(self.device, non_blocking=nb) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()}
                y_pred = self.model(batch, teacher_forcing=0.0)

                y_true = batch["y_fut"]
                y_pred, y_true = self._pair_to_orig_levels(y_pred, y_true, batch)

                loss_orig = self._loss(y_pred, y_true).item()

                total_loss_orig_sum += loss_orig * (B * T)
                total_loss_orig_cnt += (B * T)
        
        val_loss_orig  = float(total_loss_orig_sum / max(1, total_loss_orig_cnt))

        return val_loss_orig

    # ---------- training ----------
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_epochs: int = 20, 
        lr: float = 1e-3,
        patience: int = 5,
        es_rel_delta: float = 0.0, # relative delta for early stopping
        use_tf: bool = True,
        tf_mode: str = "linear",
        tf_drop_epochs: int = 15, 
        use_lr_drop: bool = True,
        lr_drop_epoch: Optional[int] = None, # 1-based epoch
        lr_drop_factor: float = 0.5,              
        es_start_epoch: int = 1, # 1-based
        grad_clip_max_norm: Optional[float] = 10.0,
        weight_decay: float = 0.0,
        trial: Optional[optuna.Trial] = None,
        max_walltime_sec: Optional[float] = None,
        log_times: bool = True,
    ) -> dict[str, Any]:
        """
        Train the model and (optionally) early-stop on validation.

        Parameters
        ----------
        train_loader : DataLoader
            Batches for training (on normalized/engineered data).
        val_loader : DataLoader | None, default=None
            Validation batches; enables early stopping & Optuna reporting if provided.
        n_epochs : int, default=20
            Max training epochs.
        lr : float, default=1e-3
            Initial learning rate for Adam/AdamW.
        patience : int, default=5
            Early-stopping patience (epochs without improvement after `es_start_epoch`).
        es_rel_delta : float, default=0.0
            Required relative improvement to reset patience (e.g., 0.01 = 1% better).
        use_tf : bool, default=True
            If True, use scheduled teacher forcing.
        tf_mode : {"linear","step"}, default="linear"
            Teacher-forcing schedule shape.
        tf_drop_epochs : int, default=15
            Horizon (in epochs) over which TF decays from 1.0 to its late value.
        use_lr_drop : bool, default=True
            If True, multiply LR by `lr_drop_factor` at `lr_drop_epoch` (or when TF hits zero).
        lr_drop_epoch : int | None, default=None
            1-based epoch to drop LR; defaults to TF-zero boundary if None.
        lr_drop_factor : float, default=0.5
            LR multiplier when dropping (e.g., 0.5 halves the LR).
        es_start_epoch : int, default=1
            1-based epoch after which early-stopping patience starts counting.
        grad_clip_max_norm : float | None, default=10.0
            Max global grad-norm; None disables clipping.
        weight_decay : float, default=0.0
            AdamW weight decay; if 0.0 uses Adam without decay.
        trial : optuna.Trial | None, default=None
            If set, reports validation loss each epoch and supports pruning.
        max_walltime_sec : float | None, default=None
            Per-call time budget; triggers Optuna prune on timeout when set.
        log_times : bool, default=False
            If True, logs per-epoch time breakdowns.

        Returns
        -------
        dict
            If early-stopped:
                {"best_val_loss_orig", "best_epoch", "duration_until_best", "avg_near_best"}
            Else:
                {"last_train_loss_orig", "n_epochs", "total_duration", "avg_train_near_end"}
        """
        # Validate input parameters
        if trial is not None:
            if not isinstance(trial, optuna.Trial):
                raise ValueError("trial must be an instance of optuna.Trial or None.")
            if val_loader is None:
                raise ValueError("val_loader must be provided when using optuna trial.")
        if not isinstance(tf_drop_epochs, int) or tf_drop_epochs < 1:
            raise ValueError("tf_drop_epochs must be a positive integer.")
        if not isinstance(es_start_epoch, int) or es_start_epoch < 1:
            raise ValueError("es_start_epoch must be a positive integer.")
        es_start_epoch_zb = es_start_epoch - 1  # zero-based
        if lr_drop_epoch is not None and (not isinstance(lr_drop_epoch, int) or lr_drop_epoch < 1):
            raise ValueError("lr_drop_epoch must be a positive integer.")
        lr_drop_at_epoch_zb = (lr_drop_epoch - 1) if (lr_drop_epoch is not None) else None
        if not isinstance(patience, int) or patience < 0:
            raise ValueError("patience must be a non-negative integer.")
        if not isinstance(n_epochs, int) or n_epochs < 1:
            raise ValueError("n_epochs must be a positive integer.")
        if not isinstance(lr_drop_factor, float) or lr_drop_factor <= 0 or lr_drop_factor >= 1.0:
            raise ValueError("lr_drop_factor must be a float in (0,1).")
        if grad_clip_max_norm is not None and (not isinstance(grad_clip_max_norm, float) or grad_clip_max_norm <= 0):
            raise ValueError("grad_clip_max_norm must be a positive float or None.")
        if grad_clip_max_norm is None:
            grad_clip_max_norm = float('inf')

        # log start
        with_val_txt = "without" if val_loader is None else "with"
        use_es = (val_loader is not None) and (patience and patience > 0)
        with_es_txt = "with" if use_es else "without"
        self._logger.info(f"[train] starting training {with_val_txt} validation and {with_es_txt} early stopping...")
        self._logger.info(
            f"  Train params: epochs={n_epochs}, lr={lr}, patience={patience}, es_rel_delta={es_rel_delta}, use_tf={use_tf}, "
            + (f"tf_mode='{tf_mode}', tf_ep='{tf_drop_epochs}', " if use_tf else "")
            + (f"use_lr_drop={use_lr_drop}, lr_ep={lr_drop_epoch}, lr_drop_factor={lr_drop_factor}, " if use_lr_drop else "")
            + (f"es_start_epoch={es_start_epoch}, " if val_loader is not None else "")
            + f"max_grad_norm={grad_clip_max_norm}, weight_decay={weight_decay}"
        )
        self._logger.info(f"  Data params: norm_mode={self.norm_mode}.")
        start_time = time.perf_counter()
        len_str_n_epochs = len(str(n_epochs))

        # update model debug settings
        if self.debug.enabled and self.debug.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        # Setup optimizer and TF scheduler
        if weight_decay > 0:
            param_groups = self._build_param_groups(weight_decay)
            optimizer = torch.optim.AdamW(param_groups, lr=lr)
        else:
            param_groups = self.model.parameters()
            optimizer = torch.optim.Adam(param_groups, lr=lr)
        if use_tf:
            tf_scheduler = TFScheduler(
                drop_epochs=tf_drop_epochs,
                mode=tf_mode,
                logger=self._logger,
            )

        # Default LR drop epoch when TF reaches zero
        # Note: _zero_start_ep is the boundary in "relative" epochs since start_epoch.
        if lr_drop_at_epoch_zb is None and use_lr_drop:
            if use_tf:
                lr_drop_at_epoch_zb = tf_scheduler.start_epoch + tf_scheduler._zero_start_ep  # 0-based comparison below
            else:
                raise ValueError("lr_drop_epoch must be provided when use_tf is False and use_lr_drop is True.")

        if use_lr_drop:
            self._logger.info(f"[lr] will drop at epoch {lr_drop_at_epoch_zb+1} by x{lr_drop_factor}")

        best_state = None
        best_epoch_zb = None
        best_val_loss_orig = float("inf")
        last_train_loss_orig = None
        wait = 0
        first_time_decreasing_lr = True

        # small helper to get accurate GPU timings
        def _maybe_sync():
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        for epoch in range(n_epochs):
            self.model.train()

            # ---- timing accumulators for this epoch ----
            t_data = t_fwd = t_bwd = t_opt = t_eval = 0.0
            t_epoch_start = time.perf_counter()
            t_prev_end = t_epoch_start

            # ---- training loss accumulators ----
            total_train_loss_orig_sum = 0.0
            total_train_loss_cnt = 0
            n_batches = 0
            grad_norm_acc = []

            # ---- teacher forcing ---
            tf_gen = None
            if use_tf:
                tf_ratio = tf_scheduler.get_tf_ratio(epoch)  # 0-based epoch

                # number of auto regressive decoder (teacher-forcing < 1.0) to hit for this epoch (based on tf_ratio)
                N = len(train_loader.dataset)                    # total samples this epoch
                target_ar_samples = int(round((1.0 - tf_ratio) * N))
                ar_samples_left = target_ar_samples
                samples_left = N

                if self.seed is not None:
                    seed_tf_epoch = derive_seed(self.seed, f"tf_epoch_{epoch}")
                    tf_gen = torch.Generator(device=self.device).manual_seed(seed_tf_epoch)

            # ---- learning rate ---
            if use_lr_drop and first_time_decreasing_lr and epoch == lr_drop_at_epoch_zb:
                for pg in optimizer.param_groups:
                    pg["lr"] *= lr_drop_factor
                self._logger.info(
                    f"[lr] dropping learning rate by x{lr_drop_factor} at epoch {epoch+1} "
                    f"-> {optimizer.param_groups[0]['lr']:.3e}"
                )
                first_time_decreasing_lr = False

            for bidx, batch in enumerate(train_loader):
                # time since previous iter ended ≈ dataloader wait
                t_iter_start = time.perf_counter()
                t_data += (t_iter_start - t_prev_end)

                nb = (self.device.type == "cuda")
                batch = {k: (v.to(self.device, non_blocking=nb) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)

                # enforce exact count of AR samples
                tf_for_this_batch = 0.0  # default: no teacher forcing
                if use_tf:
                    B = batch["y_fut"].shape[0]  # batch size
                    p = ar_samples_left / max(1, samples_left) # probability to choose AR for this batch
                    force_ar = ar_samples_left >= samples_left - B
                    rand_u = torch.rand((), device=self.device, generator=tf_gen) if tf_gen is not None else torch.rand((), device=self.device)
                    take_ar = force_ar or (rand_u < p)
                    tf_for_this_batch = 0.0 if take_ar else 1.0

                    if take_ar:
                        ar_samples_left -= B
                    samples_left -= B   

                # forward
                _maybe_sync()
                t0 = time.perf_counter()
                y_pred = self.model(batch, teacher_forcing=tf_for_this_batch, tf_generator=tf_gen) # tf_gen is not actually needed since the ratio is 0 or 1
                _maybe_sync()
                t_fwd += (time.perf_counter() - t0)

                if self.debug.check_nan_every_step:
                    # y_pred.isfinite().all() returns a scalar tensor; .item() to avoid ambiguity
                    if not torch.isfinite(y_pred).all().item():
                        raise RuntimeError(f"NaN/Inf in y_pred at epoch {epoch+1}, batch {bidx}.")

                # move to original scale (your helper)
                y_pred, y_true = self._pair_to_orig_levels(y_pred, batch["y_fut"], batch)
                loss_orig = self._loss(y_pred, y_true)
                if not torch.isfinite(loss_orig):
                    raise RuntimeError(f"NaN/Inf in loss at epoch {epoch+1}, batch {bidx}.")

                # backward
                _maybe_sync()
                t0 = time.perf_counter()
                loss_orig.backward()
                _maybe_sync()
                t_bwd += (time.perf_counter() - t0)

                # optimizer step (+ grad clipping)
                _maybe_sync()
                t0 = time.perf_counter()
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_max_norm)
                grad_val = float(total_norm) if math.isfinite(total_norm) else float('nan')
                grad_norm_acc.append(grad_val)
                if (not math.isfinite(total_norm)) or float(total_norm) > 1e6:
                    self._logger.warning(f"Skipping optimizer.step() due to bad grad norm: {float(total_norm):.3e}")
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.step()
                _maybe_sync()
                t_opt += (time.perf_counter() - t0)

                n_batches += 1
                loss_orig_item = float(loss_orig.item())
                B, T = y_pred.shape[0], y_pred.shape[1]
                total_train_loss_orig_sum += loss_orig_item * (B * T)
                total_train_loss_cnt += (B * T)

                if self.debug.log_grad_norm_every and (bidx % self.debug.log_grad_norm_every == 0):
                    self._logger.debug(
                        f"[ep {epoch+1} b{bidx}] grad_norm={float(total_norm):.4f} " \
                        + (f"tf={tf_ratio:.3f}" if use_tf else "")
                    )

                # end-of-iteration timestamp for next loader wait measure
                t_prev_end = time.perf_counter()

            train_loss_orig = total_train_loss_orig_sum / max(1, total_train_loss_cnt)
            self.train_losses_orig.append(train_loss_orig)
            last_train_loss_orig = train_loss_orig

            # validation (time it)
            val_loss_orig = None
            if val_loader is not None:
                _maybe_sync()
                t0 = time.perf_counter()
                val_loss_orig = self.evaluate(val_loader)
                _maybe_sync()
                t_eval += (time.perf_counter() - t0)
                self.val_losses_orig.append(val_loss_orig)
                self._logger.info(
                    f"Epoch {epoch+1:{len_str_n_epochs}d}/{n_epochs} " \
                    + f"- train(orig)={train_loss_orig:.4f} " \
                    + f"- val(orig)={val_loss_orig:.4f} " \
                    + (f"- tf_ratio={tf_ratio:.3f} " if use_tf else "") \
                    + f"- lr={optimizer.param_groups[0]['lr']:.3e} " \
                    + f"- grad(mean-of-current)={np.mean(grad_norm_acc):.3f}"
                )
            else:
                self._logger.info(
                    f"Epoch {epoch+1:{len_str_n_epochs}d}/{n_epochs} "
                    f"- train(orig)={train_loss_orig:.4f} " \
                    + (f"- tf_ratio={tf_ratio:.3f} " if use_tf else "") \
                    + f"- lr={optimizer.param_groups[0]['lr']:.3e} "
                )

            # epoch totals
            t_epoch_total = time.perf_counter() - t_epoch_start
            if log_times:
                # guard against division by zero
                denom = max(t_epoch_total, 1e-9)
                self._logger.info(
                    f"[time {epoch+1:{len_str_n_epochs}d}/{n_epochs}] "
                    f"total={t_epoch_total:.3f}s | "
                    f"data={t_data:.3f}s ({t_data/denom*100:.1f}%) | "
                    f"fwd={t_fwd:.3f}s ({t_fwd/denom*100:.1f}%) | "
                    f"bwd={t_bwd:.3f}s ({t_bwd/denom*100:.1f}%) | "
                    f"opt={t_opt:.3f}s ({t_opt/denom*100:.1f}%)"
                    + (f" | eval={t_eval:.3f}s ({t_eval/denom*100:.1f}%)" if val_loader is not None else "")
                )

            # Save history row
            row = dict(epoch=epoch+1, train_orig=train_loss_orig, tf=tf_ratio if use_tf else None,
                    grad_mean=float(np.mean(grad_norm_acc)),
                    epoch_time_sec=float(t_epoch_total))
            if val_loader is not None:
                row.update(val_orig=self.val_losses_orig[-1])
            if log_times:
                row.update(time_data_s=t_data, time_fwd_s=t_fwd, time_bwd_s=t_bwd, time_opt_s=t_opt, time_eval_s=t_eval)
            self.history.append(row)

            # --- optuna reporting ---
            if trial is not None:
                trial.report(val_loss_orig, step=epoch)  # 0-based
                self._logger.info(f"Reported trial {trial.number} step={epoch}")
                if trial.should_prune():
                    # Let pruner stop the trial if it wants to.
                    raise optuna.TrialPruned()

            # --- time budget check ---
            if (max_walltime_sec is not None) and (time.perf_counter() - start_time > max_walltime_sec):
                if trial is not None:
                    trial.set_user_attr("timeout", True)
                # prune so it doesn't count as a failure
                raise optuna.TrialPruned("Per-trial wall-time budget exceeded")
            
            # --- early stopping block AFTER reporting/pruning ---
            if use_es:
                if val_loss_orig < best_val_loss_orig * (1.0 - es_rel_delta):
                    best_val_loss_orig = val_loss_orig
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    best_epoch_zb = epoch
                    duration_until_best = time.perf_counter() - start_time
                    wait = 0
                else:
                    # Do not accumulate patience before es_start_epoch_zb 
                    if epoch >= es_start_epoch_zb:
                        wait += 1
                    else:
                        wait = 0

                    if epoch >= es_start_epoch_zb and wait >= patience:
                        self._logger.info(
                            f"Early stopping triggered at epoch {epoch+1} "
                            f"(es_start_epoch={es_start_epoch_zb+1}, patience={patience})."
                        )
                        break

        if use_es and best_state is not None:
            self.model.load_state_dict(best_state)
            self.best_val_loss_orig = best_val_loss_orig
            self.best_epoch_zb = best_epoch_zb
            
            secs = float(duration_until_best)  # seconds as float (no .copy())
            duration_until_best = pd.to_timedelta(secs, unit="s")
            self.duration_until_best = duration_until_best

            # mm'ss" formatting
            mins, secs_rem = divmod(int(round(secs)), 60)

            vals = self.val_losses_orig
            best_idx = best_epoch_zb
            lo = max(0, best_idx - 1)
            hi = min(len(vals), best_idx + 1 + 2)
            avg_near_best = float(np.mean(vals[lo:hi])) if len(vals) else float('nan')

            self._logger.info(
                f"Restored best weights from epoch {best_epoch_zb+1} "
                f"(val_loss(orig)={best_val_loss_orig:.6f}, dur_until_best={mins:02d}'{secs_rem:02d}\", "
                f"avg_near_best={avg_near_best:.2f})"
            )

            return dict(
                best_val_loss_orig=best_val_loss_orig, 
                best_epoch=best_epoch_zb+1, 
                duration_until_best=duration_until_best, 
                avg_near_best=avg_near_best
            )
        
        else:
            total_duration = pd.to_timedelta(time.perf_counter() - start_time, unit="s")
            avg_train_near_end = np.mean(self.train_losses_orig[-5:] if len(self.train_losses_orig) >= 5 else self.train_losses_orig)
            return dict(
                last_train_loss_orig=last_train_loss_orig,
                n_epochs=n_epochs,
                total_duration=total_duration,
                avg_train_near_end=avg_train_near_end
            )

    def history_as_dataframe(self):
        """
        Return training/validation timeline with per-epoch metrics.

        Returns
        -------
        pd.DataFrame
            Columns include: epoch, train_orig, val_orig (if any), tf,
            grad_mean, epoch_time_sec, and timing breakdowns if enabled.
        """
        return pd.DataFrame(self.history)


# Utilities to Prepare Dataloaders


# --- Helper: deterministic seeding for DataLoader workers ---
def _make_worker_init_fn(base_seed: int):
    """
    Create a deterministic worker init function for PyTorch DataLoader.
    Each worker is seeded with `base_seed + worker_id` for Python, NumPy, Torch.

    Returns
    -------
    Callable[[int], None]
        Function suitable for `DataLoader(..., worker_init_fn=...)`.
    """
    def _init_fn(worker_id: int):
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return _init_fn

def _slice_with_history(y, endog, exog, start: pd.Timestamp, end: pd.Timestamp, T_in: int,
                        y_levels: Optional[pd.Series] = None):
    """
    Slice a block [start, end] plus `T_in` hours of history before `start` (used e.g. to build the validation slice).

    Parameters
    ----------
    y, endog, exog : pd.Series | pd.DataFrame
        Target and feature frames with aligned indices.
    start, end : pd.Timestamp
        Inclusive bounds for the decoder horizon.
    T_in : int
        Encoder length (hours) to prepend.
    y_levels : pd.Series | None
        Optional level target (returned if provided).

    Returns
    -------
    tuple
        (y_slice, endog_slice, exog_slice[, y_levels_slice]) with history.
    """
    hist_start = start - pd.Timedelta(hours=T_in)
    y_s  = y.loc[hist_start:end]
    en_s = endog.loc[hist_start:end]
    ex_s = exog.loc[hist_start:end]
    if y_levels is None:
        return (y_s, en_s, ex_s)
    yl_s = y_levels.loc[hist_start:end]
    return (y_s, en_s, ex_s, yl_s)

# ─────────────────────────────────────────────────────────────────────────────
# Config classes (split into model/data/features/train + top-level)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    Architecture hyperparameters for the EncDecLSTM.

    Attributes
    ----------
    input_len : int, default=168   
        Encoder context length (T_in) in timesteps (e.g., hours).
    output_len : int, default=24
        Forecast horizon (T_out) in timesteps produced by the decoder.
    hidden_size : int, default=64
        Number of hidden units per LSTM layer (shared by encoder/decoder and the head input size).
    num_layers : int, default=1
        Stacked LSTM layers in both encoder and decoder.
    head : {"linear", "mlp"}, default="linear"
        Per-step prediction head applied to decoder outputs:
        - "linear": (optional Dropout) → Linear(H, 1)
        - "mlp":    Linear(H, H) → ReLU → (optional Dropout) → Linear(H, 1)
    dropout : float, default=0.0
        Dropout rate. Applied:
        - between LSTM layers when num_layers > 1 (PyTorch LSTM inter-layer dropout)
        - inside the head as described above.
    use_ar : {"prev", "24h", "none"}, default="prev"
        How the decoder is conditioned on past target values:
        - "prev": feed y_{t-1} at every step (classic AR decoding)
        - "24h":  feed a rolling window of the previous 24 targets (requires T_out > 24)
        - "none": no target feedback; decoder sees only future-safe endog + exog

    Notes
    -----
    • Encoder input size  = (#endog features + #exog features).
    • Decoder input size  = (#future-safe endog + #exog + 1 if use_ar ∈ {"prev","24h"} else 0).
    • The dataset always provides `y_last_24h`; choose `use_ar` to match your intended conditioning.
    """
    input_len: int = 168
    output_len: int = 24
    hidden_size: int = 64
    num_layers: int = 1
    head: Literal["linear", "mlp"] = "linear"  # output head type
    dropout: float = 0.0
    use_ar: Literal["prev", "24h", "none"] = "prev"

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class DataConfig:
    """
    Data loader and sampling options.

    Attributes
    ----------
    stride : int, default=1
        Step (in timesteps) between consecutive training windows.
        E.g., stride=1 yields every possible window; stride=24 samples daily.
    gap_hours : int, default=0
        Guard gap between the end of the train slice and the start of validation
        to avoid leakage (hours removed from the end of train when building loaders).
    batch_size : int, default=64
        Mini-batch size for both train/val loaders.
    num_workers : int, default=0
        Number of DataLoader worker processes.
    shuffle_train : bool, default=True
        Whether to shuffle windows in the training loader.
    pin_memory : Optional[bool], default=None
        If None, auto-enable when CUDA is available; otherwise passed to DataLoader.
    """
    stride: int = 1
    gap_hours: int = 0
    batch_size: int = 64
    num_workers: int = 0
    shuffle_train: bool = True
    pin_memory: Optional[bool] = None      # None => auto on CUDA

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class FeatureConfig:
    """
    Feature engineering switches for endogenous/exogenous variables.

    Attributes
    ----------
    exog_vars : Tuple[str, ...]
        Names of exogenous columns to use from aux_df.
        Allowed: {"temperature","dew_point","wind_speed","pressure","humidity"}.
    hour_averages : Tuple[int, ...]
        Rolling means (in hours) computed for both endog (except 'y_diff') and exog.
        Example: (24, 168) adds *_avg1d and *_avg7d features.
    endog_hour_lags : Tuple[int, ...]
        Lags (in hours) for endogenous features (e.g., y_lag24, y_lag168, y_diff_lag24 if present).
    include_exog_lags : bool
        If True, also creates lagged versions of exogenous variables for each lag in `endog_hour_lags`.
    use_differences : bool
        If True, target becomes Δ24 (y - y.shift(24)); also adds 'y_diff' endogenous feature.
        Requires input_len ≥ 24.
    time_vars : Tuple[str, ...]
        Time/cyclical features to add:
        - "hod":  hour-of-day (sin, cos)
        - "dow":  day-of-week (sin, cos)
        - "moy":  month-of-year (sin, cos)
        - "wss":  weekday/sat/sun 3-class cycle (sin, cos)
    use_cold_season : bool
        If True, adds a binary 'is_cold_season' feature.
        Uses temperature threshold if provided, otherwise calendar months (NDJFM).
    cold_temp_threshold : Optional[float]
        Temperature threshold (°C). If set, 'is_cold_season' is (rolling mean ≤ threshold).
    cold_avg_hours : int
        Window (hours) for the rolling temperature average used by the cold-season flag.
    """
    exog_vars: Tuple[str, ...] = ("temperature",)
    hour_averages: Tuple[int, ...] = ()
    endog_hour_lags: Tuple[int, ...] = (24, 168)
    include_exog_lags: bool = False
    use_differences: bool = False
    time_vars: Tuple[str, ...] = ("hod", "dow", "moy", "wss")
    use_cold_season: bool = False
    cold_temp_threshold: Optional[float] = 15.0
    cold_avg_hours: int = 96

    def __post_init__(self):
        # Validate exog_vars
        allowed_exog = {"temperature", "dew_point", "wind_speed", "pressure", "humidity"}
        bad = set(self.exog_vars) - allowed_exog
        if bad:
            raise ValueError(f"Unknown exog_vars {sorted(bad)}; allowed: {sorted(allowed_exog)}")

        # Validate time_vars
        allowed_time = {"hod", "dow", "moy", "wss"}
        bad = set(self.time_vars) - allowed_time
        if bad:
            raise ValueError(f"Unknown time_vars {sorted(bad)}; allowed: {sorted(allowed_time)}")

        # Validate hour_averages
        if not all(isinstance(hours, int) and hours > 0 for hours in self.hour_averages):
            raise ValueError(f"hour_averages must contain positive ints, got {self.hour_averages}")
        
        # Validate endog_hour_lags
        if not all(isinstance(lag, int) and lag > 0 for lag in self.endog_hour_lags):
            raise ValueError(f"endog_hour_lags must contain positive ints, got {self.endog_hour_lags}")

        # Validate cold_avg_hours
        if self.use_cold_season and (not isinstance(self.cold_avg_hours, int) or self.cold_avg_hours <= 0):
            raise ValueError("cold_avg_hours must be a positive int.")

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class TrainConfig:
    """
    Training loop hyperparameters and schedules.

    Attributes
    ----------
    learning_rate : float
        Initial LR for Adam/AdamW.
    n_epochs : int
        Maximum number of epochs.
    patience : int
        Early-stopping patience (in epochs). 0 disables ES.
    es_rel_delta : float
        Minimum relative improvement to reset patience (e.g., 0.01 = 1%).
    tf_mode : {"linear","step"}
        Teacher forcing schedule shape (see TFScheduler).
    tf_drop_epochs : int
        Horizon (epochs) used by TFScheduler to move from initial TF to late TF.
    use_lr_drop : bool
        If True, multiply LR by `lr_drop_factor` at `lr_drop_epoch` (or when TF reaches zero).
    lr_drop_epoch : Optional[int]
        1-based epoch index to drop LR. If None, defaults to TF zero-start epoch.
    lr_drop_factor : float
        Factor in (0,1): new_lr = old_lr * lr_drop_factor.
    es_start_epoch : int
        1-based epoch from which ES can start counting patience.
    grad_clip_max_norm : float
        Max L2 norm for gradient clipping.
    weight_decay : float
        AdamW weight decay (0 → Adam without grouped decay logic).
    max_walltime_sec : Optional[float]
        If set, prune/stop the run after this wall-clock budget (per train call).
    """
    learning_rate: float = 1e-3
    n_epochs: int = 20
    patience: int = 5
    es_rel_delta: float = 0.0 # relative delta for early stopping
    tf_mode: Literal["linear", "step"] = "linear"
    tf_drop_epochs: int = 15
    use_lr_drop: bool = True
    lr_drop_epoch: Optional[int] = None # 1-based epoch
    lr_drop_factor: float = 0.5              
    es_start_epoch: int = 1 # 1-based
    grad_clip_max_norm: float = 10.0
    weight_decay: float = 0.0
    max_walltime_sec: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class LSTMRunConfig:
    """
    End-to-end configuration bundle for the LSTM pipeline.

    Attributes
    ----------
    model : ModelConfig
        Architecture and decoder conditioning.
    data : DataConfig
        Window sampling and DataLoader parameters.
    features : FeatureConfig
        Feature engineering switches (lags, rolling means, time vars, diffs).
    train : TrainConfig
        Optimizer/schedule/ES/regularization settings.
    norm : NormalizeConfig
        Normalization policy and exclusions.
    seed : Optional[int]
        Global seed for reproducibility (data shuffling, TF coin flips, etc.).
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    norm: NormalizeConfig = field(default_factory=NormalizeConfig)
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "data": self.data.to_dict(),
            "features": self.features.to_dict(),
            "train": self.train.to_dict(),
            "norm": self.norm.to_dict(),
            "seed": self.seed,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline Class
# ─────────────────────────────────────────────────────────────────────────────

class LSTMPipeline:
    def __init__(
            self, 
            target_df: pd.DataFrame,  
            config: LSTMRunConfig,
            aux_df: Optional[pd.DataFrame] = None, 
            logger: Optional[logging.Logger] = None,
            device: Optional[torch.device] = None,
        ):
        """
        Assumptions:
        - `target_df` and optional `aux_df` contain a single `unique_id`.
        - Index column is `ds` (timestamp). The pipeline sorts and sets it as index.
        - Exogenous variables requested in `FeatureConfig` must be present in `aux_df`.
        """
        self._target_df = target_df
        self._aux_df = aux_df
        self.config = config
        self._logger = logger or logging.getLogger(__name__)
        if device is not None and isinstance(device, torch.device):
            self.device = device
            lines_to_log = [f"[pipe init] using provided device: {self.device.type}"]
        else:
            if device is not None: # -> device was provided but not a torch.device
                self._logger.warning(f"device argument must be torch.device or None; got {type(device)}. Auto-selecting device.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            lines_to_log = [f"[pipe init] auto-selected device: {self.device.type}"]

        # internal attributes initialized during data preparation
        self._endog_df = None
        self._exog_df = None
        self._y = None
        self._y_levels = None
        self._endog_vars_not_for_future = None

        # internal attributes initialized during data preparation / loading
        self._start_train = None
        self._end_train = None
        self._start_val = None
        self._end_val = None
        self._norm_global_stats = None

        # internal attributes initialized during fit
        self._trainer = None
        self._history_df = None
        self._model = None
        self._n_params = None
        self._train_losses_orig = None
        self._val_losses_orig = None


        # --- Set seeds if requested ---

        if self.config.seed is not None:
            lines_to_log.append(f"setting random seed: {self.config.seed}")
            set_global_seed(self.config.seed)

        # --- Check coherence of params ---

        # Make sure a single id is used
        if not target_df['unique_id'].nunique() == 1:
            raise ValueError("The target_df must contain a single unique_id. Found: {}".format(target_df['unique_id'].unique()))
        
        # Make sure aux_df is provided if exog_vars is used
        if self.config.features.exog_vars and aux_df is None:
            raise ValueError("aux_df must be provided when exog_vars is used.")

        # Make sure the aux_df has the same unique_id
        if aux_df is not None:
            if not aux_df['unique_id'].nunique() == 1:
                raise ValueError("The aux_df must contain a single unique_id. Found: {}".format(aux_df['unique_id'].unique()))
            if not aux_df['unique_id'].unique()[0] == target_df['unique_id'].unique()[0]:
                raise ValueError("The unique_id in aux_df must match the one in target_df. Found: {} and {}".format(
                    aux_df['unique_id'].unique()[0], target_df['unique_id'].unique()[0]
                ))
        unique_id = target_df['unique_id'].unique()[0]
        if not unique_id in ['F1', 'F2', 'F3', 'F4', 'F5']:
            raise ValueError(f"Invalid unique_id: {unique_id}. Expected one of ['F1', 'F2', 'F3', 'F4', 'F5'].")
        
        # Save unique_id and target merged with aux
        self.unique_id = unique_id
        if aux_df is not None:
            self._target_plus_aux_df = target_df.merge(
                aux_df, 
                on=['unique_id', 'ds'], 
                how='inner', 
            )
        else:
            self._target_plus_aux_df = target_df.copy()

        # Set ds as index in target_plus_aux_df
        self._target_plus_aux_df['ds'] = pd.to_datetime(self._target_plus_aux_df['ds'])
        self._target_plus_aux_df = self._target_plus_aux_df.sort_values('ds').set_index('ds')

        if self.config.features.exog_vars:
            # Make sure exog vars exist in aux_df
            missing_vars = [var for var in self.config.features.exog_vars if var not in aux_df.columns]
            if missing_vars:
                raise ValueError(f"The following exogenous variables are missing in aux_df: {missing_vars}")

        self._logger.info("; ".join(lines_to_log) + '.')
            
    # ------------------------------
    # Data Preparation
    # ------------------------------

    def generate_vars(self) -> None:
        """
        Create endogenous/exogenous matrices and target series.

        Steps
        -----
        - Optional Δ24 differencing (`use_differences`).
        - Rolling means for configured windows.
        - Lagged features (endog and, optionally, exog).
        - Time features (sin/cos of HOD/DOW/MOY, WSS, cold-season flag).
        - Head-trim to drop rows made invalid by lags/diffs/rolls.
        - NaN row drop with a warning.
        - Track endog columns not safe for decoder (leakage control).
        """
        df = self._target_plus_aux_df
        y_levels  = df['y']
        y_target = y_levels.copy() # default: level target

        # Base frames
        endog_df = df[['y']].copy()
        exog_df  = df[list(self.config.features.exog_vars)].copy() if self.config.features.exog_vars else pd.DataFrame(index=df.index)

        # --- Seasonal differencing ---
        seasonal_lag = 24 if self.config.features.use_differences else 0
        if self.config.features.use_differences:
            y_target = y_target - y_target.shift(seasonal_lag)
            endog_df['y_diff'] = endog_df['y'] - endog_df['y'].shift(24)

        # --- Rolling means (endog & exog) block ---
        def set_name(base: str, hours: int) -> str:
            if hours < 24: return f"{base}_avg{hours}h"
            d, h = divmod(hours, 24)
            return f"{base}_avg{d}d" if h == 0 else f"{base}_avg{d}d{h}h"

        endog_roll, exog_roll = {}, {}
        for hours in (self.config.features.hour_averages or []):
            win = f"{hours}h"
            # Endog: roll each col except y_diff
            for c in [c for c in endog_df.columns if c != 'y_diff']:
                endog_roll[set_name(c, hours)] = endog_df[c].rolling(window=win, min_periods=1).mean()
            # Exog
            for c in exog_df.columns:
                exog_roll[set_name(c, hours)] = exog_df[c].rolling(window=win, min_periods=1).mean()

        if endog_roll:
            endog_df = pd.concat([endog_df, pd.DataFrame(endog_roll, index=endog_df.index)], axis=1)
        if exog_roll:
            exog_df  = pd.concat([exog_df,  pd.DataFrame(exog_roll,  index=exog_df.index)],  axis=1)

        # If using differences as encoder features, consider dropping raw y (only now to keep derived cols)
        # if self.config.features.use_differences and 'y' in endog_df.columns:
        #     endog_df.drop(columns=['y'], inplace=True)

        # --- Lagged endogenous features ---
        base_endog_for_lags = [c for c in ['y','y_diff'] if c in endog_df.columns]
        lag_cols = {}
        max_lag = 0
        for lag in self.config.features.endog_hour_lags or []:
            if not isinstance(lag, int) or lag <= 0:
                raise ValueError(f"endog_hour_lags must contain positive ints, got {lag}")
            max_lag = max(max_lag, lag)
            for c in base_endog_for_lags:
                lag_cols[f"{c}_lag{lag}"] = endog_df[c].shift(lag)

        if lag_cols:
            endog_df = pd.concat([endog_df, pd.DataFrame(lag_cols, index=endog_df.index)], axis=1)
        
        # --- Lagged exogenous features ---
        base_exog_for_lags = list(self.config.features.exog_vars)
        if self.config.features.include_exog_lags and not exog_df.empty:
            exog_lag_cols = {}
            for lag in (self.config.features.endog_hour_lags or []):
                for c in base_exog_for_lags:
                    exog_lag_cols[f"{c}_lag{lag}"] = exog_df[c].shift(lag)

            if exog_lag_cols:
                exog_df = pd.concat([exog_df, pd.DataFrame(exog_lag_cols, index=exog_df.index)], axis=1)

        # --- Trim head once (for diffs/rolling/lags) ---
        head_trim = seasonal_lag + max_lag
        if head_trim > 0:
            endog_df = endog_df.iloc[head_trim:]
            exog_df  = exog_df.iloc[head_trim:]
            y_levels  = y_levels.iloc[head_trim:]
            y_target = y_target.iloc[head_trim:]

        # --- Time features (cyclical + cold flag) ---
        idx = endog_df.index
        assert idx.equals(exog_df.index) and idx.equals(y_levels.index) and idx.equals(y_target.index), "Indexes must be aligned."

        def sincos(x, period):
            x = np.asarray(x, dtype=np.float32)
            ang = 2.0 * np.pi * (x / period)
            return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)
        
        # Figure out which time primitives we need
        tv = set(self.config.features.time_vars)
        need_hod = "hod" in tv
        need_dow = ("dow" in tv) or ("wss" in tv)  # wss is derived from dow
        need_moy = ("moy" in tv) or (self.config.features.use_cold_season and 
                                    self.config.features.cold_temp_threshold is None)

        if need_hod:
            hour = idx.hour.values
        if need_dow:
            dow = idx.dayofweek.values
        if need_moy:
            month = idx.month.values

        feats_dict = {}

        if "hod" in self.config.features.time_vars:
            hour_sin, hour_cos = sincos(hour, 24)
            feats_dict["hour_sin"], feats_dict["hour_cos"] = hour_sin, hour_cos

        if "dow" in self.config.features.time_vars:
            dow_sin, dow_cos = sincos(dow, 7)
            feats_dict["dow_sin"], feats_dict["dow_cos"] = dow_sin, dow_cos

        if "moy" in self.config.features.time_vars:
            month_sin, month_cos = sincos(month - 1, 12)
            feats_dict["month_sin"], feats_dict["month_cos"] = month_sin, month_cos

        if "wss" in self.config.features.time_vars:
            wss   = np.where(dow == 5, 1, np.where(dow == 6, 2, 0)).astype(np.int32)  # weekday/sat/sun
            wss_sin, wss_cos = sincos(wss, 3)
            feats_dict["wss_sin"], feats_dict["wss_cos"] = wss_sin, wss_cos

        # cold season flag (independent of time_vars)
        if getattr(self.config.features, 'use_cold_season', False):
            cold_thresh = getattr(self.config.features, 'cold_temp_threshold', None)
            cold_hours  = int(getattr(self.config.features, 'cold_avg_hours', 96))
            if cold_thresh is not None:
                if 'temperature' not in exog_df.columns:
                    raise ValueError("Exogenous 'temperature' required for temperature-based cold-season flag.")
                temp_avg = exog_df['temperature'].rolling(window=f'{cold_hours}h', min_periods=1).mean()
                feats_dict['is_cold_season'] = (temp_avg <= float(cold_thresh)).astype(np.float32).values
            else:
                feats_dict['is_cold_season'] = np.isin(month, (11, 12, 1, 2, 3)).astype(np.float32)

        if feats_dict:
            feats_df = pd.DataFrame(feats_dict, index=idx).astype(np.float32)
            exog_df  = pd.concat([exog_df, feats_df], axis=1)

        # 6) --- Final cleanup & assignments ---
        if endog_df.isna().any().any() or exog_df.isna().any().any() or y_levels.isna().any() or y_target.isna().any():
            n_endog = int(endog_df.isna().sum().sum())
            n_exog  = int(exog_df.isna().sum().sum())
            n_yl    = int(y_levels.isna().sum())
            n_yt    = int(y_target.isna().sum())
            self._logger.warning(
                f"NaNs detected before final masking "
                f"(endog: {n_endog}, exog: {n_exog}, y_levels: {n_yl}, y_target: {n_yt}). "
                f"Rows containing NaNs will be dropped."
            )

        mask_ok = endog_df.notna().all(axis=1) & exog_df.notna().all(axis=1) & y_levels.notna() & y_target.notna()
        self._endog_df = endog_df.loc[mask_ok].astype(np.float32)
        self._exog_df  = exog_df.loc[mask_ok].astype(np.float32)
        self._y        = y_target.loc[mask_ok].astype(np.float32)
        self._y_levels = y_levels.loc[mask_ok].astype(np.float32)
        self._endog_vars_not_for_future = [c for c in self._endog_df.columns if not is_for_endog_fut(c, self.config.model.output_len)]
        self._logger.info(f"[gvars] features ready: endog={self._endog_df.shape} exog={self._exog_df.shape} "
                          f"| y={self._y.shape} y_levels={self._y_levels.shape}")
        
    def _norm_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute column indices to be normalized, honoring skip lists in `NormalizeConfig`.
        Returns (endog_norm_idx, exog_norm_idx).
        """
        exog_cols  = list(self._exog_df.columns)
        endog_cols = list(self._endog_df.columns)

        def _idx(cols, to_skip):
            skip = set(to_skip or ())
            return np.array([i for i, c in enumerate(cols) if c not in skip], dtype=np.int64)

        exog_norm_idx  = _idx(exog_cols,  getattr(self.config.norm, "exog_skip_cols", ()))
        endog_norm_idx = _idx(endog_cols, getattr(self.config.norm, "endog_skip_cols", ()))
        return endog_norm_idx, exog_norm_idx

    def _dataset_from_slices(
        self,
        y_s: pd.Series | pd.DataFrame,
        endog_s: pd.DataFrame,
        exog_s: pd.DataFrame,
        y_levels_s: Optional[pd.Series],
        *,
        global_stats: Optional[GlobalNormStats],
        norm_cfg: Optional[NormalizeConfig]
    ) -> TimeSeriesDataset:
        """
        Construct a `TimeSeriesDataset` from provided aligned slices, ensuring
        consistency with the pipeline's configuration (Δ24, normalization indices, etc.).
        """
        T_in  = int(self.config.model.input_len)
        T_out = int(self.config.model.output_len)
        if self.config.features.use_differences and T_in < 24:
            raise ValueError("input_len must be ≥ 24 when use_differences=True (Δ24).")
        endog_norm_idx, exog_norm_idx = self._norm_indices()

        return TimeSeriesDataset(
            y_s, endog_s, exog_s,
            input_len=T_in, output_len=T_out,
            stride=self.config.data.stride,
            norm_cfg=norm_cfg or NormalizeConfig(),
            global_stats=global_stats,
            y_levels=y_levels_s,
            target_is_diff24=self.config.features.use_differences,
            norm_idx_endog=endog_norm_idx,
            norm_idx_exog=exog_norm_idx,
        )

    def _build_loader(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: Optional[bool],
        seed: Optional[int],
    ) -> DataLoader:
        """Build a PyTorch DataLoader with deterministic worker seeding (if a seed is provided)."""
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()

        loader_generator = None
        worker_init_fn = None
        if seed is not None:
            loader_generator = torch.Generator(device="cpu").manual_seed(int(seed))
            worker_init_fn = _make_worker_init_fn(int(seed))  # reuse your existing helper

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            generator=loader_generator,
        )
    
    def _resolve_ranges(
        self,
        *,
        start_train: pd.Timestamp | None,
        end_train:   pd.Timestamp | None,
        start_val:   pd.Timestamp | None,
        end_val:     pd.Timestamp | None,
    ) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Fill missing bounds with the simple rules:
        - missing start train -> first available timestamp
        - missing end train   -> start val - 1h or last available timestamp
        - missing start val   -> end train + 1h or first available timestamp
        - missing end val     -> last available timestamp

        Returns
        -------
        dict
            Mapping {"train": (start, end), "val": (start, end)} for requested splits.
        """
        one_h = pd.Timedelta(hours=1)

        train_split = {"name": "train", "start": start_train, "end": end_train,
            "want": (start_train is not None) or (end_train is not None)}
        val_split = {"name": "val",   "start": start_val,   "end": end_val,
            "want": (start_val   is not None) or (end_val   is not None)}
        
        if train_split["want"]:
            if train_split["start"] is None:
                if self._y is None: raise ValueError("Generate vars before calling ._resolve_ranges()")
                train_split["start"] = self._y.index.min()
            if train_split["end"] is None:
                if val_split["start"] is not None:
                    train_split["end"] = val_split["start"] - one_h
                elif val_split["end"] is not None:
                    raise ValueError("Unable to resolve ranges: missing end_train and start_val but both loaders were required (provide at least one of the two)")
                elif self._y is None: 
                    raise ValueError("Generate vars before calling ._resolve_ranges()")
                else:
                    train_split["end"] = self._y.index.max()

        if val_split["want"]:
            if val_split["start"] is None:
                if train_split["end"] is not None:
                    val_split["start"] = train_split["end"] + one_h
                elif train_split["start"] is not None:
                    raise ValueError("Unable to resolve ranges: missing end_train and start_val but both loaders were required (provide at least one of the two)")
                elif self._y is None: 
                    raise ValueError("Generate vars before calling ._resolve_ranges()")
                else:
                    val_split["start"] = self._y.index.min()
            if val_split["end"] is None:
                if self._y is None: raise ValueError("Generate vars before calling ._resolve_ranges()")
                val_split["end"] = self._y.index.max()
        
        # final validation
        out: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        splits = [train_split, val_split]
        for s in splits:
            if not s["want"]:
                continue
            if s["end"] < s["start"]:
                raise ValueError(f"{s['name']}: end must be ≥ start (got {s['start']} → {s['end']}).")
            out[s["name"]] = (s["start"], s["end"])
        return out
    
    def make_loaders(
        self,
        *,
        start_train: pd.Timestamp | None = None,
        end_train:   pd.Timestamp | None = None,
        start_val:   pd.Timestamp | None = None,
        end_val:     pd.Timestamp | None = None,
    ) -> tuple[DataLoader | None, DataLoader | None]:
        "Make train / val (or test) loaders"

        logger = self._logger
        gap_hours = int(self.config.data.gap_hours)

        # prep & sanity
        if self._endog_df is None or self._exog_df is None or self._y is None or self._y_levels is None:
            self.generate_vars()
        if self.config.features.use_differences and self.config.model.input_len < 24:
            raise ValueError("input_len must be ≥ 24 when use_differences=True (Δ24).")
        for name, ts in {
            "start_train": start_train, "end_train": end_train,
            "start_val": start_val,     "end_val": end_val,
        }.items():
            if ts is not None and not isinstance(ts, pd.Timestamp):
                raise ValueError(f"{name} must be a pd.Timestamp or None.")

        # resolve ranges with the tiny helper
        ranges = self._resolve_ranges(
            start_train=start_train, end_train=end_train,
            start_val=start_val,     end_val=end_val,
        )

        # persist to pipeline (set to None if not requested)
        (self._start_train, self._end_train) = ranges.get("train", (None, None))
        (self._start_val,   self._end_val)   = ranges.get("val",   (None, None))

        want_train = "train" in ranges
        want_val   = "val"   in ranges
        T_in = int(self.config.model.input_len)
        T_out = int(self.config.model.output_len)

        norm_mode = getattr(self.config.norm, "mode", "none")
        use_global = (norm_mode == "global")
        global_stats = None
        norm_cfg_for_build = self.config.norm

        if use_global:
            if want_train:
                # compute from train (labels trimmed by gap_hours)
                cut = self._end_train - pd.Timedelta(hours=gap_hours)
                y_tr, en_tr, ex_tr = self._y.loc[self._start_train:cut], self._endog_df.loc[self._start_train:cut], self._exog_df.loc[self._start_train:cut]
                if y_tr.empty:
                    raise ValueError("Train slice after applying gap_hours is empty; reduce gap_hours or adjust ranges.")
                global_stats = compute_global_stats(
                    norm_type=norm_cfg_for_build.norm_type,
                    endog_tr=en_tr, exog_tr=ex_tr, y_tr=y_tr, ddof=getattr(self.config.norm, "std_ddof", 0)
                )
                self._norm_global_stats = global_stats
            elif getattr(self, "_norm_global_stats", None) is not None:
                # reuse cached
                global_stats = self._norm_global_stats
            else:
                # raise error
                raise ValueError("norm.mode='global' but no train slice and no cached stats. "
                                "Build a train loader first (to compute stats) or switch to per_sample/none.")

        # dataloader options / seeds
        bs, nw = self.config.data.batch_size, self.config.data.num_workers
        pin, shuf = self.config.data.pin_memory, self.config.data.shuffle_train
        seed_train = derive_seed(self.config.seed, "train_loader") if (self.config.seed is not None and want_train) else None
        seed_val   = derive_seed(self.config.seed, "val_loader")   if (self.config.seed is not None and want_val)   else None

        train_loader = val_loader = None
        parts = [f"T_in={T_in}, T_out={T_out}"]

        if want_train:
            cut = self._end_train - pd.Timedelta(hours=gap_hours)
            y_tr  = self._y.loc[self._start_train:cut]
            en_tr = self._endog_df.loc[self._start_train:cut]
            ex_tr = self._exog_df.loc[self._start_train:cut]
            yl_tr = self._y_levels.loc[self._start_train:cut]
            ds_tr = self._dataset_from_slices(y_tr, en_tr, ex_tr, yl_tr, global_stats=global_stats, norm_cfg=norm_cfg_for_build)
            train_loader = self._build_loader(ds_tr, batch_size=bs, shuffle=shuf, num_workers=nw, pin_memory=pin, seed=seed_train)
            parts.append(f"train(n={len(ds_tr)}, n_batches={len(train_loader)}, start={self._start_train}, end={cut})")

        if want_val:
            if want_train:
                assert self._start_val >= cut + pd.Timedelta(hours=gap_hours + 1), "Leakage of data. Validation start must be after training end"
            y_v, en_v, ex_v, yl_v = _slice_with_history(self._y, self._endog_df, self._exog_df,
                                                        start=self._start_val, end=self._end_val, T_in=T_in, y_levels=self._y_levels)
            ds_v = self._dataset_from_slices(y_v, en_v, ex_v, yl_v, global_stats=global_stats, norm_cfg=norm_cfg_for_build)
            val_loader = self._build_loader(ds_v, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, seed=seed_val)
            parts.append(f"val(n={len(ds_v)}, n_batches={len(val_loader)}, start={self._start_val}, end={self._end_val})")

        logger.info("[loaders] loaders built: " + (" | ".join(parts) if parts else "none")
                    + f" | bs={bs}, norm_mode={'global' if (use_global and global_stats is not None) else 'none/per-sample'}, "
                    + f"norm_type={norm_cfg_for_build.norm_type}, gap_hours={gap_hours}, stride={self.config.data.stride}")

        return train_loader, val_loader

    def describe_dataset(self, max_cols: int = 20, return_dict: bool = False, to_logger: bool = True):
        """
        Summarize prepared features, normalization policy, and model I/O sizes.

        Parameters
        ----------
        max_cols : int
            Truncate long column lists for readability.
        return_dict : bool
            If True, return a dict instead of a formatted string.
        to_logger : bool
            If True, also log the report via `self._logger.info`.

        Returns
        -------
        str | dict
            Human-readable report or a metadata dictionary.
        """
        # Ensure features exist
        if self._endog_df is None or self._exog_df is None or self._y is None:
            self.generate_vars()

        idx = self._y.index
        freq = pd.infer_freq(idx) or "unknown"
        endog_cols = list(self._endog_df.columns)
        exog_cols  = list(self._exog_df.columns)

        # Normalization info
        norm = getattr(self.config, "norm", None)
        norm_mode = getattr(norm, "mode", "none") if norm else "none"
        endog_skip = set(getattr(norm, "endog_skip_cols", ())) if norm else set()
        exog_skip  = set(getattr(norm, "exog_skip_cols",  ())) if norm else set()
        norm_endog = [c for c in endog_cols if c not in endog_skip]
        norm_exog  = [c for c in exog_cols  if c not in exog_skip]

        # Time-feature names we add in generate_vars()
        known_time_feats = {
            "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos","wss_sin","wss_cos","is_cold_season"
        }
        time_feats = [c for c in exog_cols if c in known_time_feats]

        # Climate base features
        climate_feats = [c for c in exog_cols if c not in known_time_feats]

        # Configured features
        feats = self.config.features
        exog_cfg = tuple(getattr(feats, "exog_vars", ()) or ())
        hour_avgs = tuple(getattr(feats, "hour_averages", ()) or ())
        endog_lags = tuple(getattr(feats, "endog_hour_lags", ()) or ())
        incl_ex = getattr(feats, "include_exog_lags", False)
        uses_diff = bool(getattr(feats, "use_differences", False))

        # Model I/O sizes
        mc = self.config.model
        T_in  = int(mc.input_len)
        T_out = int(mc.output_len)
        enc_in = int(self._endog_df.shape[1] + self._exog_df.shape[1])
        C_endog_fut = len(self._endog_df.columns) - len(self._endog_vars_not_for_future)
        use_ar = getattr(mc, "use_ar", "none")
        add_prev = 1 if use_ar in ["prev", "24h"] else 0
        dec_in = int(add_prev + C_endog_fut + self._exog_df.shape[1])

        # Splits (if already set via make_loaders)
        train_range = "—"
        val_range   = "—"
        if self._end_train is not None:
            train_start = self._start_train or idx.min()
            train_range = f"{pd.Timestamp(train_start)} → {pd.Timestamp(self._end_train)}"
        if self._end_val is not None:
            val_range = f"{pd.Timestamp(self._start_val)} → {pd.Timestamp(self._end_val)}"

        def _fmt(lst):
            if not lst: 
                return "—"
            if len(lst) <= max_cols: 
                return ", ".join(map(str, lst))
            head = ", ".join(map(str, lst[:max_cols]))
            return f"{head}, … (+{len(lst)-max_cols} more)"

        summary = {
            "Rows:": int(len(idx)),
            "Date range:": f"{pd.Timestamp(idx.min())} → {pd.Timestamp(idx.max())}",
            "Train range:": train_range,
            "Val range:": val_range,
            "Frequency:": freq,
            "Target:": "y",
            "Uses differences:": uses_diff,
            "Endog/Exog counts:": f"{len(endog_cols)} / {len(exog_cols)}",
            "Endog columns:": _fmt(endog_cols),
            "Exog columns:": _fmt(exog_cols),
            "Time features:": f"{len(time_feats)} | " + _fmt(time_feats),
            "Climate features:": f"{len(climate_feats)} | " + _fmt(climate_feats),
            "Decoder safe endog:": f"{len(endog_cols) - len(self._endog_vars_not_for_future)} | " + _fmt([c for c in endog_cols if c not in self._endog_vars_not_for_future]),
            "Decoder blocked endog:": f"{len(self._endog_vars_not_for_future)} | " + _fmt(self._endog_vars_not_for_future),
            "Normalization:": f"{norm_mode} (endog={len(norm_endog)}, exog={len(norm_exog)})",
            "Model I/O:": f"T_in={T_in}, T_out={T_out} | enc_in={enc_in}, dec_in={dec_in}",
            "Configured features:": (
                f"exog_vars={_fmt(exog_cfg)} | hour_averages={_fmt(hour_avgs)} | endog_hour_lags={_fmt(endog_lags)} (include_exog_lags={incl_ex})"
            ),
            "Device:": str(self.device),
            "Seed:": self.config.seed,
        }

        # Align into a clean text block 
        max_key_len = max(len(k) for k in summary.keys())
        lines = [f"{k.ljust(max_key_len)} {v}" for k, v in summary.items()]
        text = "\n".join(lines)

        if to_logger:
            self._logger.info("[dataset]\n" + text)

        if return_dict:
            return summary
        return text

    
    # ------------------------------
    # Training
    # ------------------------------

    def _build_model(self, *, silent: bool = False) -> EncDecLSTM:
        """
        Instantiate `EncDecLSTM` with encoder/decoder input sizes derived from
        prepared frames and AR mode. Also computes and stores parameter counts.

        Returns
        -------
        EncDecLSTM
            Model moved to `self.device`.
        """
        if self._endog_df is None or self._exog_df is None:
            # Ensure features are generated before building the model
            self.generate_vars()

        C_endog = int(self._endog_df.shape[1])
        C_endog_fut = C_endog - len(self._endog_vars_not_for_future)
        D_exog  = int(self._exog_df.shape[1]) if self._exog_df is not None else 0

        input_size_enc = C_endog + D_exog
        add_prev = 1 if self.config.model.use_ar in ["prev", "24h"] else 0
        input_size_dec = add_prev + C_endog_fut + D_exog

        model = EncDecLSTM(
            input_size_enc=input_size_enc,
            input_size_dec=input_size_dec,
            hidden_size=self.config.model.hidden_size,
            n_layers=self.config.model.num_layers,
            head=self.config.model.head,
            dropout=self.config.model.dropout,
            use_ar=self.config.model.use_ar,
        )
        nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._n_params = nparams

        if not silent:
            self._logger.info(f"[model] EncDecLSTM(T_in={self.config.model.input_len}, T_out={self.config.model.output_len}, "
                            f"hidden={self.config.model.hidden_size}, "
                            f"n_layers={self.config.model.num_layers}, "
                            f"head={self.config.model.head}, "
                            f"dropout={self.config.model.dropout}, "
                            f"in_enc={input_size_enc}, in_dec={input_size_dec}, "
                            f"use_ar={self.config.model.use_ar}, n_learnable_params={nparams})")
        
        return model.to(self.device)

    def fit(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            trial: Optional[optuna.trial.Trial] = None,
        ) -> dict[str, Any]:
        """
        Train the model under the current configuration.

        - Seeds TF scheduler deterministically from `config.seed` (if set).
        - Stores: trainer, training history, final model weights, losses.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader | None
        trial : optuna.Trial | None

        Returns
        -------
        dict
            See `LSTMTrainer.train` return schema.
        """
        # Build model
        self._model = self._build_model()

        # Seed for teacher forcing
        tf_seed = None
        if self.config.seed is not None:
            tf_seed = derive_seed(self.config.seed, "trainer")

        # Trainer
        trainer = LSTMTrainer(
            self._model,
            self.device,
            logger=self._logger,
            norm_cfg=self.config.norm,
            global_stats=self._norm_global_stats,
            target_is_diff24=self.config.features.use_differences,
            tf_seed=tf_seed,
        )

        # Train
        use_tf = self.config.model.use_ar in ["prev", "24h"]
        out = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=self.config.train.n_epochs,
            lr=self.config.train.learning_rate,
            patience=self.config.train.patience,
            es_rel_delta=self.config.train.es_rel_delta,
            tf_mode=self.config.train.tf_mode,
            tf_drop_epochs=self.config.train.tf_drop_epochs,
            use_lr_drop=self.config.train.use_lr_drop,
            lr_drop_epoch=self.config.train.lr_drop_epoch,
            lr_drop_factor=self.config.train.lr_drop_factor,
            es_start_epoch=self.config.train.es_start_epoch,
            grad_clip_max_norm=self.config.train.grad_clip_max_norm,
            weight_decay=self.config.train.weight_decay,
            trial=trial,
            max_walltime_sec=self.config.train.max_walltime_sec,
            use_tf=use_tf
        )

        # Store diagnostics
        self._trainer = trainer
        self._history_df = trainer.history_as_dataframe()
        self._model = trainer.model
        self._train_losses_orig = trainer.train_losses_orig
        self._val_losses_orig   = trainer.val_losses_orig

        # --- Logging summary (robust to ES/no-ES) ---
        if val_loader is not None:
            best_val = out.get("best_val_loss_orig")
            best_epoch = out.get("best_epoch")  # already 1-based per trainer return
            td = out.get("duration_until_best", None)
            secs = (td.total_seconds() if isinstance(td, pd.Timedelta) else float(td or 0))
            mins, secs_rem = divmod(int(round(secs)), 60)
            to_log_0 = f"Best val loss (orig): {best_val:.2f}" if best_val is not None else "Best val loss (orig): n/a"
            to_log_1 = f"Best epoch: {best_epoch:d}" if isinstance(best_epoch, int) else "Best epoch: n/a"
            to_log_2 = f"Time until best: {mins:02d}'{secs_rem:02d}\""
            avg_nb = out.get("avg_near_best")
            to_log_3 = f"Avg val loss near best: {avg_nb:.2f}" if avg_nb is not None else "Avg val loss near best: n/a"
        else:
            last_train = out.get("last_train_loss_orig")
            td = out.get("total_duration", None)
            secs = (td.total_seconds() if isinstance(td, pd.Timedelta) else float(td or 0))
            mins, secs_rem = divmod(int(round(secs)), 60)
            n_ep = out.get("n_epochs")
            to_log_0 = f"Last train loss (orig): {last_train:.2f}" if last_train is not None else "Last train loss (orig): n/a"
            to_log_1 = f"Total epochs: {n_ep:d}" if isinstance(n_ep, int) else "Total epochs: n/a"
            to_log_2 = f"Total training time: {mins:02d}'{secs_rem:02d}\""
            avg_te = out.get("avg_train_near_end")
            to_log_3 = f"Avg train loss near end: {avg_te:.2f}" if avg_te is not None else "Avg train loss near end: n/a"

        self._logger.info(f"[train] Training finished. {to_log_0}. {to_log_1}. {to_log_2}. {to_log_3}.")

        return out
    
    @property
    def training_history(self):
        return getattr(self, "_history_df", None)

    def sanity_overfit_one_batch(
        self,
        train_loader: DataLoader,
        *,
        max_epochs: int = 300,
        tol_rel_drop: float = 0.8,
        restore_weights: bool = True,
        n_samples: Optional[int] = None,
        gen_seed: Optional[int] = None,    # None -> derive from config.seed
    ):
        """
        Quick overfit test on a tiny deterministic subset to catch pipeline bugs.
        Procedure:
        - Sample a fixed subset (size `n_samples` or batch size).
        - Train for up to `max_epochs` with ES disabled.
        - Check relative drop in training loss vs. after first epoch.

        Parameters
        ----------
        train_loader : DataLoader
        max_epochs : int, default=300
        tol_rel_drop : float, default=0.8
            Required relative improvement for PASS (e.g., 0.95 = 95% drop).
        restore_weights : bool, default=True
            If True, restore original model params after the test.
        n_samples : int | None
        gen_seed : int | None

        Returns
        -------
        dict
            {passed: bool, final_loss: float, history: pd.DataFrame, indices: List[int]}
        """
        # -- capture original state/config
        init_state = None
        if restore_weights and hasattr(self, "_model") and self._model is not None:
            init_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}

        old_epochs  = self.config.train.n_epochs
        old_tf_mode = self.config.train.tf_mode
        old_patience = self.config.train.patience

        # -- build fixed subset
        ds = train_loader.dataset
        n = len(ds)
        if n == 0:
            raise ValueError("[sanity] train dataset is empty.")
        bs = n_samples or (train_loader.batch_size or min(32, n))
        bs = min(bs, n)

        seed = gen_seed if gen_seed is not None else derive_seed(self.config.seed or 0, "sanity_one_batch")
        g = torch.Generator().manual_seed(int(seed))
        idxs = torch.randperm(n, generator=g)[:bs].tolist()

        tiny_dl = DataLoader(
            Subset(ds, idxs),
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # -- temporary training overrides
        self.config.train.n_epochs = int(max_epochs)
        self.config.train.tf_mode  = "linear"
        self.config.train.patience = 0  

        self._logger.info(f"[sanity] Overfitting one batch (bs={bs}, epochs={self.config.train.n_epochs}, tol_rel_drop={tol_rel_drop}) idxs[:8]={idxs[:8]}")

        # -- run the fit on the tiny loader (no val)
        self.fit(tiny_dl, val_loader=None)

        # -- collect results
        hist = self.training_history
        final_orig = float(hist["train_orig"].iloc[-1])
        start_orig = float(hist["train_orig"].iloc[0])  # baseline from epoch 1 (close enough)
        rel_improve = (start_orig - final_orig) / max(start_orig, 1e-12)
        passed_rel = rel_improve >= tol_rel_drop  # 95% drop

        self._logger.info(f"[sanity] Relative improvement on train(orig) is {rel_improve:.1%}; threshold is {tol_rel_drop:.1%} → {'PASS' if passed_rel else 'FAIL'}")

        # -- restore model + config
        if restore_weights and init_state is not None and hasattr(self, "_model"):
            self._model.load_state_dict(init_state)
        self.config.train.n_epochs = old_epochs
        self.config.train.tf_mode  = old_tf_mode
        self.config.train.patience = old_patience

        return {"passed": passed_rel, "final_loss": final_orig, "history": hist, "indices": idxs}

    def _build_single_cutoff_loader(self, cutoff: pd.Timestamp) -> DataLoader:
        """
        Build a val loader with exactly one item whose encoder cutoff is `cutoff`
        (i.e., last observed time is `cutoff`, horizon starts at cutoff+1h).
        """
        if not isinstance(cutoff, pd.Timestamp):
            raise ValueError("cutoff must be a pandas.Timestamp")

        # Ensure features exist
        if self._endog_df is None or self._exog_df is None or self._y is None or self._y_levels is None:
            self.generate_vars()

        T_in  = int(self.config.model.input_len)
        T_out = int(self.config.model.output_len)

        # Validate there is enough history and future for the requested cutoff
        idx_min = self._y.index.min()
        idx_max = self._y.index.max()
        need_hist_start = cutoff - pd.Timedelta(hours=T_in - 1)
        need_future_end = cutoff + pd.Timedelta(hours=T_out)
        if need_hist_start < idx_min:
            raise ValueError(f"Not enough history for cutoff {cutoff}. Need data starting at {need_hist_start}, "
                            f"but earliest available is {idx_min}.")
        if need_future_end > idx_max:
            raise ValueError(f"Not enough future for cutoff {cutoff}. Need data through {need_future_end}, "
                            f"but latest available is {idx_max}.")

        # Build the exact slice that yields ONE window
        start_val = cutoff + pd.Timedelta(hours=1)          # first forecast time
        end_val   = cutoff + pd.Timedelta(hours=T_out)      # last forecast time (inclusive)

        y_blk, en_blk, ex_blk, yl_blk = _slice_with_history(
            self._y, self._endog_df, self._exog_df,
            start=start_val, end=end_val, T_in=T_in, y_levels=self._y_levels
        )

        # Use cached global stats if available (required for norm.mode == "global")
        if getattr(self.config.norm, "mode", "none") == "global" and getattr(self, "_norm_global_stats", None) is None:
            raise RuntimeError("norm.mode='global' but no cached stats. Train first to compute global stats.")

        ds = self._dataset_from_slices(
            y_blk, en_blk, ex_blk, yl_blk,
            global_stats=self._norm_global_stats,
            norm_cfg=self.config.norm
        )
        if len(ds) != 1:
            # By construction, this should always be 1; guard just in case index spacing is irregular.
            raise RuntimeError(f"Expected single-window dataset, got {len(ds)} windows.")

        # Deterministic, single-item loader
        return self._build_loader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            seed=None,
        )


    def forward(
        self,
        val_loader: DataLoader | None = None,
        cutoff: pd.Timestamp | None = None,
        alias: str = "LSTM"
    ) -> pd.DataFrame:
        """
        Generate predictions on original target scale.

        Modes
        -----
        - Provide `val_loader` (from `make_loaders`) to batch-predict; returns
          columns: ['unique_id','ds','cutoff', alias].
        - Provide `cutoff` to predict a *single* horizon; returns
          ['unique_id','ds', alias].

        Parameters
        ----------
        val_loader : DataLoader | None
        cutoff : pd.Timestamp | None
        alias : str
            Output column name for predictions.

        Returns
        -------
        pd.DataFrame
            Predictions sorted by (cutoff, ds) or by ds for single-cutoff mode.
        """
        if (val_loader is None) == (cutoff is None):
            raise ValueError("Provide exactly one of: val_loader OR cutoff.")

        # Build a temporary loader if a cutoff is given
        drop_cutoff_column = False
        if cutoff is not None:
            val_loader = self._build_single_cutoff_loader(cutoff)
            drop_cutoff_column = True

        if not hasattr(self, "_model"):
            raise RuntimeError("Model not trained. Call .fit(...) first.")
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            raise RuntimeError("Trainer not available. Train with .fit(...) before predicting.")

        ds = val_loader.dataset
        if not hasattr(ds, "starts") or not hasattr(ds, "index") or not hasattr(ds, "T_in") or not hasattr(ds, "T_out"):
            raise ValueError("val_loader.dataset must be a TimeSeriesDataset built by this pipeline.")

        self._model.eval()
        device = self.device
        nb = (device.type == "cuda")

        unique_id = getattr(self, "unique_id", getattr(self, "_unique_id", None))
        if unique_id is None:
            raise RuntimeError("unique_id not set on the pipeline.")

        records = []
        starts: List[int] = ds.starts
        idx_series: pd.Index = ds.index # datetime index of the original series
        T_in: int = ds.T_in
        T_out: int = ds.T_out

        seen = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device, non_blocking=nb) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()}

                # Always do AR decoding at inference
                y_hat = self._model(batch, teacher_forcing=0.0)  # (B, T_out, 1)

                # Bring back to original scale using trainer helpers (handles global/per-sample + Δ24)
                y_true = batch["y_fut"]
                y_hat_orig, _ = trainer._pair_to_orig_levels(y_hat, y_true, batch)  # (B, T_out, 1)
                y_hat_np = y_hat_orig.detach().cpu().numpy()

                B = y_hat_np.shape[0]
                for i in range(B):
                    j = seen + i
                    s = starts[j]
                    e = s + T_in
                    k = e + T_out

                    cutoff_ts = pd.Timestamp(idx_series.iloc[e - 1])
                    horizon_ts = idx_series.iloc[e:k]
                    preds = y_hat_np[i, :, 0].tolist()

                    if drop_cutoff_column:
                        # single-cutoff mode: no 'cutoff' column
                        for ts, yhat in zip(horizon_ts, preds):
                            records.append({
                                "unique_id": unique_id,
                                "ds": pd.Timestamp(ts),
                                f"{alias}": float(yhat),
                            })
                    else:
                        for ts, yhat in zip(horizon_ts, preds):
                            records.append({
                                "unique_id": unique_id,
                                "ds": pd.Timestamp(ts),
                                "cutoff": cutoff_ts,  # keep double 't' to match your earlier schema
                                f"{alias}": float(yhat),
                            })
                seen += B

        df = pd.DataFrame.from_records(records)
        if not df.empty:
            sort_cols = ["ds"] if drop_cutoff_column else ["cutoff", "ds"]
            df.sort_values(sort_cols, inplace=True, kind="mergesort")
            df.reset_index(drop=True, inplace=True)
        return df

    def cross_validation(
        self,
        *,
        test_size: int,
        end_test: Optional[pd.Timestamp] = None,
        step_size: int = 1,
        input_size: Optional[int] = None,   # hours of training history
        levels: List[int] = None,           # not supported for LSTM; will warn if provided
        refit: Union[bool, int] = True,     # True = refit every window; False = fit once; int = refit every k windows
        verbose: bool = True,
        alias: str = "LSTM",
    ) -> pd.DataFrame:
        """
        Rolling-window cross-validation.

        For each window:
          - train on [start .. cutoff], predict next `h` steps,
          - merge with ground truth,
          - return one row per horizon point.

        Parameters
        ----------
        test_size : int
            Total hours to evaluate at the end of the series.
        end_test : pd.Timestamp | None
            Last timestamp to consider; defaults to the latest available.
        step_size : int, default=1
            Spacing (in hours) between consecutive cutoffs.
        input_size : int | None
            Training history length (hours) for each window; defaults to all past.
        levels : list[int] | None
            Ignored (no PI support); a warning is logged if provided.
        refit : bool | int, default=True
            True or None: refit every window; False: fit once; int k: refit every k windows.
        verbose : bool, default=True
            Show progress bar with `tqdm`.
        alias : str, default="LSTM"
            Prediction column name.

        Returns
        -------
        pd.DataFrame
            Columns: [unique_id, ds, y, alias, cutoff].

        Raises
        ------
        ValueError
            On inconsistent arguments or insufficient data coverage.
        """
        # ---------- sanity & prep ----------
        if levels:
            self._logger.warning("[cv] `levels` were provided but LSTM does not produce intervals; ignoring.")

        h = self.config.model.output_len # TO DO: add support for a multiple using regressive predictions
        if h <= 0 or test_size <= 0 or step_size <= 0:
            raise ValueError("h, test_size, and step_size must be positive integers.")
        if (test_size - h) % step_size != 0:
            raise ValueError("`test_size - h` must be a multiple of `step_size`.")

        # Make sure features are ready
        if self._y is None or self._endog_df is None or self._exog_df is None or self._y_levels is None:
            if verbose:
                self._logger.info("[cv] Features not prepared; calling generate_vars() now.")
            self.generate_vars()

        # Build relative offsets for window cutoffs
        steps = list(range(-test_size, -h + 1, step_size))  # last cutoff is end_test - h

        # Determine end_test
        idx_all = self._y.index  # trimmed to rows usable after feature engineering
        if end_test is None:
            end_test = idx_all.max()
        else:
            if not isinstance(end_test, pd.Timestamp):
                raise ValueError("end_test must be a pandas.Timestamp.")
            if end_test not in idx_all:
                raise ValueError(f"end_test ({end_test}) must be present in the prepared target index.")

        # Check if the given input size (amount of data for each train) is valid
        if input_size is not None:
            if not isinstance(input_size, int) or input_size <= 0:
                raise ValueError("input_size must be a positive integer (hours).")
            first_cutoff = end_test + pd.Timedelta(hours=steps[0])
            first_start_train = first_cutoff - pd.Timedelta(hours=input_size) + pd.Timedelta(hours=1)
            if first_start_train < self._y.index.min():
                raise ValueError(
                    f"input_size ({input_size} hours) is too large for the given end test ({end_test}). "
                    f"Training would start at {first_start_train}, but the earliest available data in .prepared_data is {self._y.index.min()}."
                )

        all_results = []
        prev_pipeline: Optional["LSTMPipeline"] = None

        iterator = tqdm(steps, disable=not verbose, desc="CV windows", leave=True)
        for i, offset in enumerate(iterator):
            cutoff = end_test + pd.Timedelta(hours=offset)

            # training slice for this window
            if input_size is not None:
                start_train = cutoff - pd.Timedelta(hours=input_size) + pd.Timedelta(hours=1)
            else:
                start_train = self._y.index.min()
            end_train = cutoff

            # decide whether to refit
            do_refit = (
                i == 0
                or (isinstance(refit, int) and not isinstance(refit, bool) and i % int(refit) == 0)
                or (refit is True)
                or (refit is None)
            )

            if do_refit or prev_pipeline is None:
                # Build a fresh pipeline but reuse precomputed features to avoid recomputing
                pipeline = LSTMPipeline(
                    target_df=self._target_df,
                    config=copy.deepcopy(self.config), 
                    aux_df=self._aux_df,
                    logger=self._logger,
                )
                # copy internals (features already generated)
                pipeline._target_plus_aux_df = self._target_plus_aux_df
                pipeline.unique_id = self.unique_id
                pipeline._endog_df = self._endog_df
                pipeline._exog_df = self._exog_df
                pipeline._y = self._y
                pipeline._y_levels = self._y_levels
                pipeline._endog_vars_not_for_future = self._endog_vars_not_for_future

                # Train (final mode: no early stopping)
                train_loader, _ = pipeline.make_loaders(
                    start_train=start_train,
                    end_train=end_train,
                    start_val=None,
                    end_val=None,
                )
                pipeline.fit(train_loader, val_loader=None)
                prev_pipeline = pipeline
            else:
                pipeline = prev_pipeline

            # forecast h steps after cutoff
            fc = pipeline.forward(cutoff=cutoff, alias=alias)  # returns columns: [unique_id, ds, LSTM]

            # ground truth for (cutoff, cutoff + h]
            mask = (self._target_plus_aux_df.index > cutoff) & (self._target_plus_aux_df.index <= cutoff + pd.Timedelta(hours=h))
            val_df = self._target_plus_aux_df.loc[mask, ["y"]].copy()
            val_df["unique_id"] = self.unique_id
            val_df = val_df.reset_index().rename(columns={"index": "ds"})

            # merge forecast & truth
            merged = (
                val_df.merge(fc[["unique_id", "ds", alias]], on=["unique_id", "ds"], how="left")
                    .assign(cutoff=cutoff)
            )
            all_results.append(merged)

            # Make tqdm show progress even if nothing printed
            try:
                iterator.refresh()
            except Exception:
                pass

        result = pd.concat(all_results, ignore_index=True)

        # optional CV metadata for traceability
        self._last_cv_metadata = {
            "pipeline_class": self.__class__.__name__,
            "unique_id": self.unique_id,
            "cv_params": {
                "h": h,
                "test_size": test_size,
                "end_test": str(end_test),
                "step_size": step_size,
                "input_size": input_size if input_size is not None else self.config.model.input_len,
                "refit": refit,
                "alias": alias,
            },
        }
        return result
    
    @property
    def n_params(self):
        """
        Number of trainable parameters in the current model (int) or None if
        the model has not been built yet.
        """
        return self._n_params

    def describe_model(self, *, to_logger: bool = True, return_dict: bool = False):
        """
        Summarize architecture, input sizes, AR mode, dropout locations,
        and parameter counts (estimated & exact).
        """
        # Ensure features and model are built
        if self._endog_df is None or self._exog_df is None:
            self.generate_vars()
        if getattr(self, "_model", None) is None:
            self._model = self._build_model(silent=True)

        mc = self.config.model
        endog_cols = list(self._endog_df.columns)
        not_for_future = list(self._endog_vars_not_for_future or [])
        dec_endog_cols = [c for c in endog_cols if c not in not_for_future]

        C_endog     = int(self._endog_df.shape[1])
        D_exog      = int(self._exog_df.shape[1])
        C_endog_fut = len(dec_endog_cols)
        enc_in      = C_endog + D_exog
        use_ar = getattr(mc, "use_ar", "none")
        add_prev = 1 if use_ar in ["prev", "24h"] else 0
        dec_in   = add_prev + C_endog_fut + D_exog

        # ---- Estimated params ----
        def _lstm_params(I, H, L):
            total = 0
            for layer in range(L):
                in_size = I if layer == 0 else H
                total += 4 * H * (in_size + H) + 8 * H
            return total

        def _head_params(H, head):
            if head == "linear":
                return H * 1 + 1
            elif head == "mlp":
                return (H * H + H) + (H * 1 + 1)
            return 0

        enc_est = _lstm_params(enc_in, mc.hidden_size, mc.num_layers)
        dec_est = _lstm_params(dec_in, mc.hidden_size, mc.num_layers)
        head_est = _head_params(mc.hidden_size, mc.head)
        total_est = enc_est + dec_est + head_est

        # ---- Exact params ----
        exact_enc   = sum(p.numel() for p in self._model.encoder.parameters() if p.requires_grad)
        exact_dec   = sum(p.numel() for p in self._model.decoder.parameters() if p.requires_grad)
        exact_head  = sum(p.numel() for p in self._model.head.parameters() if p.requires_grad)
        exact_total = exact_enc + exact_dec + exact_head

        # ---- Dropout where? ----
        drop_locs = []
        if mc.dropout > 0.0:
            if mc.num_layers > 1:
                drop_locs.append("between LSTM layers")
            if mc.head == 'linear':
                drop_locs.append("before linear head")
            elif mc.head == 'mlp':
                drop_locs.append("within MLP head after activation")
        else: 
            drop_locs.append("none")


        # ---- Organize into a dict ----
        summary = {
            "Architecture:": "EncDecLSTM",
            "Device:": str(self.device),
            "Seq lengths:": f"T_in={mc.input_len}, T_out={mc.output_len}",
            "Encoder input size:": f"{enc_in} (endog({C_endog}) + exog({D_exog}))",
            "Decoder input size:": f"{dec_in} "
                       f"({'prev_pred(1) + ' if add_prev else ''}"
                       f"safe_endog({C_endog_fut}) + exog({D_exog}))",
            "Endog/Exog counts:": f"{C_endog} / {D_exog}",
            "Decoder safe endog:": f"{len(dec_endog_cols)} | " + ", ".join(dec_endog_cols) if dec_endog_cols else "—",
            "Decoder blocked endog:": f"{len(not_for_future)} | " + ", ".join(not_for_future) if not_for_future else "—",
            "Hidden size:": mc.hidden_size,
            "Layers:": mc.num_layers,
            "Dropout:": mc.dropout,
            "Dropout locations:": " | ".join(drop_locs),
            "Head:": mc.head,
            "Params (estimated):": f"enc={enc_est:,} | dec={dec_est:,} | head={head_est:,} | total={total_est:,}",
            "Params (exact):": f"enc={exact_enc:,} | dec={exact_dec:,} | head={exact_head:,} | total={exact_total:,}",
        }

        max_key_len = max(len(k) for k in summary.keys())
        lines = [f"{k.ljust(max_key_len)} {v}" for k, v in summary.items()]
        text = "\n".join(lines)

        if to_logger:
            self._logger.info("[model]\n" + text)

        if return_dict:
            return summary
        return text


# ─────────────────────────────────────────────────────────────────────────────
# For Optuna objective function
# ─────────────────────────────────────────────────────────────────────────────

def fit_for_optuna(
    *,
    trial: optuna.trial.Trial,
    target_df: pd.DataFrame,
    aux_df: pd.DataFrame,
    cfg: LSTMRunConfig,
    start_train: pd.Timestamp | None, end_train: pd.Timestamp | None,
    start_val: pd.Timestamp | None, end_val: pd.Timestamp | None,
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    """
    Fit the LSTM model for Optuna hyperparameter optimization.
    """
    logger = logger or logging.getLogger(__name__)
    pipe = LSTMPipeline(
        target_df=target_df,
        aux_df=aux_df,
        config=cfg,
        logger=logger,
    )
    # loaders (computes global stats if needed)
    train_loader, val_loader = pipe.make_loaders(
        start_train=start_train, end_train=end_train,
        start_val=start_val,     end_val=end_val,
    )
    # train with pruning support (pipeline forwards `trial` to trainer)
    out = pipe.fit(train_loader, val_loader=val_loader, trial=trial)
    out_optuna: Dict[str, Any] = {}
    out_optuna['n_params'] = pipe.n_params
    trainer = pipe._trainer

    best_val = out.get('best_val_loss_orig')
    out_optuna['best_val'] = float(best_val) if best_val is not None else None

    best_epoch = out.get('best_epoch')
    out_optuna['best_epoch'] = int(best_epoch) if best_epoch is not None else None
    
    dur = out.get('duration_until_best')
    out_optuna['duration_until_best'] = dur

    out_optuna['duration_until_best_s'] = float(dur.total_seconds()) if dur is not None else None

    avg_near_best = out.get('avg_near_best')
    out_optuna['avg_near_best'] = float(avg_near_best) if avg_near_best is not None else None
    
    last_val = trainer.val_losses_orig[-1] if trainer.val_losses_orig else None
    out_optuna['last_val'] = float(last_val) if last_val is not None else None

    return out_optuna

def config_builder(cfg_dict: Mapping['str', Any]) -> LSTMRunConfig:
    """
    Build an LSTMRunConfig from a generic dict.
    """
    return LSTMRunConfig(
        model = ModelConfig.from_dict(cfg_dict.get("model", {})),
        training = TrainConfig.from_dict(cfg_dict.get("training", {})),
        data = DataConfig.from_dict(cfg_dict.get("data", {})),
        features = FeatureConfig.from_dict(cfg_dict.get("features", {})),
        norm = NormalizeConfig.from_dict(cfg_dict.get("norm", {})),
        seed = cfg_dict.get("seed", None)
    )

