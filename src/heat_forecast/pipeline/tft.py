# tft_pipeline.py
from __future__ import annotations
import torch
from torch import cuda
from dataclasses import dataclass, field, asdict
import math
from typing import Optional, Tuple, Literal, Dict, Any, List, Mapping
import logging
import numpy as np
import pandas as pd
import optuna
from tqdm.notebook import tqdm

from pathlib import Path
import pickle

from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler

import torch.nn as nn

from heat_forecast.pipeline.lstm import set_global_seed, is_for_endog_fut

# ---------------------------------------------------
# Configs
# ---------------------------------------------------

@dataclass
class TFTModelConfig:
    """
    Architecture/fit hyperparameters for Darts' TFTModel.
    """
    input_chunk_length: int = 168
    output_chunk_length: int = 24
    hidden_size: int = 64
    lstm_layers: int = 1
    dropout: float = 0.1
    num_attention_heads: int = 4
    torch_device_str: Optional[str] = "auto"  # "cuda" | "cpu" | "auto" | None (-> auto)
    show_warnings: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class DataConfig:
    """
    Data and split options (similar semantics to your pipeline).
    """
    stride: int = 1
    batch_size: int = 64

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class FeatureConfig:
    """
    Endog/exog feature engineering switches.

    Conventions
    -----------
    • We treat "future-safe" features as future_covariates (deterministic calendar or known-in-advance exog).
    • All others go to past_covariates.
    """
    exog_vars: Tuple[str, ...] = ("temperature",)
    endog_hour_lags: Tuple[int, ...] = ()
    include_exog_lags: bool = True
    time_vars: Tuple[str, ...] = ("hod", "dow", "moy", "wss")

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class TrainConfig:
    """
    Training hyperparameters.
    """
    n_epochs: int = 10
    lr: float = 1e-3
    gradient_clip_val: Optional[float] = None
    use_es: bool = True   
    es_patience: int = 3
    es_min_delta: float = 0.0
    es_rel_min_delta: float = 0.0
    es_warmup_epochs: int = 0
    loss_fn_str: Optional[str] = "L1"  # "L1" | "MSE" | None (-> L1)

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class NormalizeConfig:
    """
    Scaling policy (train-only, then apply to val/test).
    """
    scaler: Optional[Any] = None # to pass to Darts' Scaler()
    scale_time_vars: bool = False
    verbose: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class TFTRunConfig:
    model: TFTModelConfig = field(default_factory=TFTModelConfig)
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

# ---------------------------------------------------
# PL Callbacks
# ---------------------------------------------------
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc

class InspectMetricsCallback(plc.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        print("Current callback_metrics keys:", trainer.callback_metrics.keys())
        for k, v in trainer.callback_metrics.items():
            print(f"  {k}: {v}")

class FitTracker(plc.Callback):
    def __init__(self, avg_window=5, best_idx_in_window=1):
        if not (isinstance(avg_window, int) and avg_window >= 0):
            raise ValueError("avg_window must be a non-negative integer.")
        if not (isinstance(best_idx_in_window, int) and 0 <= best_idx_in_window < avg_window):
            raise ValueError("best_idx_in_window must be a non-negative integer less than avg_window.")
        self.avg_window = avg_window
        self.best_idx_in_window = best_idx_in_window
        self.start = None
        self.val_losses = [] # unscaled when available, else scaled
        self.epochs = []
        self.best_epoch = None
        self.best_val = math.inf
        self.dur_until_best = None
        # set from pipeline after _fit_transformers()
        self._transformer_t = None
        self._grad_norm_sum = 0.0        # sum of grad norms in current epoch
        self._grad_norm_steps = 0        # how many steps contributed
        self.avg_grad_norms = []         # one value per epoch (average grad norm)

    def on_fit_start(self, trainer, pl_module):
        self.start = pd.Timestamp.now()

    # ---------------- helpers ----------------

    def _infer_loss_power(self, pl_module) -> int | None:
        """
        Return the power p such that:
           loss_orig = (scale_factor ** p) * loss_scaled
        - L1/MAE  -> p = 1
        - MSE  -> p = 2
        - RMSE    -> p = 1   (sqrt(MSE) -> scales like L1)
        Returns None if unknown (no rescaling performed).
        """
        # try common attributes
        for attr in ("loss_fn", "criterion", "loss"):
            fn = getattr(pl_module, attr, None)
            if fn is None:
                continue

            # plain torch losses
            if isinstance(fn, nn.L1Loss):
                return 1
            if isinstance(fn, nn.MSELoss):
                return 2

            name = fn.__class__.__name__.lower()
            if "l1" in name or "mae" in name:
                return 1
            if "mse" in name or "l2" in name:
                return 2
            if "rmse" in name or "rootmeansquare" in name:
                return 1

        try:
            h = getattr(pl_module, "hparams", None)
            if h is not None:
                n = str(getattr(h, "loss", getattr(h, "loss_fn", ""))).lower()
                if "l1" in n or "mae" in n: return 1
                if "mse" in n or "l2" in n: return 2
                if "rmse" in n: return 1
        except Exception:
            pass

        return None

    def _get_fitted_sklearn_scaler(self):
        """
        Darts Scaler keeps the fitted scaler(s) in _fitted_params (seq).
        Return the first fitted scaler for our single-series case.
        """
        tr = self._transformer_t
        if tr is None:
            return None
        # If not fitted yet, _fitted_params may be None
        fitted = getattr(tr, "_fitted_params", None)
        if fitted is None:
            return None
        if isinstance(fitted, (list, tuple)) and len(fitted) > 0:
            return fitted[0]
        return fitted

    def _linear_S(self) -> float | None:
        """
        Compute the linear factor S so that:
          MAE_orig = S * MAE_scaled,  MSE_orig = S^2 * MSE_scaled.
        """
        skl = self._get_fitted_sklearn_scaler()
        if skl is None:
            return None

        name = skl.__class__.__name__.lower()

        # StandardScaler path
        if "standardscaler" in name:
            if hasattr(skl, "scale_"):
                s = skl.scale_[0] if hasattr(skl.scale_, "__len__") else skl.scale_
                return float(s)

        # MinMaxScaler path
        if "minmaxscaler" in name:
            if hasattr(skl, "data_range_") and hasattr(skl, "feature_range"):
                data_range = skl.data_range_[0] if hasattr(skl.data_range_, "__len__") else skl.data_range_
                fr = skl.feature_range
                span = float(fr[1] - fr[0])
                # S = data_range / span  (for default (0,1) this is == data_range)
                return float(data_range) / (span if span != 0.0 else 1.0)
            if hasattr(skl, "scale_"):
                sc = skl.scale_[0] if hasattr(skl.scale_, "__len__") else skl.scale_
                if sc != 0:
                    return 1.0 / float(sc)

        # Generic fallback: cannot infer
        return None

    # --------------- main hook ----------------

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """
        Compute L2 norm of gradients for all parameters and accumulate it.
        This runs once per optimizer step (after backward and grad clipping).
        """
        total_norm_sq = 0.0
        for p in pl_module.parameters():
            if p.grad is None:
                continue
            param_norm = p.grad.detach().data.norm(2)
            total_norm_sq += float(param_norm) ** 2

        if total_norm_sq > 0.0:
            total_norm = math.sqrt(total_norm_sq)
            self._grad_norm_sum += total_norm
            self._grad_norm_steps += 1

            # log per step
            pl_module.log(
                "train_grad_norm",
                total_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=False,
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if not metrics:
            return

        # fetch the scaled loss Darts/Lightning logged
        val_scaled = None
        for k in ("val_loss", "val_loss_epoch"):  # common keys
            if k in metrics and metrics[k] is not None:
                v = metrics[k]
                val_scaled = float(v.item()) if hasattr(v, "item") else float(v)
                break
        if val_scaled is None:
            return

        # figure out loss scaling power
        p = self._infer_loss_power(pl_module)
        S = self._linear_S()

        # compute original-scale loss if possible
        if (p is not None) and (S is not None):
            val_orig = val_scaled * (S ** p)
            pl_module.log("val_loss_orig", val_orig, prog_bar=True, on_epoch=True, logger=False)
            use_val = val_orig
        else:
            # still log something 
            pl_module.log("val_loss_orig", val_scaled, prog_bar=True, on_epoch=True, logger=False)
            use_val = val_scaled

        # track stats
        e = trainer.current_epoch
        self.epochs.append(e)
        self.val_losses.append(use_val)

        if use_val < self.best_val:
            self.best_val = use_val
            self.best_epoch = e
            self.dur_until_best = pd.Timestamp.now() - self.start

        # average gradient norm for this epoch
        if self._grad_norm_steps > 0:
            avg_grad_norm = self._grad_norm_sum / self._grad_norm_steps
            self.avg_grad_norms.append(avg_grad_norm)

            # log 
            pl_module.log(
                "avg_grad_norm",
                avg_grad_norm,
                prog_bar=True,   
                on_epoch=True,
                logger=False,
            )

            self._grad_norm_sum = 0.0
            self._grad_norm_steps = 0

    @property
    def last_val(self):
        return float(self.val_losses[-1]) if self.val_losses else None

    @property
    def avg_near_best(self):
        if self.best_epoch is None or self.avg_window == 0:
            return None
        first_window_epoch = self.best_epoch - self.best_idx_in_window
        window_epochs = set(range(first_window_epoch, first_window_epoch + self.avg_window))
        vals = [v for e, v in zip(self.epochs, self.val_losses) if e in window_epochs]
        return float(sum(vals) / len(vals)) if vals else None

    def as_optuna_attrs(self):
        dur = getattr(self, "dur_until_best", None)
        return {
            "best_val": float(self.best_val) if self.best_epoch is not None else None,
            "best_epoch": int(self.best_epoch) if self.best_epoch is not None else None,
            "duration_until_best": dur,
            "dur_until_best_s": float(dur.total_seconds()) if dur is not None else None,
            "avg_near_best": self.avg_near_best,
            "last_val": self.last_val,
        }

class OptunaReportPruneCallback(pl.callbacks.Callback):
    """
    Reports a monitored metric to Optuna every validation end and optionally prunes.
    """
    def __init__(self, trial: optuna.trial.Trial, monitor: str = "val_loss", prune: bool = True):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.prune = prune
        self._last_reported_epoch = -1

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if getattr(trainer, "sanity_checking", False):
            return # skip during validation sanity run
        epoch = int(trainer.current_epoch)
        if epoch == self._last_reported_epoch:
            return  # already reported this epoch

        if self.monitor not in trainer.callback_metrics:
            return
        val = trainer.callback_metrics[self.monitor]
        val = float(val.item() if hasattr(val, "item") else val)

        step = int(trainer.current_epoch) 
        self.trial.report(val, step=step)

        if self.prune and self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {step} with {self.monitor}={val}")

class RelativeEarlyStopping(plc.EarlyStopping):
    """
    Early stopping with relative improvement and warmup.
    rel_min_delta=0.01 means “require ≥1% improvement” (in 'min' mode).
    """
    def __init__(self, *, monitor: str, mode: str = "min", min_delta: float = 0.0,
                 rel_min_delta: float = 0.0, warmup_epochs: int = 0, **kwargs):
        super().__init__(monitor=monitor, mode=mode, min_delta=min_delta, **kwargs)
        self.rel_min_delta_user = abs(float(rel_min_delta))  
        sign = 1.0 if self.monitor_op == torch.gt else -1.0  # mirror Lightning's sign flip
        self.rel_min_delta = sign * self.rel_min_delta_user
        self.warmup_epochs = int(warmup_epochs)
        self._check_on_train_epoch_end = False  # val_end only

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.current_epoch = getattr(trainer, "current_epoch", 0)
        self.min_epochs = getattr(trainer, "min_epochs", 0)
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            reasons = []
            if self._check_on_train_epoch_end:
                reasons.append("check_on_train_epoch_end=True → skip val_end check")

            if getattr(trainer, "sanity_checking", False):
                reasons.append("trainer.sanity_checking=True (sanity val run)")

            state = getattr(trainer, "state", None)
            if state is not None and getattr(state, "fn", None) != "fit":
                reasons.append(f"trainer.state.fn={getattr(state, 'fn', None)} (not fitting)")

            try:
                val_loop = trainer._fit_loop.epoch_loop.val_loop
                if not getattr(val_loop._data_source, "is_defined", lambda: False)():
                    reasons.append("no validation dataloader defined")
            except Exception:
                pass

            if not reasons:
                reasons.append("_should_skip_check(trainer) returned True (unspecified)")

            logging.debug(f"[ES skip] epoch={trainer.current_epoch} → {'; '.join(reasons)}")
            return
        self._run_early_stopping_check(trainer)

    def _improvement_message_rel(self, current: torch.Tensor, best: torch.Tensor) -> str:
        eps = best.new_tensor(1e-12)
        denom = best.abs().clamp_min(eps)
        frac = (best - current) / denom if self.mode == "min" else (current - best) / denom
        return (f"Metric {self.monitor} improved relatively by {float(frac):.3%} "
                f">= {self.rel_min_delta_user:.3%}. New best score: {float(current):.3f}")

    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> Tuple[bool, Optional[str]]:
        should_stop, reason = False, None
        epoch = self.current_epoch
        min_epochs = self.min_epochs

        best = self.best_score.to(current.device)
        eps = best.new_tensor(1e-12)

        # safety checks (don’t advance patience during warmup if metric is non-finite)
        if self.check_finite and not torch.isfinite(current):
            msg = (f"Monitored metric {self.monitor} = {current} is not finite. "
                   f"Previous best: {float(self.best_score):.3f}.")
            logging.debug(f"[ES] {msg}")
            return (self._allow_stop(epoch, min_epochs), msg)

        if self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            msg = (f"Stopping threshold reached: {self.monitor} "
                   f"{self.order_dict[self.mode]} {self.stopping_threshold} (current={float(current):.3f}).")
            logging.debug(f"[ES] {msg}")
            return (self._allow_stop(epoch, min_epochs), msg)

        if self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            msg = (f"Divergence threshold reached: {self.monitor} "
                   f"{self.order_dict[self.mode]} {self.divergence_threshold} (current={float(current):.3f}).")
            logging.debug(f"[ES] {msg}")
            return (self._allow_stop(epoch, min_epochs), msg)

        # first valid metric initializes best (Lightning-compatible)
        if not torch.isfinite(best):
            self.best_score = current
            self.wait_count = 0
            logging.debug(f"[ES] New best score (epoch={epoch}): {float(current):.3f}")
            return False, self._improvement_message(current)

        # absolute improvement (Lightning logic; min_delta already sign-flipped upstream)
        if self.min_delta == 0.0 or self.monitor_op(current - self.min_delta, best):
            logging.debug(f"[ES] Min delta test passed, diff={best - current:.2e}>{abs(self.min_delta):.2e}.")
            # relative improvement
            rel_target = best + self.rel_min_delta * best.abs().clamp_min(eps)
            if self.rel_min_delta == 0.0 or self.monitor_op(current, rel_target):
                self.best_score = current
                self.wait_count = 0
                logging.debug(f"[ES] Rel delta test passed, rel diff={(best - current)/best.abs().clamp_min(eps):.2e}>{abs(self.rel_min_delta):.2e}. New best score (epoch={epoch}): {float(current):.3f}")
                return False, self._improvement_message_rel(current, best)
            else:
                logging.debug(f"[ES] Rel delta test failed, rel diff={(best - current)/best.abs().clamp_min(eps):.2e}<{abs(self.rel_min_delta):.2e}.")
        else:
            logging.debug(f"[ES] Min delta test failed, diff={best - current:.2e}<{abs(self.min_delta):.2e}.")

        # no improvement
        self.wait_count += 1
        if self.wait_count >= self.patience:
            if self._allow_stop(epoch, min_epochs):
                should_stop = True
                reason = (f"Monitored metric {self.monitor} did not improve in the last "
                          f"{self.wait_count} checks. Best score: {float(self.best_score):.3f}.")
            else:
                reason = f"Patience exhausted but warmup/min_epochs not reached (epoch={epoch})."
            logging.debug(f"[ES] {reason}")
        logging.debug(f"[ES] Wait count: {self.wait_count}/{self.patience} (epoch={epoch}).")
        return should_stop, reason

    def _allow_stop(self, epoch: int, min_epochs: int) -> bool:
        # don't stop until both warmup AND Trainer.min_epochs are satisfied
        return (epoch >= self.warmup_epochs) and ((epoch + 1) >= (min_epochs or 0))



# -------------------------
# Pipeline
# -------------------------

class TFTPipeline:
    """
    End-to-end TFT pipeline using Darts.

    Input
    -----
    target_df: DataFrame with columns ['unique_id','ds','y']
    aux_df:    DataFrame with the same ['unique_id','ds'] index cols and exogenous features.
    """

    def __init__(
        self,
        target_df: pd.DataFrame,
        config: TFTRunConfig,
        aux_df: Optional[pd.DataFrame] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._target_df = target_df
        self._aux_df = aux_df
        self.config = config
        self._logger = logger or logging.getLogger(__name__)

        # internal attributes initialized during data preparation
        self._endog_components = []
        self._time_components = []
        self._climate_components = []
        self._y = None
        self._past_covs = None
        self._endog_vars_not_for_future = None
        self._future_covs = None
        self._target = None
        self._target_transf = None
        self._past_covs_transf = None
        self._future_covs_transf = None

        # internal attributes initialized during fit
        self._start_train = None
        self._end_train = None
        self._trainer = None
        self._history_df = None
        self._model = None
        self._n_params = None
        self._train_losses_orig = None
        self._val_losses_orig = None


        # --- Set seeds if requested ---
        lines_to_log = []
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
        
        tv = set(self.config.features.time_vars)
        allowed_time_vars = {"hod", "dow", "moy", "wss"}
        invalid_time_vars = tv - allowed_time_vars
        if invalid_time_vars:
            raise ValueError(f"Invalid time_vars: {invalid_time_vars}. Allowed values are {allowed_time_vars}.")
        allowed_and_present = tv & allowed_time_vars
        if not allowed_and_present:
            raise ValueError("At least one valid time_var must be specified.")
        
        # Persist device
        mc = self.config.model
        if hasattr(mc, 'torch_device_str') and (mc.torch_device_str is not None) and (mc.torch_device_str in ["cuda", "cpu", "gpu"]):
            device_str = mc.torch_device_str
            device_str = "gpu" if device_str == "cuda" else device_str
        else:
            if cuda.is_available():
                device_str = "gpu"
            else:
                device_str = "cpu"
        lines_to_log.append(f"using device: {device_str}")
        self.device_str = device_str

        self._logger.info("[pipe init] " + "; ".join(lines_to_log) + '.')

    def generate_vars(self) -> None:
        """
        Create endogenous/exogenous matrices and target series.

        Steps
        -----
        - Lagged features (endog and, optionally, exog).
        - Time features (sin/cos of HOD/DOW/MOY, WSS, cold-season flag).
        - Head-trim to drop rows made invalid by lags/diffs/rolls.
        - NaN row drop with a warning.
        - Track endog columns not safe for decoder (leakage control).
        """
        df = self._target_plus_aux_df

        # Base frames
        endog_df = df[['y']].copy()
        exog_df  = df[list(self.config.features.exog_vars)].copy() if self.config.features.exog_vars else pd.DataFrame(index=df.index)
        self._climate_components = list(exog_df.columns) if not exog_df.empty else []

        # --- Lagged endogenous features ---
        lag_cols = {}
        max_lag = 0
        for lag in self.config.features.endog_hour_lags or []:
            if not isinstance(lag, int) or lag <= 0:
                raise ValueError(f"endog_hour_lags must contain positive ints, got {lag}")
            max_lag = max(max_lag, lag)
            lag_cols[f"y_lag{lag}"] = endog_df['y'].shift(lag)
            self._endog_components.append(f"y_lag{lag}")

        if lag_cols:
            endog_df = pd.concat([endog_df, pd.DataFrame(lag_cols, index=endog_df.index)], axis=1)
        
        # --- Lagged exogenous features ---
        base_exog_for_lags = list(self.config.features.exog_vars)
        if self.config.features.include_exog_lags and not exog_df.empty:
            exog_lag_cols = {}
            for lag in (self.config.features.endog_hour_lags or []):
                for c in base_exog_for_lags:
                    exog_lag_cols[f"{c}_lag{lag}"] = exog_df[c].shift(lag)
                    self._climate_components.append(f"{c}_lag{lag}")

            if exog_lag_cols:
                exog_df = pd.concat([exog_df, pd.DataFrame(exog_lag_cols, index=exog_df.index)], axis=1)

        # --- Trim head once (for lags) ---
        head_trim = max_lag
        if head_trim > 0:
            endog_df = endog_df.iloc[head_trim:]
            exog_df  = exog_df.iloc[head_trim:]

        # --- Time features (cyclical) ---
        idx = endog_df.index
        assert idx.equals(exog_df.index), "Indexes must be aligned."

        def sincos(x, period):
            x = np.asarray(x, dtype=np.float32)
            ang = 2.0 * np.pi * (x / period)
            return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)
        
        # Figure out which time primitives we need
        tv = set(self.config.features.time_vars)
        need_hod = "hod" in tv
        need_dow = ("dow" in tv) or ("wss" in tv)  # wss is derived from dow
        need_moy = "moy" in tv

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

        if feats_dict:
            feats_df = pd.DataFrame(feats_dict, index=idx).astype(np.float32)
            exog_df  = pd.concat([exog_df, feats_df], axis=1)
            self._time_components = list(feats_df.columns)

        # --- Final cleanup & assignments ---
        if endog_df.isna().any().any() or exog_df.isna().any().any():
            n_endog = int(endog_df.isna().sum().sum())
            n_exog  = int(exog_df.isna().sum().sum())
            raise ValueError(f"NaNs detected after feature generation (endog: {n_endog}, exog: {n_exog}). "
                             f"Please check your configuration and input data.")

        # --- Build TimeSeries objects ---
        endog_df = endog_df.astype(np.float32) if not endog_df.empty else endog_df
        exog_df  = exog_df.astype(np.float32) if not exog_df.empty else exog_df
        
        self._target = TimeSeries.from_dataframe(endog_df[['y']])

        self._endog_vars_not_for_future = [
            c for c in endog_df.columns
            if not (is_for_endog_fut(c, self.config.model.output_chunk_length) or c == 'y')
        ]
        past_endog_df  = endog_df[self._endog_vars_not_for_future]
        fut_endog_df = endog_df[[c for c in endog_df.columns if c not in self._endog_vars_not_for_future and c != 'y']]
        fut_covs_df   = exog_df if fut_endog_df.empty else pd.concat([fut_endog_df, exog_df], axis=1)

        self._future_covs = None if fut_covs_df.shape[1] == 0 else TimeSeries.from_dataframe(fut_covs_df)
        self._past_covs   = None if past_endog_df.shape[1] == 0 else TimeSeries.from_dataframe(past_endog_df)

        
        n_endog_past = past_endog_df.shape[-1] if past_endog_df is not None else 0
        n_endog_fut = fut_endog_df.shape[-1] if fut_endog_df is not None else 0
        n_past_covs = self._past_covs.width if self._past_covs is not None else 0
        n_future_covs = self._future_covs.width if self._future_covs is not None else 0
        n_time = len(self.config.features.time_vars)*2
        n_climate_fut = len(exog_df.columns) - n_time
        self._logger.info(f"[gvars] features ready: "
            f"past_covs={n_past_covs} (endog={n_endog_past}, climate=0) | future_covs={n_future_covs} (endog={n_endog_fut}, climate={n_climate_fut}, time={n_time})")

    @property
    def target(self) -> Optional[TimeSeries]: return self._target
    @property
    def past_covs(self) -> Optional[TimeSeries]: return self._past_covs
    @property
    def future_covs(self) -> Optional[TimeSeries]: return self._future_covs


    def _fit_transformers(self, ts_train: TimeSeries, past_train: Optional[TimeSeries], fut_train: Optional[TimeSeries]) -> None:        
        """fit scalers on training data only. Return already transformed train data."""
        # Inizialize transformers
        transformer_t = Scaler(self.config.norm.scaler, **self.config.norm.kwargs)

        need_p = past_train is not None and past_train.width > 0
        transformer_p = Scaler(self.config.norm.scaler, **self.config.norm.kwargs) if need_p else None

        fc_set = set(self._future_covs.components) if self._future_covs is not None else set()
        tf_set = set(self._time_components)
        need_f = (len(fc_set - tf_set) > 0) or \
                (len(fc_set & tf_set) > 0 and self.config.norm.scale_time_vars) 
        transformer_f = Scaler(self.config.norm.scaler, **self.config.norm.kwargs) if need_f else None

        # Optionally mask time vars from scaling
        mask_f = None
        if need_f and not self.config.norm.scale_time_vars and self._future_covs is not None:
            mask_f = np.array([c not in self._time_components for c in self._future_covs.components])

        # Fit scalers on training data only and apply to train data only
        ts_tt   = transformer_t.fit_transform(ts_train)
        past_tt = transformer_p.fit_transform(past_train) if need_p else past_train
        fut_tt  = transformer_f.fit_transform(fut_train, component_mask=mask_f) if need_f else fut_train

        # Persist in self
        self._need_p = need_p
        self._need_f = need_f
        self._mask_f = mask_f
        self._transformer_t = transformer_t
        self._transformer_p = transformer_p if need_p else None
        self._transformer_f = transformer_f if need_f else None

        return ts_tt, past_tt, fut_tt

    def _apply_transforms(self, ts: Optional[TimeSeries], past: Optional[TimeSeries], fut: Optional[TimeSeries]) -> Tuple[Optional[TimeSeries], Optional[TimeSeries], Optional[TimeSeries]]:
        """Apply fitted scalers to given data."""
        ts_transf = self._transformer_t.transform(ts) if ts is not None else None
        past_transf = self._transformer_p.transform(past) if (past is not None and self._need_p) else past
        fut_transf = self._transformer_f.transform(fut, component_mask=self._mask_f) if (fut is not None and self._need_f) else fut
        return ts_transf, past_transf, fut_transf

    def _build_model(self, callbacks) -> None:
        mc = self.config.model
        tc = self.config.train
        dc = self.config.data
        if tc.loss_fn_str is not None:
            if tc.loss_fn_str.lower() == "l1":
                loss_fn = nn.L1Loss()
            elif tc.loss_fn_str.lower() == "mse":
                loss_fn = nn.MSELoss()
            else:
                raise ValueError(f"Unsupported loss function string: {tc.loss_fn_str}. Supported: 'l1', 'mse'.")
        else:
            loss_fn = nn.L1Loss()

        self._model = TFTModel(
            input_chunk_length=mc.input_chunk_length,
            output_chunk_length=mc.output_chunk_length,
            hidden_size=mc.hidden_size,
            lstm_layers=mc.lstm_layers,
            dropout=mc.dropout,
            num_attention_heads=mc.num_attention_heads,
            random_state=self.config.seed,
            batch_size=dc.batch_size,
            n_epochs=tc.n_epochs,
            pl_trainer_kwargs={
                "accelerator": self.device_str,
                "callbacks": callbacks,
                "gradient_clip_val": tc.gradient_clip_val,
                "gradient_clip_algorithm": "norm" if tc.gradient_clip_val is not None else None,
            },
            optimizer_kwargs={"lr": tc.lr},
            loss_fn=loss_fn,
        )

    @staticmethod
    def _split_opt(ts: Optional[TimeSeries], t1, t2=None, t0=None):
        """
        Split TimeSeries `ts` into one or two chunks:
        - The first is after `t0` (inclusive) if provided (or from the start of `ts` otherwise) up to `t1` (inclusive).
        - The second is after `t1` (exclusive) up to `t2` (inclusive), only if `t2` is provided (else None).
        """
        if ts is None:
            return (None, None)
        if t0 is not None:
            _, ts = ts.split_before(t0)
        a, b = ts.split_after(t1)
        if t2 is None:
            return (a, None)
        c, _ = b.split_after(t2)
        return (a, c)

    def fit(self, 
            end_train: pd.Timestamp, 
            end_val: Optional[pd.Timestamp] = None,
            start_train: Optional[pd.Timestamp] = None,
            trial: Optional[optuna.trial.Trial] = None,
        ) -> None:
        if self._target is None:
            self.generate_vars()

        tc = self.config.train

        h = self.config.model.output_chunk_length
        ts_train, ts_val = self._split_opt(self._target, t1=end_train, t2=end_val, t0=start_train)
        past_train, past_val = self._split_opt(self._past_covs, t1=end_train, t2=end_val, t0=start_train)
        fut_train, _ = self._split_opt(self._future_covs, t1=end_train+pd.Timedelta(hours=h), t0=start_train)
        if end_val is not None:
            _, fut_val = self._split_opt(self._future_covs, t1=end_train, t2=end_val+pd.Timedelta(hours=h))
        else:
            fut_val = None

        ts_tt, past_tt, fut_tt = self._fit_transformers(ts_train, past_train, fut_train)
        ts_vt, past_vt, fut_vt = self._apply_transforms(ts_val, past_val, fut_val)

        tracker = FitTracker()
        tracker._transformer_t = self._transformer_t  # for scaling info
        callbacks = [
            tracker
        ]

        if trial is not None:
            optuna_pruner = OptunaReportPruneCallback(
                trial=trial,
                monitor="val_loss",
            )
            callbacks.append(optuna_pruner)

        if (end_val is not None) and tc.use_es:
            early_stop = RelativeEarlyStopping(
                monitor="val_loss",   
                mode="min",
                patience=tc.es_patience,
                min_delta=tc.es_min_delta,
                rel_min_delta=tc.es_rel_min_delta,
                warmup_epochs=tc.es_warmup_epochs,
            )
            callbacks.append(early_stop)

        self._build_model(callbacks)

        self._logger.info(f"[fit] training from {ts_tt.start_time()} to {ts_tt.end_time()} "
                          f"({len(ts_tt)} points); "
                          f"validating from {ts_vt.start_time() if ts_vt is not None else 'N/A'} "
                          f"to {ts_vt.end_time() if ts_vt is not None else 'N/A'} "
                          f"({len(ts_vt) if ts_vt is not None else 'N/A'} points).")
        self._model.fit(
            series = ts_tt,
            val_series = ts_vt,
            past_covariates = past_tt,
            val_past_covariates = past_vt,
            future_covariates = fut_tt,
            val_future_covariates = fut_vt,
            stride = self.config.data.stride,
            verbose = True
        )
        self._end_train = end_train
        return tracker.as_optuna_attrs()

    def predict(self, n: int, cutoff: pd.Timestamp, alias: str = "TFT") -> pd.DataFrame:
        """
        Forecast next `n`, from `cutoff`+ 1h to `cutoff` + `n` hours. 
        Uses `n`= `output_chunk_length`, `cutoff` = last training timestamp by default.
        """
        if self._model is None:
            raise ValueError("Model is not trained yet. Please call fit() before predict().")
        if self._target_transf is None:
            tt, pt, ft = self._apply_transforms(self._target, self._past_covs, self._future_covs)
            self._target_transf = tt
            self._past_covs_transf = pt
            self._future_covs_transf = ft
        
        if n is None:
            n = self.config.model.output_chunk_length
        if cutoff is None:
            cutoff = self._end_train

        ts_context, _ = self._target_transf.split_after(cutoff)
        past_context, _ = self._past_covs_transf.split_after(cutoff) if self._past_covs_transf is not None else (None, None)
        fut_context, _ = self._future_covs_transf.split_after(cutoff + pd.Timedelta(hours=n)) if self._future_covs_transf is not None else (None, None)

        pred_t = self._model.predict(
            n=n,
            series=ts_context,
            past_covariates=past_context,
            future_covariates=fut_context,
        )

        # invert transform
        pred = self._transformer_t.inverse_transform(pred_t)

        # Back to DataFrame
        s = pred.to_series()
        pred_df = s.reset_index()
        pred_df.columns = ["ds", alias]  # robust
        pred_df.insert(0, "unique_id", self.unique_id)
        return pred_df
    
    def predict_many(
        self,
        n: int,
        start: pd.Timestamp,          # first cutoff to evaluate
        end: pd.Timestamp | None = None,   # optional last cutoff
        stride_hours: int = 1,        # gap between consecutive cutoffs
        alias: str = "TFT",
    ) -> pd.DataFrame:
        """
        Produce forecasts of length `n` at multiple cutoffs in [start, end],
        spaced by `stride_hours`. Returns a DataFrame with columns:
        unique_id, cutoff, ds, <alias>.
        """
        if self._model is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        if self._target_transf is None:
            tt, pt, ft = self._apply_transforms(self._target, self._past_covs, self._future_covs)
            self._target_transf = tt
            self._past_covs_transf = pt
            self._future_covs_transf = ft

        # contexts are already transformed in your pipeline
        if end is None:
            ts = self._target_transf
            pc = self._past_covs_transf
            fc = self._future_covs_transf
        else:
            ts, _ = self._split_opt(self._target_transf, end)
            pc, _ = self._split_opt(self._past_covs_transf, end)
            fc, _ = self._split_opt(self._future_covs_transf, end + pd.Timedelta(hours=n))

        # run rolling forecasts
        fcsts = self._model.historical_forecasts(
            series=ts,
            past_covariates=pc,
            future_covariates=fc,
            start=start,
            forecast_horizon=n,
            stride=stride_hours,
            last_points_only=False,   # keep full n-step forecasts for each cutoff
            retrain=False,            # reuse the fitted weights
            verbose=True,
        )

        # fcsts is a list of TimeSeries, one per cutoff
        out_frames = []
        for f in fcsts:
            f = self._transformer_t.inverse_transform(f)
            cutoff = f.start_time() - pd.Timedelta(hours=1)  # first pred is cutoff + 1 step
            s = f.to_series().rename(alias).reset_index()
            s.insert(0, "cutoff", cutoff)
            s.insert(0, "unique_id", self.unique_id)
            s.columns = ["unique_id", "cutoff", "ds", alias]
            out_frames.append(s)

        return pd.concat(out_frames, ignore_index=True)

    def cross_validation(
        self,
        *,
        h: int | None = None,
        test_size: int,
        end_test: pd.Timestamp | None = None,
        step_size: int = 1,
        input_size: int | None = None,
        val_size: int | None = 0,
        refit: bool | int | None = True,
        alias: str = "TFT",
        verbose: bool = True,
        checkpoint_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Perform rolling-window cross-validation for the TFTPipeline,
        optionally supporting checkpoint-based resume and periodic refits.

        Parameters
        ----------
        h: Forecast horizon (hours). Defaults to `config.model.output_chunk_length`
            if not provided.
        test_size: Total size of the CV test period (in hours).
        end_test: Final timestamp of the CV period. If None, defaults to the last
            available day at 23:00.
        step_size: Step size (hours) between consecutive CV cutoffs.
        input_size: Length (hours) of each training window. If None, uses all
            available data up to the cutoff.
        val_size: Optional validation window size (hours) before each cutoff.
            Set to 0 to disable validation.
        refit: Controls model re-training frequency.
        alias: Model alias used as forecast column name.
        verbose: If True, log progress and checkpoint status.
        checkpoint_path: Optional path to a pickle file for incremental checkpointing.
            If provided, CV progress is periodically saved and can be resumed
            from this file in future runs.

        Returns
        -------
        A pandas DataFrame containing one row per timestamp per window with columns:
        `[unique_id, ds, y, <alias>, cutoff]`.
        """

        # --- sanity checks ---
        if h is None:
            h = self.config.model.output_chunk_length
        if not isinstance(h, int) or h <= 0:
            raise ValueError("h must be a positive integer (forecast horizon in hours).")
        if not isinstance(test_size, int) or test_size <= h:
            raise ValueError("test_size must be an integer > h (in hours).")
        if not isinstance(step_size, int) or step_size <= 0:
            raise ValueError("step_size must be a positive integer (in hours).")
        if (test_size - h) % step_size != 0:
            raise ValueError("`test_size - h` must be a multiple of `step_size`.")
        if input_size is not None and (not isinstance(input_size, int) or input_size <= 0):
            raise ValueError("input_size must be a positive integer (hours) or None.")
        val_size = val_size or 0
        if not isinstance(val_size, int) or val_size < 0:
            raise ValueError("val_size must be a non-negative integer (hours) or None.")

        if self._target is None:
            if verbose:
                self._logger.info("target not generated yet, calling generate_vars() before CV.")
            self.generate_vars()

        # --- change refit to int ---
        if (refit is True) or (refit is None):
            refit = 1
        elif refit is False:
            refit = 0
        elif isinstance(refit, int):
            if refit < 0:
                raise ValueError("refit as int must be non-negative.")
        else:
            raise ValueError("refit must be bool or non-negative int or None.")

        # --- definition of end_test ---
        # last T23:00 available
        if end_test is None:
            last_ds = pd.to_datetime(self._target_df["ds"]).max()
            end_test = (last_ds - pd.Timedelta(hours=23)).floor("D") + pd.Timedelta(hours=23)
        else:
            if not isinstance(end_test, pd.Timestamp):
                raise ValueError("end_test must be a pd.Timestamp.")
            if end_test not in pd.to_datetime(self._target_df["ds"]).tolist():
                raise ValueError(f"end_test ({end_test}) must be in target_df['ds'].")

        t_min = self._target.start_time()
        t_max = self._target.end_time()

        # --- build relative offsets for cutoffs ---
        # offsets
        steps = list(range(-test_size, -h + 1, step_size))

        # --- checkpoint: load partial results if available ---  
        checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        all_results: list[pd.DataFrame] = []
        completed = 0

        current_cv_params = {
            "h": h,
            "test_size": test_size,
            "end_test": str(end_test),
            "step_size": step_size,
            "input_size": input_size,
            "val_size": val_size,
            "refit": refit,
            "alias": alias,
        }

        if checkpoint_path is not None and checkpoint_path.exists():
            all_results, completed = self._validate_cv_checkpoint(
                checkpoint_path, current_cv_params,
                steps, verbose
            )

        # --- define first_end_train based on input_size & val_size ---
        if input_size is not None:
            first_cutoff = end_test + pd.Timedelta(hours=steps[0])
            if val_size > 0:
                first_end_train = first_cutoff - pd.Timedelta(hours=val_size)
            else:
                first_end_train = first_cutoff
            first_start_train = first_end_train - pd.Timedelta(hours=input_size) + pd.Timedelta(hours=1)
            if first_start_train < t_min:
                raise ValueError(
                    f"input_size ({input_size}) + val_size ({val_size}) too large for the given end_test ({end_test}). "
                    f"First training window would start at {first_start_train}, but target starts at {t_min}."
                )

        prev_pipeline: TFTPipeline | None = None
        t0 = pd.Timestamp.now()

        # progress bar starts at 'completed'
        start_idx = completed
        end_idx = len(steps)

        iterator = tqdm(
            range(start_idx, end_idx),
            disable=not verbose,
            desc="TFT CV windows",
            total=len(steps),
            initial=completed,  # already done
            leave=True,
        )

        for i in iterator:
            # i runs from completed .. len(steps)-1

            # --- build cutoff ---
            offset = steps[i]
            cutoff = end_test + pd.Timedelta(hours=offset)

            # --- build start_train / end_train / end_val ---
            if val_size > 0:
                end_train = cutoff - pd.Timedelta(hours=val_size)
                end_val = cutoff
            else:
                end_train = cutoff
                end_val = None

            if input_size is not None:
                start_train = end_train - pd.Timedelta(hours=input_size) + pd.Timedelta(hours=1)
            else:
                start_train = t_min

            if start_train < t_min:
                raise ValueError(
                    f"Training window for cutoff={cutoff} would start at {start_train}, "
                    f"but available target starts at {t_min}. Reduce input_size/val_size or move end_test later."
                )
            if end_train <= start_train:
                raise ValueError(
                    f"Invalid training window [{start_train}, {end_train}] for cutoff={cutoff}. "
                    "Check input_size and val_size."
                )
            if cutoff + pd.Timedelta(hours=h) > t_max:
                raise ValueError(
                    f"Test horizon for cutoff={cutoff} (up to {cutoff + pd.Timedelta(hours=h)}) "
                    f"exceeds available data end ({t_max}). Reduce test_size/h or end_test."
                )

            # --- decide whether to refit ---
            do_fit = (
                i == 0
                or (refit > 0 and i % refit == 0)
            )

            if do_fit or prev_pipeline is None:
                # new pipeline & fit
                pipe = TFTPipeline(
                    target_df=self._target_df,
                    aux_df=self._aux_df,
                    config=self.config,
                    logger=self._logger,
                )

                # reuse already generated variables (TimeSeries and meta)
                pipe._target = self._target
                pipe._past_covs = self._past_covs
                pipe._future_covs = self._future_covs
                pipe._time_components = self._time_components
                pipe._climate_components = self._climate_components
                pipe._endog_components = self._endog_components
                pipe._endog_vars_not_for_future = self._endog_vars_not_for_future

                # fit 
                pipe.fit(
                    end_train=end_train,
                    end_val=end_val,
                    start_train=start_train,
                    trial=None,
                )
                prev_pipeline = pipe
            else:
                # reuse already fitted model
                pipe = prev_pipeline

            # --- forecast h steps after cutoff ---
            fc = pipe.predict(
                n=h,
                cutoff=cutoff,
                alias=alias,
            )
            # Sanity check
            assert fc["ds"].min() == cutoff + pd.Timedelta(hours=1)
            assert fc["ds"].max() == cutoff + pd.Timedelta(hours=h)

            # --- ground truth for test (cutoff, cutoff+h] ---
            mask = (
                (pd.to_datetime(self._target_df["ds"]) > cutoff)
                & (pd.to_datetime(self._target_df["ds"]) <= cutoff + pd.Timedelta(hours=h))
            )
            val_df = self._target_df.loc[mask, ["unique_id", "ds", "y"]]

            # merge forecast & truth
            merged = (
                val_df
                .merge(
                    fc[["unique_id", "ds", alias]],
                    on=["unique_id", "ds"],
                    how="left",
                )
                .assign(cutoff=cutoff)
            )

            all_results.append(merged)

            # --- update checkpoint on disk ---            
            if checkpoint_path is not None:
                ckpt = {
                    "cv_params": current_cv_params,
                    "steps": steps,
                    "results": all_results,
                }
                try:
                    with checkpoint_path.open("wb") as f:
                        pickle.dump(ckpt, f)
                except Exception as e:
                    if verbose:
                        self._logger.warning(
                            f"[CV checkpoint] Failed to save checkpoint to {checkpoint_path}: {e}"
                        )

            iterator.refresh()

        iterator.close()

        # --- metadata CV ---
        self._last_cv_metadata = {
            "pipeline_init": {
                "target_df_uids": self._target_df["unique_id"].unique().tolist(),
                "aux_df_uids": self._aux_df["unique_id"].unique().tolist() if self._aux_df is not None else [],
            },
            "pipeline_class": self.__class__.__name__,
            "cv_params": current_cv_params,
            "run_timestamp": pd.Timestamp.now().isoformat(),
            "elapsed_seconds": (pd.Timestamp.now() - t0).total_seconds(),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        }

        return pd.concat(all_results, ignore_index=True)

    def _validate_cv_checkpoint(
        self,
        checkpoint_path: Path,
        current_params: dict,
        steps: list[int],
        verbose: bool = False,
    ) -> tuple[list[pd.DataFrame], int]:
        """
        Load and validate an existing CV checkpoint, ensuring consistency with
        the current cross-validation configuration and refit schedule.
        """
        all_results: list[pd.DataFrame] = []
        completed: int = 0

        try:
            with checkpoint_path.open("rb") as f:
                ckpt = pickle.load(f)

            saved_params = ckpt.get("cv_params", {})
            saved_steps = ckpt.get("steps", [])
            saved_results = ckpt.get("results", [])

            # Parametri e steps devono coincidere esattamente
            if (
                saved_params == current_params
                and list(saved_steps) == steps
                and isinstance(saved_results, list)
            ):
                all_results = list(saved_results)
                K = len(all_results)
                refit_int = int(current_params.get("refit", 1))

                if K == 0:
                    # Checkpoint is valid but no windows completed yet
                    if verbose:
                        self._logger.info(
                            f"[CV checkpoint] Checkpoint at {checkpoint_path} is valid "
                            "but no windows completed yet. Starting fresh."
                        )
                    return [], 0

                if refit_int == 0:
                    # refit=0 not compatible with partial resume
                    if verbose:
                        self._logger.warning(
                            "[CV checkpoint] refit=0 (single global fit) is not compatible with "
                            "partial resume. Restarting CV from scratch."
                        )
                    return [], 0

                # Find last index where a refit was done
                last_completed_idx = K - 1
                refit_indices = [
                    i for i in range(len(steps))
                    if i <= last_completed_idx and (i == 0 or (refit_int > 0 and i % refit_int == 0))
                ]
                if not refit_indices:
                    # should not happen
                    true_completed = 0
                else:
                    true_last_completed_idx = max(refit_indices) - 1 # -> the next requires refit
                    true_completed = true_last_completed_idx + 1

                if true_completed < K:
                    if verbose:
                        self._logger.info(
                            f"[CV checkpoint] Adjusting completed windows from {K} to {true_completed} "
                            f"to enforce a refit at the next step (with zero-based index: {true_completed})."
                        )
                    all_results = all_results[:true_completed]
                completed = true_completed

                if verbose:
                    self._logger.info(
                        f"[CV checkpoint] Resuming from {checkpoint_path}: "
                        f"{completed}/{len(steps)} windows already completed."
                    )

            else:
                # Checkpoint non compatibile con i parametri correnti
                if verbose:
                    self._logger.warning(
                        f"[CV checkpoint] Checkpoint at {checkpoint_path} does not match "
                        "current CV parameters. Ignoring and starting fresh."
                    )

        except Exception as e:
            if verbose:
                self._logger.warning(
                    f"[CV checkpoint] Failed to load checkpoint from {checkpoint_path}: {e}. "
                    "Starting fresh."
                )

        return all_results, completed


    # ------------------------------
    # Summaries
    # ------------------------------

    def describe_dataset(self, max_cols: int = 20, return_dict: bool = False, to_logger: bool = True):
        """
        Summarize prepared features, normalization policy, and model I/O sizes.
        """
        # Ensure features exist
        if self._target is None:
            self.generate_vars()

        # Get time index from target series
        idx = self._target.time_index
        freq = pd.infer_freq(idx) or "unknown"
        
        # Extract column names from TimeSeries objects
        target_cols = list(self._target.components) if self._target is not None else []
        past_cov_cols = list(self._past_covs.components) if self._past_covs is not None else []
        future_cov_cols = list(self._future_covs.components) if self._future_covs is not None else []

        # Normalization info
        norm = getattr(self.config, "norm", None)
        scale_time_vars = getattr(norm, "scale_time_vars", False) if norm else False
        if norm and hasattr(norm, 'scaler') and norm.scaler is not None:
            # Get the scaler type name
            scaler_name = norm.scaler.__class__.__name__
            # Try to extract parameters if available
            try:
                params = norm.scaler.get_params()
                # Format key parameters for common scalers
                if hasattr(norm.scaler, 'feature_range'):
                    params_str = f"feature_range={norm.scaler.feature_range}"
                elif hasattr(norm.scaler, 'with_mean') and hasattr(norm.scaler, 'with_std'):
                    params_str = f"with_mean={norm.scaler.with_mean}, with_std={norm.scaler.with_std}"
                else:
                    params_str = str(params)
                norm_mode = f"{scaler_name}({params_str})"
            except:
                norm_mode = f"{scaler_name}"
        else:
            norm_mode = "none"
        
        # Feature component tracking
        time_feats = list(self._time_components)
        climate_feats = list(self._climate_components)
        endog_feats = list(self._endog_components)

        # Configured features
        feats = self.config.features
        exog_cfg = tuple(getattr(feats, "exog_vars", ()) or ())
        endog_lags = tuple(getattr(feats, "endog_hour_lags", ()) or ())
        incl_ex = getattr(feats, "include_exog_lags", False)
        time_vars = tuple(getattr(feats, "time_vars", ()) or ())

        # Model I/O sizes
        T_in = int(self.config.model.input_chunk_length)
        T_out = int(self.config.model.output_chunk_length)
        
        # Calculate encoder/decoder input dimensions
        enc_in = len(target_cols) + len(past_cov_cols) + len(future_cov_cols)
        dec_in = len(future_cov_cols)

        # Handle decoder-safe endogenous features
        safe_endog = [
            c for c in future_cov_cols if not (c in time_feats or c in climate_feats)
        ]

        # Splits (if already set via fit)
        train_range = "—"
        if self._end_train is not None:
            train_start = idx.min() if self._start_train is None else self._start_train
            train_range = f"{pd.Timestamp(train_start)} → {pd.Timestamp(self._end_train)}"
        # Note: val range tracking would need to be added to the class

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
            "Frequency:": freq,
            "Target:": "y",
            "Target/Past/Future cov counts:": f"{len(target_cols)} / {len(past_cov_cols)} / {len(future_cov_cols)}",
            "Target columns:": _fmt(target_cols),
            "Past cov columns:": _fmt(past_cov_cols),
            "Future cov columns:": _fmt(future_cov_cols),
            "Time features:": f"{len(time_feats)} | " + _fmt(time_feats),
            "Climate features:": f"{len(climate_feats)} | " + _fmt(climate_feats),
            "Endogenous features:": f"{len(endog_feats)} | " + _fmt(endog_feats),
            "Endog safe for decoder:": f"{len(safe_endog)} | " + _fmt(safe_endog),
            "Normalization:": f"{norm_mode} (scale_time_vars={scale_time_vars})",
            "Model I/O:": f"T_in={T_in}, T_out={T_out} | enc_in={enc_in}, dec_in={dec_in}",
            "Configured features:": (
                f"exog_vars={_fmt(exog_cfg)} | endog_hour_lags={_fmt(endog_lags)} | "
                f"include_exog_lags={incl_ex} | time_vars={_fmt(time_vars)}"
            ),
            "Device:": self.device_str,
            "Seed:": self.config.seed,
            "Unique ID:": self.unique_id,
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

    @property
    def n_params(self) -> Optional[int]:
        """
        Number of trainable parameters in the current (built) model, or None
        if the underlying torch module hasn't been materialized by Darts yet.
        """
        n_params = getattr(self, "_n_params", None)
        if n_params is not None:
            return n_params
        # Exact parameter count, if the internal torch module exists
        exact_total = None
        try:
            if getattr(self, "_model", None) is not None and getattr(self._model, "model", None) is not None:
                exact_total = sum(p.numel() for p in self._model.model.parameters() if p.requires_grad)
                self._n_params = int(exact_total)
        except Exception:
            exact_total = None
        self._n_params = exact_total
        return exact_total

    def describe_model(self, to_logger: bool = True, return_dict: bool = False):
        """
        Summarize TFT architecture and input sizes, plus exact parameter count
        (if the underlying torch module has been built).
        """
        # Ensure features exist so we can compute enc/dec input sizes
        if self._target is None:
            self.generate_vars()

        # Ensure there's at least a TFTModel instance (this does NOT guarantee
        # the internal torch module is created yet; Darts usually does that on fit())
        if getattr(self, "_model", None) is None:
            self._build_model(callbacks=[])

        mc = self.config.model

        # Component lists (may be empty if that group is unused)
        target_cols     = list(self._target.components) if self._target is not None else []
        past_cov_cols   = list(self._past_covs.components) if self._past_covs is not None else []
        future_cov_cols = list(self._future_covs.components) if self._future_covs is not None else []

        # Encoder/decoder input dimensionalities for Darts.TFTModel
        enc_in = len(target_cols) + len(past_cov_cols) + len(future_cov_cols)
        dec_in = len(future_cov_cols)

        # Device string decided at __init__
        device_str = getattr(self, "device_str", "cpu")

        # Exact parameter count, if the internal torch module exists
        exact_total = self.n_params

        summary = {
            "Architecture:": "Temporal Fusion Transformer (Darts.TFTModel)",
            "Device:": device_str,
            "Seq lengths:": f"T_in={mc.input_chunk_length}, T_out={mc.output_chunk_length}",
            "Encoder input size:": f"{enc_in} (target({len(target_cols)}) + past_cov({len(past_cov_cols)}) + future_cov({len(future_cov_cols)}))",
            "Decoder input size:": f"{dec_in} (future_cov only)",
            "Hidden size:": mc.hidden_size,
            "LSTM layers:": mc.lstm_layers,
            "Attention heads:": mc.num_attention_heads,
            "Dropout:": mc.dropout,
            "Params (exact):": f"{exact_total:,}" if exact_total is not None else "— (materialized after first fit())",
        }

        # Pretty formatting or dict return
        if return_dict:
            if to_logger:
                self._logger.info("[model]\n" + "\n".join(f"{k} {v}" for k, v in summary.items()))
            return summary

        max_key_len = max(len(k) for k in summary.keys())
        text = "\n".join(f"{k.ljust(max_key_len)} {v}" for k, v in summary.items())
        if to_logger:
            self._logger.info("[model]\n" + text)
        return text
    
def fit_for_optuna(
    *,
    trial: optuna.trial.Trial,
    target_df: pd.DataFrame,
    aux_df: pd.DataFrame,
    cfg: TFTRunConfig,
    start_train: pd.Timestamp | None, end_train: pd.Timestamp | None,
    start_val: pd.Timestamp | None, end_val: pd.Timestamp | None,
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    """
    Fit the TFT model for Optuna hyperparameter optimization.
    """
    logger = logger or logging.getLogger(__name__)
    pipe = TFTPipeline(
        target_df=target_df,
        aux_df=aux_df,
        config=cfg,
        logger=logger,
    )
    if start_val is not None:
        logger.warning("start_val was provided but it is ignored in fit_for_optuna (tft).")
    out = pipe.fit(start_train=start_train, end_train=end_train, end_val=end_val, trial=trial)

    out_optuna: Dict[str, Any] = {}

    out_optuna['n_params'] = pipe.n_params

    best_val = out.get('best_val')
    out_optuna['best_val'] = float(best_val) if best_val is not None else None

    best_epoch = out.get('best_epoch')
    out_optuna['best_epoch'] = int(best_epoch) if best_epoch is not None else None
    
    dur = out.get('duration_until_best')
    out_optuna['duration_until_best'] = dur if dur is not None else None

    dur_s = out.get('duration_until_best_s')
    out_optuna['duration_until_best_s'] = float(dur_s.total_seconds()) if dur_s is not None else None

    avg_near_best = out.get('avg_near_best')
    out_optuna['avg_near_best'] = float(avg_near_best) if avg_near_best is not None else None
    
    last_val = out.get('last_val')
    out_optuna['last_val'] = float(last_val) if last_val is not None else None

    return out_optuna

def config_builder(cfg_dict: Mapping['str', Any]) -> TFTRunConfig:
    """
    Build an TFTRunConfig from a generic dict.
    """
    return TFTRunConfig(
        model = TFTModelConfig(**cfg_dict.get("model", {})),
        train = TrainConfig(**cfg_dict.get("train", {})),
        data = DataConfig(**cfg_dict.get("data", {})),
        features = FeatureConfig(**cfg_dict.get("features", {})),
        norm = NormalizeConfig(**cfg_dict.get("norm", {})),
        seed = cfg_dict.get("seed", None)
    )
        
