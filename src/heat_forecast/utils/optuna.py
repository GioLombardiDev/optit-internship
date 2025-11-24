import importlib
import sys
import optuna
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
import hashlib
import gc
import torch
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
import itertools
from plotly.subplots import make_subplots

import numbers
import numpy as np
import math
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_float_dtype, is_timedelta64_dtype, is_bool_dtype
from pandas.io.formats.style import Styler

from typing import Callable, Dict, List, Optional, Sequence, Any, Literal, Iterable, SupportsFloat, TypeVar, Mapping

import warnings
import logging
_LOGGER = logging.getLogger(__name__)
from dataclasses import dataclass, replace, asdict
from IPython.display import display, HTML


from ..pipeline import lstm, tft

# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
## GENERAL OPTUNA UTILITIES
# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

# -----------------------------
# Registering suggesters
# -----------------------------

@dataclass(frozen=True)
class ParamInt:
    name: str
    low: int
    high: int
    step: Optional[int] = None
    log: bool = False
    condition: Optional[str] = None  # e.g., "model.head == 'mlp'"

@dataclass(frozen=True)
class ParamFloat:
    name: str
    low: float
    high: float
    step: Optional[float] = None
    log: bool = False
    condition: Optional[str] = None

@dataclass(frozen=True)
class ParamCat:
    name: str
    choices: Sequence[Any]
    condition: Optional[str] = None

ParamSpec = ParamInt | ParamFloat | ParamCat

@dataclass(frozen=True)
class SuggesterDoc:
    """Optional description block attached to a suggester."""
    summary: str
    params: Sequence[ParamSpec]
    notes: Optional[Sequence[str]] = None  # invariants, constraints, etc.

# ---- Registry ----

RunConfig = TypeVar(
    "RunConfig",
    lstm.LSTMRunConfig,
    tft.TFTRunConfig,
)
SuggesterFn = Callable[[optuna.trial.Trial, RunConfig], RunConfig]

class _Entry:
    __slots__ = ("fn", "doc")
    def __init__(self, fn: SuggesterFn, doc: Optional[SuggesterDoc]):
        self.fn = fn
        self.doc = doc

# preserve across importlib.reload
REGISTRY: Dict[str, _Entry] = globals().get("REGISTRY", {})

def register_suggester(name: str, *, doc: Optional[SuggesterDoc] = None):
    """Decorator registering a suggester under a stable name, with optional docs."""
    def _decorator(fn: SuggesterFn) -> SuggesterFn:
        # if name in REGISTRY and REGISTRY[name].fn is not fn:
        #     _LOGGER.warning("A suggester with name '%s' was already registered. Re-writing.", name)
        REGISTRY[name] = _Entry(fn, doc)
        return fn  # no wrapping
    return _decorator

def get_registered_entry(name: str) -> _Entry:
    """
    Return the registered entry for a suggester.

    On a cache miss, this lazily imports the suggester module to populate the
    registry, then tries again.
    """
    entry = REGISTRY.get(name)
    if entry is not None:
        return entry

    importlib.invalidate_caches()
    mod = sys.modules.get("heat_forecast.suggesters")
    if mod is None:
        importlib.import_module("heat_forecast.suggesters")
    else:
        importlib.reload(mod)

    entry = REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"Unknown suggester '{name}'. Known: {sorted(REGISTRY)}")
    return entry

def get_suggester(name: str) -> SuggesterFn:
    entry = get_registered_entry(name)
    return entry.fn

def describe_suggester(name: str, *, format: str = "markdown") -> str | Dict[str, Any]:
    """Return a human-readable description of the suggester's parameter space."""
    entry = get_registered_entry(name)
    if entry.doc is None:
        return f"(no description registered for '{name}')"

    doc = entry.doc
    if format == "markdown":
        lines: list[str] = []
        lines.append(f"### {name}")
        lines.append("")
        lines.append(doc.summary.strip())
        lines.append("")
        lines.append("**Parameters:**")
        for p in doc.params:
            if isinstance(p, ParamInt):
                rng = f"[{p.low}, {p.high}]"
                step = f", step={p.step}" if p.step is not None else ""
                log  = ", log" if p.log else ""
                cond = f" _(if {p.condition})_" if p.condition else ""
                lines.append(f"- `{p.name}` (int) {rng}{step}{log}{cond}")
            elif isinstance(p, ParamFloat):
                rng = f"[{p.low}, {p.high}]"
                step = f", step={p.step}" if p.step is not None else ""
                log  = ", log" if p.log else ""
                cond = f" _(if {p.condition})_" if p.condition else ""
                lines.append(f"- `{p.name}` (float) {rng}{step}{log}{cond}")
            else:  # ParamCat
                choices = ", ".join(repr(c) for c in p.choices)
                cond = f" _(if {p.condition})_" if p.condition else ""
                lines.append(f"- `{p.name}` (categorical): {{{choices}}}{cond}")
        if doc.notes:
            lines.append("")
            lines.append("**Notes / constraints:**")
            for n in doc.notes:
                lines.append(f"- {n}")
        return "\n".join(lines)

    elif format == "dict":
        # simple machine-readable dump
        def _one(p: ParamSpec) -> dict:
            base = {"name": p.name, "condition": getattr(p, "condition", None)}
            if isinstance(p, ParamInt):
                base |= {"type": "int", "low": p.low, "high": p.high, "step": p.step, "log": p.log}
            elif isinstance(p, ParamFloat):
                base |= {"type": "float", "low": p.low, "high": p.high, "step": p.step, "log": p.log}
            else:
                base |= {"type": "categorical", "choices": list(p.choices)}
            return base
        return {
            "name": name,
            "summary": doc.summary,
            "params": [_one(p) for p in doc.params],
            "notes": list(doc.notes) if doc.notes else [],
        }
    else:
        raise ValueError("format must be 'markdown' or 'dict'")
    
# ---------------------------------
# To set a random seed at each trial
# ---------------------------------

def trial_based_seed(base_seed: int | None, trial: optuna.trial.Trial) -> int:
    # Seeding strategy: combine base seed (if any) with study name and trial number
    base = 0 if base_seed is None else int(base_seed)
    key = f"{trial.study.study_name}:{trial.number}:{base}"
    h = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return h % (2**31 - 1)

def param_based_seed(base_seed: int | None, trial: optuna.trial.Trial) -> int:
    # Seeding strategy: combine base seed (if any) with trial params (useful for grid search with different repeat_ids)
    base = 0 if base_seed is None else int(base_seed)
    items = "|".join(f"{k}={trial.params[k]!r}" for k in sorted(trial.params))
    h = int(hashlib.sha256(f"{items}:{base}".encode()).hexdigest()[:16], 16)
    return h % (2**31 - 1)

# --------------------------------------------------------
# Optuna objective function
# --------------------------------------------------------

def cleanup_after_trial():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _duration_fmt(td: pd.Timedelta) -> str:
    if pd.isna(td): return ""
    total = int(td.total_seconds())
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02}:{m:02}:{s:02}"

FitFn = Callable[
    [
        optuna.trial.Trial,
        pd.DataFrame,                    # target_df
        Optional[pd.DataFrame],          # aux_df
        RunConfig,                       # cfg (concrete type at runtime)
        logging.Logger,                  # logger
        Optional[pd.Timestamp], Optional[pd.Timestamp],  # start_train, end_train
        Optional[pd.Timestamp], Optional[pd.Timestamp],  # start_val, end_val
    ],
    dict,  # out_optuna
]

def make_objective(
    target_df: pd.DataFrame,
    aux_df: pd.DataFrame | None,
    base_config: RunConfig,
    *,
    objective_val: Literal["best", "avg_near_best", "last"] = "best",
    suggest_config: Callable[[optuna.trial.Trial, RunConfig], RunConfig],
    start_train: pd.Timestamp | None,
    end_train: pd.Timestamp | None,
    start_val: pd.Timestamp | None,
    end_val: pd.Timestamp | None,
    logger: logging.Logger | None = None,
):
    """
    Returns an Optuna objective(trial) that:
      1) builds a per-trial config,
      2) prepares loaders,
      3) trains with early stopping + pruning,
      4) returns the best validation loss on original scale.
    """
    logger = logger or _LOGGER
    def objective(trial: optuna.trial.Trial) -> float:
        cfg = suggest_config(trial, base_config)
        logger.info(f"Starting trial {trial.number} with hyperparameters {trial.params}")

        # derive a deterministic seed; decide the strategy based on the sampler (tpe -> per trial, grid -> param based)
        is_grid = isinstance(trial.study.sampler, optuna.samplers.GridSampler)
        if is_grid:
            seed = param_based_seed(base_config.seed, trial)  # stable per (params, repeat_idx)
        else:
            seed = trial_based_seed(base_config.seed, trial)  # distinct per trial in TPE
        trial.set_user_attr("device", "cuda" if torch.cuda.is_available() else "cpu")
        trial.set_user_attr("config_seed", int(seed))
        cfg = replace(cfg, seed=seed)

        pipe = None
        train_loader = None
        val_loader = None

        FIT_FUNCS: Dict[type, FitFn] = {
            lstm.LSTMRunConfig: lstm.fit_for_optuna,
            tft.TFTRunConfig: tft.fit_for_optuna,
        }

        logging.info(f"Config type for trial {trial.number}: {type(cfg)}")
        fit_for_optuna = FIT_FUNCS.get(type(cfg))
        try:
            out_optuna = fit_for_optuna(
                trial=trial,
                target_df=target_df,
                aux_df=aux_df,
                cfg=cfg,
                logger=logger,
                start_train=start_train, end_train=end_train,
                start_val=start_val,     end_val=end_val,
            )

            trial.set_user_attr("n_params", out_optuna.get("n_params"))
            best_val = out_optuna.get("best_val")
            trial.set_user_attr("best_val", best_val)
            trial.set_user_attr("best_epoch", out_optuna.get("best_epoch"))
            dur_td = out_optuna.get('duration_until_best')
            trial.set_user_attr("duration_until_best", _duration_fmt(dur_td) if dur_td is not None else None)
            trial.set_user_attr("duration_until_best_s", out_optuna.get("duration_until_best_s"))
            avg_near_best = out_optuna.get("avg_near_best")
            trial.set_user_attr("avg_near_best", avg_near_best)
            last_val = out_optuna.get("last_val")
            trial.set_user_attr("last_val", last_val)

            if objective_val == "avg_near_best":
                return float(avg_near_best) if avg_near_best is not None else float("inf")
            if objective_val == "best":
                return float(best_val) if best_val is not None else float("inf")
            if objective_val == "last":
                return float(last_val) if last_val is not None else float("inf")
            raise ValueError(f"objective must be 'best', 'avg_near_best' or 'last'. Found: {objective_val}")

        except RuntimeError as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            if "out of memory" in str(e).lower():
                cleanup_after_trial()
                if isinstance(trial.study.sampler, optuna.samplers.GridSampler):
                    trial.set_user_attr("oom", True)
                    return float("inf")  # keep grid complete
                else:
                    raise optuna.TrialPruned()  # skip in TPE
            raise
        finally:
            # Drop large refs to help GC
            del pipe, train_loader, val_loader
            cleanup_after_trial()
    return objective

# --------------------------------------------------------
# Utilities to configure and run a Optuna study
# --------------------------------------------------------

def best_params_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    if study.best_trial.number == trial.number:
        n_params = trial.user_attrs.get("n_params")
        if n_params is not None:
            study.set_user_attr("best_n_params", int(n_params))

@dataclass
class OptunaStudyConfig:
    """
    Configuration for running an Optuna study. Parameters are grouped into three main categories:

    General
    -------
    study_name: str
        Name of the Optuna study (used for storage + logging).
    objective: Literal["best", "avg_near_best", "last"] = "best"
        Whether to minimize or maximize the objective function.
    n_trials: Optional[int] = 50
        Number of trials to run. Ignored for grid search (all grid points run).
    timeout: Optional[float] = None
        Maximum optimization time in seconds. None = unlimited.
    seed: Optional[int] = None
        Global random seed for reproducibility (affects sampler, not training).
    storage: Optional[str] = None
        Storage backend (e.g., SQLite URL or RDB for distributed optimization).

    Pruner
    ------
    pruner: {"percentile", "median", "nop"} = "percentile"
        Strategy for trial pruning:
        - "percentile": prune if trial is worse than Xth percentile.
        - "median": prune if trial is worse than running median.
        - "nop": disable pruning.
    pruner_percentile: float = 60.0
        For percentile pruner: cutoff percentile (lower = more aggressive).
    n_warmup_steps: int = 7
        Minimum number of steps (epochs) before pruning is considered.
    n_startup_trials_pruner: Optional[int] = None
        Number of initial trials to complete before pruning is enabled.
        Defaults to ~20% of n_trials (min 15).
    interval_steps: int = 1
        Frequency (in steps/epochs) at which pruning checks are made.
    patience: Optional[int] = 3
        Wraps the pruner with a "PatientPruner":
        waits for `patience` failed checks before actually pruning.
        None disables patience wrapper.

    Sampler 
    ------
    sampler: {"tpe", "grid"} = "tpe"
        Which sampler to use:
        - "tpe": Tree-structured Parzen Estimator (Bayesian optimizer).
        - "grid": exhaustive grid search (evaluates all param combos).
    n_startup_trials_sampler: Optional[int] = None
        Number of random trials before TPE starts modeling the search space.
        Defaults to ~20% of n_trials (min 10).
    n_ei_candidates: int = 128
        Number of candidate samples evaluated at each TPE step.
        Higher = better search accuracy but more compute overhead.
    multivariate: bool = True
        If True, TPE models joint distributions (captures param interactions).
    constant_liar: bool = False
        For distributed/parallel runs:
        - If True, mark running trials with "fake" losses to avoid duplicate suggestions.
        - For sequential runs, leave False (default).

    Internals (autogenerated)
    ---------
    grid: dict | None
        Parameter grid (only set when using GridSampler).
    grid_size: int | None
        Number of parameter combinations in the grid.
    """
    study_name: str
    objective: Literal["best", "avg_near_best", "last"] = "best"
    n_trials: Optional[int] = 50
    timeout: Optional[float] = None
    seed: Optional[int] = None
    storage: Optional[str] = None

    pruner: Literal["percentile", "median", "nop"] = "percentile"
    pruner_percentile: float = 60.0
    n_warmup_steps: int = 7
    n_startup_trials_pruner: Optional[int] = None
    interval_steps: int = 1
    patience: Optional[int] = 3  # None disables PatientPruner

    sampler: Literal["tpe", "grid"] = "tpe"  
    n_startup_trials_sampler: Optional[int] = None  # for TPE
    n_ei_candidates: int = 40                       # for TPE
    multivariate: bool = True                       # for TPE
    constant_liar: bool = False                     # for TPE, True only if parallel
    consider_endpoints: bool = True                 # for TPE

    def __post_init__(self):
        if self.pruner not in ["percentile", "median", "nop"]:
            raise ValueError("Invalid pruner: choose 'percentile', 'median' or 'nop'.")
        if self.n_warmup_steps < 0 or self.interval_steps < 1:
            raise ValueError("n_warmup_steps >= 0 and interval_steps >= 1 required.")
        if self.sampler not in ["tpe", "grid"]:
            raise ValueError("Invalid sampler: choose 'tpe' or 'grid'.")

        # Grid: ignore n_trials and force NopPruner
        if self.sampler == "grid":
            if self.n_trials is not None:
                _LOGGER.warning("GridSampler ignores n_trials; evaluating full grid.")
                self.n_trials = None
            if self.pruner != "nop":
                _LOGGER.warning("Forcing pruner='nop' for grid search.")
                self.pruner = "nop"
        
        # Defaults for TPE
        if self.sampler == "tpe":
            # 20% or 25% of budget, min 10
            if self.n_startup_trials_sampler is None:
                self.n_startup_trials_sampler = 10 if self.n_trials is None else max(10, self.n_trials * 0.25)
            # Pruner startup: 20% or 25% of budget, min 15
            if self.pruner != "nop" and self.n_startup_trials_pruner is None:
                self.n_startup_trials_pruner = 15 if self.n_trials is None else max(15, self.n_trials * 0.25)

        # internals
        self.grid = None       # for grid search, set by make_sampler
        self.grid_size = None  # for grid search, set by make_sampler

    # --- Sampler ---
    def make_sampler(self, *, suggester_name: Optional[str]) -> optuna.samplers.BaseSampler:
        if self.sampler == "tpe":
            return optuna.samplers.TPESampler(
                seed=self.seed,
                multivariate=self.multivariate,
                n_startup_trials=self.n_startup_trials_sampler,
                n_ei_candidates=self.n_ei_candidates,
                constant_liar=self.constant_liar,
                consider_endpoints=self.consider_endpoints,
            )
    
        if self.sampler == "grid":
            # infer grid from suggester documentation
            if suggester_name is None:
                raise ValueError("suggester_name must be provided when using grid sampler.")
            entry = get_registered_entry(suggester_name)
            if entry.doc is None:
                raise ValueError(f"Suggester '{suggester_name}' has no registered doc; can't infer grid.")
            # Make sure every param is categorical without conditions
            for p in entry.doc.params:
                if not isinstance(p, ParamCat):
                    raise ValueError(f"Suggester '{suggester_name}' has non-categorical param '{p.name}'; can't do grid search.")
                if p.condition is not None:
                    raise ValueError(f"Suggester '{suggester_name}' has conditional param '{p.name}'; can't do grid search.")
            # Infer grid from categorical params
            grid = {p.name: p.choices for p in entry.doc.params}
            self.grid = grid
            self.grid_size = math.prod(len(v) for v in grid.values())
            return optuna.samplers.GridSampler(grid)

    def make_pruner(self) -> optuna.pruners.BasePruner:
        if self.pruner == "nop":
            return optuna.pruners.NopPruner()
        
        # If using a pruner but suggester is grid, warn
        if self.sampler == "grid":
            _LOGGER.warning("Using a pruner with grid search is unusual; changing to pruner='nop'.")
            self.pruner = "nop"
            return optuna.pruners.NopPruner()

        if self.pruner == "percentile":
            base = optuna.pruners.PercentilePruner(
                percentile=self.pruner_percentile,
                n_startup_trials=self.n_startup_trials_pruner,
                n_warmup_steps=self.n_warmup_steps,
                interval_steps=self.interval_steps,
            )
        elif self.pruner == "median":
            base = optuna.pruners.MedianPruner(
                n_startup_trials=self.n_startup_trials_pruner,
                n_warmup_steps=self.n_warmup_steps,
                interval_steps=self.interval_steps,
            )

        return optuna.pruners.PatientPruner(base, patience=self.patience) if self.patience is not None else base
    
    def to_dict(self) -> dict:
        d = asdict(self)
        return d
    
    def to_structured_dict(self) -> dict:
        d = self.to_dict()
        structured = {
            "General": {
                "study_name": d.pop("study_name"),
                "n_trials": d.pop("n_trials"),
                "objective": d.pop("objective", "best"),
                "timeout": d.pop("timeout"),
                "seed": d.pop("seed"),
                "storage": d.pop("storage"),
            },
            "Sampler": {
                "sampler": d.pop("sampler"),
                "n_startup_trials_sampler": d.pop("n_startup_trials_sampler"),
                "n_ei_candidates": d.pop("n_ei_candidates"),
                "multivariate": d.pop("multivariate"),
                "constant_liar": d.pop("constant_liar"),
                "consider_endpoints": d.pop("consider_endpoints"),
            },
            "Pruner": {
                "pruner": d.pop("pruner"),
                "pruner_percentile": d.pop("pruner_percentile"),
                "n_warmup_steps": d.pop("n_warmup_steps"),
                "n_startup_trials_pruner": d.pop("n_startup_trials_pruner"),
                "interval_steps": d.pop("interval_steps"),
                "patience": d.pop("patience"),
            },
        }
        return structured

def run_study(
        unique_id: str,
        heat_df: pd.DataFrame, 
        aux_df: pd.DataFrame | None, 
        base_cfg: RunConfig,
        *,
        start_train: pd.Timestamp | None,
        end_train: pd.Timestamp | None,
        start_val: pd.Timestamp | None,
        end_val: pd.Timestamp | None,
        optuna_cfg: OptunaStudyConfig,
        suggest_config_name: str,
    ) -> optuna.Study:
    CONFIG_MODEL_NAME: Dict[type, str] = {
        lstm.LSTMRunConfig: "LSTM",
        tft.TFTRunConfig: "TFT",
    }
    
    suggest_config = get_suggester(suggest_config_name)
    sampler = optuna_cfg.make_sampler(suggester_name=suggest_config_name)
    pruner = optuna_cfg.make_pruner()

    heat_id_df = heat_df[heat_df['unique_id'] == unique_id]
    aux_id_df = aux_df[aux_df['unique_id'] == unique_id] if aux_df is not None else None

    objective_fn = make_objective(
        heat_id_df, 
        aux_id_df, 
        base_cfg,
        objective_val=optuna_cfg.objective,
        suggest_config=suggest_config,
        start_train=start_train, end_train=end_train, 
        start_val=start_val, end_val=end_val,
        logger=logging.getLogger("optuna_run"),
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=optuna_cfg.study_name,
        storage=optuna_cfg.storage,
        load_if_exists=bool(optuna_cfg.storage),
    )
    study.set_user_attr("splits", {
        "start_train": str(start_train), "end_train": str(end_train),
        "start_val": str(start_val), "end_val": str(end_val)
    })
    study.set_user_attr("unique_id", unique_id)
    study.set_user_attr("base_cfg", base_cfg.to_dict())
    study.set_user_attr("optuna_cfg", optuna_cfg.to_dict())
    study.set_user_attr("suggest_config_name", suggest_config_name)
    study.set_user_attr("env", {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.version.cuda else None,
        "cudnn": torch.backends.cudnn.version()
    })
    study.set_user_attr("model_name", CONFIG_MODEL_NAME.get(type(base_cfg), "unknown"))

    study.optimize(
        objective_fn,
        n_trials=optuna_cfg.n_trials,
        timeout=optuna_cfg.timeout,
        gc_after_trial=True,
        show_progress_bar=True,
        callbacks=[best_params_callback]
    )
    return study

def _validate_combos(combos: Iterable[Dict[str, Any]], suggest_config_name: str) -> None:
    entry = get_registered_entry(suggest_config_name)
    if entry.doc is None:
        raise ValueError(f"Suggester '{suggest_config_name}' has no registered doc.")
    # Make sure every param name exists
    for combo in combos:
        for k in combo:
            if k not in {p.name for p in entry.doc.params}:
                raise ValueError(f"Combo has unknown param '{k}'; suggester '{suggest_config_name}' knows {[p.name for p in entry.doc.params]}.")

ConfigBuilder = Callable[Mapping[str, Any], RunConfig]

def continue_study(
        study_name: str,
        storage_url: str,
        *,
        n_new_trials: int,
        target_df: pd.DataFrame,
        aux_df: pd.DataFrame | None,
        suggest_config_name: str | None = None,
        combos_to_enqueue: Iterable[Dict[str, Any]] | None = None,
        trials_per_combo: int | None = None,

    ) -> optuna.Study:
    CONFIG_BUILDERS: Dict[str, ConfigBuilder] = {
        "LSTM": lstm.config_builder,
        "TFT": tft.config_builder,
    }
    
    # retrieve and update pruner
    tmp = optuna.load_study(
        study_name=study_name,
        storage=storage_url
    ) # Note: this loads the study with the default pruner and sampler; 
      # we will need to reload the study with the correct ones

    # resolve suggester
    if suggest_config_name is None:
        suggest_config_name = tmp.user_attrs.get("suggest_config_name")
        if not suggest_config_name:
            raise KeyError("study missing 'suggest_config_name' user_attr")
    if combos_to_enqueue is not None:
        _validate_combos(combos_to_enqueue, suggest_config_name)
    suggest_config = get_suggester(suggest_config_name)

    # rebuild study with correct pruner and sampler
    optuna_cfg_dict = tmp.user_attrs.get("optuna_cfg")
    if optuna_cfg_dict is None:
        raise KeyError("study missing 'optuna_cfg' user_attr; can't rebuild pruner/sampler")
    optuna_cfg = OptunaStudyConfig(**optuna_cfg_dict)
    pruner = optuna_cfg.make_pruner()
    sampler = optuna_cfg.make_sampler(suggester_name=suggest_config_name)
    
    # final study load
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url,
        pruner=pruner,
        sampler=sampler
    )

    # enqueue specific combos if requested 
    if combos_to_enqueue is not None:
        if trials_per_combo is None:
            raise ValueError("If combos_to_enqueue is provided, trials_per_combo must also be provided.")
        if not isinstance(trials_per_combo, int) or trials_per_combo < 1:
            raise ValueError("trials_per_combo must be a positive integer.")
        if n_new_trials < len(combos_to_enqueue) * trials_per_combo:
            raise ValueError(f"n_new_trials={n_new_trials} is less than the number of combos to enqueue ({len(combos_to_enqueue)}x{trials_per_combo}={len(combos_to_enqueue)*trials_per_combo}).")
        n_enqueued = 0
        for combo in combos_to_enqueue:
            for _ in range(trials_per_combo):
                study.enqueue_trial(combo)
                n_enqueued += 1

    if combos_to_enqueue is None:
        _LOGGER.info("Continuing study '%s' with %d new trials.", study_name, n_new_trials)
    else:
        _LOGGER.info(
            "Continuing study '%s' with %d new trials; %d are enqueued (%dx%d).",
            study_name, n_new_trials, n_enqueued, len(list(combos_to_enqueue)), trials_per_combo
        )

    sd = optuna_cfg.to_structured_dict()
    lines = ", ".join(f"{k}={v}" for k, v in sd["General"].items())
    _LOGGER.info("Using general config: %s.", lines)

    lines = ", ".join(f"{k}={v}" for k, v in sd["Pruner"].items())
    _LOGGER.info("Using pruner config: %s.", lines)

    lines = ", ".join(f"{k}={v}" for k, v in sd["Sampler"].items())
    _LOGGER.info("Using sampler config: %s.", lines)

    # retrieve base config
    base_cfg_dict = study.user_attrs["base_cfg"]
    model_name = tmp.user_attrs.get("model_name", "LSTM") 
        # default to LSTM if attr is missing 
        # (the studies for LSTM were created before using this attr)
    config_builder = CONFIG_BUILDERS.get(model_name)
    if config_builder is None:
        raise ValueError(f"Unknown model_name '{model_name}' in study user_attrs; can't rebuild base config.")
    base_cfg = config_builder(base_cfg_dict)

    # rebuild train sets
    unique_id = study.user_attrs["unique_id"]
    heat_id_df = target_df[target_df['unique_id'] == unique_id]
    aux_id_df = None
    if aux_df is not None:
        aux_id_df = aux_df[aux_df['unique_id'] == unique_id]

    # retrieve splits
    to_ts = lambda s: None if s in (None, "None") else pd.Timestamp(s)
    start_train = to_ts(study.user_attrs["splits"]["start_train"])
    end_train   = to_ts(study.user_attrs["splits"]["end_train"])
    start_val   = to_ts(study.user_attrs["splits"]["start_val"])
    end_val     = to_ts(study.user_attrs["splits"]["end_val"])

    objective_fn = make_objective(
        heat_id_df, 
        aux_id_df, 
        base_cfg,
        objective_val=optuna_cfg.objective,
        suggest_config=suggest_config,
        start_train=start_train, end_train=end_train, 
        start_val=start_val, end_val=end_val,
        logger=logging.getLogger("optuna_run"),
    )

    # continue an existing study with new trials
    study.optimize(
        objective_fn, 
        n_trials=n_new_trials, 
        gc_after_trial=True, 
        show_progress_bar=True, 
        callbacks=[best_params_callback]
    )
    return study

def copy_study_with_first_n_trials(
        *,
        src_study_name: str,
        dst_study_name: str,
        storage_url: str,
        n_trials: int
    ) -> optuna.Study:
    # Load the source study
    src = optuna.load_study(study_name=src_study_name, storage=storage_url)
    optuna_cfg_dict = src.user_attrs.get("optuna_cfg")
    suggest_config_name = src.user_attrs.get("suggest_config_name")
    if not suggest_config_name:
        raise KeyError("study missing 'suggest_config_name' user_attr")
    if optuna_cfg_dict is None:
        raise KeyError("study missing 'optuna_cfg' user_attr; can't rebuild pruner/sampler")
    optuna_cfg = OptunaStudyConfig(**optuna_cfg_dict)
    pruner = optuna_cfg.make_pruner()
    sampler = optuna_cfg.make_sampler(suggester_name=suggest_config_name)

    # Create a destination study with the same objective directions
    dst = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=dst_study_name,
        storage=optuna_cfg.storage,
    )

    # Keep the first n_trials by trial number (0..99); skip any RUNNING trials
    allowed = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    first_n = [
        t for t in src.get_trials(deepcopy=True)
        if t.number < n_trials and t.state in allowed
    ]
    dst.add_trials(first_n)
    _LOGGER.info("Copied the first %d trials (and all user_attrs) from '%s' to '%s'", len(first_n), src_study_name, dst_study_name)

    # Copy study-level user attrs
    for k, v in src.user_attrs.items():
        dst.set_user_attr(k, v)
        if k == "optuna_cfg":
            sd = optuna_cfg.to_structured_dict()
            lines = ", ".join(f"{k}={v}" for k, v in sd["General"].items())
            _LOGGER.info("Copied general config: %s.", lines)

            lines = ", ".join(f"{k}={v}" for k, v in sd["Pruner"].items())
            _LOGGER.info("Copied pruner config: %s.", lines)

            lines = ", ".join(f"{k}={v}" for k, v in sd["Sampler"].items())
            _LOGGER.info("Copied sampler config: %s.", lines)

    # After verifying everything looks right, you can remove the old study:
    # optuna.delete_study(study_name=SRC_NAME, storage=STORAGE)

    return dst

def rename_study(
    *,
    storage_url: str,
    old_name: str,
    new_name: str,
    keep_old: bool = False,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None,
) -> optuna.Study:
    """
    Safely 'renames' an Optuna study by cloning it to a new study name and
    (optionally) deleting the old one.

    Parameters
    ----------
    storage_url : str
        RDB storage URL (sqlite:///..., postgresql://..., mysql://..., etc.).
    old_name : str
        Existing study name.
    new_name : str
        Desired new study name (must not exist already).
    keep_old : bool, default False
        If True, do NOT delete the old study after cloning.
    dry_run : bool, default False
        If True, perform all checks and report what would happen, but do not
        create/delete anything.
    logger : logging.Logger | None
        Optional logger for status messages.

    Returns
    -------
    optuna.Study
        The *new* study object (loaded from storage). In dry-run mode, this
        just returns the *old* study object.

    Notes
    -----
    - Refuses to proceed if the old study has RUNNING trials.
    - Copies all trials (COMPLETE/PRUNED/FAIL) and study-level user attrs.
    - System attrs are copied when possible.
    - Directions are preserved (multi-objective supported).
    - If your code depends on custom pruner/sampler, they're not persisted in
      the storage itself; you'll set them when you call `optimize()` again,
      just like in your `continue_study()` helper.
    """
    log = logger or _LOGGER

    # --- Load the source study and sanity-checks
    src = optuna.load_study(study_name=old_name, storage=storage_url)

    # Check RUNNING trials (cloning those is undefined / unsafe)
    running = [t for t in src.get_trials(deepcopy=False)
               if t.state == optuna.trial.TrialState.RUNNING]
    if running:
        log.warning("Found %d RUNNING trial(s) in study '%s'; skipping.", len(running), old_name)

    # New name must not already exist
    try:
        _ = optuna.load_study(study_name=new_name, storage=storage_url)
    except Exception:
        pass  # likely doesn't exist
    else:
        raise ValueError(f"A study named '{new_name}' already exists in this storage.")

    # Gather everything we need from src
    directions = getattr(src, "directions", None)
    if not directions:
        # Older optuna exposes .direction (single objective)
        directions = [src.direction]

    # Trials to copy (skip RUNNING by construction)
    allowed_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.WAITING,  # usually none left, but safe to include
    }
    trials_to_copy = [
        t for t in src.get_trials(deepcopy=True)
        if t.state in allowed_states
    ]

    if dry_run:
        log.info("[DRY RUN] Would create study '%s' and copy %d trial(s) and %d user_attr(s).",
                 new_name, len(trials_to_copy), len(src.user_attrs))
        if not keep_old:
            log.info("[DRY RUN] Would delete old study '%s'.", old_name)
        return src

    # --- Create destination study with the same objective directions
    if len(directions) == 1:
        dst = optuna.create_study(
            study_name=new_name,
            storage=storage_url,
            direction=directions[0].name.lower(),  # "minimize"/"maximize"
        )
    else:
        dst = optuna.create_study(
            study_name=new_name,
            storage=storage_url,
            directions=[d.name.lower() for d in directions],
        )

    # --- Copy trials
    # Note: add_trials() preserves numbers/params/values/states/timings.
    if trials_to_copy:
        dst.add_trials(trials_to_copy)

    # --- Copy study-level attributes
    for k, v in src.user_attrs.items():
        dst.set_user_attr(k, v)

    # --- Quick verification
    src_trials = [t for t in src.get_trials(deepcopy=False) if t.state in allowed_states]
    dst_trials = dst.get_trials(deepcopy=False)
    if len(src_trials) != len(dst_trials):
        raise RuntimeError(
            f"Clone verification failed: expected {len(src_trials)} trials, "
            f"found {len(dst_trials)} in destination."
        )

    # --- Optionally delete the old study
    if not keep_old:
        optuna.delete_study(study_name=old_name, storage=storage_url)
        log.info("Deleted old study '%s'.", old_name)

    log.info("Renamed study '%s' -> '%s' (copied %d trials).",
             old_name, new_name, len(dst_trials))
    # Return a loaded handle to the *new* study
    return optuna.load_study(study_name=new_name, storage=storage_url)

def clone_filtered_study(
    *,
    # Source (choose ONE): either pass src_study OR (storage_url + src_name)
    src_study: Optional[optuna.Study] = None,
    storage_url: Optional[str] = None,
    src_name: Optional[str] = None,

    # Destination controls
    new_name: Optional[str] = None,
    save_to_storage: bool = False,  # default: return in-memory study

    # Filtering
    predicate: Optional[Callable[[optuna.trial.FrozenTrial], bool]] = None,

    # Misc
    remember_original_numbers: bool = True,  # <- annotate cloned trials with original numbers
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None,
) -> optuna.Study:
    """
    Clone an Optuna study while filtering trials with `predicate`.
    - Source can be a loaded Study (src_study) OR storage_url+src_name.
    - By default, returns an IN-MEMORY study (not persisted).
    - If save_to_storage=True, requires storage_url and new_name.

    Skips RUNNING trials. Copies study-level user attrs.
    Preserves single/multi-objective directions.
    """
    log = logger or logging.getLogger(__name__)

    # --- Resolve source
    if src_study is not None:
        src = src_study
    else:
        if not (storage_url and src_name):
            raise ValueError("Provide either src_study OR (storage_url and src_name).")
        src = optuna.load_study(study_name=src_name, storage=storage_url)

    # --- Sanity checks
    running = [t for t in src.get_trials(deepcopy=False)
               if t.state == optuna.trial.TrialState.RUNNING]
    if running:
        log.warning("Found %d RUNNING trial(s); these will be skipped.", len(running))

    # Directions (single or multi)
    directions = getattr(src, "directions", None)
    if not directions:
        directions = [src.direction]

    # Eligible trials (skip RUNNING)
    allowed_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.WAITING,
    }
    candidates = [t for t in src.get_trials(deepcopy=True) if t.state in allowed_states]

    # Apply predicate safely
    if predicate is not None:
        def _passes(t: optuna.trial.FrozenTrial) -> bool:
            try:
                return bool(predicate(t))
            except Exception as e:
                log.debug("Predicate error on trial %s: %s; skipping.", t.number, e)
                return False
        trials = [t for t in candidates if _passes(t)]
    else:
        trials = candidates

    # Keep stable order (by original trial number)
    trials.sort(key=lambda t: t.number)

    # --- DRY RUN
    if dry_run:
        dst_desc = "in-memory study" if not save_to_storage else f"'{new_name}' in storage"
        log.info("[DRY RUN] Would clone -> %s; copy %d/%d trial(s); copy %d user_attr(s).",
                 dst_desc, len(trials), len(candidates), len(src.user_attrs))
        return src

    # --- Create destination (in-memory by default)
    if save_to_storage:
        if not (storage_url and new_name):
            raise ValueError("When save_to_storage=True, provide storage_url and new_name.")
        # Ensure destination doesn't already exist
        try:
            _ = optuna.load_study(study_name=new_name, storage=storage_url)
        except Exception:
            pass
        else:
            raise ValueError(f"A study named '{new_name}' already exists in this storage.")

        if len(directions) == 1:
            dst = optuna.create_study(
                study_name=new_name, storage=storage_url,
                direction=directions[0].name.lower(),
            )
        else:
            dst = optuna.create_study(
                study_name=new_name, storage=storage_url,
                directions=[d.name.lower() for d in directions],
            )
    else:
        # Pure in-memory
        if len(directions) == 1:
            dst = optuna.create_study(direction=directions[0].name.lower())
        else:
            dst = optuna.create_study(directions=[d.name.lower() for d in directions])

    # --- Copy trials
    if trials:
        if remember_original_numbers:
            # annotate each trial with provenance before adding
            for t in trials:
                # avoid mutating src object: user_attrs is copied in deepcopy already
                t.user_attrs = dict(t.user_attrs)  # ensure it's our own dict
                t.user_attrs.setdefault("orig_trial_number", t.number)
                # keep study name if available
                try:
                    src_name_val = src.study_name  # may raise if not available
                except Exception:
                    src_name_val = None
                if src_name_val is not None:
                    t.user_attrs.setdefault("orig_study_name", src_name_val)

        # add_trials preserves params/values/states/timings/attrs/intermediates
        dst.add_trials(trials)

    # --- Copy study-level attrs
    for k, v in src.user_attrs.items():
        dst.set_user_attr(k, v)

    log.info(
        "Cloned filtered study; copied %d/%d trial(s). Destination: %s",
        len(dst.get_trials(deepcopy=False)), len(candidates),
        (new_name if save_to_storage else "<in-memory>")
    )

    return dst

# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
## FOR STUDY INSPECTION
# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

# ---------------------------
# Basics
# ---------------------------

def trials_df(
        study: optuna.study.Study,
        states: tuple[str, ...] = ("COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING"),
    ) -> tuple[pd.DataFrame, str]:
    """Collect trial rows with params_ columns and the objective value."""
    # Build last_epoch from intermediate_values (epoch reported as step, 0-based)
    last_epoch_by_number = {
        t.number: (max(t.intermediate_values.keys()) + 1) if t.intermediate_values else None
        for t in study.trials
    }

    # Get DataFrame
    df = study.trials_dataframe()

    # Add last_epoch column (not persisted in storage)
    df["last_epoch"] = df["number"].map(last_epoch_by_number)

    # Rename value to objective (more intuitive)
    if "value" in df.columns:
        val_name = study.user_attrs.get("optuna_cfg", {}).get("objective", "best MAE")
        df = df.rename(columns={"value": val_name})
    else:
        raise KeyError("DataFrame missing 'value' column; is the study empty?")

    df = df[df["state"].isin(states)].copy()
    return df, val_name


def _param_kinds_from_study(study: optuna.study.Study) -> dict[str, str]:
    """
    Map param name -> 'numeric' | 'categorical' from Optuna distributions.
    Falls back to dtype later where distributions are missing.
    """
    kinds: dict[str, str] = {}
    for t in study.trials:
        for name, dist in t.distributions.items():
            if isinstance(dist, (FloatDistribution, IntDistribution)):
                kinds[name] = "numeric"
            elif isinstance(dist, CategoricalDistribution):
                kinds[name] = "categorical"
    return kinds

def _guess_param_kind(series: pd.Series) -> str:
    # treat integer/float categoricals as numeric
    return "numeric" if is_numeric_dtype(series) and (not is_bool_dtype(series)) and (not series.nunique() <= 5) else "categorical"


def study_minimize(study: optuna.study.Study) -> bool:
    """True if the first objective is MINIMIZE; sensible default True."""
    try:
        if hasattr(study, "direction"):
            return study.direction == optuna.study.StudyDirection.MINIMIZE
        if hasattr(study, "directions"):
            return study.directions[0] == optuna.study.StudyDirection.MINIMIZE
    except Exception:
        pass
    return True


def _param_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("params_")]

def _param_meta(study: optuna.study.Study) -> dict[str, dict]:
    """param name -> metadata extracted from Optuna distributions (bounds, log, step, choices)."""
    out: dict[str, dict] = {}
    # scan trials to collect distributions (some trials may not have all)
    for t in study.trials:
        for name, dist in t.distributions.items():
            if name in out:
                continue
            if isinstance(dist, IntDistribution) or isinstance(dist, FloatDistribution):
                out[name] = dict(kind="numeric", low=float(dist.low), high=float(dist.high),
                                 log=getattr(dist, "log", False), step=getattr(dist, "step", None))
            elif isinstance(dist, CategoricalDistribution):
                choices = list(dist.choices)
                out[name] = dict(kind="categorical", choices=choices, n_choices=len(choices))
    return out

def _float_fmt(x):
    """Scientific if |x|<1e-2 or very large; fixed otherwise."""
    if pd.isna(x):
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)
    return f"{v:.2e}" if (abs(v) < 1e-2 or abs(v) >= 1e6) else f"{v:.3f}"

def _int_fmt(x):
    if pd.isna(x): return ""
    return f"{int(np.round(float(x))):,d}"

def _bool_fmt(x):
    if pd.isna(x):
        return ""
    return "True" if bool(x) else "False"

def _pct_fmt(x):
    if pd.isna(x): return ""
    return f"{100*float(x):.1f}%"

def _is_integerish_series(s: pd.Series, tol: float = 1e-12) -> bool:
    if is_integer_dtype(s):
        return True
    if is_float_dtype(s):
        arr = pd.to_numeric(s, errors="coerce").to_numpy()
        arr = arr[np.isfinite(arr)]
        return arr.size > 0 and np.all(np.abs(arr - np.round(arr)) <= tol)
    return False

def _best_num_formatter_for_series(s):
    """Return appropriate formatter (_int_fmt, _float_fmt or _bool_fmt) for Series or 1D array."""
    if isinstance(s, np.ndarray):
        s = pd.Series(s)
    elif isinstance(s, pd.Index):
        s = s.to_series().reset_index(drop=True)
    elif not isinstance(s, pd.Series):
        raise TypeError(f"Expected pandas Series or numpy ndarray, got {type(s)}")

    if is_bool_dtype(s):
        return _bool_fmt
    elif _is_integerish_series(s):
        return _int_fmt
    else:
        return _float_fmt

def _param_col(p: str) -> str:
    if "user_attrs_" in p or "system_attrs_" in p:
        return p
    # Treat as parameter name
    if p.startswith("params_"):
        return p
    return f"params_{p}"

def _get_param(df: pd.DataFrame, param: str) -> pd.Series:
    if param in df.columns:
        return df[param]
    col = _param_col(param)
    if col in df.columns:
        return df[col]
    raise KeyError(f"Parameter '{col}' not found in DataFrame.")

# ---------------------------
# Parameter tables (high-level)
# ---------------------------

def trials_df_for_display(
    df: pd.DataFrame, 
    val_name: str,
    *,              
    to_exclude: tuple[str, ...] | None = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """Display a trials DataFrame with params_ columns and the objective value."""
    # create a working cop
    df = df.copy()

    # exclude uninformative columns
    if to_exclude is None:
        to_exclude = (
            "user_attrs_repeat_idx",
            "system_attrs_grid_id",
            "system_attrs_search_space",
            "datetime_start",
            "datetime_complete",
            "user_attrs_orig_trial_number",
            "user_attrs_orig_study_name",
        )
    df = df.drop(columns=[c for c in to_exclude if c in df.columns], errors="ignore")

    # nicer state
    if "state" in df.columns:
        df["state"] = df["state"].astype(str)

    # format duration
    if "duration" in df.columns and is_timedelta64_dtype(df["duration"]):
        df["duration"] = df["duration"].apply(_duration_fmt)

    # sort by original 'value', then rename to value_col
    if val_name in df.columns:
        df.sort_values(by=val_name, ascending=ascending, inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)
    else:
        raise KeyError(f"Value column '{val_name}' not found in DataFrame.")

    # format numeric params
    param_cols = _param_cols(df)
    for c in param_cols:
        if is_numeric_dtype(df[c]):
            _fmt = _best_num_formatter_for_series(df[c])
            df[c] = df[c].map(_fmt)

    # format objective
    if is_numeric_dtype(df[val_name]):
        fmt_val = _best_num_formatter_for_series(df[val_name])
        df[val_name] = df[val_name].map(fmt_val)

    # format user_attrs columns
    user_attrs_cols = [c for c in df.columns if c.startswith("user_attrs_")]
    for c in user_attrs_cols:
        if is_numeric_dtype(df[c]):
            _fmt = _best_num_formatter_for_series(df[c])
            df[c] = df[c].map(_fmt)

    # strip 'params_' prefix for display
    if param_cols:
        df.rename(columns={c: c.replace("params_", "", 1) for c in param_cols}, inplace=True)

    return df

def _style_coverage_tables(
    num_df: pd.DataFrame | None,
    cat_df: pd.DataFrame | None,
    *,
    cmap_good: str = "Greens",
    cmap_bad: str = "Reds",
) -> tuple[Styler | None, Styler | None]:
    """Return styled versions of numeric + categorical coverage tables."""

    # --- Numeric coverage
    if num_df is None or num_df.empty:
        num_sty = None
    else:
        num_fmt = {}

        # integer-ish count columns
        for col in [
            "n_with_value", "n_complete", "n_pruned",
            "n_fail", "n_running", "unique", "non_empty_bins"
        ]:
            if col in num_df:
                num_fmt[col] = _int_fmt

        # percentage-like columns
        for col in ["unique_ratio", "span_ratio", "bin_coverage"]:
            if col in num_df:
                num_fmt[col] = _pct_fmt

        # continuous metrics
        for col in [
            "min", "p25", "median", "p75", "max",
            "search_low", "search_high", "search_step"
        ]:
            if col in num_df:
                num_fmt[col] = _float_fmt

        num_sty = (
            num_df.style
            .format(num_fmt, na_rep="")
            .hide(axis="index")
            .set_properties(subset=["parameter"],
                            **{"font-weight": "600", "white-space": "nowrap"})
        )

        # heatmaps / bars
        if "span_ratio" in num_df:
            num_sty = num_sty.background_gradient(
                subset=["span_ratio"], cmap=cmap_good, vmin=0, vmax=1
            )
        for col in ["unique_ratio", "bin_coverage"]:
            if col in num_df:
                num_sty = num_sty.bar(subset=[col], vmin=0, vmax=1, color=cmap_good)

    # --- Categorical coverage
    if cat_df is None or cat_df.empty:
        cat_sty = None
    else:
        cat_fmt = {}

        # integer-ish counts
        for col in [
            "n_with_value", "n_complete", "n_pruned",
            "n_fail", "n_running", "unique_levels", "choices_declared"
        ]:
            if col in cat_df:
                cat_fmt[col] = _int_fmt

        # ratios / continuous
        if "coverage_ratio" in cat_df:
            cat_fmt["coverage_ratio"] = _pct_fmt
        if "imbalance_ratio" in cat_df:
            cat_fmt["imbalance_ratio"] = _float_fmt
        if "entropy_ratio" in cat_df:
            cat_fmt["entropy_ratio"] = _pct_fmt

        cat_sty = (
            cat_df.style
            .format(cat_fmt, na_rep="")
            .hide(axis="index")
            .set_properties(subset=["parameter"],
                            **{"font-weight": "600", "white-space": "nowrap"})
        )

        # visual cues
        if "coverage_ratio" in cat_df:
            cat_sty = cat_sty.background_gradient(subset=["coverage_ratio"], cmap=cmap_good, vmin=0.0, vmax=1.0)
        if "imbalance_ratio" in cat_df:
            cat_sty = cat_sty.background_gradient(subset=["imbalance_ratio"], cmap=cmap_bad, vmin=1.0)
        if "entropy_ratio" in cat_df:
            cat_sty = cat_sty.background_gradient(subset=["entropy_ratio"], cmap=cmap_good, vmax=1.0, vmin=0.0)

    return num_sty, cat_sty

def summarize_params_coverage(
    study: optuna.study.Study,
    df: pd.DataFrame,
    val_name: str,
    *,
    max_levels: int = 8,
    bins_for_gap: int = 10,
) -> tuple[Styler | None, Styler | None]:
    """
    Return (num_df, cat_df) coverage diagnostics.

    Coverage counts use ALL trials; performance stats (median/IQR/mean/std) use COMPLETE trials only.
    Adds per-state counts, missing rate, and comparison vs the declared search space when available.
    """
    kinds = _param_kinds_from_study(study)
    meta = _param_meta(study)
    param_cols = _param_cols(df)

    # state masks
    if val_name not in df.columns:
        raise KeyError(f"Value column '{val_name}' not found in DataFrame.")
    st = df["state"].astype(str)
    m_complete_inf = (
        (st == "COMPLETE") &
        df[val_name].apply(lambda v: pd.isna(v) or np.isinf(v))
    )
    if any(m_complete_inf):
        _LOGGER.warning("There are COMPLETE trials with missing or infinite values in column '%s'", val_name)
    m_complete = (st == "COMPLETE") & (~m_complete_inf)
    m_pruned   = (st == "PRUNED")
    m_fail     = (st == "FAIL")
    m_running  = (st == "RUNNING")

    num_rows, cat_rows = [], []

    for col in sorted(param_cols):
        pname = col.replace("params_", "", 1)
        # coverage series: use all trials (any state)
        s_all = df[col]
        has_val = s_all.notna()
        n_any = int(has_val.sum())
        if n_any == 0:
            # no value ever sampled for this param
            continue

        # counts by state where param present
        n_complete_param = int((m_complete & has_val).sum())
        n_pruned_param = int((m_pruned & has_val).sum())
        n_fail_param = int((m_fail & has_val).sum())
        n_running_param = int((m_running & has_val).sum())

        # determine kind (prefer distribution info)
        kind = _guess_param_kind(df[col]) # or meta.get(pname, {}).get("kind") or kinds.get(pname)

        if kind == "numeric":
            vals_all = pd.to_numeric(s_all[has_val], errors="coerce").astype(float).dropna()
            if vals_all.empty:
                continue

            # bounds from distribution if available
            low = meta.get(pname, {}).get("low")
            high = meta.get(pname, {}).get("high")
            log_flag = bool(meta.get(pname, {}).get("log", False))
            step = meta.get(pname, {}).get("step")

            vmin = float(vals_all.min())
            v25  = float(vals_all.quantile(0.25))
            v50  = float(vals_all.median())
            v75  = float(vals_all.quantile(0.75))
            vmax = float(vals_all.max())
            nunique = int(vals_all.nunique())
            unique_ratio = nunique / n_any

            # span ratio vs declared space
            span_ratio = np.nan
            if low is not None and high is not None and high > low:
                span_ratio = (vmax - vmin) / (high - low)

            # gap/coverage via quantile bins: fraction of non-empty bins
            non_empty_bins = np.nan
            bin_coverage = np.nan
            try:
                q = min(bins_for_gap, max(1, vals_all.nunique()))
                cats = pd.qcut(vals_all, q=q, duplicates="drop")
                non_empty_bins = int(cats.cat.categories.size)
                bin_coverage = non_empty_bins / q
            except Exception:
                pass

            num_rows.append({
                "parameter": col,
                "n_with_value": n_any,
                "n_complete": n_complete_param,
                "n_pruned": n_pruned_param,
                "n_fail": n_fail_param,
                "n_running": n_running_param,
                "unique": nunique,
                "unique_ratio": float(unique_ratio),
                "min": vmin, "p25": v25, "median": v50, "p75": v75, "max": vmax,
                "search_low": low, "search_high": high, "search_log": log_flag, "search_step": step,
                "span_ratio": float(span_ratio),
                "non_empty_bins": non_empty_bins, "bin_coverage": bin_coverage,
            })

        else:
            # categorical
            levels = s_all[has_val]
            counts = levels.value_counts(dropna=False)  # include actual nulls if any slipped
            nunique_obs = int(counts.size)

            # distribution info
            n_choices_decl = meta.get(pname, {}).get("n_choices")
            # Use declared choices if available; otherwise observed uniques
            n_base = int(n_choices_decl) if n_choices_decl else nunique_obs

            # coverage ratio
            coverage_ratio = np.nan
            if n_choices_decl:
                coverage_ratio = nunique_obs / n_choices_decl

            # imbalance metrics
            imbalance_ratio = float(counts.max() / max(1.0, counts.mean()))
            probs = (counts / counts.sum()).values
            ent = float(-(probs * np.log(probs + 1e-12)).sum())  # natural log
            max_ent = np.log(n_base) if n_base and n_base > 0 else np.nan
            ent_ratio = float(ent / max_ent) if (max_ent and np.isfinite(max_ent) and max_ent > 0) else np.nan

            # string of top levels
            top = [f"{lvl}({int(counts[lvl])})" for lvl in counts.index[:max_levels]]
            if counts.size > max_levels:
                top.append(f"… +{counts.size - max_levels}")

            cat_rows.append({
                "parameter": col,
                "n_with_value": n_any,
                "n_complete": n_complete_param,
                "n_pruned": n_pruned_param,
                "n_fail": n_fail_param,
                "n_running": n_running_param,
                "unique_levels": nunique_obs,
                "choices_declared": n_choices_decl,
                "coverage_ratio": coverage_ratio,
                "imbalance_ratio": imbalance_ratio,
                "entropy_ratio": ent_ratio,
                "top_levels(count)": ", ".join(top),
            })

    num_df = pd.DataFrame(num_rows)
    if "parameter" in num_df.columns and not num_df.empty:
        num_df = num_df.sort_values("parameter").reset_index(drop=True)

    cat_df = pd.DataFrame(cat_rows)
    if "parameter" in cat_df.columns and not cat_df.empty:
        cat_df = cat_df.sort_values("parameter").reset_index(drop=True)
    return _style_coverage_tables(num_df, cat_df)

# ---------------------------
# 1D marginals
# ---------------------------

@dataclass
class Binned:
    bins: pd.Categorical              # ordered categorical mapping row -> bin
    level: list[str]                  # human-readable labels in bin order
    x_center: np.ndarray              # centers aligned to bin order
    x_left: np.ndarray                # left edges (float) or NaNs
    x_right: np.ndarray               # right edges (float) or NaNs
    effective: str                    # "categorical" | "quantile" | "uniform" | "unique" | "degenerate"
    treat_as_categorical: bool
    degenerate: bool
    degenerate_info: dict | None      # for the <=1 unique numeric case

def _interval_midpoints(index: pd.Index) -> np.ndarray:
    mids = np.full(len(index), np.nan, float)
    if isinstance(index, pd.IntervalIndex):
        mids = (index.left.astype(float) + index.right.astype(float)) / 2.0
    return mids

def _clamp_first_left_edge(intervals: pd.IntervalIndex, xn: pd.Series, *, floor_to_zero: bool = False) -> pd.IntervalIndex:
    # finite-only min
    xf = pd.to_numeric(xn, errors="coerce").astype(float)
    xf = xf[np.isfinite(xf)]
    if xf.empty:
        return intervals  # nothing to clamp sensibly

    lo = float(xf.min())
    if floor_to_zero:
        lo = max(0.0, lo)

    L = intervals.left.astype(float).to_numpy()
    R = intervals.right.astype(float).to_numpy()

    # clamp first left edge to lo
    L[0] = lo

    # ensure the first interval is valid even if rounding produced R[0] < lo
    if R[0] <= lo:
        R[0] = lo

    return pd.IntervalIndex.from_arrays(L, R, closed=intervals.closed)

def _bin_param(
    d: pd.DataFrame,
    param_col: str,
    *,
    binning: str = "quantile",                               # "quantile" | "uniform" | "unique" | "custom"
    bins: int = 10,
    param_kind: str | None = None,                           # 'categorical' or 'numeric' (override dtype)
    custom_edges: Optional[Sequence[SupportsFloat]] = None,          # interior cut points, by value
    custom_quantiles: Optional[Sequence[float]] = None,      # interior cut points, by quantile in (0,1)
) -> Binned:
    x = _get_param(d, param_col)

    # Validate inputs
    if binning not in ("quantile", "uniform", "unique", "custom"):
        raise ValueError(f"Error while binning {param_col}: binning must be one of: 'quantile', 'uniform', 'unique', 'custom'")
    if binning == "custom" and (custom_edges is None and custom_quantiles is None):
        raise ValueError(f"Error while binning {param_col}: for binning='custom', provide custom_edges or custom_quantiles.")
    if binning == "custom" and (custom_edges is not None and custom_quantiles is not None):
        raise ValueError(f"Error while binning {param_col}: for binning='custom', provide only one of custom_edges or custom_quantiles.")
    if binning != "custom" and (custom_edges is not None or custom_quantiles is not None):
        _LOGGER.warning(f"Error while binning {param_col}: ignoring custom_edges/custom_quantiles since binning != 'custom'.")
    if param_kind not in (None, "categorical", "numeric"):
        raise ValueError(f"Error while binning {param_col}: param_kind must be one of: None, 'categorical', 'numeric'")
    
    # Decide treatment 
    if param_kind == "categorical":
        treat_as_categorical = True
    elif param_kind == "numeric":
        treat_as_categorical = False
    else: # param_kind is None
        treat_as_categorical = (not is_numeric_dtype(x)) or is_bool_dtype(x) or (x.nunique() <= bins)

    # --------- Categorical treatment ----------
    if treat_as_categorical:
        # Boolean special-case: enforce [False, True] order and include both levels
        if is_bool_dtype(x) or set(pd.Series(x).dropna().unique()).issubset({0, 1, True, False, "True", "False"}):
            x_norm = (
                pd.Series(x)
                .map({"True": True, "False": False, 1: True, 0: False, True: True, False: False})
            )
            bins_ordered = pd.Categorical(x_norm, categories=[False, True], ordered=True)
            level_index = bins_ordered.categories
            x_center = np.array([0.0, 1.0])
            x_left = np.array([np.nan, np.nan])
            x_right = np.array([np.nan, np.nan])
            level = [str(v) for v in level_index]

            return Binned(
                bins=bins_ordered,
                level=level,
                x_center=x_center,
                x_left=x_left,
                x_right=x_right,
                effective="categorical",
                treat_as_categorical=True,
                degenerate=False,
                degenerate_info=None,
            )
        
        # General categorical path (non-boolean)
        # Each distinct level is its own bin; preserve numeric order if labels are numeric-like
        bins_idx = x.astype("category")
        # Try to order by numeric value of the labels; else keep label order
        try:
            level_values = pd.to_numeric(bins_idx.cat.categories, errors="raise")
            order = np.argsort(level_values)
            ordered_levels = bins_idx.cat.categories[order]
            level_index = pd.CategoricalIndex(ordered_levels, ordered=True)
        except Exception:
            ordered_levels = bins_idx.cat.categories
            level_index = pd.CategoricalIndex(ordered_levels, ordered=True)

        # Recode bins to the ordered categories 
        bins_ordered = bins_idx.cat.set_categories(level_index, ordered=True)

        level_index = bins_ordered.cat.categories
        x_center = np.arange(len(level_index), dtype=float)
        x_left = np.full(len(level_index), np.nan)
        x_right = np.full(len(level_index), np.nan)
        level = level_index.astype(str).to_list()

        return Binned(
            bins=bins_ordered,
            level=level,
            x_center=x_center,
            x_left=x_left,
            x_right=x_right,
            effective="categorical",
            treat_as_categorical=True,
            degenerate=False,
            degenerate_info=None,
        )

    # --------- Numeric treatment ----------
    xn = pd.to_numeric(x, errors="coerce")
    if np.isinf(xn).any():
        raise ValueError(f"Error while binning {param_col}: non-finite values found.")
    xn_finite = xn
    n_unique = xn_finite.dropna().nunique()

    if n_unique <= 1:
        single_center = float(xn_finite.median())
        degenerate_info = {
            "level": ["all"],
            "x_center": np.array([single_center], dtype=float),
            "x_left": np.array([np.nan]),
            "x_right": np.array([np.nan]),
        }
        # ordered categorical with a single category
        bins_c = pd.Categorical(["all"] * len(d), categories=["all"], ordered=True)
        return Binned(
            bins=bins_c,
            level=["all"],
            x_center=np.array([single_center], dtype=float),
            x_left=np.array([np.nan]),
            x_right=np.array([np.nan]),
            effective="degenerate",
            treat_as_categorical=False,
            degenerate=True,
            degenerate_info=degenerate_info,
        )
    
    # Custom quantiles to numeric edges
    if binning == "custom" and custom_quantiles is not None:
        try:
            qs = np.asarray(list(custom_quantiles), dtype=float)
        except Exception as e:
            raise ValueError(f"Error while binning {param_col}: custom_quantiles must be a sequence of floats.") from e
        if qs.ndim != 1 or qs.size == 0:
            raise ValueError(f"Error while binning {param_col}: custom_quantiles must be a non-empty 1D sequence.")
        if np.any(~np.isfinite(qs)) or np.any((qs <= 0.0) | (qs >= 1.0)):
            raise ValueError(f"Error while binning {param_col}: custom_quantiles must be finite and in the open interval (0, 1).")
        qs = np.unique(qs)
        if xn_finite.empty:
            raise ValueError(f"Error while binning {param_col}: all values are NaN/inf; cannot compute quantiles.")
        q_vals = xn_finite.quantile(qs, interpolation="linear").to_numpy(dtype=float)
        custom_edges = q_vals  # fall-through to the custom_edges path

    # Custom numeric edges
    if binning == "custom" and custom_edges is not None:
        # accept ints, floats, numpy scalars → coerce by trying float conversion
        try:
            edges = np.asarray(list(custom_edges), dtype=float)
        except Exception as e:
            raise ValueError(f"Error while binning {param_col}: custom_edges must be a sequence of numeric values.") from e
        if edges.ndim != 1 or edges.size == 0:
            raise ValueError(f"Error while binning {param_col}: custom_edges must be a non-empty 1D sequence.")

        # dedupe + sort
        edges = np.unique(edges)

        # establish observed finite range
        if xn_finite.empty:
            raise ValueError(f"Error while binning {param_col}: all values are NaN/inf; cannot build custom bins.")
        lo = float(xn_finite.min())
        hi = float(xn_finite.max())
        if not (hi > lo):
            # degenerate case will be handled below, but guard anyway
            hi = lo

        # keep only strict interior edges
        edges = edges[(edges > lo) & (edges < hi)]
        breaks = np.concatenate(([lo], edges, [hi]))

        # build intervals, clamp first-left edge
        intervals = pd.IntervalIndex.from_breaks(breaks, closed="right")
        b = pd.cut(xn, bins=intervals, include_lowest=True)
        b = b.cat.set_categories(intervals, ordered=True)

        intervals = b.cat.categories
        intervals = _clamp_first_left_edge(intervals, xn, floor_to_zero=True)

        centers = _interval_midpoints(intervals)
        x_left = intervals.left.astype(float)
        x_right = intervals.right.astype(float)
        _fmtL = _best_num_formatter_for_series(x_left)
        _fmtR = _best_num_formatter_for_series(x_right)
        level = [f"[{_fmtL(x_left[0])}, {_fmtR(x_right[0])}]"] \
            + [f"({_fmtL(L)}, {_fmtR(R)}]" for L, R in zip(x_left[1:], x_right[1:])]

        return Binned(
            bins=b, level=level, x_center=centers, x_left=x_left, x_right=x_right,
            effective="custom", treat_as_categorical=False, degenerate=False, degenerate_info=None,
        )

    # Choose effective binning
    effective = "unique" if (binning == "quantile" and bins >= n_unique) else binning

    if effective == "quantile":
        # Build edges from finite data, then apply to full series
        q = int(min(bins, max(1, n_unique))) 
        b = pd.qcut(xn_finite, q=q, duplicates="drop")
        b = b.cat.set_categories(b.cat.categories, ordered=True)

        intervals = b.cat.categories  # IntervalIndex
        intervals = _clamp_first_left_edge(intervals, xn, floor_to_zero=True)  # set False if negatives are allowed

        centers = _interval_midpoints(intervals)
        x_left = intervals.left.astype(float)
        x_right = intervals.right.astype(float)
        _fmtL = _best_num_formatter_for_series(x_left)
        _fmtR = _best_num_formatter_for_series(x_right)
        level = [f"[{_fmtL(x_left[0])}, {_fmtR(x_right[0])}]"] \
            + [f"({_fmtL(L)}, {_fmtR(R)}]" for L, R in zip(x_left[1:], x_right[1:])]

        return Binned(
            bins=b, level=level, x_center=centers, x_left=x_left, x_right=x_right,
            effective="quantile", treat_as_categorical=False, degenerate=False, degenerate_info=None,
        )

    elif effective == "uniform":
        # Same heuristic cap + uniform bins via cut
        nb = min(bins, max(2, int(np.sqrt(n_unique))))
        b = pd.cut(xn, bins=nb, include_lowest=True)
        b = b.cat.set_categories(b.cat.categories, ordered=True)

        intervals = b.cat.categories  # IntervalIndex
        intervals = _clamp_first_left_edge(intervals, xn, floor_to_zero=True)  # set False if negatives are allowed

        centers = _interval_midpoints(intervals)
        x_left = intervals.left.astype(float)
        x_right = intervals.right.astype(float)
        _fmtL = _best_num_formatter_for_series(x_left)
        _fmtR = _best_num_formatter_for_series(x_right)
        level = [f"[{_fmtL(x_left[0])}, {_fmtR(x_right[0])}]"] \
            + [f"({_fmtL(L)}, {_fmtR(R)}]" for L, R in zip(x_left[1:], x_right[1:])]

        return Binned(
            bins=b,
            level=level,
            x_center=centers,
            x_left=x_left,
            x_right=x_right,
            effective="uniform",
            treat_as_categorical=False,
            degenerate=False,
            degenerate_info=None,
        )


    else:  # "unique"
        idx = pd.Index(xn_finite.dropna().unique()).astype(float).sort_values()
        bins_c = pd.Categorical(xn, categories=idx, ordered=True)

        x_center = idx.to_numpy(dtype=float)
        x_left = np.full(len(idx), np.nan)
        x_right = np.full(len(idx), np.nan)
        _fmt = _best_num_formatter_for_series(xn_finite)
        level = [_fmt(v) for v in idx]

        return Binned(
            bins=bins_c, level=level, x_center=x_center, x_left=x_left, x_right=x_right,
            effective="unique", treat_as_categorical=False, degenerate=False, degenerate_info=None,
        )

def marginal_1d(
    df: pd.DataFrame,
    param_col: str,
    val_name: str,
    *,
    objective: str | None = None, # -> val_name if None
    binning: str = "quantile",
    bins: int = 10,
    min_count: int = 2,
    compute_shares: bool = True,
    top_k: int = 10,
    top_frac: float = 0.20,
    minimize: bool = True,
    param_kind: str | None = None,
    custom_edges: Optional[Sequence[SupportsFloat]] = None,          
    custom_quantiles: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    
    # Validation
    if not isinstance(val_name, str) or not val_name or val_name not in df.columns:
        raise KeyError(f"Value column '{val_name}' not found in DataFrame or not a string.")
    if objective is None:
        objective = val_name
    elif not isinstance(objective, str) or objective not in df.columns:
        raise KeyError(f"Objective column '{objective}' not found in DataFrame or not a string.")

    # Filter df
    param_col = _param_col(param_col)
    if param_col not in df.columns:
        raise KeyError(f"Parameter column '{param_col}' not found in DataFrame.")
    d = df[(df["state"] == "COMPLETE") & df[objective].notna() & df[param_col].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=[
            "level","x_center","x_left","x_right","count","mean","median","std","p25","p75",
            "share_topK","share_topFrac"
        ])

    y = d[objective]

    # Bin and return
    binfo = _bin_param(
        d, param_col,
        binning=binning,
        bins=bins,
        param_kind=param_kind,
        custom_edges=custom_edges,
        custom_quantiles=custom_quantiles,
    )

    if binfo.degenerate:
        out = pd.DataFrame({
            "level": binfo.degenerate_info["level"],
            "x_center": binfo.degenerate_info["x_center"],
            "x_left": binfo.degenerate_info["x_left"],
            "x_right": binfo.degenerate_info["x_right"],
            "count": [len(d)],
            "mean": [y.mean()],
            "median": [y.median()],
            "std": [y.std(ddof=0)],
            "p25": [y.quantile(0.25)],
            "p75": [y.quantile(0.75)],
            "share_topK": [np.nan],
            "share_topFrac": [np.nan],
        })
        out.loc[out["count"] < int(min_count), ["mean","median","std","p25","p75"]] = np.nan
        return out

    d["__bin__"] = binfo.bins  # ordered categorical
    grp = d.groupby("__bin__", dropna=False, observed=False)

    # Full category index for all bins, including empty ones
    bin_index = d["__bin__"].cat.categories

    # Aggregations, reindexed so lengths match the bin metadata
    count  = grp.size().reindex(bin_index, fill_value=0)
    mean   = grp[objective].mean().reindex(bin_index)
    median = grp[objective].median().reindex(bin_index)
    std    = grp[objective].std(ddof=0).reindex(bin_index)
    p25    = grp[objective].quantile(0.25).reindex(bin_index)
    p75    = grp[objective].quantile(0.75).reindex(bin_index)

    out = pd.DataFrame({
        "level": binfo.level,
        "x_center": np.asarray(binfo.x_center, dtype=float),
        "x_left": np.asarray(binfo.x_left, dtype=float),
        "x_right": np.asarray(binfo.x_right, dtype=float),
        "count": count.to_numpy(),
        "mean": mean.to_numpy(),
        "median": median.to_numpy(),
        "std": std.to_numpy(),
        "p25": p25.to_numpy(),
        "p75": p75.to_numpy(),
    })

    if compute_shares:
        d_sorted = d.sort_values(objective, ascending=minimize)

        k = max(1, min(top_k, len(d_sorted)))
        if "number" in d_sorted.columns:
            best_ids = set(d_sorted.head(k)["number"])
            d["__is_topK"] = d["number"].isin(best_ids).astype(float)
        else:
            # fall back to rank-by-index if you want exact K even without an id
            d["__is_topK"] = 0.0
            d.loc[d_sorted.index[:k], "__is_topK"] = 1.0

        frac = max(1e-6, min(1.0, float(top_frac)))
        k_cut = max(1, int(np.ceil(frac * len(d_sorted))))
        cutoff = d_sorted.iloc[k_cut - 1][objective]
        d["__is_topFrac"] = (
            (d[objective] <= cutoff) if minimize else (d[objective] >= cutoff)
        ).astype(float)

        # --- Aggregate per bin ---
        bin_index = d["__bin__"].cat.categories
        grp = d.groupby("__bin__", observed=False)

        bin_count = grp.size().reindex(bin_index, fill_value=0)
        top_sums = grp[["__is_topK", "__is_topFrac"]].sum().reindex(bin_index, fill_value=0.0)

        # Prevalence (what you already expose)
        out["share_topK"]   = (top_sums["__is_topK"]   / bin_count.replace(0, np.nan)).to_numpy()
        out["share_topFrac"]= (top_sums["__is_topFrac"]/ bin_count.replace(0, np.nan)).to_numpy()

        # --- Composition: how the *global* top is composed by bins ---
        total_topK   = float(d["__is_topK"].sum())
        total_topFrac= float(d["__is_topFrac"].sum())

        out["contrib_topK"]    = (top_sums["__is_topK"]   / (total_topK   if total_topK   > 0 else np.nan)).to_numpy()
        out["contrib_topFrac"] = (top_sums["__is_topFrac"]/ (total_topFrac if total_topFrac> 0 else np.nan)).to_numpy()
    else:
        out["share_topK"] = out["share_topFrac"] = np.nan
        out["contrib_topK"] = out["contrib_topFrac"] = np.nan

    out.loc[out["count"] < int(min_count), ["mean","median","std","p25","p75","share_topK","share_topFrac"]] = np.nan
    out = out.sort_values("x_center", kind="mergesort").reset_index(drop=True)
    return out

def _build_xticklabels_from_table(
        tbl: pd.DataFrame | dict, 
        *,
        max_labels: int = 16, 
        force_levels_for_labels: bool = False
    ) -> tuple[np.ndarray, list[str]]:
    """Return (xticks, xlabels) built from table's x_center and interval edges if present."""
    x = tbl["x_center"].to_numpy()

    if force_levels_for_labels:
        labels = tbl["level"].astype(str).tolist()
    else:
        # treat as categorical (and use levels) if no edges, else use centers with best numeric formatting
        if "x_left" not in tbl.columns or tbl["x_left"].isna().all() or \
           "x_right" not in tbl.columns or tbl["x_right"].isna().all():
            labels = tbl["level"].astype(str).tolist()
        else:
            _fmt = _best_num_formatter_for_series(pd.Series(x))
            labels = [_fmt(v) for v in x]

    # downsample tick labels to avoid clutter
    n = len(x)
    if n > max_labels and n > 0:
        step = math.ceil(n / max_labels)
        keep_idx = np.arange(0, n, step, dtype=int)
        return x[keep_idx], [labels[i] for i in keep_idx]
    return x, labels

def plot_marginal_1d_on_ax(
    ax: plt.Axes,
    tbl: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    ylabel: str = "objective",
    use_semilogx: bool = False,
    use_median: bool = True,
    show_std: bool = True,
):
    """Same as plot_marginal_1d_from_table, but draws on a provided Matplotlib Axes."""
    if tbl.empty or tbl["x_center"].isna().all():
        ax.set_axis_off()
        return

    y = (tbl["median"] if use_median else tbl["mean"]).to_numpy()
    x = tbl["x_center"].to_numpy()
    s = tbl["std"].to_numpy()

    if use_semilogx:
        ax.set_xscale("symlog", linthresh=1e-2)
    ax.plot(x, y, marker="o")
    if show_std and np.isfinite(s).any():
        ax.fill_between(x, y - s, y + s, alpha=0.20, linewidth=0)

    xticks, xlabels = _build_xticklabels_from_table(tbl, max_labels=16)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def _make_grid(n_panels, *, panel_size=None, ncols=None):
    # auto columns
    if ncols is None:
        ncols = 1 if n_panels == 1 else (2 if n_panels == 2 else 3)
    nrows = math.ceil(n_panels / ncols)

    if panel_size is None:
        panel_size = (8,4) if ncols == 1  else (4,3)

    # compute global figsize
    fig_w = panel_size[0] * ncols
    fig_h = panel_size[1] * nrows

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        layout="constrained"   
    )
    return fig, axes

def _compute_allowed_cols(
    df: pd.DataFrame,
    params: Iterable[str] | None,
    non_params_to_allow: Iterable[str]
):
    if params is not None:
        params = [_param_col(p) for p in params]
    else:
        params = _param_cols(df)
    params_allowed = params
    
    non_params_allowed = []
    param_cols_df = _param_cols(df)
    for p in non_params_to_allow:
        if p not in df.columns:
            print(f"Warning: non-parameter column '{p}' not found in DataFrame; ignoring.")
        elif p in param_cols_df or _param_col(p) in param_cols_df:
            print(f"Warning: '{p}' looks like a parameter column but was included in `non_params_to_allow`; ignoring.")
        else:
            non_params_allowed.append(p)

    # all allowed columns
    allowed = params_allowed + non_params_allowed
    return allowed

def plot_marginals_1d(
    df: pd.DataFrame,
    val_name: str,
    *,
    objective: str = None, # -> val_name if None
    params: list[str] | None = None,
    non_params_to_allow: Iterable[str] = [],
    ncols: int | None = None,              # auto: 1 if 1 param, 2 if 2, else 3
    panel_size: tuple[float, float] | None = None,
    # marginal/plot settings
    binning_numeric: str = "quantile",
    custom_edges: Optional[Sequence[SupportsFloat]] = None,
    custom_quantiles: Optional[Sequence[float]] = None,
    bins_numeric: int = 8,
    min_count: int = 2,
    use_median: bool = True,
    show_std: bool = True,
    use_semilogx: list[str] = ["train.learning_rate"],
    param_kinds: dict[str, str] | None = None,
    title_prefix: str = "Marginal of ",

):
    # Validation
    if not isinstance(val_name, str) or not val_name or val_name not in df.columns:
        raise KeyError(f"Value column '{val_name}' not found in DataFrame or not a string.")
    if objective is None:
        objective = val_name
    elif not isinstance(objective, str) or objective not in df.columns:
        raise KeyError(f"Objective column '{objective}' not found in DataFrame or not a string.")
    
    # Ensure using full params names
    use_semilogx_cols = [_param_col(p) for p in use_semilogx]

    # Compute allowed cols
    allowed = _compute_allowed_cols(df, params, non_params_to_allow)
    if not allowed:
        print("No parameters or non-parameter columns to plot.")
        return

    # Prepare grid
    n = len(allowed)
    fig, axes = _make_grid(n, panel_size=panel_size, ncols=ncols)
    nrows, ncols = axes.shape

    plotted = False
    for i, pcol in enumerate(allowed):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        kind = (param_kinds or {}).get(pcol, _guess_param_kind(_get_param(df, pcol)))
        tbl = marginal_1d(
            df, pcol, val_name,
            objective=objective,
            binning=(binning_numeric if kind == "numeric" else "unique"),
            custom_edges=custom_edges,
            custom_quantiles=custom_quantiles,
            bins=bins_numeric, min_count=min_count,
            compute_shares=False, minimize=True,
            param_kind=kind
        )

        semilog_this = (pcol in use_semilogx_cols) and (kind == "numeric")
        plot_marginal_1d_on_ax(
            ax, tbl,
            title=f"{title_prefix}{pcol.replace('params_', '', 1)}",
            xlabel="",
            ylabel=val_name if c == 0 else "",
            use_median=use_median,
            show_std=show_std,
            use_semilogx=semilog_this,
        )
        if not tbl.empty:
            plotted = True

    # hide unused
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_axis_off()

    if plotted:
        fig.suptitle("Marginal Parameter Effects")
        plt.show()
    else:
        plt.close(fig)
        print("Nothing to plot.")

def display_marginal_1d(
    tbl: pd.DataFrame,
    *,
    minimize: bool = True,
    include_spread: bool = True,   # include p25/p75/std if supported
    include_shares: bool = True,   # include share_topK/share_topFrac if present
) -> Styler:
    """
    Pretty display for a 1D marginal table produced by `marginal_1d`.

    - Auto-detects kind when not provided.
    - Hides irrelevant columns.
    - Formats floats compactly; percentages for share columns.
    - Green gradient on the chosen stat, blue gradient on Std.
    - For interval-binned numerics, Level is rendered as "(left, right]" with formatted bounds.
    """

    # helper
    def _has(col):
        return (col in tbl.columns) and (not tbl[col].isna().all())

    # Build a working copy
    df = tbl.copy()

    # --- pick columns
    cols = ["level", "count", "mean", "median"]

    # spread
    if include_spread:
        for c in ("p25","p75","std"):
            if _has(c): cols.append(c)

    # shares
    if include_shares:
        for c in ("share_topK","share_topFrac", "contrib_topK","contrib_topFrac"):
            if _has(c): cols.append(c)

    # ensure existence & filter
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # nice display names
    rename_map = {
        "level": "Level",
        "count": "Count",
        "mean": "Mean Value",
        "median": "Median Value",
        "p25": "P25",
        "p75": "P75",
        "std": "Std",
        "share_topK": "Top-K share",
        "share_topFrac": "Top-Frac share",
        "contrib_topK": "Top-K composition",
        "contrib_topFrac": "Top-Frac composition",
    }
    df = df.rename(columns=rename_map)
    print(df.columns)

    # formatting
    fmt_map = {}
    for c in df.columns:
        if c in ("Level",):
            continue
        if c in ("Count",):
            fmt_map[c] = "{:,.0f}".format
        elif c in ("Top-K composition","Top-Frac composition"):
            fmt_map[c] = _pct_fmt
        else:
            fmt_map[c] = _float_fmt

    # color gradients
    styler = (
        df.style
        .hide(axis="index")
        .format(fmt_map, na_rep="—") 
    )
    # green on chosen stat
    to_prettify = [rename_map[s] for s in ("median", "mean", "p25", "p75") if s in cols]
    if to_prettify:
        styler = styler.background_gradient(
            subset=to_prettify,
            cmap=("Greens_r" if minimize else "Greens")
        )
    to_prettify_shares = [rename_map[s] for s in ("share_topK","share_topFrac","contrib_topK","contrib_topFrac") if s in cols]
    if to_prettify_shares:
        styler = styler.background_gradient(
            subset=to_prettify_shares,
            cmap=("Greens")
        )

    # blue on Std if present
    if "Std" in df.columns:
        styler = styler.background_gradient(subset=["Std"], cmap="Blues")

    return styler

def display_marginals_1d(
    df: pd.DataFrame,
    val_name: str,
    *,
    objective: str | None = None, # -> val_name if None
    params: list[str] | None = None,
    non_params_to_allow: Iterable[str] = [],
    binning_numeric: str = "quantile",
    custom_edges_dict: Dict[str, Optional[Sequence[SupportsFloat]]] = {},
    custom_quantiles_dict: Dict[str, Optional[Sequence[float]]] = {},
    bins_numeric: int = 8,
    top_k: int = 10,
    top_frac: float = 0.20,
    minimize: bool = True,
    include_spread: bool = True,
    include_shares: bool = True,
    param_kinds: dict[str, str] | None = None,
    display_tbls: bool = True,
) -> dict[str, Styler]:
    """
    Return dict of {param_name: Styler} for 1D marginals of specified params.

    If no params specified, all param columns in df are used.
    """
    # Validation
    if not isinstance(val_name, str) or not val_name or val_name not in df.columns:
        raise KeyError(f"Value column '{val_name}' not found in DataFrame or not a string.")
    if objective is None:
        objective = val_name
    elif not isinstance(objective, str) or objective not in df.columns:
        raise KeyError(f"Objective column '{objective}' not found in DataFrame or not a string.")
    
    non_params_to_allow = tuple(non_params_to_allow or ())
    custom_edges_dict = custom_edges_dict or {}
    custom_quantiles_dict = custom_quantiles_dict or {}
    
    # all allowed columns
    allowed = _compute_allowed_cols(df, params, non_params_to_allow)
    if not allowed:
        print("No parameters or non-parameter columns to display.")
        return {}

    out = {}
    for pcol in allowed:
        kind = (param_kinds or {}).get(pcol, _guess_param_kind(df[pcol]))
        pname = pcol.replace("params_", "")

        # fetch custom specs without using `or` (so [] isn't treated as missing)
        custom_edges = custom_edges_dict[pcol] if pcol in custom_edges_dict else custom_edges_dict.get(pname)
        custom_quants = custom_quantiles_dict[pcol] if pcol in custom_quantiles_dict else custom_quantiles_dict.get(pname)

        # decide binning for this column only
        use_custom = (custom_edges is not None) or (custom_quants is not None)
        binning_for_this = ("custom" if use_custom else (binning_numeric if kind == "numeric" else "unique"))

        tbl = marginal_1d(
            df, pcol, val_name,
            objective=objective,
            binning=binning_for_this,
            bins=bins_numeric, min_count=2,
            compute_shares=True, 
            top_k=top_k, top_frac=top_frac,
            minimize=minimize,
            param_kind=kind,
            custom_edges=custom_edges,
            custom_quantiles=custom_quants,
        )
        sty = display_marginal_1d(
            tbl, minimize=minimize,
            include_spread=include_spread,
            include_shares=include_shares
        )
        if display_tbls:
            _LOGGER.info(f" === {pcol.replace('params_', '', 1)} === ")
            display(sty)
        out[pcol.replace("params_", "", 1)] = sty
    return out

def plot_param_importances(
    imps: pd.Series,
    *,
    top_n: int | None = None,
    normalize: bool = True,
    annotate: bool = True,
):
    # make sure we have importances as an ordered pd.Series
    s = pd.Series(imps, dtype=float).sort_values(ascending=False)

    if top_n is not None:
        s = s.iloc[:top_n]

    if normalize:
        total = s.sum()
        if total > 0:
            s = s / total

    # plot (height scales with number of params)
    n = len(s)
    fig_h = max(2.5, min(0.45 * n + 1.0, 12))  # compact but readable
    fig, ax = plt.subplots(figsize=(8, fig_h), constrained_layout=True)

    ax.barh(s.index, s.values) 
    ax.invert_yaxis()           # largest on top
    ax.set_xlabel("Importance")
    ax.set_title("Parameter Importances")

    # x axis as percent if normalized
    if normalize:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _ : f"{x:.0%}"))
        ax.set_xlim(0, 1)

    # annotate bars
    if annotate:
        x_max = ax.get_xlim()[1]
        for y, v in enumerate(s.values):
            label = f"{v:.1%}" if normalize else f"{v:.3g}"
            ax.text(min(v, x_max) + 0.01 * x_max, y, label, va="center")

    return fig, ax

# ---------------------------
# Two-parameter interaction views
# ---------------------------

def _assign_left_edge_nans_to_first_bin(
    d: pd.DataFrame,
    *,
    value_col: str,    # the raw numeric series we binned (e.g., param_a)
    bin_col: str,      # the categorical bins column (e.g., "__A__")
) -> None:
    """
    If any rows have NaN in `bin_col` but the underlying value is <= first bin's right edge,
    force-assign them to the first bin. Works for IntervalIndex bins only. In-place on `d`.
    """
    b = d[bin_col]
    if not isinstance(getattr(b, "dtype", None), pd.CategoricalDtype):
        return  # nothing to do

    cats = b.cat.categories
    if not isinstance(cats, pd.IntervalIndex) or len(cats) == 0:
        return  # only meaningful for interval bins

    first = cats[0]
    # numeric values (coerce and ignore non-finite)
    x = pd.to_numeric(d[value_col], errors="coerce").astype(float)

    # robust "≤ first.right" with a tiny tolerance
    # nextafter handles exact-endpoint and floating rounding
    right_edge = float(np.nextafter(float(first.right), np.inf))

    # mask: currently NaN bin AND value ≤ first.right
    mask = b.isna() & x.le(right_edge)

    if mask.any():
        b = b.copy()                  # make it writeable
        b.loc[mask] = first           # assign the first category
        d[bin_col] = b                # write back


def marginal_2d(
    df: pd.DataFrame,
    param_a: str,
    param_b: str,
    val_name: str,
    *,
    objective: str | None = None, # -> val_name if None
    binning: str = "quantile",   # "quantile" | "uniform" | "unique"
    bins_a: int = 5,
    bins_b: int = 5,
    custom_edges_a: Optional[Sequence[SupportsFloat]] = None,
    custom_quantiles_a: Optional[Sequence[float]] = None,
    custom_edges_b: Optional[Sequence[SupportsFloat]] = None,
    custom_quantiles_b: Optional[Sequence[float]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Build pivot tables of objective statistics for interactions between two params.
    Returns dict with keys: 'median', 'mean', 'std', 'count'.
    Index: bins/levels of A. Columns: bins/levels of B.
    """
    # Validation
    if not isinstance(val_name, str) or not val_name or val_name not in df.columns:
        raise KeyError(f"Value column '{val_name}' not found in DataFrame or not a string.")
    if objective is None:
        objective = val_name
    elif not isinstance(objective, str) or objective not in df.columns:
        raise KeyError(f"Objective column '{objective}' not found in DataFrame or not a string.")
    if not binning in ("quantile", "uniform", "unique"):
        raise ValueError(f"Invalid binning method '{binning}'; must be one of 'quantile', 'uniform', or 'unique'.")

    # filter rows
    param_a = _param_col(param_a)
    param_b = _param_col(param_b)
    d = df[
        (df["state"] == "COMPLETE")
        & df[objective].notna()
        & df[param_a].notna()
        & df[param_b].notna()
    ].copy()
    if d.empty:
        empty = {"median": pd.DataFrame(), "mean": pd.DataFrame(), "std": pd.DataFrame(), "count": pd.DataFrame()}
        return empty

    # bin both params (ordered categoricals)
    if custom_edges_a is not None or custom_quantiles_a is not None:
        binning_a = "custom"
    else:
        binning_a = binning
    if custom_edges_b is not None or custom_quantiles_b is not None:
        binning_b = "custom"
    else:
        binning_b = binning
    binfo_a = _bin_param(d, param_a, binning=binning_a, bins=bins_a,
                         custom_edges=custom_edges_a, custom_quantiles=custom_quantiles_a)
    binfo_b = _bin_param(d, param_b, binning=binning_b, bins=bins_b,
                         custom_edges=custom_edges_b, custom_quantiles=custom_quantiles_b)

    d["__A__"] = binfo_a.bins
    d["__B__"] = binfo_b.bins
    _assign_left_edge_nans_to_first_bin(d, value_col=param_a, bin_col="__A__")
    _assign_left_edge_nans_to_first_bin(d, value_col=param_b, bin_col="__B__")

    # observed=False -> include empty categories (full A×B grid)
    grp = d.groupby(["__A__", "__B__"], dropna=False, observed=False)
    agg = grp[objective].agg(["median", "mean", "std", "count"])

    # pivot each stat (no reset_index / no manual reindex needed)
    piv_median = agg["median"].unstack("__B__")
    piv_mean   = agg["mean"].unstack("__B__")
    piv_std    = agg["std"].unstack("__B__")
    piv_count  = agg["count"].unstack("__B__").fillna(0).astype(int)

    # replace axis tick labels with human-readable levels & rename clumns/index
    pa_name = param_a.replace("params_", "", 1)
    pb_name = param_b.replace("params_", "", 1)
    def _relabel_axis(piv: pd.DataFrame) -> pd.DataFrame:
        # Row labels (A)
        if len(binfo_a.level) == len(piv.index):
            piv.index = pd.Index(binfo_a.level, name=piv.index.name)
        # Column labels (B)
        if len(binfo_b.level) == len(piv.columns):
            piv.columns = pd.Index(binfo_b.level, name=piv.columns.name)
        piv = piv.rename_axis(index=pa_name)
        piv = piv.rename_axis(columns=pb_name)
        return piv

    return {
        "median": _relabel_axis(piv_median.copy()),
        "mean":   _relabel_axis(piv_mean.copy()),
        "std":    _relabel_axis(piv_std.copy()),
        "count":  _relabel_axis(piv_count.copy()),
        "binfo": (binfo_a, binfo_b),
    }

def _format_intervals_like_user_wants(index_like) -> list[str]:
    """
    If index_like is an IntervalIndex, return strings like "(L, R]” using:
      _fmtL = _best_num_formatter_for_series(x_left)
      _fmtR = _best_num_formatter_for_series(x_right)
    Else, return str(label) for each entry.
    """
    if isinstance(index_like, pd.IntervalIndex):
        x_left = index_like.left.to_numpy()
        x_right = index_like.right.to_numpy()
        _fmtL = _best_num_formatter_for_series(x_left)
        _fmtR = _best_num_formatter_for_series(x_right)
        return [f"({_fmtL(L)}, {_fmtR(R)}]" for L, R in zip(x_left, x_right)]
    else:
        return [str(v) for v in index_like]

import re

# Match interval-like strings: "(a, b]", "[a,b)", etc. (flexible whitespace & number formats)
_INTERVAL_RE = re.compile(
    r"""^\s*[\(\[]\s*         # opening bracket
        [+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*,\s*  # left number
        [+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*      # right number
        [\)\]]\s*$         # closing bracket
    """,
    re.VERBOSE,
)

def _infer_center_labels_from_levels(
    levels: Sequence[str],
    centers: Sequence[float],
) -> List[str]:
    """
    If *all* `levels` look like interval strings (e.g., "(a, b]", "[a, b)"),
    return the numeric `centers` sequence converted to strings. Otherwise return the original `levels`.

    Returns:
        list[float] if interval-like; else list[str].
    """
    # Fast exit for empty input
    if not levels:
        return list(levels)  # empty list

    # Check if every level matches the interval pattern
    all_interval = True
    for s in levels:
        if not isinstance(s, str) or _INTERVAL_RE.match(s) is None:
            all_interval = False
            break
    if all_interval:
        _fmt = _best_num_formatter_for_series(np.array(centers))
        return [_fmt(v) for v in centers]

    return list(levels)

def plot_marginal_2d(
    pivots: dict[str, Any],
    statistic: str,
    objective: str, # make sure it alignes with the one used in marginal_2d
    *,
    title: str,
    minimize: bool = True,
    colorscale_minimize: str = "Viridis",
    colorscale_maximize: str = "Viridis_r",
    colorbar_title: str | None = None,
    show_text: bool = False,
    text_fmt: str = ".3g",
    hover_stats: Sequence[str] | None = None,
    hover_value_fmt: str = ".4g",
) -> go.Figure:
    """
    Simple Plotly heatmap for a two_param_summary pivot.

    - Tick *positions* use bin centers from pivots["binfo"] = (binfo_a, binfo_b), else positional centers.
    - Tick *labels* prefer centers (formatted) when available, else the pivot's level labels.
    - Hover shows param names + *level labels* (not centers).
    - Axis titles use the param names when available.
    """
    if statistic not in pivots:
        raise ValueError(f"Statistic '{statistic}' not in pivots: {list(pivots.keys())}")

    pivot: pd.DataFrame = pivots[statistic]
    if pivot.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} — (empty)")
        return fig

    Z = pivot.to_numpy(dtype=float)
    n_rows, n_cols = Z.shape

    # infer param names (A=rows/index, B=cols/columns)
    pname_a = pivot.index.name if pivot.index.name else "param A"
    pname_b = pivot.columns.name if pivot.columns.name else "param B"

    # centers: prefer from binfo (binfo_a = rows, binfo_b = cols), else from IntervalIndex, else positional
    binfo = pivots.get("binfo")

    def _safe_centers(b) -> np.ndarray | None:
        if not hasattr(b, "x_center"):
            return None
        arr = np.asarray(getattr(b, "x_center"), dtype=float)  # coerce to float array
        if arr.ndim != 1 or not np.isfinite(arr).all():
            return None
        return arr

    x_centers = y_centers = None

    if isinstance(binfo, tuple) and len(binfo) == 2:
        binfo_a, binfo_b = binfo  # rows=A, cols=B
        yc = _safe_centers(binfo_a)
        xc = _safe_centers(binfo_b)

        if yc is not None and len(yc) == n_rows:
            y_centers = yc
        if xc is not None and len(xc) == n_cols:
            x_centers = xc

    # If still None (should never happen), fall back to positional
    if x_centers is None:
        x_centers = np.arange(n_cols, dtype=float)  # positional fallback
    if y_centers is None:
        y_centers = np.arange(n_rows, dtype=float)  # positional fallback

    if len(x_centers) != n_cols or len(y_centers) != n_rows:
        raise ValueError("x_centers / y_centers lengths must match pivot shape.")

    # z-range & colorscale
    if not np.isfinite(Z).any():
        fig = go.Figure()
        fig.update_layout(title=f"{title} — (all values are NaN)")
        return fig

    zmin = float(np.nanmin(Z))
    zmax = float(np.nanmax(Z))
    if zmin == zmax:
        eps = 1e-12
        zmin, zmax = zmin - eps, zmax + eps

    colorscale = colorscale_minimize if minimize else colorscale_maximize

    # Optional per-cell text
    text = None
    if show_text:
        text = np.empty_like(Z, dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                text[i, j] = (format(Z[i, j], text_fmt) if np.isfinite(Z[i, j]) else "")

    # tick labels: centers if available, else level labels
    x_ticktext = _infer_center_labels_from_levels(pivot.columns.astype(str).tolist(), x_centers)
    y_ticktext = _infer_center_labels_from_levels(pivot.index.astype(str).tolist(), y_centers)

    # Hover uses param names + interval labels (formatted), else raw labels if not intervals
    row_hover_labels = _format_intervals_like_user_wants(pivot.index)
    col_hover_labels = _format_intervals_like_user_wants(pivot.columns)

    row_labels_arr = np.asarray(row_hover_labels, dtype=object)
    col_labels_arr = np.asarray(col_hover_labels, dtype=object)

    # Find all DF-like stats with matching shape; default to all of them
    def _is_eligible_stat(k: str, v: Any) -> bool:
        return isinstance(v, pd.DataFrame) and v.shape == pivot.shape and list(v.index) == list(pivot.index) and list(v.columns) == list(pivot.columns)

    all_stat_names = [k for k, v in pivots.items() if _is_eligible_stat(k, v)]
    # If user specified some, respect order and intersect
    if hover_stats is not None:
        wanted = [s for s in hover_stats if s in all_stat_names]
        # ensure we still show at least the heatmap's statistic
        if statistic not in wanted and statistic in all_stat_names:
            wanted = [statistic] + wanted
        hover_stat_names = wanted
    else:
        hover_stat_names = all_stat_names  # show everything we can

    # Build arrays (reindex_like for safety)
    stat_arrays = []
    for s in hover_stat_names:
        arr = pivots[s].reindex_like(pivot).to_numpy(dtype=float)
        stat_arrays.append(arr)

    # customdata = [row_label, col_label, stat1, stat2, ...]
    # we keep labels as strings, stats as floats (so we can format with :<fmt>)
    # Shape: (n_rows, n_cols, 2 + num_stats)
    customdata = np.concatenate(
        [
            np.stack(
                [
                    np.broadcast_to(row_labels_arr[:, None], Z.shape),
                    np.broadcast_to(col_labels_arr[None, :], Z.shape),
                ],
                axis=-1,
            ),
            np.stack(stat_arrays, axis=-1) if stat_arrays else np.empty(Z.shape + (0,), dtype=float),
        ],
        axis=-1,
    )

    # Build hovertemplate dynamically:
    # first lines = param labels; then one line per statistic with formatting
    lines = [
        f"{pname_a}: %{{customdata[0]}}",
        f"{pname_b}: %{{customdata[1]}}",
    ]
    # Add one line per stat (order matches hover_stat_names)
    # customdata indices for stats start at 2
    for i, s in enumerate(hover_stat_names):
        # If this stat is the color stat, show "(heatmap)" tag so it's obvious
        tag = " (heatmap)" if s == statistic else ""
        lines.append(f"{s}{tag}: %{{customdata[{2+i}]:{hover_value_fmt}}}")
    hovertemplate = "<br>".join(lines) + "<extra></extra>"

    # --- heatmap ---
    hm = go.Heatmap(
        x=x_centers, y=y_centers, z=Z,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(title=colorbar_title or f"{objective} ({statistic})"),
        hovertemplate=hovertemplate,
        customdata=customdata,
        text=text,
        texttemplate="%{text}" if show_text else None,
        showscale=True,
    )
    fig = go.Figure(hm)

    # --- axes: ticks at centers; titles from param names; labels from x/y_ticktext ---
    fig.update_xaxes(
        title_text=pname_b,
        tickmode="array", tickvals=x_centers, ticktext=x_ticktext,
        tickangle=45
    )
    fig.update_yaxes(
        title_text=pname_a,
        tickmode="array", tickvals=y_centers, ticktext=y_ticktext
    )

    fig.update_layout(
        title=title,
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig

def plot_marginals_2d(
    df: pd.DataFrame,
    val_name: str,
    *,
    objective: str | None = None,
    params: Iterable[str] | None = None,
    non_params_to_allow: Iterable[str] = [],
    as_first: str | None = None,          # param to pin to first axis (if present)
    statistic: Literal["median", "mean", "std", "count"] = "median",
    title: str | None = None,
    # binning / pivot controls (passed into marginal_2d)
    binning: Literal["quantile", "uniform", "unique"] = "quantile",
    bins_a: int = 5,
    bins_b: int | None = None,  # if None -> bins_a
    custom_edges_dict_a: Dict[str, Sequence[SupportsFloat]] = {},
    custom_quantiles_dict_a: Dict[str, Sequence[float]] = {},
    custom_edges_dict_b: Dict[str, Sequence[SupportsFloat]] = {},
    custom_quantiles_dict_b: Dict[str, Sequence[float]] = {},
    # plotting controls (passed into plot_marginal_2d)
    minimize: bool = True,
    colorscale_minimize: str = "Viridis",
    colorscale_maximize: str = "Viridis_r",
    show_text: bool = False,
    text_fmt: str = ".3g",
    # layout controls
    ncols: int = 2,
    share_colorscale: bool = True,      # uniform zmin/zmax across all subplots
    show_single_colorbar: bool = True,  # show one colorbar (right side)
) -> tuple[go.Figure, dict[tuple[str, str], dict[str, pd.DataFrame]]]:
    """
    Build a grid of heatmaps for all pairwise combos of `params` using `marginal_2d`
    and `plot_marginal_2d` (your single-heatmap helper).

    Returns (figure, pivots_by_pair). Keys of pivots_by_pair are (param_a, param_b).
    """
    # Validation
    if not isinstance(val_name, str) or not val_name or val_name not in df.columns:
        raise KeyError(f"Value column '{val_name}' not found in DataFrame or not a string.")
    if objective is None:
        objective = val_name
    elif not isinstance(objective, str) or objective not in df.columns:
        raise KeyError(f"Objective column '{objective}' not found in DataFrame or not a string.")
    
    bins_b = bins_b if bins_b is not None else bins_a

    # all allowed columns
    allowed = _compute_allowed_cols(df, params, non_params_to_allow)
    if len(allowed) < 2:
        _LOGGER.warning("Need at least two parameters or non-parameter columns to plot pairwise marginals.")
        fig = go.Figure()
        fig.update_layout(title=title or "Pairwise heatmaps (not enough parameters)")
        return fig, {}

    # --- build & order pairs, with optional pinning of one param to an axis ---
    all_pairs = list(itertools.combinations(allowed, 2))

    def _pin(pair: tuple[str, str]) -> tuple[str, str]:
        a, b = pair
        if as_first is None:
            return a, b
        if a == as_first:
            return b, a
        return a, b

    if as_first:
        as_first = _param_col(as_first)
        if as_first not in allowed:
            _LOGGER.warning(f"as_first param '{as_first}' not found in allowed params; ignoring pinning.")
            as_first = None
            ordered_pairs = all_pairs
        else:
            priority = []
            others = []
            for p in all_pairs:
                if as_first in p:
                    priority.append(_pin(p))
                else:
                    others.append(p)
            # keep priority first (already pinned), then the rest (original orientation)
            ordered_pairs = priority + others
    else:
        ordered_pairs = all_pairs

    if not ordered_pairs:
        fig = go.Figure()
        fig.update_layout(title=title or "Pairwise heatmaps (no parameter pairs)")
        return fig, {}

    # --- compute pivots + global z-range ---
    pivots_by_pair: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    global_min = float("inf")
    global_max = float("-inf")

    for a, b in ordered_pairs:
        piv = marginal_2d(
            df,
            param_a=a,
            param_b=b,
            val_name=val_name,
            objective=objective,
            binning=binning,
            bins_a=bins_a,
            bins_b=bins_b,
            custom_edges_a=custom_edges_dict_a.get(a) or custom_edges_dict_a.get(a.replace("params_", "", 1)),
            custom_quantiles_a=custom_quantiles_dict_a.get(a) or custom_quantiles_dict_a.get(a.replace("params_", "", 1)),
            custom_edges_b=custom_edges_dict_b.get(b) or custom_edges_dict_b.get(b.replace("params_", "", 1)),
            custom_quantiles_b=custom_quantiles_dict_b.get(b) or custom_quantiles_dict_b.get(b.replace("params_", "", 1)),
        )
        pivots_by_pair[(a, b)] = piv

        if statistic in piv and not piv[statistic].empty:
            Z = piv[statistic].to_numpy(dtype=float)
            if np.isfinite(Z).any():
                zmin = float(np.nanmin(Z))
                zmax = float(np.nanmax(Z))
                global_min = min(global_min, zmin)
                global_max = max(global_max, zmax)

    # fallback if everything was empty/NaN or flat
    if not np.isfinite(global_min) or not np.isfinite(global_max) or global_min == global_max:
        global_min, global_max = 0.0, 1.0

    # --- create subplot grid ---
    n = len(ordered_pairs)
    ncols = max(1, int(ncols))
    nrows = math.ceil(n / ncols)

    subplot_titles = [f"{a.replace('params_', '')} x {b.replace('params_', '')}" for (a, b) in ordered_pairs]
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
    )

    # --- add each subplot using your helper ---
    for k, (a, b) in enumerate(ordered_pairs, start=1):
        r = (k - 1) // ncols + 1
        c = (k - 1) % ncols + 1

        piv = pivots_by_pair[(a, b)]
        subfig = plot_marginal_2d(  # your single-heatmap function
            pivots=piv,
            statistic=statistic,
            objective=objective,
            title="",  # we'll use subplot_titles
            minimize=minimize,
            colorscale_minimize=colorscale_minimize,
            colorscale_maximize=colorscale_maximize,
            colorbar_title=statistic,
            show_text=show_text,
            text_fmt=text_fmt,
        )

        # Empty pivot → annotate and still propagate axis titles if present
        if len(subfig.data) == 0:
            fig.add_annotation(row=r, col=c, text="(empty)", showarrow=False, font=dict(size=12))
            fig.update_xaxes(title_text=subfig.layout.xaxis.title.text or b, row=r, col=c)
            fig.update_yaxes(title_text=subfig.layout.yaxis.title.text or a, row=r, col=c)
            continue

        trace = subfig.data[0]

        # Share colorscale: either via per-trace zmin/zmax, or a true shared coloraxis
        if share_colorscale and not show_single_colorbar:
            trace.update(zmin=global_min, zmax=global_max)
        if show_single_colorbar:
            # we'll attach all traces to a single coloraxis later
            trace.update(showscale=False)

        fig.add_trace(trace, row=r, col=c)

        # Copy axis formatting (tickvals/ticktext, titles) from the subfig
        xax = subfig.layout.xaxis
        yax = subfig.layout.yaxis
        fig.update_xaxes(
            title_text=(xax.title.text or b),
            tickmode=getattr(xax, "tickmode", "array"),
            tickvals=getattr(xax, "tickvals", None),
            ticktext=getattr(xax, "ticktext", None),
            tickangle=getattr(xax, "tickangle", 45),
            automargin=True,                 # <<< let Plotly make room for ticks/title
            title_standoff=8,                # <<< small gap between axis and title
            row=r, col=c,
        )
        fig.update_yaxes(
            title_text=(yax.title.text or a),
            tickmode=getattr(yax, "tickmode", "array"),
            tickvals=getattr(yax, "tickvals", None),
            ticktext=getattr(yax, "ticktext", None),
            automargin=True,
            title_standoff=8,
            row=r, col=c,
        )

    # --- layout + spacing polish ---
    main_title = title or f"Pairwise heatmaps — {statistic}"
    fig.update_layout(
        title=dict(text=main_title),
        height=375*nrows, 
        width=min(1200, 500*ncols),  
    )
    fig.update_xaxes(automargin=True, title_standoff=20)
    fig.update_yaxes(automargin=True, title_standoff=20)

    # --- de-collide subplot titles (they're annotations) ---
    # anchor from the bottom and push up a bit
    for ann in (fig.layout.annotations or []):
        ann.update(yanchor="bottom", yshift=5)

    # --- single shared colorbar via coloraxis (cleaner spacing) ---
    if show_single_colorbar:
        colorscale = colorscale_minimize if minimize else colorscale_maximize
        fig.update_layout(
            coloraxis=dict(
                cmin=global_min, cmax=global_max,
                colorscale=colorscale,
                colorbar=dict(title=statistic, x=1.02)
            )
        )
        # Attach all heatmaps to that shared coloraxis
        for tr in fig.data:
            if isinstance(tr, go.Heatmap):
                tr.update(coloraxis="coloraxis")

    return fig, pivots_by_pair

# ---------------------------
# Wrappers around Optuna visualizations
# ---------------------------

import optuna.visualization as ov

def _fmt_params(*, params: dict, max_param_items: int, sort_params: bool) -> str:
    items = sorted(params.items()) if sort_params else list(params.items())
    if max_param_items and len(items) > max_param_items:
        items = items[:max_param_items] + [("…", f"+{len(params)-max_param_items} more")]
    return "<br>".join(f"{k}: {v!r}" for k, v in items)

def plot_intermediate_values(
    study: optuna.study.Study,
    *,
    val_name: str = "value",
    max_param_items: int = 30,
    sort_params: bool = True,
    include_params: Optional[Dict[str, Any]] = None,
    exclude_params: Optional[Dict[str, Any]] = None,
    predicate: Optional[Callable[[optuna.trial.FrozenTrial], bool]] = None,
    dim_excluded: bool = False,
    dim_factor: float = 0.1,
    semilogy: bool = False,
):
    """
    Plot intermediate values with hover text showing trial params, with optional filtering.

    Parameters
    ----------
    study : optuna.study.Study
        The Optuna study to visualize.
    val_name : str, default "value"
        Label for target on the Y axis (also shown in hover).
    max_param_items : int, default 30
        Maximum number of params to display in tooltip (truncated if exceeded).
    sort_params : bool, default True
        Sort params by key name in tooltip.
    include_params : dict[str, Any], optional
        Keep only trials whose params match ALL key/value pairs (==).
    exclude_params : dict[str, Any], optional
        Drop (or dim) trials whose params match ANY key/value pairs (==).
    predicate : callable(trial)->bool, optional
        Custom boolean filter applied after include/exclude dicts.
    dim_excluded : bool, default False
        If True, excluded trials are shown at opacity=0.3 instead of removed.
    dim_factor : float, default 0.1
        If dim_excluded is True, opacity of excluded trials is set to this value.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Interactive Plotly figure.
    """
    # Base plot from Optuna
    fig = ov.plot_intermediate_values(study)

    # Align trials with traces
    trials_with_iv = [t for t in study.trials if t.intermediate_values]

    def _passes_include(trial: optuna.trial.FrozenTrial) -> bool:
        if not include_params:
            return True
        for k, v in include_params.items():
            if trial.params.get(k, None) != v:
                return False
        return True

    def _passes_exclude(trial: optuna.trial.FrozenTrial) -> bool:
        if not exclude_params:
            return True
        for k, v in exclude_params.items():
            if trial.params.get(k, object()) == v:
                return False
        return True

    def _passes_predicate(trial: optuna.trial.FrozenTrial) -> bool:
        return True if predicate is None else bool(predicate(trial))

    keep_mask = [
        (_passes_include(t) and _passes_exclude(t) and _passes_predicate(t))
        for t in trials_with_iv
    ]

    # If not dimming, just drop excluded trials
    if not dim_excluded:
        if keep_mask and (not all(keep_mask)):
            new_traces = [tr for keep, tr in zip(keep_mask, fig.data) if keep]
            fig.data = tuple(new_traces)
            trials_with_iv = [t for t, keep in zip(trials_with_iv, keep_mask) if keep]
    else:
        # Keep all traces, but reduce opacity on excluded ones
        for keep, trace in zip(keep_mask, fig.data):
            if not keep:
                trace.update(opacity=dim_factor)

    # Attach hover text to each (remaining or all) trial
    for trial, trace in zip(trials_with_iv, fig.data):
        params_str = _fmt_params(
            params=trial.params,
            max_param_items=max_param_items,
            sort_params=sort_params
        ) if trial.params else "(no params)"
        td = getattr(trial, "duration", None)

        hover_text = []
        for step, val in trial.intermediate_values.items():
            parts = [
                f"Trial {trial.number}",
                f"state: {trial.state.name}",
                f"step: {step}",
            ]
            if val is not None and isinstance(val, (int, float)) and math.isfinite(val):
                parts.append(f"{val_name}: {val:g}")
            parts.append(f"best_epoch: {trial.user_attrs.get('best_epoch', 'n/a')}")
            parts.append(f"duration: {_duration_fmt(td)}" if td else "duration: n/a")
            parts.append("--- params ---")
            parts.append(params_str)
            hover_text.append("<br>".join(parts))

        if hover_text:
            trace.hovertext = hover_text
            trace.hoverinfo = "text"
            trace.hovertemplate = "%{hovertext}<extra></extra>"
    
    if semilogy:
        fig.update_yaxes(type="log")

    html = fig.to_html(include_plotlyjs="inline", full_html=False)  # offline, self-contained
    display(HTML(html))
    return



def plot_optimization_history(
    study: optuna.study.Study,
    *,
    target = None,
    target_name: str = "value",
    max_param_items: int = 30,
    sort_params: bool = True,
):
    """
    Like optuna.visualization.plot_optimization_history, but each point's hover shows
    the trial's params (and optionally state/duration).

    Parameters
    ----------
    study : optuna.study.Study
        Source study.
    target : Callable[[FrozenTrial], float] | None
        Same as in Optuna's plot_optimization_history (for custom metrics).
    target_name : str
        Label for target on the Y axis (also shown in hover).
    max_param_items : int
        Limit number of param key/values shown (helps with very large spaces).
    sort_params : bool
        If True, params are sorted by key for stable display.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The augmented Plotly figure. Call `fig.show()` to display.
    """
    fig = ov.plot_optimization_history(study, target=target, target_name=target_name)

    trial_by_num = {t.number: t for t in study.trials}

    # Go through traces; add hover text for marker traces (skip best-line etc.)
    for tr in fig.data:
        # Only update scatter-like traces with per-point markers
        mode = getattr(tr, "mode", "") or ""
        if "markers" not in mode:
            continue
        xs = list(getattr(tr, "x", []))
        ys = list(getattr(tr, "y", []))
        if not xs or not ys:
            continue

        texts = []
        for x, y in zip(xs, ys):
            # X is trial number for Optuna's optimization history
            try:
                num = int(x)
            except (ValueError, TypeError):
                num = None
            t = trial_by_num.get(num) if num is not None else None

            parts = []
            parts.append(f"Trial {num}" if num is not None else "Trial")
            
            if t is not None:
                parts.append(f"state: {t.state.name}") 
            
            if y is not None and (isinstance(y, (int, float)) and math.isfinite(y)):
                parts.append(f"{target_name}: {y:g}")
            
            if t is not None:
                best_epoch = t.user_attrs.get("best_epoch", None)
                parts.append(f"best_epoch: {best_epoch if best_epoch is not None else 'n/a'}")
                td = getattr(t, "duration", None)
                parts.append(f"duration: {_duration_fmt(td)}" if td else "duration: n/a")    
                parts.append("--- params ---")
                params_str = _fmt_params(params=t.params, max_param_items=max_param_items, sort_params=sort_params) if t.params else "(no params)"
                parts.append(params_str)

            texts.append("<br>".join(parts))

        # Attach as hover text; hide the default extra box
        tr.hovertext = texts
        tr.hoverinfo = "text"
        tr.hovertemplate = "%{hovertext}<extra></extra>"
    
    html = fig.to_html(include_plotlyjs="inline", full_html=False)  # offline, self-contained
    display(HTML(html))

    return 

def _debug_bins_and_pivot(df, param_a, param_b, objective, binning="quantile", bins_a=5, bins_b=5,
                          custom_edges_a=None, custom_quantiles_a=None,
                          custom_edges_b=None, custom_quantiles_b=None):
    pa = _param_col(param_a)
    pb = _param_col(param_b)

    # 1) Build the same filtered frame `d` as in marginal_2d
    print(f"[debug] rows before filter: {len(df)}")
    d = df[
        (df["state"] == "COMPLETE")
        & df[objective].notna()
        & df[pa].notna()
        & df[pb].notna()
    ].copy()
    print(f"[debug] rows after filter: {len(d)}")

    # 2) Bin exactly like marginal_2d
    binning_a = binning if (custom_edges_a is None and custom_quantiles_a is None) else "custom"
    binfo_a = _bin_param(d, pa, binning=binning_a, bins=bins_a,
                         custom_edges=custom_edges_a, custom_quantiles=custom_quantiles_a)
    binning_b = binning if (custom_edges_b is None and custom_quantiles_b is None) else "custom"
    binfo_b = _bin_param(d, pb, binning=binning_b, bins=bins_b,
                         custom_edges=custom_edges_b, custom_quantiles=custom_quantiles_b)
    print(f"[debug] Param A bins: {binfo_a.bins.dtype}, levels: {binfo_a.level}")
    print(f"[debug] Param B bins: {binfo_b.bins.dtype}, levels: {binfo_b.level}")

    d["__A__"] = binfo_a.bins
    d["__B__"] = binfo_b.bins

    # 3) Check for NaN assignments in the binned columns
    nan_A = d["__A__"].isna().sum()
    nan_B = d["__B__"].isna().sum()
    print(f"[debug] NaN assignments: __A__={nan_A}, __B__={nan_B}")

    # 4) How many categories were defined vs. how many got at least one row?
    A_cats = binfo_a.bins.dtype.categories
    B_cats = binfo_b.bins.dtype.categories

    cntA = d["__A__"].value_counts(dropna=False).reindex(list(A_cats) + [np.nan], fill_value=0)
    cntB = d["__B__"].value_counts(dropna=False).reindex(list(B_cats) + [np.nan], fill_value=0)
    print(f"[debug] A bins defined: {len(A_cats)}, with rows: {(cntA.iloc[:-1] > 0).sum()}, NaN-bin rows: {int(cntA.iloc[-1])}")
    print(f"[debug] B bins defined: {len(B_cats)}, with rows: {(cntB.iloc[:-1] > 0).sum()}, NaN-bin rows: {int(cntB.iloc[-1])}")

    # 5) Build the same groupby as marginal_2d, two ways, to spot the NaN "bin" and dropped-all-NaN bins
    grp_keep_nan = d.groupby(["__A__", "__B__"], dropna=False, observed=False)
    grp_drop_nan = d.groupby(["__A__", "__B__"], dropna=True , observed=False)

    agg_keep = grp_keep_nan[objective].agg(["median", "mean", "std", "count"])
    agg_drop = grp_drop_nan[objective].agg(["median", "mean", "std", "count"])

    # 6) Unstack both ways and compare shapes/labels
    piv_count_keep = agg_keep["count"].unstack("__B__")               # default dropna=True for all-NaN cols
    piv_count_keep_full = agg_keep["count"].unstack("__B__")
    piv_count_drop = agg_drop["count"].unstack("__B__")

    print(f"[debug] pivot shapes (count): keep_nan+default_unstack={piv_count_keep.shape}, "
          f"keep_nan+full_unstack={piv_count_keep_full.shape}, drop_nan+full_unstack={piv_count_drop.shape}")

    # 7) Which axis labels are NaN (i.e., NaN "bin" is present)?
    has_nan_row_keep = piv_count_keep_full.index.isna().any()
    has_nan_col_keep = piv_count_keep_full.columns.isna().any()
    print(f"[debug] NaN row in pivot? {has_nan_row_keep} | NaN col in pivot? {has_nan_col_keep}")

    # 8) Explicitly enforce full A×B grid, then compare to binfo lengths
    piv_full = (piv_count_drop
                .reindex(index=A_cats, columns=B_cats))  # full grid, no NaN bin
    print(f"[debug] enforced grid shape={piv_full.shape} vs A={len(A_cats)} x B={len(B_cats)}")
    if (len(A_cats), len(B_cats)) != piv_full.shape:
        print("[debug] MISMATCH after reindex — categories changed unexpectedly")

    # 9) Return to let you inspect if needed
    return dict(
        rows=len(d),
        nan_A=int(nan_A),
        nan_B=int(nan_B),
        A_bins_defined=len(A_cats),
        B_bins_defined=len(B_cats),
        A_bins_with_rows=int((cntA.iloc[:-1] > 0).sum()),
        B_bins_with_rows=int((cntB.iloc[:-1] > 0).sum()),
        has_nan_row=bool(has_nan_row_keep),
        has_nan_col=bool(has_nan_col_keep),
        piv_keep_shape=piv_count_keep.shape,
        piv_keep_full_shape=piv_count_keep_full.shape,
        piv_drop_full_shape=piv_count_drop.shape,
    )
