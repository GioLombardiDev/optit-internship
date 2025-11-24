from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Mapping, Optional, Sequence, List, Union
import logging
from tqdm.notebook import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from contextlib import nullcontext

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsforecast import StatsForecast  
from statsforecast.models import AutoETS, SeasonalNaive  
from statsmodels.tsa.seasonal import MSTL 

from ..utils.transforms import make_transformer, transform_column, get_lambdas, make_is_winter
from ..utils.datasplit import generate_sets
from ..utils.decomposition import decompose_annual_seasonality, remove_annual_component, add_annual_component
from ..utils.plotting import configure_time_axes

_LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

@dataclass(frozen=True)
class MSTLConfig:
    """Configuration for :class:`MSTLPipeline`.  Immutable & serialisable."""

    # data‑prep & transform 
    transform: str = "boxcox"  # choices: "boxcox", "log", "arcsinh", "arcsinh2", "arcsinh10", or "none"
    lam_method: str = "loglik"  # relevant for Box‑Cox
    winter_focus: bool = True
    season_length: int = 365  # days, only for lambda estimation

    # decomposition params
    decomposition_method: str = "CD"  # "STL" or "CD"
    smooth_loess: bool = False
    window_loess: int = 7
    robust_stl: bool = False

    # MSTL settings
    train_horizon: int = 7 * 11 * 24  # hours for MSTL to capture daily and weekly seasonalities
    windows_mstl: Sequence[int] = field(default_factory=lambda: [11, 15])
    stl_kwargs_mstl: Mapping[str, Any] = field(default_factory=dict) # additional kwargs for STL used for annual decomposition
    periods_mstl: Sequence[int] = field(default_factory=lambda: [24, 24 * 7])

    # behavior toggle
    # If True, AutoETS is trained on (trend + remainder + annual_seasonal),
    # and annual seasonality will NOT be added back later in predict().
    treat_annual_as_trend: bool = False

    # plotting
    palette: Optional[Sequence[str]] = None  # optional colour set

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

# --------------------------------------------------------------------
# Main pipeline class
# --------------------------------------------------------------------

class MSTLPipeline:
    """Multi-seasonal STL pipeline for forecasting.
    """

    def __init__(
        self,
        *,
        target_df: pd.DataFrame,
        config: MSTLConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        
        self._target_df = target_df.copy()
        self.cfg = config or MSTLConfig()
        self._log = logger or _LOGGER

        # internal state 
        self._train_df = None
        self._test_df = None
        self._lambdas: Dict[str, float] = {}
        self._results_mstl: Dict[str, pd.DataFrame] = {}
        self._results_stl: Dict[str, pd.DataFrame] = {}
        self._forecast_df: pd.DataFrame | None = None
        self._last_cv_metadata: Dict[str, Any] = {}
        self._alias = None

    def fit(
            self,
            *,
            start_train: pd.Timestamp | None = None,
            end_train: pd.Timestamp | None = None,
            silent: bool = False,
            alias: str = "MSTL"
        ) -> MSTLPipeline:
        """Fit the full pipeline and return *self* for chaining."""

        self._alias = alias # it's not actually used during fit, but used in predict
                            # we still pass it here for coherence with other classes and statsforecast

        if start_train is not None and end_train is not None:
            # use provided train range
            if not (start_train < end_train): 
                raise ValueError("Dates must satisfy start_train < end_train")
            
            self._train_df = self._split(
                self._target_df, start_train, end_train
            )
            if not silent:
                self._log.info("Fitting MSTL pipeline …")
        else:
            # use default train range
            self._train_df = self._target_df.copy()
            if not silent:
                self._log.info("Fitting MSTL pipeline on the full dataset …")

        if self.cfg.transform == "boxcox":
            self._estimate_lambdas()

        self._train_df = self._apply_transform(self._train_df, forward=True)

        self._results_stl = self._decompose_annual()

        season_df = self._season_df(self._results_stl)
        self._train_df = remove_annual_component(
            df=self._train_df,
            season_df=season_df,
            target_col="y_transformed",
            target_col_deseason="y_transformed_deseason",
        )

        self._results_mstl = self._decompose_multi()
        if not silent:
            self._log.info("\u2713 Fit complete (%d series)", len(self._results_mstl))
        return self

    def predict(
        self,
        *,
        h: int,
        level: Sequence[int] | None = None,
        trend_forecaster_kwargs: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Forecast *h* hours ahead using the fitted model."""
        if not self._results_mstl:
            raise RuntimeError("Call `fit()` before `predict()`.")
        level = list(level or [])
        tf_kwargs = trend_forecaster_kwargs or {}

        y_df = self._trend_plus_remainder_df(
            include_annual=self.cfg.treat_annual_as_trend
        )

        trend_fc = self._forecast_trend(
            y_df=y_df,
            h=h,
            level=level,
            trend_kwargs=tf_kwargs,
        )

        seas_fc = self._forecast_short_seasonal(h)

        fc = self._merge_seasonals(trend_fc, seas_fc, level)

        if not self.cfg.treat_annual_as_trend:
            season_df = self._season_df(self._results_stl)
            fc = add_annual_component(forecast_df=fc, season_df=season_df)

        fc = self._apply_transform(fc, forward=False)
        self._forecast_df = fc
        return fc.copy()
    
    def forward(
        self,
        context_end: pd.Timestamp,
        h: int,
        context_start: Optional[pd.Timestamp] = None, # equivalent of start_train (different from train_horizon from the config)
        level: Optional[List[int]] = None,
        trend_forecaster_kwargs: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Forecast *h* hours ahead using the fitted model."""
        if not self._results_mstl:
            raise RuntimeError("Call `fit()` before `forward()`.")
        level = list(level or [])
        tf_kwargs = trend_forecaster_kwargs or {}

        self._forward_seas_to_context_end(context_end)
        y_df = self._build_context_trend_remainder(
            context_end=context_end,
            context_start=context_start,
        )

        trend_fc = self._forecast_trend(
            y_df=y_df,
            h=h,
            level=level,
            trend_kwargs=tf_kwargs,
            use_forward=True
        )

        # Seasonal (24 h / 7 d) naive forecasts 
        seas_fc = self._forecast_short_seasonal(h, use_forward=True)

        # Merge & add seasonal components 
        fc = self._merge_seasonals(trend_fc, seas_fc, level)

        # Add annual seasonality 
        if not self.cfg.treat_annual_as_trend:
            season_df = self._season_df(self._active_results_stl(use_forward=True))
            fc = add_annual_component(forecast_df=fc, season_df=season_df)

        # Inverse transform 
        fc = self._apply_transform(fc, forward=False)
        self._forecast_df = fc
        return fc.copy()

    def cross_validation(
        self,
        *,
        h: int,
        test_size: int,
        end_test: Optional[pd.Timestamp] = None,
        step_size: int = 1,
        input_size: Optional[int] = None,
        level: List[int] = None,
        refit: Union[bool, int] = True,
        trend_forecaster_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        alias: str = 'MSTL',
    ) -> pd.DataFrame:
        """
        Rolling-window cross-validation using this MSTL pipeline.

        Returns one forecast-ground-truth pair per (unique_id, window),
        with columns [unique_id, ds, y, <alias>, (<alias>-lo-<L>, <alias>-hi-<L>…), cutoff].
        """
        level = list(level or [])

        # sanity check
        if (test_size - h) % step_size != 0:
            raise ValueError("`test_size - h` must be a multiple of `step_size`")

        # figure out the final timestamp
        if end_test is None:
            end_test = self._target_df["ds"].max()

        # relative offsets for each window
        steps = list(range(-test_size, -h + 1, step_size))

        all_results = []
        prev_pipeline = None
        t0 = pd.Timestamp.now()

        iterator = tqdm(steps, disable=not verbose, desc="CV windows", leave=True)
        for i, offset in enumerate(iterator):
            cutoff = end_test + pd.Timedelta(hours=offset)

            # build training window
            if input_size is not None:
                start_train = cutoff - pd.Timedelta(hours=input_size) + pd.Timedelta(hours=1)
            else:
                start_train = self._target_df["ds"].min()
            end_train = cutoff

            # decide whether to refit
            do_fit = (
                i == 0
                or (isinstance(refit, int) and not isinstance(refit, bool) and i % refit == 0)
                or (refit is True)
                or (refit is None)
            )

            if do_fit or prev_pipeline is None:
                # new pipeline for this window
                pipeline = MSTLPipeline(
                    target_df=self._target_df,
                    config=self.cfg,
                    logger=self._log,
                )
                pipeline.fit(
                    start_train=start_train,
                    end_train=end_train,
                    silent=True,  # no logging during fit
                    alias=alias,  
                )
                prev_pipeline = pipeline
                fc = pipeline.predict(h=h, level=level, trend_forecaster_kwargs=trend_forecaster_kwargs)
            else:
                # reuse last fitted
                pipeline = prev_pipeline
                fc = pipeline.forward(context_end=cutoff, h=h, context_start=start_train, level=level, trend_forecaster_kwargs=trend_forecaster_kwargs)

            # build validation set for exactly h hours after cutoff
            mask = (
                (self._target_df["ds"] > cutoff)
                & (self._target_df["ds"] <= cutoff + pd.Timedelta(hours=h))
            )
            val_df = self._target_df.loc[mask, ["unique_id", "ds", "y"]]

            # merge forecast & truth
            lo_cols = [f"{alias}-lo-{L}" for L in level]
            hi_cols = [f"{alias}-hi-{L}" for L in level]
            merged = (
                val_df
                .merge(
                    fc[["unique_id", "ds", alias] + lo_cols + hi_cols],
                    on=["unique_id", "ds"],
                    how="left",
                )
                .assign(cutoff=cutoff)
            )

            all_results.append(merged)
            iterator.refresh()

        iterator.close()

        # Build metadata automatically
        self._last_cv_metadata = {
            "pipeline_init": {
                "target_df_start": str(self._target_df["ds"].min()),
                "target_df_end": str(self._target_df["ds"].max()),
                "target_df_uids": self._target_df["unique_id"].unique().tolist(),
            },
            "pipeline_class": self.__class__.__name__,
            "cv_params": {
                "h": h,
                "test_size": test_size,
                "end_test": str(end_test or self._target_df["ds"].max()),
                "step_size": step_size,
                "input_size": input_size,
                "level": level,
                "refit": refit,
                "trend_forecaster_kwargs": trend_forecaster_kwargs or {},
            },
            "run_timestamp": pd.Timestamp.now().isoformat(),
            "elapsed_seconds": (pd.Timestamp.now() - t0).total_seconds(),
            "heat_df_range": {
                "min_ds": str(self._target_df["ds"].min()),
                "max_ds": str(self._target_df["ds"].max()),
            },
        }

        return pd.concat(all_results, ignore_index=True)

    @property
    def forecast(self) -> pd.DataFrame:
        """Return the last computed forecast."""
        if self._forecast_df is None:
            raise AttributeError("No forecast available; call `predict()` first.")
        return self._forecast_df
    
    @property
    def last_cv_metadata(self) -> pd.DataFrame:
        """Return the metadata for the last computed cv."""
        if self._last_cv_metadata is None:
            raise AttributeError("No metadata available; call `cross_validation()` first.")
        return self._last_cv_metadata

    # INTERNALS

    def _estimate_lambdas(self) -> None:
        """Estimate Box-Cox λ per series, optionally winter-only."""
        self._log.debug("Estimating Box-Cox lambdas …")

        self._lambdas = get_lambdas(
            df=self._train_df,
            method=self.cfg.lam_method,
            winter_focus=self.cfg.winter_focus,
            make_is_winter=make_is_winter,
            season_length=self.cfg.season_length,
        )   

    def _apply_transform(self, df: pd.DataFrame, forward: bool) -> pd.DataFrame:
        col = "y_transformed" if forward else df.select_dtypes("number").columns
        if forward:
            f = make_transformer(self.cfg.transform, "y", self._lambdas, inv=False)
            df = df.copy()
            df["y_transformed"] = transform_column(df, f)
            return df
        # else: inverse over *all* numeric cols in forecast
        out = df.copy()
        for c in col:
            bwd = make_transformer(self.cfg.transform, c, self._lambdas, inv=True)
            out[c] = transform_column(out, bwd)
        return out

    def _decompose_annual(self) -> Dict[str, pd.DataFrame]:
        self._log.debug("Decomposing annual seasonality …")
        daily_train = (
            self._train_df
            .groupby(["unique_id", pd.Grouper(key="ds", freq="D")], as_index=False)
            .mean(numeric_only=True)
        )
        return decompose_annual_seasonality(
            decomposition_method=self.cfg.decomposition_method,
            df=daily_train,
            target_col="y_transformed",
            robust=self.cfg.robust_stl,
            seasonal_STL=11,
            smooth_loess=self.cfg.smooth_loess,
            window_loess=self.cfg.window_loess,
        )

    @staticmethod
    def _season_df(results_stl: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract annual seasonal component from STL results (ds has daily frequency)."""
        return pd.concat(
            [df[["ds", "seasonal"]].assign(unique_id=uid) for uid, df in results_stl.items()],
            ignore_index=True,
        )

    def _decompose_multi(self) -> Dict[str, pd.DataFrame]:
        self._log.debug("Multi‑season MSTL per series …")
        tail_df = (
            self._train_df
            .groupby("unique_id", group_keys=False)
            .tail(self.cfg.train_horizon)
            .sort_values("ds")
        )
        res: Dict[str, pd.DataFrame] = {}
        for uid, grp in tail_df.groupby("unique_id"):
            m = MSTL(
                endog=grp["y_transformed_deseason"],
                periods=self.cfg.periods_mstl,
                windows=self.cfg.windows_mstl,
                lmbda=None,
                stl_kwargs=self.cfg.stl_kwargs_mstl,
            ).fit()
            res[uid] = pd.DataFrame({
                "ds": grp["ds"].to_numpy(),
                "trend": m.trend,
                "seasonal_24h": m.seasonal.iloc[:, 0],
                "seasonal_7d": m.seasonal.iloc[:, 1],
                "remainder": m.resid,
            })
        return res

    def _trend_plus_remainder_df(self, include_annual: bool = False) -> pd.DataFrame:
        rows = []

        results_mstl = self._results_mstl
        results_stl  = self._results_stl

        if include_annual:
            season_df = self._season_df(results_stl).rename(columns={"ds": "ds_day"}).copy()
            season_df["ds_day"] = pd.to_datetime(season_df["ds_day"])
            season_df["ds_day"] = season_df["ds_day"].dt.floor("D")

        for uid, d in results_mstl.items():
            d = d.reset_index(drop=True).copy()

            y_tr = (d["trend"].to_numpy() + d["remainder"].to_numpy())

            if include_annual:
                tmp = pd.DataFrame({"ds": d["ds"]})
                tmp["unique_id"] = uid
                tmp["ds_day"] = pd.to_datetime(tmp["ds"])
                tmp["ds_day"] = tmp["ds_day"].dt.floor("D")

                ann = season_df.loc[season_df["unique_id"] == uid, ["ds_day", "seasonal"]]
                tmp = tmp.merge(ann, on="ds_day", how="left")

                y_tr = y_tr + tmp["seasonal"].to_numpy()

            rows.append(pd.DataFrame({
                "unique_id": uid,
                "ds": d["ds"].to_numpy(),
                "y_trend_rem": y_tr,
            }))

        tpr = pd.concat(rows, ignore_index=True)

        # final sanity check
        if tpr["y_trend_rem"].isna().any():
            bad = tpr[tpr["y_trend_rem"].isna()].head()
            raise ValueError(
                "NaNs found in trend+remainder even after positional alignment. "
                "Check seasonal merge coverage / timestamp alignment. "
                f"Examples:\n{bad}"
            )
        return tpr


    def _forecast_trend(
        self,
        h: int,
        y_df: pd.DataFrame,
        level: Sequence[int],
        trend_kwargs: Mapping[str, Any],
        use_forward: bool = False,
    ) -> pd.DataFrame:
        """Forecast trend + remainder (+ annual if configured) using AutoETS."""
        alias = self._alias or "MSTL"
        level = list(level or [])
        model_spec = trend_kwargs.get("model", "ZZN")
        pi_custom = trend_kwargs.get("prediction_intervals", None)

        if not use_forward:
            # Predict path: fit fresh model(s) on y_df
            trend_model = AutoETS(model=model_spec, prediction_intervals=pi_custom, alias=self._alias)
            sf = StatsForecast(models=[trend_model], freq="h")
            sf = sf.fit(df=y_df, target_col="y_trend_rem") 
            self._sf_trend = sf
            fc = sf.predict(h=h, level=level)

            # Cache fitted model(s) by uid for forward()
            self._ets_fitted_by_uid = {}
            # sf.fitted_ has shape [n_series, n_models]
            uids = y_df["unique_id"].drop_duplicates().tolist()
            for i, uid in enumerate(uids):
                self._ets_fitted_by_uid[uid] = sf.fitted_[i, 0]
        else:
            # Forward path: reuse cached fitted models and pass a fresh context y
            if not hasattr(self, "_ets_fitted_by_uid") or not self._ets_fitted_by_uid:
                raise RuntimeError(
                    "No cached ETS models found. Call predict() (fit path) at least once before forward, "
                    "or initialize the cache by calling _forecast_trend with use_forward=False first."
                )
            out_rows = []
            for uid, grp in y_df.groupby("unique_id", sort=False):
                if uid not in self._ets_fitted_by_uid:
                    raise KeyError(f"No cached ETS model for uid={uid!r}.")
                ets_model = self._ets_fitted_by_uid[uid]
                # ensure correct column name
                y_ctx = grp["y_trend_rem"].to_numpy()
                # model-level forward
                fc_dict = ets_model.forward(y=y_ctx, h=h, level=level)
                fc_dict = self._rename_forward_keys(fc_dict, alias, level)
                fc_uid = pd.DataFrame(fc_dict)
                last_ds = grp["ds"].max()
                horizon = pd.date_range(
                    last_ds + pd.Timedelta(hours=1),
                    periods=h, freq="h",
                    tz=getattr(last_ds, "tz", None),
                )
                fc_uid["ds"] = horizon.to_numpy()
                fc_uid["unique_id"] = uid
                out_rows.append(fc_uid)

            fc = pd.concat(out_rows, ignore_index=True)
        return fc

    def _forecast_short_seasonal(self, h: int, use_forward: bool = False) -> pd.DataFrame:
        results_mstl = self._active_results_mstl(use_forward=use_forward)
        seas_df = pd.concat([
            pd.DataFrame({
                "unique_id": uid,
                "ds": dcmp["ds"],
                "seasonal_24h": dcmp["seasonal_24h"],
                "seasonal_7d": dcmp["seasonal_7d"],
            })
            for uid, dcmp in results_mstl.items()
        ], ignore_index=True)

        naive24 = SeasonalNaive(season_length=24, alias="Naive24h")
        naive7d = SeasonalNaive(season_length=24 * 7, alias="Naive7d")
        fc24 = StatsForecast(models=[naive24], freq="h").forecast(df=seas_df, h=h, target_col="seasonal_24h")
        fc7d = StatsForecast(models=[naive7d], freq="h").forecast(df=seas_df, h=h, target_col="seasonal_7d")
        return fc24.merge(fc7d, on=["unique_id", "ds"], how="left")

    def _merge_seasonals(
        self,
        trend_fc: pd.DataFrame,
        seas_fc: pd.DataFrame,
        level: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Merge trend and seasonal forecasts"""

        alias = self._alias
        out = trend_fc.merge(seas_fc, on=["unique_id", "ds"], how="left")
        out[alias] += out["Naive24h"] + out["Naive7d"]
        for lvl in level or []:
            lo, hi = f"{alias}-lo-{lvl}", f"{alias}-hi-{lvl}"
            out[lo] += out["Naive24h"] + out["Naive7d"]
            out[hi] += out["Naive24h"] + out["Naive7d"]

        return out.drop(columns=["Naive24h", "Naive7d"])
    
    @staticmethod
    def _rename_forward_keys(d: dict, alias: str, levels):
        out = {}
        # point forecast
        if 'mean' in d:
            out[alias] = d['mean']
        # lo/hi style
        for L in levels:
            lo_k, hi_k = f'lo-{L}', f'hi-{L}'
            if lo_k in d:
                out[f'{alias}-lo-{L}'] = d[lo_k]
            if hi_k in d:
                out[f'{alias}-hi-{L}'] = d[hi_k]
        for L in levels:
            lvl_k = f'level_{L}'
            if lvl_k in d:
                vals = d[lvl_k]
                # accept (lo, hi) tuple/array or dict with 'lo'/'hi'
                if isinstance(vals, (list, tuple)) and len(vals) == 2:
                    out[f'{alias}-lo-{L}'], out[f'{alias}-hi-{L}'] = vals
                elif isinstance(vals, dict):
                    if 'lo' in vals: out[f'{alias}-lo-{L}'] = vals['lo']
                    if 'hi' in vals: out[f'{alias}-hi-{L}'] = vals['hi']
        return out
    
    def _forward_seas_to_context_end(
        self,
        context_end: pd.Timestamp,
    ) -> None:
        """
        Build forward versions of results_mstl / results_stl whose time axes end at `context_end`
        by splicing the last available history with a phase-advanced tail of seasonals.
        """
        results_mstl_fwd: Dict[str, pd.DataFrame] = {}
        for uid, dcmp in self._results_mstl.items():
            dcmp = dcmp.sort_values("ds").reset_index(drop=True)
            if dcmp.empty:
                raise ValueError(f"Empty MSTL decomposition for uid={uid}")

            # ensure context_end is at or after the end of the MSTL window
            if context_end < dcmp["ds"].max():
                raise ValueError(
                    "According to the last saved MSTL decomposition, "
                    f"context_end {context_end} is before end of the training window ({dcmp['ds'].max()})."    
                )
            hist = dcmp.copy()

            last_hist_ds = hist["ds"].iloc[-1]
            # how many hours missing to reach context_end
            missing_len = int((context_end - last_hist_ds) / pd.Timedelta(hours=1))
            if missing_len < 0:
                missing_len = 0

            # build phase-advanced tails (just to fill the gap up to context_end)
            if missing_len > 0:
                add24 = self._repeat_tail(hist["seasonal_24h"], 24, missing_len)
                add7d = self._repeat_tail(hist["seasonal_7d"], 24 * 7, missing_len)
                add_ds = pd.date_range(last_hist_ds + pd.Timedelta(hours=1), periods=missing_len, freq="h")

                ext = pd.DataFrame({
                    "ds": add_ds,
                    "seasonal_24h": add24,
                    "seasonal_7d": add7d,
                })
                hist = pd.concat([hist[["ds", "seasonal_24h", "seasonal_7d"]], ext], ignore_index=True)

            # rebuild a clean index
            hist = hist.sort_values("ds").reset_index(drop=True)
            results_mstl_fwd[uid] = hist.tail(self.cfg.train_horizon)

        self._results_mstl_fwd = results_mstl_fwd

        results_stl_fwd: Dict[str, pd.DataFrame] = {}
        for uid, stl in self._results_stl.items():
            stl = stl.sort_values("ds").reset_index(drop=True)
            if stl.empty:
                raise ValueError(f"Empty STL decomposition for uid={uid}")

            # ensure context_end is at or after the end of the MSTL window
            if context_end < stl["ds"].max():
                raise ValueError(
                    "According to the last saved STL decomposition, "
                    f"context_end {context_end} is before end of the training window ({stl['ds'].max()})."
                )
            hist = stl.copy()

            last_hist_ds = hist["ds"].iloc[-1]
            missing_days = int((context_end.normalize() - last_hist_ds.normalize()) / pd.Timedelta(days=1))
            if missing_days < 0:
                missing_days = 0

            if missing_days > 0:
                # Use 365 as the annual cycle length your STL assumed
                add_ann = self._repeat_tail(hist["seasonal"], 365, missing_days)
                add_ds = pd.date_range(last_hist_ds + pd.Timedelta(days=1), periods=missing_days, freq="D")
                ext = pd.DataFrame({"ds": add_ds, "seasonal": add_ann})
                hist = pd.concat([hist[["ds", "seasonal"]], ext], ignore_index=True)

            # rebuild a clean index
            hist = hist.sort_values("ds").reset_index(drop=True)
            results_stl_fwd[uid] = hist.tail(self.cfg.train_horizon)

        self._results_stl_fwd = results_stl_fwd

    @staticmethod
    def _repeat_tail(series: pd.Series, season_len: int, steps: int) -> np.ndarray:
        """
        Return an array that continues the *cycle* of the last `season_len` values
        for exactly `steps` steps. If history < season_len, raise.
        """
        arr = series.to_numpy()
        if arr.size == 0:
            return np.zeros(steps, dtype=float)
        if arr.size < season_len:
            raise ValueError("Insufficient history to repeat tail.")
        else:
            base = arr[-season_len:]
        reps = math.ceil(steps / season_len)
        return np.tile(base, reps)[:steps]
    
    def _active_results_mstl(self, use_forward: bool):
        if use_forward:
            if hasattr(self, "_results_mstl_fwd"):
                return self._results_mstl_fwd
            raise AttributeError("No forward MSTL results available; call `_forward_seas_to_context_end()` first.")
        return self._results_mstl

    def _active_results_stl(self, use_forward: bool):
        if use_forward:
            if hasattr(self, "_results_stl_fwd"):
                return self._results_stl_fwd
            raise AttributeError("No forward STL results available; call `_forward_seas_to_context_end()` first.")
        return self._results_stl
    
    def _build_context_trend_remainder(
        self,
        *,
        context_end: pd.Timestamp,
        context_start: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        """
        Build y_trend_rem for the context window directly from data:
        y_trend_rem = y_transformed - (annual if treat_annual_as_trend else 0) - (seasonal_24h + seasonal_7d)
        If treat_annual_as_trend=True, annual is NOT removed here.
        Uses the *forward* seasonal dicts that end at context_end.
        """
        if not hasattr(self, "_results_mstl_fwd") or not hasattr(self, "_results_stl_fwd"):
            raise RuntimeError("Call _forward_seas_to_context_end(context_end) first.")
        
        # slice raw and transform forward
        df = self._target_df.copy()
        mask = (df["ds"] <= context_end)
        if context_start is not None:
            mask &= (df["ds"] >= context_start)
        df = df.loc[mask, ["unique_id", "ds", "y"]].copy()
        df = self._apply_transform(df, forward=True)  # adds y_transformed

        # get daily annual seasonal aligned (forward STL)
        season_daily = self._season_df(self._results_stl_fwd).rename(columns={"ds": "ds_day"})
        season_daily["ds_day"] = pd.to_datetime(season_daily["ds_day"]).dt.floor("D")

        ctx = df.copy()
        ctx["ds_day"] = pd.to_datetime(ctx["ds"]).dt.floor("D")
        ctx = ctx.merge(season_daily, on=["unique_id", "ds_day"], how="left")

        # make sure we only keep at most last `train_horizon` hours up to context_end
        ctx = ctx[ctx["ds"] > context_end - pd.Timedelta(hours=self.cfg.train_horizon)].copy()

        # get short seasonals aligned (forward MSTL)
        seas_short = pd.concat([
            pd.DataFrame({
                "unique_id": uid,
                "ds": d["ds"].to_numpy(),
                "seasonal_24h": d["seasonal_24h"].to_numpy(),
                "seasonal_7d": d["seasonal_7d"].to_numpy(),
            })
            for uid, d in self._results_mstl_fwd.items()
        ], ignore_index=True)

        ctx = ctx.merge(seas_short, on=["unique_id", "ds"], how="left")

        # build y_trend_rem in transformed space
        seasonal_short = ctx["seasonal_24h"].to_numpy() + ctx["seasonal_7d"].to_numpy()
        annual = ctx["seasonal"].to_numpy()
        if self.cfg.treat_annual_as_trend:
            # do NOT remove annual here → ETS will learn it
            y_trend_rem = ctx["y_transformed"].to_numpy() - seasonal_short
        else:
            # remove annual (as in fit)
            y_trend_rem = ctx["y_transformed"].to_numpy() - annual - seasonal_short
        if np.isnan(y_trend_rem).any():
            bad = ctx[np.isnan(y_trend_rem)].head()
            raise ValueError(
                "NaNs found in context trend+remainder calculation. Check seasonal merge coverage / timestamp alignment. "
                f"Examples:\n{bad}"
            )

        out = ctx[["unique_id", "ds"]].copy()
        out["y_trend_rem"] = y_trend_rem
        return out

    @staticmethod
    def _split(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return df[(df['ds'] >= start) & (df['ds'] <= end)].copy()
    
    def plot_decomposition(
        self,
        uid: str,
        *,
        same_yaxis_span: bool = False,
        return_fig: bool = False,
    ):
        """Plot decomposition for a single series."""

        if uid not in self._results_mstl:
            raise KeyError(f"Unknown uid {uid!r}. Call `fit()` first or check id.")
        dcmp_mslt = self._results_mstl[uid]
        dcmp_stl = self._results_stl[uid]
        dcmp_stl = dcmp_stl[dcmp_stl['ds'].isin(dcmp_mslt['ds'])].copy()
        train_whole_period = self._train_df[self._train_df["unique_id"] == uid]
        train = train_whole_period[train_whole_period["ds"].isin(dcmp_mslt["ds"])]

        fig, axes = plt.subplots(6, 1, figsize=(12, 12))
        color = "black"
        sns.lineplot(data=train, x="ds", y="y_transformed", ax=axes[0], color=color)
        sns.lineplot(data=dcmp_mslt, x="ds", y="trend", ax=axes[1], color=color)
        sns.lineplot(data=dcmp_mslt, x="ds", y="seasonal_24h", ax=axes[2], color=color)
        sns.lineplot(data=dcmp_mslt, x="ds", y="seasonal_7d", ax=axes[3], color=color)
        sns.lineplot(data=dcmp_stl, x="ds", y="seasonal", ax=axes[4], color=color)
        sns.lineplot(data=dcmp_mslt, x="ds", y="remainder", ax=axes[5], color=color)

        configure_time_axes(axes, dcmp_mslt['ds'])

        axes[0].set_ylabel('Data')
        axes[1].set_ylabel('Trend')
        axes[2].set_ylabel('Seasonal 24h')
        axes[3].set_ylabel('Seasonal 7d')
        axes[4].set_ylabel('Seasonal 1y')
        axes[5].set_ylabel('Remainder')

        if same_yaxis_span:
            spans = [ax.get_ylim()[1] - ax.get_ylim()[0] for ax in axes]
            max_span = max(spans)
            for ax in axes:
                ymin, ymax = ax.get_ylim()
                center = 0.5 * (ymin + ymax)
                ax.set_ylim(center - max_span / 2, center + max_span / 2)

        fig.suptitle(f"Series {uid}: MSTL Decomposition")
        fig.supxlabel('Date')
        fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])

        if return_fig:
            return fig

