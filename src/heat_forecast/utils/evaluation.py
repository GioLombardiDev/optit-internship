from functools import partial
from typing import Sequence, List, Optional, Iterable, Tuple
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse, mae, mape, mase
from typing import Literal, Dict, Callable, Any
import logging
from tqdm.notebook import tqdm
from IPython.display import display, HTML

from .plotting import display_scrollable

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# -------------------------------------------------------------------------------
# FUNCTIONS FOR EVALUATING FORECASTS AND DISPLAY / PLOT EVALUATION RESULTS
# -------------------------------------------------------------------------------

def me(
    df: pd.DataFrame,
    models: Sequence[str],
    id_col: str = "unique_id",
    target_col: str = "y",
):
    """
    Mean Error (ME)

    Negative ME -> the model *over-forecasts* on average  
    Positive ME -> the model *under-forecasts* on average

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that contains the ground-truth column target_col,
        the identifier column id_col, and one forecast column for each
        entry in models.

    models : Sequence[str]
        List or tuple of column names holding the model forecasts whose
        ME should be computed.

    id_col : str, default "unique_id"
        Name of the column that uniquely identifies each series.

    target_col : str, default "y"
        Name of the ground-truth column.

    Returns
    -------
    wide : pandas.DataFrame
        Wide-format table with columns::

            [id_col, "metric", *models]

        One row per series; each model column contains that model's ME.
    """
    # Compute errors for every model at once
    err_block = pd.DataFrame(
        {m: df[target_col].to_numpy() - df[m].to_numpy() for m in models},
        index=df.index,
    )
    err_df = pd.concat([df[[id_col]], err_block], axis=1)

    # Average over the time dimension
    wide = (
        err_df
        .groupby(id_col, as_index=False)
        .mean(numeric_only=True)          
        .assign(metric="me")              
        [[id_col, "metric", *models]]     
    )
    return wide

from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd

def nmae(
    df: pd.DataFrame,
    models: Sequence[str],
    target_df: pd.DataFrame,
    id_col: str = "unique_id",
    target_col: str = "y",
    time_col: str = "ds",
    period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> pd.DataFrame:
    r"""
    NMAE per series.

    Numerator (per-series MAE): computed from `df` using the forecast rows
    (i.e., where both the forecast columns and `target_col` are present).

    Denominator (per-series mean |y|): computed from `target_df` restricted to `period`.
    """
    # ---- validate columns ----
    need_df = {id_col, target_col, *models}
    need_tgt = {id_col, time_col, target_col}
    miss_df = need_df - set(df.columns)
    miss_tgt = need_tgt - set(target_df.columns)
    if miss_df:
        raise KeyError(f"`df` missing columns: {sorted(miss_df)}")
    if miss_tgt:
        raise KeyError(f"`target_df` missing columns: {sorted(miss_tgt)}")

    # ---- period default/validation ----
    if period is None:
        period = (pd.Timestamp("2019-01-01"), pd.Timestamp("2025-01-01"))
    if not (isinstance(period, tuple) and len(period) == 2 and
            isinstance(period[0], pd.Timestamp) and isinstance(period[1], pd.Timestamp) and
            period[0] < period[1]):
        raise ValueError("period must be (start_ts, end_ts) with start < end.")

    # ---- numerator: per-series MAE over forecast rows in df ----
    err_block = pd.DataFrame({m: (df[m] - df[target_col]).abs().to_numpy() for m in models},
                             index=df.index)
    mae_per_series = (
        pd.concat([df[[id_col]], err_block], axis=1)
          .groupby(id_col, as_index=False)
          .mean(numeric_only=True)
    )

    # ---- denominator: per-series mean |y| over target_df within period ----
    target_df = target_df.copy()
    target_df[time_col] = pd.to_datetime(target_df[time_col])
    mask = target_df[time_col].between(period[0], period[1])
    norm_df = (
        target_df.loc[mask, [id_col, target_col]]
                 .groupby(id_col, as_index=False)
                 .agg(norm=(target_col, lambda s: np.abs(s).mean()))
    )

    # ---- merge and normalise ----
    wide = mae_per_series.merge(norm_df, on=id_col, how="left")
    wide.loc[wide["norm"] == 0, "norm"] = np.nan
    wide[models] = wide[models].div(wide["norm"], axis=0)
    wide = wide.drop(columns="norm")
    wide.insert(1, "metric", "nmae")
    return wide[[id_col, "metric", *models]]

def smape(
    df: pd.DataFrame,
    models: Sequence[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    use_factor_2: bool = True,
    as_percent: bool = False,
):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE) per series.
    NOTE: in other defs the denominator may be defined slightly differently,
    in our case it doesn't matter since the data is positive

    By default this uses the conventional SMAPE:
        SMAPE = mean( 2 * |y - yhat| / (|y| + |yhat|) )
    """
    # ground-truth vector
    y = df[target_col].to_numpy()

    # build block of smape contributions per row for every model
    smape_block = {}
    for m in models:
        yhat = df[m].to_numpy()
        num = (y - yhat)
        num = abs(num)
        if use_factor_2:
            num = 2 * num
        den = abs(y) + abs(yhat)

        # safe division: where den == 0 set contribution to 0
        contrib = num.copy()
        contrib = np.where(den <= 1e-7, 0.0, num / den)

        smape_block[m] = contrib

    err_df = pd.concat([df[[id_col]].reset_index(drop=True),
                        pd.DataFrame(smape_block, index=df.index).reset_index(drop=True)],
                       axis=1)

    # Average over the time dimension (groupby series id)
    wide = (
        err_df
        .groupby(id_col, as_index=False)
        .mean(numeric_only=True)          # mean SMAPE per series & model
    )

    wide = wide.assign(metric="smape")[[id_col, "metric", *models]]
    return wide


def compute_pct_increase(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each model's percentage difference from the Naive24h baseline.

    Parameters
    ----------
    evaluation_df : pandas.DataFrame
        Must contain
        * 'unique_id' - series identifier  
        * 'metric'    - name of the accuracy metric (e.g. 'MAE')  
        * 'Naive24h'  - baseline values  
        * one column per additional model

    Returns
    -------
    pandas.DataFrame
        Wide-format frame with the original metrics plus new rows where
        metric is suffixed by '_pct_inc' and each model column holds
        the percentage increase
    """
    # Melt to long form
    long = evaluation_df.melt(
        id_vars=['unique_id','metric'],
        var_name='model',
        value_name='value'
    )

    # Pull out the Naive24h baseline
    baseline = (
        long[long['model']=='Naive24h']
          .rename(columns={'value':'naive'})
          .loc[:, ['unique_id','metric','naive']]
    )

    # Merge baseline back onto all rows
    merged = long.merge(baseline, on=['unique_id','metric'])

    # Compute pct-inc only for non-Naive
    pct = merged[merged['model']!='Naive24h'].copy()
    pct['value'] = (pct['value'] - pct['naive']) / pct['naive'] * 100
    pct['metric'] = pct['metric'] + '_pct_inc'

    # Stack originals + pct rows
    all_long = pd.concat([long, pct[['unique_id','metric','model','value']]], ignore_index=True)

    # One final pivot back to wide
    wide = (
        all_long
        .pivot_table(
            index=['unique_id','metric'],
            columns='model',
            values='value',
            fill_value=0 # fill with 0s for missing values (corresponding to the pct inc of the Naive24h model itself)
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .sort_values(['unique_id','metric'])
        .reset_index(drop=True)
    )

    # Drop me_pct_inc rows if they exist
    wide = wide[~(wide['metric']=='me_pct_inc')].copy()

    return wide

def custom_evaluate(
    forecast_df: pd.DataFrame,  
    target_df: pd.DataFrame,  
    insample_size: Optional[int] = None,  
    metrics: Optional[List[str]] = None,  
    with_naive: bool = True,  
    with_pct_increase: bool = False,  # Whether to compute percentage increase compared to Naive24h
    period_for_nmae: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,  
) -> pd.DataFrame:
    """
    Evaluate forecasts against the ground truth and return a
    tidy table of accuracy metrics.

    Parameters
    ----------
    forecast_df : pandas.DataFrame
        Model output with columns
        * 'unique_id' - series identifier  
        * 'ds'        - forecast timestamp  
        * one column per model containing point forecasts
    target_df : pandas.DataFrame
        Complete history of the target variable.  Must include
        'unique_id', 'ds', 'y'.
    insample_size : int, optional
        Limit the Naive24h training window to the last insample_size
        observations per series when computing MASE.  
    metrics : list[str], optional
        Accuracy measures to compute. 
    with_naive : bool, default True
        Fit and evaluate a 24-hour seasonal naive benchmark unless it is
        already present in *forecast_df*.
    with_pct_increase : bool, default False
        Append extra rows where metric is suffixed by '_pct_inc' and the
        values are percentage differences from Naive24h.  Requires
        with_naive to be True.
    period_for_nmae : tuple of (pd.Timestamp, pd.Timestamp), optional
        Time range over which to compute the normaliser for NMAE.
        Defaults to (pd.Timestamp("2019-01-01"),
        pd.Timestamp("2025-01-01")).  Ignored when 'nmae' is not
        among metrics.

    Returns
    -------
    pandas.DataFrame
        Wide-format table with columns

        ['unique_id', 'metric', <model1>, <model2>, …].
    """
    available_metrics = ['mae', 'rmse', 'mase', 'nmae', 'mape', 'smape', 'me']
    if metrics is not None:
        if not set(metrics).issubset(set(available_metrics)):
            raise ValueError(f"metrics must be a subset of {available_metrics}.")
    else:
        metrics = ['mae', 'rmse', 'smape', 'nmae']

    str_to_func = {
        'mae': mae,
        'rmse': rmse,
        'mase': partial(mase, seasonality=24),  # Assuming a 24-hour seasonality for the naive model
        'nmae': partial(nmae, target_df=target_df, period=period_for_nmae),
        'mape': mape,
        'smape': smape,
        'me': me
    }
    metrics_func = [str_to_func[key] for key in metrics if key in str_to_func]

    # Add the true values to the forecast DataFrame for evaluation
    forecast_and_val_df = forecast_df.merge(
        target_df[['unique_id', 'ds', 'y']],
        on=['unique_id', 'ds'],
        how='left'
    )

    if 'mase' in metrics or (with_naive and 'Naive24h' not in forecast_and_val_df.columns):
        start_test = forecast_df['ds'].min()
        heat_train_df = target_df[target_df['ds'] < start_test].copy()
        if insample_size is not None:
            # If insample_size is provided, limit the training data to the last `insample_size` hours
            heat_train_df = heat_train_df.groupby('unique_id').apply(
                lambda x: x.tail(insample_size)
            ).reset_index(drop=True)

    if with_naive and 'Naive24h' not in forecast_and_val_df.columns:
        h = forecast_df['ds'].nunique()  # Number of hours to forecast
        
        # Compute forecasts using the naive method
        naive_model24 = SeasonalNaive(season_length=24, alias='Naive24h')
        naive_forecast_df = StatsForecast(
            models=[naive_model24], 
            freq='h'
        ).forecast(h, heat_train_df)

        # Merge the forecasts into a single DataFrame
        forecast_and_val_df = (
            forecast_and_val_df
            .merge(naive_forecast_df, on=['unique_id', 'ds'], how='left')
        )

    # Evaluate
    if 'mase' in metrics: 
        evaluation_df = evaluate(df=forecast_and_val_df, metrics=metrics_func, train_df=heat_train_df)
    else:
        evaluation_df = evaluate(df=forecast_and_val_df, metrics=metrics_func)
    
    if with_pct_increase:
        if not with_naive:
            raise ValueError("with_naive must be True to compute percentage increase compared to Naive24h.")
        # Compute percentage increase compared to Naive24h
        evaluation_df = compute_pct_increase(evaluation_df)

    return evaluation_df

def evaluate_cv_forecasts(
    cv_df: pd.DataFrame,  # DataFrame with the forecasts of the models, with columns 'unique_id', 'ds', 'cutoff', and forecast columns
    metrics: Optional[List[str]] = None,  # List of metrics to compute
    target_df: Optional[pd.DataFrame] = None,  # Full training data with 'ds', 'unique_id', and 'y' columns, for mase if requested
    target_col: str = "y",  # Column name in `target_df` for the true values
    period_for_nmae: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,  # Time range for NMAE normalization
) -> pd.DataFrame:
    """
    Evaluate multi-model, multi-window cross-validation results and
    return both per-window scores and an aggregated summary.

    Returns
    -------
    all_results : pandas.DataFrame
        Window-level metrics: one row per ``unique_id`` x metric x cutoff,
        with a column for every model's score in that window.
    """
    uids = cv_df['unique_id'].unique()

    # select only the sieries that are in cv_df
    target_df = target_df[target_df['unique_id'].isin(uids)].copy() if target_df is not None else None

    available = ['mae', 'rmse', 'mase', 'nmae', 'mape', 'smape', 'me']
    if metrics is None:
        metrics = ['mae', 'rmse', 'smape', 'nmae']
    else:
        if not metrics:                         # forbids empty list
            raise ValueError("metrics list may not be empty.")
        bad = set(metrics) - set(available)
        if bad:
            raise ValueError(
                f"Unknown metric(s): {sorted(bad)}. "
                f"Choose from {available}."
            )
    
    if 'mase' in metrics and target_df is None:
        raise ValueError("`target_df` is required when 'mase' is requested.")

    str_to_func = {
        'mae': mae,
        'rmse': rmse,
        'mase': partial(mase, seasonality=24),  # Assuming a 24-hour seasonality for the naive model
        'nmae': partial(nmae, target_df=target_df, period=period_for_nmae),
        'mape': mape,
        'smape': smape,
        'me': me
    }
    metrics_func = [str_to_func[key] for key in metrics if key in str_to_func]
    
    cv_df = cv_df.copy()
    all_res = []

    for cutoff in sorted(cv_df['cutoff'].unique()):
        window_df = cv_df[cv_df['cutoff'] == cutoff].drop(columns='cutoff')

        kw = dict(df=window_df, metrics=metrics_func)
        if 'mase' in metrics:                          # needs training slice
            kw['train_df'] = target_df[target_df['ds'] <= cutoff]

        eval_df = evaluate(**kw, target_col=target_col).copy()       # one call per window
        eval_df = pd.concat([eval_df, pd.DataFrame({'cutoff': [cutoff] * len(eval_df)})], axis=1)
        all_res.append(eval_df)

    all_results = pd.concat(all_res, ignore_index=True).copy()

    return all_results.copy()

def cv_evaluation_summary(
    all_results: pd.DataFrame,
    stats: Literal['mean', 'median', 'std', 'wrate'] = ['mean', 'median', 'std', 'wrate'],
    ignore_cutoffs: Optional[List[pd.Timestamp]] = None
):
    """
    Summarise cross-validation results by computing mean, median,
    standard deviation, and win rate per model.
    Parameters
    ----------
    all_results : pandas.DataFrame
        DataFrame with the window-level metrics as returned by
        evaluate_cv_forecasts. 
    stats : list of str, default ['mean', 'median', 'std', 'wrate']
        List of statistics to compute for each model.  Allowed values are
        ['mean', 'median', 'std', 'wrate'].  The default is to compute all
        four statistics.
    Returns
    -------
    summary : pandas.DataFrame
        Aggregated metrics: one row per unique_id and metric, with
        three columns for every model (<model>_mean, <model>_median,
        <model>_std, <model>_win_rate) summarising performance across cutoffs.
    """
    # Remove from stats unwanted cutoffs
    if ignore_cutoffs is not None:
        if not (isinstance(ignore_cutoffs, Iterable) and 
                all(isinstance(ts, pd.Timestamp) for ts in ignore_cutoffs)):
            raise ValueError("ignore_cutoffs must be a iterable of pd.Timestamp values.")
        all_results = all_results[~all_results['cutoff'].isin(ignore_cutoffs)].copy()

    # validate stats
    allowed_stats = {'mean', 'median', 'std', 'wrate'}
    if isinstance(stats, str):
        stats = [stats]
    bad_stats = set(stats) - allowed_stats
    if bad_stats:
        raise ValueError(f"Unknown stats: {sorted(bad_stats)}. Choose from {sorted(allowed_stats)}.")
    
    # Get model columns (exclude 'unique_id', 'metric', 'cutoff')
    stats_by_agg = list(set(stats) & {'mean', 'median', 'std'})
    model_cols = [c for c in all_results.columns
              if c not in ("unique_id", "metric", "cutoff")]
    summary = (
        all_results
        .groupby(["unique_id", "metric"])[model_cols]    # ← keep cutoff out
        .agg(stats_by_agg)
        .pipe(lambda df: df.set_axis(
            [f"{col}_{stat}" for col, stat in df.columns], axis=1))
        .reset_index()
    )

    if 'wrate' not in stats:
        return summary.copy()
    
    # compute win rates
    def _compute_win_rates_for_group(group_df: pd.DataFrame, models: List[str]) -> pd.Series:
        """
        group_df: subset of all_results for one (unique_id, metric),
                  rows = different cutoffs, columns include model_cols.
        returns: pd.Series indexed by models with win rates (percent).
        """
        # Extract the matrix of model scores (rows = windows)
        scores = group_df.loc[:, models].to_numpy(dtype=float)  # shape (n_windows, n_models)

        n_windows = scores.shape[0]
        if n_windows == 0:
            # no windows: return zeros
            return pd.Series(0.0, index=models)

        # find per-row minimum (nan-safe)
        row_min = np.nanmin(scores, axis=1)  # length n_valid

        # determine ties (close to min) and give fractional credit
        # we use np.isclose to handle floating point ties
        is_best = np.isclose(scores, row_min[:, None])  # shape (n_windows, n_models)
        # count ties per row:
        tie_counts = is_best.sum(axis=1)                    # shape (n_windows,)

        # fractional credit: each best model gets 1/tie_count for that row
        fractional = is_best.astype(float) / tie_counts[:, None]  # broadcasting
        # sum fractional credits across rows -> win_counts per model
        win_counts = fractional.sum(axis=0)  # length n_models

        # normalize by total number of windows
        win_rates = (win_counts / float(n_windows))

        return pd.Series(win_rates, index=models)
    
        # We'll compute win rates per (unique_id, metric) group and attach columns to summary.
    # Because for 'me' we want to compare abs(me), we'll prepare a small helper to select the right scores.
    win_rate_frames = []  # list of DataFrames (one per group) to concat and then merge into summary

    grouped = all_results.groupby(["unique_id", "metric"])
    for (uid, metric_name), grp in grouped:
        # create a copy of model columns for comparisons
        grp_scores = grp.loc[:, model_cols].copy()

        if metric_name == "me":
            # For ME, define "best" as closest to zero -> compare on absolute values
            grp_scores = grp_scores.abs()

        # compute win rates (fractional tie-splitting)
        win_rates_series = _compute_win_rates_for_group(grp_scores, model_cols)

        # construct df row
        row = {"unique_id": uid, "metric": metric_name}
        for m in model_cols:
            row[f"{m}_wrate"] = win_rates_series[m]
        win_rate_frames.append(row)

    if win_rate_frames:
        win_rates_df = pd.DataFrame(win_rate_frames)
    else:
        # no data
        win_rates_df = pd.DataFrame(columns=["unique_id", "metric"] + [f"{m}_wrate" for m in model_cols])

    # Merge the win_rates into summary (left join on unique_id, metric)
    # If summary lacks some (unique_id, metric) combos (unlikely), we still keep summary as base.
    summary = summary.merge(win_rates_df, on=["unique_id", "metric"], how="left")

    # Fill NaNs (if any) with 0.0 for win rates
    for m in model_cols:
        col = f"{m}_wrate"
        if col in summary.columns:
            summary[col] = summary[col].fillna(0.0)

    return summary.copy()

def display_metrics(
    evaluation_df: pd.DataFrame,  # DataFrame with evaluation metrics
):
    """
    Render an interactive table that summarises model-evaluation results.
    """
    long = evaluation_df.melt(
        id_vars=['unique_id','metric'],
        var_name='model',
        value_name='value'
    )
    wide = long.pivot(
        index=['unique_id','model'],
        columns=['metric'],
        values='value'
    ).sort_index(axis=0, level=[0,1])
    wide.index.names = ['facility','model']
    wide.columns.names = ['metric']

    # Add average rows
    if evaluation_df['unique_id'].nunique() > 1:
        model_means = wide.groupby(level='model').mean()
        model_means.index = pd.MultiIndex.from_product(
            [['Average'], model_means.index],
            names=wide.index.names
        )
        combined = pd.concat([wide, model_means])
    else:
        combined = wide

    # Define the custom green—white—red colormap
    cmap = LinearSegmentedColormap.from_list(
        "GreenWhiteRed", 
        ["green", "lightgray", "red"]
    )

    bar_cols  = ['mae', 'rmse', 'mape', 'mase', 'nmae']           # bars
    grad_cols = [c + '_pct_inc' for c in bar_cols]                # add _pct_inc suffix

    grad_cols_actual = [c for c in grad_cols if c in combined.columns]
    bar_cols_actual  = [c for c in bar_cols if c in combined.columns]

    styled = combined.style

    for col in grad_cols_actual:
        rng   = combined[col].abs().max()
        styled = styled.background_gradient(
            cmap=cmap,
            subset=[col],        # ← single column
            vmin=-rng, vmax=rng  # ← scalar, so no error
        )

    styled = (
        styled
        .set_properties(subset=grad_cols_actual, **{'color': 'white'})
        .bar(subset=bar_cols_actual, align='mid')
        .format("{:.2f}")
    )

    display(styled)

def display_cv_summary(
    summary: pd.DataFrame,
    sort_metric: Optional[str] = 'mae',   # e.g. "rmse", "mae", …
    sort_stat: str = "mean",             # "mean" or "std"
    ascending: bool = True,
    by_panel: bool = False,              # True → sort within each unique_id
    show_row_numbers: bool = False,      # True → add a “#” column (0-based)
    are_loss_diffs: bool = False,        # True → indicate loss differences
    times_df: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Pretty-print cross-validation metrics (wide table) with optional sorting
    and an optional plain integer row index.
    """
    # Add times info if provided (average elapsed time per fit)
    if times_df is not None:
        ids = summary['unique_id'].unique()
        if len(ids) > 1:
            raise ValueError("times_df can only be provided when summary contains a single unique_id.")
        id = ids[0]
        if isinstance(times_df, pd.Series):
            times_df = times_df.reset_index()
        if 'model' not in times_df.columns:
            raise KeyError("times_df must a Series with index='model' or a DataFrame with a 'model' column.")
        times_df = times_df.rename(columns={'avg_el_per_fit':'time (s)'})
        if 'time (s)' not in times_df.columns:
            raise KeyError("times_df must contain a column named 'avg_el_per_fit' or 'time (s)'.")
        times_df['unique_id'] = id
        # add naive model if missing
        if 'Naive24h' not in times_df['model'].values:
            times_df = pd.concat([
                times_df,
                pd.DataFrame({
                    'unique_id': [id],
                    'model': ['Naive24h'],
                    'time (s)': [0.0]
                })
            ], ignore_index=True)
        times_df.set_index(['unique_id', 'model'], inplace=True)
        times_df.columns = pd.MultiIndex.from_product([list(times_df.columns), ['mean']])

    # Reshape to a nicer (unique_id, model) × (metric, stat) cube 
    long = (
        summary
        .melt(id_vars=["unique_id", "metric"],
              var_name="tmp", value_name="value")
        .assign(
            model=lambda d: d["tmp"].str.rsplit("_", n=1).str[0],
            stat=lambda d: d["tmp"].str.rsplit("_", n=1).str[1],
        )
        .drop(columns="tmp")
    )

    wide = (
        long
        .pivot(index=["unique_id", "model"],
               columns=["metric", "stat"],
               values="value")
        .sort_index(axis=0)                # rows: unique_id > model
        .sort_index(axis=1, level=[0, 1])  # cols: metric > stat
    )
    wide.index.names   = ["unique_id", "model"]
    wide.columns.names = ["metric", "stat"]

    # Merge times if provided
    if times_df is not None:
        wide = wide.merge(
            times_df,
            left_index=True,
            right_index=True,
            how="left"
        )

    # Optional sorting 
    if sort_metric is not None:
        sort_key = (sort_metric, sort_stat)
        if sort_key not in wide.columns:
            raise ValueError(
                f"{sort_metric!r} with stat {sort_stat!r} not present in summary."
            )

        if by_panel:
            wide = (
                wide
                .groupby(level="unique_id", group_keys=False)
                .apply(lambda df: df.sort_values(sort_key, ascending=ascending))
            )
        else:
            wide = wide.sort_values(sort_key, ascending=ascending)

    # Optional "#" column 
    if show_row_numbers:
        wide_disp = wide.copy()
        wide_disp.insert(0, "#", range(len(wide_disp)))
    else:
        wide_disp = wide

    # Styler (skip the "#" column when colouring) 
    metric_cols = [c for c in wide_disp.columns if isinstance(c, tuple)]
    mean_cols   = [c for c in metric_cols if c[1] == "mean"]
    median_cols = [c for c in metric_cols if c[1] == "median"]
    wrate_cols  = [c for c in metric_cols if c[1] == "wrate"]
    std_cols    = [c for c in metric_cols if c[1] == "std"]

    # pick columns to represent as percentages
    perc_cols = [c for c in metric_cols if c[0] in ["nmae", "smape"]]
    other_cols = [c for c in metric_cols if c not in perc_cols]

    styler = wide_disp.style

    if are_loss_diffs:
        # Use a diverging green–white–red background gradient instead of bars
        cmap = LinearSegmentedColormap.from_list(
            "GreenWhiteRed",
            ["green", "lightgray", "red"]
        )
        cmap_subset = mean_cols + median_cols
        for col in cmap_subset:
            col_vals = wide_disp[col].to_numpy()
            col_max = np.nanmax(np.abs(col_vals))
            if np.isnan(col_max) or col_max == 0:
                # nothing to color for this column; skip or set a tiny epsilon
                continue
            styler = styler.background_gradient(
                cmap=cmap,           # e.g. "RdBu_r" or your Green-White-Red
                subset=[col],        # <-- IMPORTANT: one column at a time
                vmin=-col_max,       # symmetric around 0
                vmax= col_max,
                axis=0               # column-wise (default), fine to be explicit
            )
        styler = (
            styler
            .background_gradient(cmap="Blues", subset=std_cols)
            .background_gradient(cmap="Greens", subset=wrate_cols, vmin=0, vmax=1)
            .format("{:.2%}", subset=perc_cols)
            .format(precision=2, subset=other_cols)
            .format("{:.0%}", subset=wrate_cols)
        )
    else:
        # Default styling: bar for means/medians
        to_bar = [c for c in (mean_cols + median_cols) if c != ('time (s)', 'mean')]
        styler = (
            styler
            .bar(color="lightcoral", subset=to_bar, align="mid")
            .background_gradient(cmap="Blues", subset=std_cols)
            .background_gradient(cmap="Greens", subset=wrate_cols, vmin=0, vmax=1)
            .format("{:.2%}", subset=perc_cols)
            .format(precision=2, subset=other_cols)
            .format("{:.0%}", subset=wrate_cols)
        )
        if ('time (s)', 'mean') in wide_disp.columns:
            styler = (
                styler
                .background_gradient(cmap="Oranges", subset=[('time (s)', 'mean')])
                .format("{:.0f}", subset=[('time (s)', 'mean')])
            )
        
    display(styler)
    return wide_disp

def barplot_cv(df):
    """
    Draw a grid of bar charts that compares model performance across
    series and error metrics.
    """
    long = df.melt(
        id_vars=['unique_id', 'cutoff', 'metric'],
        var_name='model', value_name='value'
    )

    ids     = long['unique_id'].unique()
    metrics = long['metric'].unique()

    nrows, ncols = len(ids), len(metrics)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(3 * ncols, 3 * nrows),
        sharey=False
    )

    # axes is 2-D even if nrows or ncols == 1
    axes = np.atleast_2d(axes)

    for i, uid in enumerate(ids):
        for j, m in enumerate(metrics):
            ax  = axes[i, j]
            sub = long.query("unique_id == @uid and metric == @m")

            sns.barplot(
                data=sub,
                x='model', y='value',
                errorbar=('pi', 80),
                ax=ax, dodge=True,
                color="#3A78EA",
            )

            if i == 0: # top row: column titles
                ax.set_title(m, weight='bold')
            if j == 0: # first column: row labels
                ax.set_ylabel(uid)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("") # model names already on ticks

    fig.suptitle("Mean performance ± 80 % interval per series / metric",
                 y=1.02)
    fig.tight_layout()
    plt.show()

def plot_cv_metric_by_cutoff(
        combined_results: pd.DataFrame,
        metric: str = "mae",
        figsize: Optional[tuple[int, int]] = None,
        models: Optional[list[str]] = None,
    ) -> plt.Figure:
    """
    Plot a grid of bar charts that visualise cross-validation scores by
    cut-off date and model.

    For every time-series (`unique_id`) a separate subplot is created.
    Inside each subplot, bars display the chosen error metric for each
    model at every back-test cut-off date.

    Parameters
    ----------
    combined_results : pandas.DataFrame
        Cross-validation results.  Must include the columns
        * ``'unique_id'`` (str) - series identifier  
        * ``'metric'``    (str) - name of the metric  
        * ``'cutoff'``    (datetime64) - forecast origin  
        * one column per model (numeric scores)
    metric : str, default ``"mae"``
        Metric to plot.  Rows with other metrics are filtered out before
        plotting.
    figsize : tuple[int, int], default ``None``
        Size of the entire figure in inches *(width, height)*.
    models : list[str] | None, default ``None``
        Subset of model columns to display.  If ``None``, all model
        columns found in *combined_results* are used.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting figure object.
    """
    
    # Filter for metric only
    metric_df = combined_results[combined_results['metric'] == metric].copy()
    
    # Get unique facilities and models
    facilities = sorted(metric_df['unique_id'].unique())
    
    # Get model columns (exclude metadata columns)
    model_cols = [col for col in metric_df.columns 
                  if col not in ['unique_id', 'metric', 'cutoff']]
    if models is not None:
        if not isinstance(models, list):
            raise ValueError("`models` should be None or a list of model names.")
        for m in models:
            if not isinstance(m, str):
                raise ValueError(f"Model name '{m}' should be a string.")
            if m not in model_cols:
                raise ValueError(f"Model '{m}' not found in the results. Available models: {model_cols}")
        model_cols = models
    
    # Create subplots - one for each facility
    fig, axes = plt.subplots(nrows=len(facilities), ncols=1, figsize=(15, len(facilities) * 7) if figsize is None else figsize,
                             sharex=True)
    axes = np.atleast_1d(axes)
    
    # Color palette for models
    colors = sns.color_palette("tab10", len(model_cols))
    
    for i, facility in enumerate(facilities):
        ax = axes[i]
        
        # Filter data for this facility
        facility_data = metric_df[metric_df['unique_id'] == facility].copy()
        
        # Sort by cutoff for proper ordering
        facility_data = facility_data.sort_values('cutoff')
        
        # Convert cutoff to string for better x-axis labels
        facility_data['cutoff_str'] = facility_data['cutoff'].dt.strftime('%Y-%m-%d')
        
        # Create bar positions
        x_positions = range(len(facility_data))
        bar_width = 0.8 / len(model_cols)

        # Set y-axis limits
        bottom = min(facility_data[model_cols].min().min(), 0)
        y_max = facility_data[model_cols].max().max()
        margin = 0.05 * y_max            # 5 % head-room
        ax.set_ylim(bottom, y_max + margin)   # text now lives inside the axes

        # Offset for test on bars
        offset = 0.01 * y_max
        
        # Plot bars for each model
        for j, model in enumerate(model_cols):
            x_pos = [x + bar_width * (j - len(model_cols)/2 + 0.5) for x in x_positions]
            bars = ax.bar(x_pos, facility_data[model], 
                         width=bar_width, 
                         label=model,
                         color=colors[j],
                         alpha=0.8)
            
            # Add value labels on bars 
            for bar, val in zip(bars, facility_data[model]):
                if not pd.isna(val):
                    #ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    #       f'{val:.1f}', ha='center', va='bottom', fontsize=8)
                    ax.text(bar.get_x() + bar.get_width()/2,
                            val + offset,
                            f'{val:.1f}',
                            ha='center', va='bottom', fontsize=8)
        
        # Customize subplot
        ax.set_title(f'Series {facility}')
        ax.set_xlabel('Cutoff Date')
        ax.set_ylabel(f'{metric}')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(facility_data['cutoff_str'], rotation=45, ha='right')

    # build a single legend for the whole figure
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, lab in zip(h, l):
            if lab not in labels:
                handles.append(hi)
                labels.append(lab)

    # place the legend above all subplots
    fig.legend(handles, labels,
               loc='upper right',
               ncol=1,
               bbox_to_anchor=(1, 0.97))

    fig.suptitle(f'Cross-Validation {metric} Results by Cutoff and Model', y=0.98, fontsize=18, fontweight='bold')     
    fig.tight_layout(rect=[0, 0, 1, 0.97])                     
    
    return fig

def plotly_cv_metric_by_cutoff(
    combined_results: pd.DataFrame,
    metric: str = "mae",
    models: Optional[List[str]] = None,
    height_per_row: int = 320,
    width: int = 1400,
    title_suffix: Optional[str] = None,
    as_lineplot: bool = False,
    grayscale_safe: bool = False,
    aux_df: Optional[pd.DataFrame] = None
) -> go.Figure:
    """
    Plot a grid of grouped bar charts (one row per `unique_id`) showing cross-validation
    scores by cutoff date and model, using Plotly (graph_objects).

    Parameters
    ----------
    combined_results : pandas.DataFrame
        Must include:
        - 'unique_id' (str): series identifier
        - 'metric' (str): metric name
        - 'cutoff' (datetime64): forecast origin
        - one column per model (numeric scores)
    metric : str, default "mae"
        Which metric rows to plot.
    models : list[str] | None, default None
        Subset of model columns to display. If None, all model columns found are used.
    height_per_row : int, default 320
        Plot height per subplot row (pixels).
    width : int, default 1400
        Overall figure width (pixels). You can update this later via fig.update_layout(width=...).

    Returns
    -------
    go.Figure
        The Plotly figure object.
    """
    # 1) Filter metric
    df = combined_results.loc[combined_results["metric"] == metric].copy()
    if df.empty:
        raise ValueError(f"No rows found for metric='{metric}'")

    # 2) Identify model columns
    meta_cols = {"unique_id", "metric", "cutoff"}
    all_model_cols = [c for c in df.columns if c not in meta_cols]
    if not all_model_cols:
        raise ValueError("No model columns found (columns other than 'unique_id','metric','cutoff').")

    if models is not None:
        if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
            raise ValueError("`models` should be None or a list[str].")
        missing = [m for m in models if m not in all_model_cols]
        if missing:
            raise ValueError(f"Models not found: {missing}. Available: {all_model_cols}")
        model_cols = models
    else:
        model_cols = all_model_cols

    # 3) Prepare list of series (facilities)
    series_list = sorted(df["unique_id"].unique())
    nrows = len(series_list)

    # 4) Build subplots
    fig = make_subplots(
        rows=nrows if aux_df is None else 2*nrows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[f"Series {sid}" for sid in series_list],
    )

    # Choose a pleasant qualitative palette and map per model
    if grayscale_safe:
        # Distinct, high-contrast grays
        palette = ["#111111", "#A7A5A5", "#C3C3C3", "#DDDDDD"]
    else:
        palette = px.colors.sequential.Viridis
    color_indices = [i * len(palette) // len(model_cols) for i in range(len(model_cols))]
    colors = {m: palette[color_indices[i] % len(palette)] for i, m in enumerate(model_cols)}
    # marker + line style cycles for variety
    line_styles = ["solid", "dot", "dash", "longdash", "dashdot"]
    markers = ["circle", "square", "diamond", "x", "triangle-up"]
    metric_names = {"mae": "MAE", "smape": "sMAPE", "nmae": "NMAE", "rmse": "RMSE"}

    if aux_df is not None:
        cutoffs = combined_results['cutoff'].unique()
        ds0 = cutoffs.min() + pd.Timedelta(hours=1)
        dsMax = cutoffs.max() + pd.Timedelta(hours=24)
        temp_hourly_df = aux_df.loc[(aux_df['ds'] >= ds0) & (aux_df['ds'] <= dsMax), ["temperature", "unique_id", "ds"]].copy()
        temp_hourly_df['cutoff'] = temp_hourly_df["ds"].dt.floor('D') - pd.Timedelta(hours=1)
        temp_df = temp_hourly_df.groupby(['unique_id', 'cutoff']).agg({'temperature': 'mean'}).reset_index()

    # 5) Add one row per series
    for r, sid in enumerate(series_list, start=1):
        sub = df.loc[df["unique_id"] == sid].copy()
        if sub.empty:
            continue

        # Sort by cutoff; create string labels for tidy x-axis
        sub = sub.sort_values("cutoff")
        sub["cutoff_str"] = sub["cutoff"].dt.strftime("%Y-%m-%d")
        x = sub["cutoff_str"].tolist()

        # Add one trace per model (grouped bars)
        for i, m in enumerate(model_cols):
            y_vals = sub[m].tolist()

            if as_lineplot:
                # ---- Line + Marker version ----
                fig.add_trace(
                    go.Scatter(
                        name=m.split('-')[1] if '-' in m else m,
                        x=x,
                        y=y_vals,
                        mode="lines", #"lines+markers"
                        line=dict(color=colors[m]),#, dash=line_styles[i % len(line_styles)]),
                        marker=dict(symbol=markers[i % len(markers)]),
                        hovertemplate=(
                            f"<b>Series:</b> {sid}<br>"
                            f"<b>Model:</b> {m}<br>"
                            "<b>Cutoff:</b> %{x}<br>"
                            f"<b>{metric.upper()}:</b> %{{y:.3f}}<extra></extra>"
                        ),
                        showlegend=(r == 1),
                    ),
                    row=r if aux_df is None else 2*(r-1)+1,
                    col=1,
                )
            else:
                # ---- Grouped Bar version ----
                fig.add_trace(
                    go.Bar(
                        name=m.split('-')[1] if '-' in m else m,
                        x=x,
                        y=y_vals,
                        marker_color=colors[m],
                        cliponaxis=False,  # helps prevent cutoff of text
                        hovertemplate=(
                            f"<b>Series:</b> {sid}<br>"
                            f"<b>Model:</b> {m}<br>"
                            "<b>Cutoff:</b> %{x}<br>"
                            f"<b>{metric.upper()}:</b> %{{y:.3f}}<extra></extra>"
                        ),
                        showlegend=(r == 1),  # show legend only once (top subplot)
                    ),
                    row=r if aux_df is None else 2*(r-1)+1,
                    col=1,
                )

        if aux_df is not None: # Plot temperature
            y_vals = temp_df['temperature'].values
            if as_lineplot:
                # ---- Line + Marker version ----
                fig.add_trace(
                    go.Scatter(
                        name="T (°C)",
                        x=x,
                        y=y_vals,
                        mode="lines+markers",
                        line=dict(color="gray" if grayscale_safe else "blue", dash="dot"),
                        marker=dict(symbol=markers[0]),
                        hovertemplate=(
                            f"<b>Series:</b> {sid}<br>"
                            "<b>Cutoff:</b> %{x}<br>"
                            f"<b>Temperature (°C):</b> %{{y:.2f}}<extra></extra>"
                        ),
                        showlegend=(r == 1),
                    ),
                    row=2*r,
                    col=1,
                )
            else:
                # ---- Grouped Bar version ----
                fig.add_trace(
                    go.Bar(
                        name="T (°C)",
                        x=x,
                        y=y_vals,
                        marker_color="gray" if grayscale_safe else "blue",
                        cliponaxis=False,  # helps prevent cutoff of text
                        hovertemplate=(
                            f"<b>Series:</b> {sid}<br>"
                            "<b>Cutoff:</b> %{x}<br>"
                            f"<b>Temperature (°C):</b> %{{y:.2f}}<extra></extra>"
                        ),
                        showlegend=(r == 1),  # show legend only once (top subplot)
                    ),
                    row=2*r,
                    col=1,
                )
            top_row = 2*(r-1) + 1
            bot_row = 2*r
            # Make bottom row's x-axis match the top row's x-axis (shared range + zoom/pan)
            fig.update_xaxes(matches=f"x{top_row}", row=bot_row, col=1)
            # Hide duplicate ticks on the top row (keep only bottom)
            fig.update_xaxes(showticklabels=False, row=top_row, col=1)

            # per-row axis labels
            fig.update_yaxes(title_text=metric_names[metric], row=top_row, col=1)
            fig.update_yaxes(title_text="Temperature (°C)", row=bot_row, col=1)
            fig.update_xaxes(title_text="Cutoff Date", tickangle=0, row=bot_row, col=1)
        else:
            fig.update_yaxes(title_text=metric_names[metric], row=r, col=1)
            fig.update_xaxes(title_text="Cutoff Date", tickangle=0, row=r, col=1)

    # 6) Global layout
    fig.update_layout(
        barmode="group",
        width=width,
        height=max(300, nrows * height_per_row),
        title_text=f"Cross-Validation {metric} Results by Cutoff and Model" + (f" - {title_suffix}" if title_suffix else ""),
        title_x=0.5,
        legend_title_text="Model",
        margin=dict(l=60, r=20, t=70, b=60),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )
    # --- White background for grayscale-safe mode ---
    if grayscale_safe:
        fig.update_layout(template="plotly_white")
        fig.update_xaxes(showgrid=True, gridcolor="rgb(235,235,235)")
        fig.update_yaxes(showgrid=True, gridcolor="rgb(235,235,235)")
    return fig


def compute_loss_diffs(
        combined_results: pd.DataFrame,
        baseline_model: str = 'Naive24h'
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute loss-differences (LD) versus a
    baseline model.

    For every model column in *combined_results* (except the baseline),
    the function subtracts the baseline's losses to create new columns
    named ``"LD-<model>"``.  
    """
    # Check if the baseline model exists in the results
    if baseline_model not in combined_results.columns:
        raise ValueError(
            f"baseline_model '{baseline_model}' not found in results."
        )
    # Get model columns (exclude 'unique_id', 'metric', 'cutoff')
    meta_cols  = {"unique_id", "metric", "cutoff"}
    models     = [c for c in combined_results.columns if c not in meta_cols]
    other_mdl  = [m for m in models if m != baseline_model]
    if not other_mdl: raise ValueError("No models other than the baseline—nothing to compare.")

    # Compute loss diff
    ld_block = (
        combined_results[other_mdl]
        .sub(combined_results[baseline_model], axis=0)   
        .add_prefix("LD-")                              
    )
    all_loss_diff = pd.concat(
        [combined_results[["unique_id", "metric", "cutoff"]], ld_block],
        axis=1,
    )

    return all_loss_diff

def adj_r2_score(y, y_hat, T, k):
    """
    Compute the adjusted R-squared score.

    Adjusted R² accounts for the number of predictors in the model and 
    penalizes excessive use of non-informative features.

    Parameters
    ----------
    y : array-like
        True target values.
    y_hat : array-like
        Predicted target values.
    T : int
        Number of observations (sample size).
    k : int
        Number of explanatory variables (not including intercept).
    """
    from sklearn.metrics import r2_score
    r2 = r2_score(y, y_hat)
    adj_r2 = 1 - (1 - r2) * (T - 1) / (T - k - 1)
    return adj_r2

def aic_score(y, y_hat, T, k):
    """
    Compute the Akaike Information Criterion (AIC) score.
    """
    sse = np.sum((y - y_hat) ** 2)
    aic = T * np.log(sse / T) + 2 * (k + 2)
    return aic

def aicc_score(y, y_hat, T, k):
    """
    Compute the corrected Akaike Information Criterion (AICc).
    """
    aic = aic_score(y, y_hat, T, k)
    aicc = aic + (2 * (k + 2) * (k + 3)) / (T - k - 3)
    return aicc

def bic_score(y, y_hat, T, k):
    """
    Compute the Bayesian Information Criterion (BIC) score.
    """
    sse = np.sum((y - y_hat) ** 2)
    bic = T * np.log(sse / T) + (k + 2) * np.log(T)
    return bic

def overforecast_over_th_score(y, y_hat, y_th):
    """
    Compute the average over-forecast error (y_hat > y).
    """
    error = y_hat - y
    over_mask = (error > 0) & (y > y_th)
    if np.any(over_mask):
        return np.mean(np.abs(error[over_mask]))
    else:
        return 0.0

def underforecast_over_th_score(y, y_hat, y_th):
    """
    Compute the average under-forecast error (y_hat < y).
    """
    error = y_hat - y
    under_mask = (error < 0) & (y > y_th)  
    if np.any(under_mask):
        return np.mean(np.abs(error[under_mask]))
    else:
        return 0.0
    
def mae_over_thr_score(y, y_hat, y_th):
    """
    Compute MAE limited to high y periods (y > y_th).
    """
    error = y_hat - y
    mask = (y > y_th)  
    if np.any(mask):
        return np.mean(np.abs(error[mask]))
    else:
        return 0.0

def by_horizon_preds(cv_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Extract hour-ahead predictions from cv_df, building a dict of DataFrames keyed by horizon in hours."""
    df = cv_df.copy()

    # Precompute horizon in hours once
    horizon_hours = ((df["ds"] - df["cutoff"]).dt.total_seconds() // 3600).astype(int)
    df["horizon_h"] = horizon_hours

    horizons = sorted(df["horizon_h"].unique())
    by_horizon_dict: Dict[int, pd.DataFrame] = {}

    for h in horizons:
        subset = df.loc[df["horizon_h"] == h].drop(columns=['horizon_h'])
        by_horizon_dict[h] = subset

    return by_horizon_dict

def compute_error_stats_by_horizon(
    by_horizon_dict: Dict[int, pd.DataFrame],  # Dict of DataFrames keyed by horizon in hours
    target_df: pd.DataFrame,                   # Ground-truth series aligned with cv_df.
    step: int = 1,                             # Evaluate every 'step' hours (e.g., 1, 3, 6).
    nmae_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    log_every: Optional[int] = 5,              # Log and optionally display every Nth horizon. Set to None to disable.
    show_per_cutoff: bool = True,              # Whether to show per-cutoff plots when logging using 'by_cutoff_plot_fn'
    show_progress: bool = True,
    evaluate_fn: Callable[..., Any] = None,    # Function to evaluate errors, must have args 'cv_df', 'target_df', 'period_for_nmae' 
                                               # (e.g. a partial of evaluate_cv_forecasts)
    summarize_fn: Callable[..., Any] = None,   # Function to summarize results, must have arg 'all_results'
                                               # (e.g. a partial of cv_evaluation_summary)
    by_cutoff_plot_fn: Optional[Callable[..., Any]] = None, 
                                               # Function to plot per-cutoff results, must have args 'combined_results', 
                                               # (e.g. a partial of plotly_cv_metric_by_cutoff)
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[int, Any], Dict[int, pd.DataFrame]]:
    logger = logger or logging.getLogger(__name__)

    iterator: Iterable[int] = sorted(by_horizon_dict.keys())[::step]
    horizons = list(iterator)
    if show_progress:
        iterator = tqdm(horizons, desc="Horizon error stats")

    by_horizon_summary: Dict[int, Any] = {}
    by_horizon_results: Dict[int, pd.DataFrame] = {}
    for idx, h in enumerate(iterator, start=1):
        subset = by_horizon_dict[h]

        results = evaluate_fn(
            cv_df=subset,
            target_df=target_df,
            period_for_nmae=nmae_period,
        )
        summary = summarize_fn(results)

        if log_every is not None and (idx - 1) % log_every == 0:
            logger.info("Error statistics for horizon h = %d", h)
            display_cv_summary(summary)
            if show_per_cutoff:
                fig = by_cutoff_plot_fn(results)
                display_scrollable(fig)

        by_horizon_summary[h] = summary
        by_horizon_results[h] = results

    return by_horizon_summary, by_horizon_results

def summaries_to_long(per_h_summ: dict) -> pd.DataFrame:
    # per_h_summ keys are horizons, values are 4-row DataFrames (metric in rows)
    df = pd.concat(per_h_summ, names=["horizon"])
    df = df.reset_index(level=0).rename(columns={"level_0": "horizon"})
    # Ensure numeric horizon
    df["horizon"] = df["horizon"].astype(int)
    return df

def plotly_models_vs_horizon(
    per_h_summ: dict,
    metric: str = "mae",
    models: list = ("LSTM", "XGBoost", "SARIMAX", "SVMR", "Naive24h"),
    agg: str = "mean",
    with_std_band: bool = False,
    with_bootstrap_band: bool = False,
    ci_level: float = 0.95,
    n_boot: int = 10000,
    block_length: Optional[int] = None,
    width: int = 900,
    height: int = 500,
    display_fig: bool = True,
    per_h_results: Optional[dict] = None,
) -> go.Figure:
    df = summaries_to_long(per_h_summ)
    dfm = df[df["metric"] == metric].sort_values("horizon")

    fig = go.Figure()
    palette = px.colors.qualitative.Set2
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(models)}

    x = dfm["horizon"].values
    xticks = sorted(dfm["horizon"].unique())

    y_lo, y_hi = None, None
    for i, model in enumerate(models):
        y_col = f"{model}_{agg}"
        std_col = f"{model}_std"
        if y_col not in dfm.columns:
            continue

        y = dfm[y_col].values
        c = color_map[model]

        # optional ±std band
        if with_std_band and std_col in dfm.columns:
            y_std = dfm[std_col].values
            y_lo = y - y_std
            y_hi = y + y_std

        # optional ±bootstrap confidence interval band
        elif with_bootstrap_band:
            from recombinator.optimal_block_length import optimal_block_length
            from recombinator.block_bootstrap import circular_block_bootstrap
            if per_h_results is None:
                raise ValueError("per_h_results must be provided when with_bootstrap_band=True")
            
            y_lo, y_hi = [], []
            for h in xticks:
                df_res = per_h_results.get(h, pd.DataFrame())
                df_res = df_res[df_res['metric']==metric]
                y_values = df_res.get(model, pd.Series()).values
                if len(y_values) == 0:
                    raise ValueError(f"No prediction values found for horizon {h} and model {model}")
                y_values = y_values[~np.isnan(y_values)]
                if len(y_values) == 0:
                    raise ValueError(f"Only NaNs for horizon {h} and model {model}")

                # Optimal block length for this series
                if block_length is not None:
                    b_cb = block_length
                else:
                    b_star = optimal_block_length(y_values)[0]
                    b_cb = int(np.ceil(b_star.b_star_cb))

                # Circular block bootstrap
                samples = circular_block_bootstrap(
                    y_values,
                    block_length=b_cb,
                    replications=n_boot,
                    replace=True,
                )  # shape = (replications, len(x))

                boot_means = samples.mean(axis=1)

                # CI (percentile bootstrap)
                alpha = 1 - ci_level
                q_low, q_high = np.percentile(
                    boot_means,
                    [100 * alpha / 2, 100 * (1 - alpha / 2)],
                )

                y_lo.append(q_low)
                y_hi.append(q_high)

        if y_lo is not None and y_hi is not None:
            def make_opaque(rgba_str: str, alpha: float) -> str:
                color_str = rgba_str.replace("rgba", "rgb")
                vals = color_str.strip("rgb() ").split(",")
                r, g, b = [v.strip() for v in vals]
                return f"rgba({r}, {g}, {b}, {alpha})"
            band_color = make_opaque(c, 0.2)

            # lower bound (no legend)
            fig.add_trace(
                go.Scatter(
                    x=x, y=y_lo,
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            # upper bound filled to previous
            fig.add_trace(
                go.Scatter(
                    x=x, y=y_hi,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=band_color,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # main line
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode="lines+markers",
                name=f"{model}",
                line=dict(width=2, color=c),
                marker=dict(size=6),
                hovertemplate=(
                    f"Model: {model}<br>"
                    "Horizon: %{x}<br>"
                    f"{metric.upper()} ({agg}): %{{y:.4g}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"{metric.upper()} vs Horizon",
        xaxis=dict(
            title="Forecast Horizon (hours)",
            tickmode="array",
            tickvals=xticks,
            showgrid=True
        ),
        yaxis=dict(
            title=metric.upper(),
            showgrid=True,
            zeroline=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        #margin=dict(l=60, r=30, t=70, b=50),
        width=width,
        height=height,
        template="plotly_white"
    )

    if display_fig:
        html = fig.to_html(include_plotlyjs="inline", full_html=False)  # offline, self-contained
        display(HTML(html))

    return fig


