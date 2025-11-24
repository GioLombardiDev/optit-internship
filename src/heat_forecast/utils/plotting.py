from pydoc import html
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd
from typing import Optional, Sequence, Dict, Any, Tuple, List, Callable, Iterable, Mapping
from datetime import datetime
import warnings
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from IPython.display import display
from matplotlib import colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from IPython.display import HTML, display
from matplotlib.ticker import FuncFormatter, MultipleLocator
import ipywidgets as widgets
from IPython.display import HTML
from statsmodels.tsa.stattools import acf
from scipy.stats import gaussian_kde
import copy

from .transforms import make_is_winter, get_lambdas, make_transformer, transform_column

# -------------------------------------------------------------------------------
# FOR PLOTTING FORECASTS / TARGET & EXOGENOUS SERIES
# -------------------------------------------------------------------------------

# Helper function to format the axes for series plots based on the displayed period
def configure_time_axes(
        axes: matplotlib.axes.Axes | Sequence[matplotlib.axes.Axes],
        period: Sequence[datetime | pd.Timestamp],
        *,
        global_legend: bool = False,
        legend_fig: Optional[matplotlib.figure.Figure] = None,
        legend_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
    """
    Format time axes of one or more matplotlib axes for time-series
    plots, adapting tick density and labels to the length of the period
    shown.

    The function chooses appropriate major and minor tick locators/
    formatters based on the total number of days between the first and
    last date in *period*.

    All axes receive identical formatting.

    Parameters
    ----------
    axes : matplotlib.axes.Axes or Sequence[matplotlib.axes.Axes]
        One axis or an iterable of axes to be formatted.

    period : Sequence[datetime | pandas.Timestamp]
        Two-element sequence ``[start, end]`` (or any iterable whose first
        and last elements represent the visible date range).  The function
        treats the elements as inclusive.
    
    global_legend : bool, default False
        If True, the function will create a global legend for all axes
    
    legend_fig : matplotlib.figure.Figure, optional
        If *global_legend* is True, this figure will be used to place
        the global legend. If None, the legend will be placed in the first
        axis of *axes*.

    Returns
    -------
    None
        The function works by side-effect; it modifies each axis in
        *axes* and returns nothing.
    """
    if isinstance(axes, matplotlib.axes.Axes):
        axes = [axes]
    else:
        try:                         
            iter(axes)
        except TypeError:
            raise ValueError(
                "`axes` must be a Matplotlib Axes instance "
                "or an iterable of Axes objects."
        )
    period = sorted(period)  # Ensure period is sorted
    start, end = period[0], period[-1]
    period_days = (end - start).days + 1  # Include the last day

    # Choose locator and formatter based on the period length
    if period_days < 1:
        locator_major = mdates.HourLocator(byhour=range(0, 24, 3))
        formatter_major = mdates.DateFormatter('%H:%M')
        locator_minor = mdates.MinuteLocator(interval=30)
        formatter_minor = None                       
    elif period_days < 11:
        locator_major = mdates.DayLocator(interval=1)
        formatter_major = mdates.DateFormatter('%d %b')
        locator_minor = mdates.HourLocator(byhour=[0, 12])
        formatter_minor = None
    elif period_days < 31:
        locator_major = mdates.WeekdayLocator(byweekday=mdates.MO)
        formatter_major = mdates.DateFormatter('Wk %W\n%d %b')
        locator_minor = mdates.DayLocator(interval=1)
        formatter_minor = None
    elif period_days < 91:
        locator_major = mdates.WeekdayLocator(byweekday=mdates.MO)
        formatter_major = mdates.DateFormatter('%d %b')
        locator_minor = mdates.DayLocator(interval=2)
        formatter_minor = None
    elif period_days < 120:
        locator_major = mdates.MonthLocator()
        formatter_major = mdates.DateFormatter('%b\n%Y')
        locator_minor = mdates.WeekdayLocator(byweekday=mdates.MO)
        formatter_minor = mdates.DateFormatter('%d')
    elif period_days < 366:
        locator_major = mdates.MonthLocator()
        formatter_major = mdates.DateFormatter('%b\n%Y')
        locator_minor = mdates.WeekdayLocator(byweekday=mdates.MO, interval=2)
        formatter_minor = None
    elif period_days < 366 * 2:
        locator_major = mdates.MonthLocator(interval=3)
        formatter_major = mdates.DateFormatter('%b\n%Y')
        locator_minor = mdates.MonthLocator(interval=1)
        formatter_minor = None
    elif period_days < 366 * 5:
        locator_major = mdates.MonthLocator(bymonth=[1,4,7,10])  
        def month_formatter(x, pos):
            dt = mdates.num2date(x)
            if dt.month == 1: return dt.strftime("%b\n%Y")
            else: return dt.strftime("%b")   
        formatter_major = FuncFormatter(month_formatter)
        locator_minor = None
        formatter_minor = None
    else:
        locator_major = mdates.YearLocator(base=2)      
        formatter_major = mdates.DateFormatter('%Y')
        locator_minor = mdates.YearLocator(base=1)
        formatter_minor = None

    with_minor = locator_minor is not None

    # Apply the locator and formatter to all axes
    for ax in axes:
        ax.xaxis.set_major_locator(locator_major)
        ax.xaxis.set_major_formatter(formatter_major)
        if with_minor:
            ax.xaxis.set_minor_locator(locator_minor)
            if formatter_minor is not None: ax.xaxis.set_minor_formatter(formatter_minor)
        pad = 12 if with_minor and (formatter_minor is not None) else 2
        ax.tick_params(axis='x', which='major', pad=pad, labelbottom=True)
        ax.tick_params(axis='x', which='minor', pad=2, labelbottom=True)
        ax.set_xlabel('')
        ax.set_ylabel('')
        if global_legend:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    
    # If global_legend is True, create a global legend
    if global_legend:
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if not isinstance(hh, matplotlib.artist.Artist): 
                    continue
                if ll not in labels:           
                    handles.append(hh)
                    labels.append(ll)
        if len(handles) > 1:
            legend_fig.legend(handles, labels,
                              **(legend_kwargs or dict(loc="upper right",
                                                       bbox_to_anchor=(1, 1))))

def plot_cutoff_results(
    target_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    start_offset: int,
    end_offset: int,
    *,
    cutoffs: Optional[Sequence[pd.Timestamp]] = None,
    cutoffs_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    levels: Optional[Sequence[int]] = None,
    models: Optional[Sequence[str]] = None,
    ids: Optional[Sequence[str]] = None,
    highlight_dayofweek: bool = False,
    order_of_models: Optional[Sequence[str]] = None,
    alpha: float = 0.9,
) -> plt.Figure:
    """
    Produce a multi-panel plot that overlays historical data with
    cross-validation forecasts for selected cut-off dates.
    """
    target_col = "y"  # Default target column
    if not isinstance(target_df, pd.DataFrame) or not isinstance(cv_df, pd.DataFrame):
        raise TypeError("Both target_df and cv_df must be pandas DataFrames.")
    if not all(col in target_df.columns for col in ['unique_id', 'ds', 'y']):
        raise ValueError(f"target_df must contain 'unique_id', 'ds', and 'y' columns.")
    if not all(col in cv_df.columns for col in ['unique_id', 'ds', 'cutoff']):
        raise ValueError(f"cv_df must contain 'unique_id', 'ds', and 'cutoff' columns.")

    if (cutoffs is None) == (cutoffs_period is None) == False:
        raise warnings.warn(
            "Both `cutoffs` and `cutoffs_period` are provided. "
            "Using `cutoffs` only, ignoring `cutoffs_period`."
        )

    levels = list(levels or [])

    # Model discovery & validation 
    all_models = [
        c for c in cv_df.columns
        if c not in {"ds", "unique_id", "y", "cutoff"}
        and "-lo-" not in c and "-hi-" not in c
    ]

    if models is None:
        models = all_models
    else:
        unknown = set(models) - set(all_models)
        if unknown:
            raise ValueError(f"Unknown model columns: {', '.join(unknown)}")

    if order_of_models:
        extra = [m for m in models if m not in order_of_models]
        ordered_models: List[str] = list(order_of_models) + extra
    else:
        ordered_models = list(models)

    # Choose cut-offs
    if cutoffs is not None:
        cutoffs = pd.to_datetime(cutoffs).tolist()
        missing = set(cutoffs) - set(cv_df["cutoff"].unique())
        if missing:
            raise ValueError(f"Cut-offs not found: {', '.join(map(str, missing))}")
        mask_cutoff = cv_df["cutoff"].isin(cutoffs)

    elif cutoffs_period is not None:  # by explicit period (inclusive)
        if not (
            isinstance(cutoffs_period, (tuple, list))
            and len(cutoffs_period) == 2
        ):
            raise TypeError(
                "`cutoffs_period` must be a tuple(start, end) of two timestamps."
            )
        start, end = map(pd.Timestamp, cutoffs_period)
        if start > end:
            raise ValueError("cutoffs_period: start date must be before end date.")
        mask_cutoff = (cv_df["cutoff"] >= start) & (cv_df["cutoff"] <= end)
    else:
        mask_cutoff = cv_df["cutoff"].notna()

    cut_df = cv_df.loc[mask_cutoff].copy()
    if cut_df.empty:
        raise ValueError("No cut-offs found in the requested period.")

    # ID filtering 
    all_ids = cut_df["unique_id"].unique()
    if ids is None:
        ids = all_ids
    else:
        unknown = set(ids) - set(all_ids)
        if unknown:
            raise ValueError(f"Unknown unique_id values: {', '.join(unknown)}")

    cut_df = cut_df[cut_df["unique_id"].isin(ids)]

    # Columns to keep
    keep_cols = ["ds", "unique_id", "cutoff"] + ordered_models
    for lv in levels:
        keep_cols += [f"{m}-lo-{lv}" for m in ordered_models]
        keep_cols += [f"{m}-hi-{lv}" for m in ordered_models]
    cut_df = cut_df[keep_cols]

    # Plotting limits 
    plot_start = cut_df["ds"].min() - pd.Timedelta(hours=start_offset)
    plot_end   = cut_df["ds"].max() + pd.Timedelta(hours=end_offset)

    # ---------- figure ----------
    n_series = len(ids)
    fig, axes = plt.subplots(n_series, 1, figsize=(12, 4 * n_series))
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    model_colors = sns.color_palette("husl", len(ordered_models))

    for ax, uid in zip(axes, ids):
        # historical line
        mask_train = (
            (target_df["unique_id"] == uid) &
            (target_df["ds"] >= plot_start) &
            (target_df["ds"] <= plot_end)
        )
        train_grp = target_df.loc[mask_train]
        ax.plot(train_grp["ds"], train_grp[target_col], color="black", label="Target")

        # weekday scatter
        if highlight_dayofweek:
            weekday_pal = (
                sns.color_palette("YlOrBr", 5) +
                sns.color_palette("YlGn", 2)
            )
            for i, day in enumerate(
                ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
            ):
                mask = train_grp["ds"].dt.day_name() == day
                ax.scatter(
                    train_grp.loc[mask, "ds"],
                    train_grp.loc[mask, target_col],
                    color=weekday_pal[i],
                    label=day,
                )

        # forecasts
        for cutoff_val in cut_df["cutoff"].unique():
            f_grp = cut_df[
                (cut_df["cutoff"] == cutoff_val) &
                (cut_df["unique_id"] == uid)
            ]
            for idx, model in enumerate(ordered_models):
                c = model_colors[idx]
                ax.plot(
                    f_grp["ds"], f_grp[model],
                    color=c, alpha=alpha,
                    label=f"{model}",
                )
                for lv in levels:
                    lo = f_grp[f"{model}-lo-{lv}"]
                    hi = f_grp[f"{model}-hi-{lv}"]
                    if not lo.empty and not hi.empty:
                        alpha_lv = 0.1 + (100 - lv) / 100 * 0.8  # narrower PI darker
                        ax.fill_between(
                            f_grp["ds"], lo, hi,
                            color=c, alpha=alpha_lv, linewidth=0,
                            label=f"{model} {lv}% PI",
                        )

        ax.set_title(f"Series {uid}")
        ax.legend().remove()  # global legend later

    # Configure time axes
    configure_time_axes(
        axes, period=[plot_start, plot_end],
        global_legend=True, legend_fig=fig,
    )

    fig.suptitle("Forecast Results", y=0.98, fontsize=16)
    fig.supxlabel('Date Time [H]')
    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.98])

    return fig

def plot_cutoff_results_with_exog(
    target_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    start_offset: int,
    end_offset: int,
    *,
    aux_df: Optional[pd.DataFrame] = None,
    exog_vars: Optional[Sequence[str]] = None,
    cutoffs: Optional[Sequence[pd.Timestamp]] = None,
    cutoffs_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    levels: Optional[Sequence[int]] = None,
    models: Optional[Sequence[str]] = None,
    id: Optional[str] = None,
    highlight_dayofweek: bool = False,
    order_of_models: Optional[Sequence[str]] = None,
    alpha: float = 0.9,
    add_context: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot forecast results with multiple cutoffs for a single time series,
    along with one panel per exogenous variable.

    This visualization is useful for evaluating model performance across
    cutoffs and understanding how exogenous variables relate to the target.

    Parameters
    ----------
    target_df : pd.DataFrame
        DataFrame containing the historical target series.
        Must include columns: 'unique_id', 'ds', 'y'.

    cv_df : pd.DataFrame
        Cross-validation results containing forecast values.
        Must include: 'unique_id', 'ds', 'cutoff', one or more model columns,
        and optionally prediction interval columns like 'model-lo-90', 'model-hi-90'.

    start_offset : int
        Hours to extend the plot before the first forecast timestamp.

    end_offset : int
        Hours to extend the plot after the last forecast timestamp.

    aux_df : Optional[pd.DataFrame], default=None
        DataFrame containing the exogenous variables.
        Must include 'unique_id', 'ds', and all variables in `exog_vars`.

    exog_vars : Optional[Sequence[str]], default=None
        List of exogenous variable names to plot in separate panels.

    cutoffs : Optional[Sequence[pd.Timestamp]], default=None
        Exact cutoffs to plot. Takes precedence over `cutoffs_period` if both are provided.

    cutoffs_period : Optional[Tuple[pd.Timestamp, pd.Timestamp]], default=None
        Inclusive period for selecting all cutoffs between `start` and `end`.

    levels : Optional[Sequence[int]], default=None
        List of prediction interval levels (e.g., [80, 95]). If None, no PIs are shown.

    models : Optional[Sequence[str]], default=None
        Subset of model columns in `cv_df` to plot. Defaults to all detected model columns.

    id : Optional[str], default=None
        `unique_id` of the series to plot. If None, the first available ID in `cv_df` is used.

    highlight_dayofweek : bool, default=False
        If True, scatter points in the target series are colored by weekday.

    order_of_models : Optional[Sequence[str]], default=None
        Specifies the order of models in the plot legend. Unlisted models are appended.

    alpha : float, default=0.9
        Opacity for forecast lines.

    add_context : bool, default=False
        If True, an additional panel is included showing the full historical context
        of the target series, and each subplot (including context) will have its own legend.
        If False, a single shared legend is shown above the plot.

    figsize : Optional[Tuple[int, int]], default=None
        Size of the figure to create, in inches. If None, defaults to (12, 4 * (number of panels)).

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib Figure with the main target series forecast panel,
        one panel per exogenous variable, and optionally a context panel.
    """
    target_col = "y"  # Default target column
    if not isinstance(target_df, pd.DataFrame) or not isinstance(cv_df, pd.DataFrame):
        raise TypeError("target_df and cv_df must be pandas DataFrames.")
    if aux_df is not None and not isinstance(aux_df, pd.DataFrame):
        raise TypeError("aux_df must be a pandas DataFrame if provided.")

    if (cutoffs is None) == (cutoffs_period is None):
        if cutoffs is None:
            raise ValueError(
                "At least one of `cutoffs` or `cutoffs_period` must be provided."
            )
        else: # both are provided
            warnings.warn(
                "Both `cutoffs` and `cutoffs_period` are provided. Using `cutoffs` only."
            )

    levels = list(levels or [])

    # Model discovery & validation
    all_models = [
        c for c in cv_df.columns
        if c not in {"ds", "unique_id", "y", "cutoff"}
        and "-lo-" not in c and "-hi-" not in c
    ]

    if models is None:
        models = all_models
    else:
        unknown = set(models) - set(all_models)
        if unknown:
            raise ValueError(f"Unknown model columns: {', '.join(unknown)}")

    if order_of_models:
        extra = [m for m in models if m not in order_of_models]
        ordered_models: List[str] = list(order_of_models) + extra
    else:
        ordered_models = list(models)

    # Choose cut-offs 
    if cutoffs is not None:
        cutoffs = pd.to_datetime(cutoffs).tolist()
        missing = set(cutoffs) - set(cv_df["cutoff"].unique())
        if missing:
            raise ValueError(f"Cut-offs not found: {', '.join(map(str, missing))}")
        mask_cutoff = cv_df["cutoff"].isin(cutoffs)

    else:  # by explicit period (inclusive)
        if not (
            isinstance(cutoffs_period, (tuple, list))
            and len(cutoffs_period) == 2
        ):
            raise TypeError(
                "`cutoffs_period` must be a tuple(start, end) of two timestamps."
            )
        start, end = map(pd.Timestamp, cutoffs_period)
        if start > end:
            raise ValueError("cutoffs_period: start date must be before end date.")
        mask_cutoff = (cv_df["cutoff"] >= start) & (cv_df["cutoff"] <= end)

    cut_df = cv_df.loc[mask_cutoff].copy()
    if cut_df.empty:
        raise ValueError("No cut-offs found in the requested period.")

    # ID filtering 
    all_ids = cut_df["unique_id"].unique()
    if len(all_ids) > 1 and id is None:
        raise ValueError("Multiple unique_id values found, please specify one with `id` parameter.")
    if id is not None:    
        unknown = set([id]) - set(all_ids)
        if unknown:
            raise ValueError(f"Unknown unique_id values: {', '.join(unknown)}")
    else:
        id = all_ids[0]

    cut_df = cut_df[cut_df["unique_id"]==id]
    target_id_df = target_df[target_df["unique_id"] == id]

    # Columns to keep 
    keep_cols = ["ds", "unique_id", "cutoff"] + ordered_models
    for lv in levels:
        keep_cols += [f"{m}-lo-{lv}" for m in ordered_models]
        keep_cols += [f"{m}-hi-{lv}" for m in ordered_models]
    cut_df = cut_df[keep_cols]

    # Plotting limits 
    plot_start = cut_df["ds"].min() - pd.Timedelta(hours=start_offset)
    plot_end   = cut_df["ds"].max() + pd.Timedelta(hours=end_offset)

    # ---------- figure ----------
    exog_vars = list(exog_vars or [])
    n_exog = len(exog_vars)
    figsize = figsize or (12, 4 * (1 + n_exog + int(add_context)))
    fig, axes = plt.subplots(1+n_exog+int(add_context), 1, figsize=figsize)
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    model_colors = sns.color_palette("husl", len(ordered_models))

    ax = axes[0]
    
    # Historical line
    mask_train = (
        (target_id_df["ds"] >= plot_start) &
        (target_id_df["ds"] <= plot_end)
    )
    train_grp = target_id_df.loc[mask_train]
    ax.plot(train_grp["ds"], train_grp[target_col], color="black", label="Target")

    # Weekday scatter
    if highlight_dayofweek:
        weekday_pal = (
            sns.color_palette("YlOrBr", 5) +
            sns.color_palette("YlGn", 2)
        )
        for i, day in enumerate(
            ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]
        ):
            mask = train_grp["ds"].dt.day_name() == day
            ax.scatter(
                train_grp.loc[mask, "ds"],
                train_grp.loc[mask, target_col],
                color=weekday_pal[i],
                label=day,
            )

    # Forecasts
    for cutoff_val in cut_df["cutoff"].unique():
        f_grp = cut_df[
            (cut_df["cutoff"] == cutoff_val) &
            (cut_df["unique_id"] == id)
        ]
        for idx, model in enumerate(ordered_models):
            c = model_colors[idx]
            ax.plot(
                f_grp["ds"], f_grp[model],
                color=c, alpha=alpha,
                label=f"{model}" if cutoff_val == cut_df['cutoff'].unique()[0] else None
            )
            for lv in levels:
                lo = f_grp[f"{model}-lo-{lv}"]
                hi = f_grp[f"{model}-hi-{lv}"]
                if not lo.empty and not hi.empty:
                    alpha_lv = 0.1 + (100 - lv) / 100 * 0.8  # narrower PI darker
                    ax.fill_between(
                        f_grp["ds"], lo, hi,
                        color=c, alpha=alpha_lv, linewidth=0,
                        label=f"{model} {lv}% PI" if cutoff_val == cut_df['cutoff'].unique()[0] else None,
                    )

    ax.set_title(f"Forecasts")
    if add_context: # in this case, the plot is clearer with per-axes legends
        ax.legend(loc="upper left")  # Per-axes legend
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    if aux_df is not None:
        # Plot the auxiliary variables
        missing_exog = set(exog_vars) - set(aux_df.columns)
        if missing_exog:
            raise ValueError(f"Missing exogenous variables in aux_df: {', '.join(missing_exog)}")
    
        units = {
            'temperature': '°C',
            'pressure': 'hPa',
            'dew_point': '°C',
            'humidity': '%',
            'wind_speed': 'm/s'
        }
        aux_grp = aux_df[(aux_df["unique_id"] == id) & 
                        (aux_df["ds"] >= plot_start) & 
                        (aux_df["ds"] <= plot_end)]
        for exog_idx, exog_var in enumerate(exog_vars):
            ax = axes[exog_idx + 1]
            ax.plot(aux_grp["ds"], aux_grp[exog_var], color="blue")
            ax.set_title(f"{exog_var} ({units.get(exog_var, '')})")

    if add_context:
        ax = axes[-1]
        ax.set_title("Full Context vs Current Range")

        ds_all_start = cv_df["ds"].min()
        ds_all_end = cv_df["ds"].max()

        ds_cut_start = cut_df["ds"].min()
        ds_cut_end = cut_df["ds"].max()

        # Full context range (gray)
        context_range = target_id_df[
            (target_id_df["ds"] >= ds_all_start) & (target_id_df["ds"] <= ds_all_end)
        ].iloc[::2]  # <- this skips 1 row out of 2 for performance
        ax.plot(
            context_range["ds"], context_range[target_col],
            color="gray", label="Full Context Range"
        )

        # Highlighted range (black)
        current_range = target_id_df[
            (target_id_df["ds"] >= ds_cut_start) & (target_id_df["ds"] <= ds_cut_end)
        ].iloc[::2]  # <- this skips 1 row out of 2 for performance
        ax.plot(
            current_range["ds"], current_range[target_col],
            color="black", label="Current Plot Range"
        )

        ax.legend(loc="upper left")  # Per-axes legend

    # Configure time axes
    configure_time_axes(
        axes, period=[plot_start, plot_end],
        global_legend=not add_context, legend_fig=fig,
    )

    if add_context:
        configure_time_axes(
            [axes[-1]], period=cv_df["ds"],
            global_legend=False,
        )

    fig.suptitle(f"Forecast Results for ID: {id}", y=0.98, fontsize=16)
    fig.supxlabel('Date Time [H]')
    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.98])

    return fig

def interactive_plot_cutoff_results(
    target_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    *,
    aux_df: Optional[pd.DataFrame] = None,
    exog_vars: Optional[Sequence[str]] = None,
    n_windows: int = 1,
    models: Optional[Sequence[str]] = None,
    id: Optional[str] = None,
    add_context: bool = False,
    levels: Optional[Sequence[int]] = None,
    alpha: float = 0.9,
    order_of_models: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    only_aligned_to_day: bool = True,
    use_slider: bool = True,
    center_cutoff_index: int = 0,
):
    """
    Create an interactive slider to visualize rolling forecast windows
    from cross-validation results.
    """
    from IPython.display import clear_output

    if only_aligned_to_day:
        cv_df = cv_df[cv_df['cutoff'].dt.hour == 23].copy()

    cutoffs = sorted(cv_df['cutoff'].unique())
    if len(cutoffs) < n_windows:
        raise ValueError("Not enough cutoffs to create the requested number of windows.")
    
    # Create the slider
    slider = widgets.IntSlider(
        value=len(cutoffs) // 2,
        min=0,
        max=len(cutoffs) - 1,
        step=1,
        description="Cutoff idx:",
        continuous_update=False
    )

    def interactive_plot(center_cutoff_index):
        clear_output(wait=True)

        half_window = (n_windows - 1) // 2 
        start = max(0, center_cutoff_index - half_window)
        end = min(len(cutoffs), center_cutoff_index + half_window + 1 + int(n_windows % 2 == 0))
        selected_cutoffs = cutoffs[start:end]

        mfig = plot_cutoff_results_with_exog(
            target_df=target_df,
            aux_df=aux_df,
            exog_vars=exog_vars,
            cv_df=cv_df,
            start_offset=48,
            end_offset=48,
            cutoffs=selected_cutoffs,
            models=models,
            id=id,
            add_context=add_context,
            levels=levels,
            alpha=alpha,
            order_of_models=order_of_models,  # Use the same models for ordering
            figsize=figsize
        )

        plt.show(mfig)
        #plt.close('all')    # Prevent Jupyter from showing it again
        
        return None

    if use_slider is False:
        # Just render the initial plot without interactivity
        interactive_plot(center_cutoff_index)
        return None
    # Hook up the slider
    widgets.interact(interactive_plot, center_cutoff_index=slider)

    return None

def display_scrollable(fig: go.Figure, container_width: str = "100%", max_height: Optional[int] = None):
    """
    Display a Plotly figure inside a horizontally scrollable container.
    You can call this any time, even after updating fig width/height.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to render.
    container_width : str, default "100%"
        CSS width of the outer container (e.g., "100%", "900px").
    max_height : int | None
        If provided, fixes the container's max height and enables vertical scrolling.
    """
    style = f"width:{container_width}; overflow-x:auto; border:0; padding:0; margin:0;"
    if max_height is not None:
        style += f" max-height:{max_height}px; overflow-y:auto;"
    html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    display(HTML(f'<div style="{style}">{html}</div>'))

def plotly_cutoffs_with_exog(
    target_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    start_offset: int,
    end_offset: int,
    *,
    aux_df: Optional[pd.DataFrame] = None,
    exog_vars: Optional[Sequence[str]] = None,
    levels: Optional[Sequence[int]] = None,
    models: Optional[Sequence[str]] = None,
    id: Optional[str] = None,
    highlight_dayofweek: bool = False,
    order_of_models: Optional[Sequence[str]] = None,
    alpha: float = 0.9,
    base_height_per_panel: int = 300,
    width_per_day: int = 60,
    grayscale_safe: bool = False,
    title: Optional[str] = "Forecast Results",
) -> go.Figure:
    """
    Interactive Plotly version that plots *all* available cutoffs (no filtering)
    and ensures consistent colors per model. No context subplot.
    """

    target_col = "y"
    if not isinstance(target_df, pd.DataFrame) or not isinstance(cv_df, pd.DataFrame):
        raise TypeError("target_df and cv_df must be pandas DataFrames.")
    if aux_df is not None and not isinstance(aux_df, pd.DataFrame):
        raise TypeError("aux_df must be a pandas DataFrame if provided.")

    levels = list(levels or [])
    exog_vars = list(exog_vars or [])

    # --- Model discovery
    all_models = [
        c for c in cv_df.columns
        if c not in {"ds", "unique_id", "y", "cutoff"}
        and "-lo-" not in c and "-hi-" not in c
    ]
    if models is None:
        models = all_models
    else:
        unknown = set(models) - set(all_models)
        if unknown:
            raise ValueError(f"Unknown model columns: {', '.join(unknown)}")

    # Order models
    if order_of_models:
        extra = [m for m in models if m not in order_of_models]
        ordered_models = list(order_of_models) + extra
    else:
        ordered_models = list(models)

    # --- Choose ID
    all_ids = cv_df["unique_id"].unique()
    if len(all_ids) > 1 and id is None:
        raise ValueError("Multiple unique_id values found, please specify one with `id`.")
    if id is not None and id not in set(all_ids):
        raise ValueError(f"Unknown unique_id: {id}")
    if id is None:
        id = all_ids[0]

    target_id_df = target_df[target_df["unique_id"] == id]
    if target_id_df.empty:
        raise ValueError(f"No target series found for unique_id={id}")

    cut_df = cv_df[cv_df["unique_id"] == id].copy()
    if cut_df.empty:
        raise ValueError("No rows in cv_df for the selected unique_id.")

    # --- Keep only required columns
    keep_cols = ["ds", "unique_id", "cutoff"] + ordered_models
    for lv in levels:
        keep_cols += [f"{m}-lo-{lv}" for m in ordered_models]
        keep_cols += [f"{m}-hi-{lv}" for m in ordered_models]
    keep_cols = [c for c in keep_cols if c in cut_df.columns]
    cut_df = cut_df[keep_cols]

    # --- Plot limits
    plot_start = cut_df["ds"].min() - pd.Timedelta(hours=start_offset)
    plot_end   = cut_df["ds"].max() + pd.Timedelta(hours=end_offset)

    # --- Create subplots (no context)
    n_exog = len(exog_vars)
    rows = 1 + n_exog
    specs = [[{}] for _ in range(rows)]
    row_titles = ["Forecasts"] + [f"{v}" for v in exog_vars]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=None
    )

    # --- Highlight weekdays in background
    if highlight_dayofweek:
        # pleasant, very light tints for the 7 weekdays
        dow_rgb = [
            (141, 211, 199),  # Mon
            (255, 255, 179),  # Tue
            (190, 186, 218),  # Wed
            (251, 128, 114),  # Thu
            (128, 177, 211),  # Fri
            (253, 180, 98),   # Sat
            (179, 222, 105),  # Sun
        ]
        def rgba(rgb, a):
            r, g, b = rgb
            return f"rgba({r},{g},{b},{a})"

        # Day boundaries spanning full plot range
        day0 = pd.to_datetime(plot_start.floor("D"))
        dayN = pd.to_datetime(plot_end.ceil("D"))
        days = pd.date_range(day0, dayN, freq="D")

        # Add one rectangle per day across ALL rows using yref='paper'
        # so it covers the full vertical canvas of the subplots.
        for d_start in days:
            d_end = d_start + pd.Timedelta(days=1)
            # mid point for annotation
            x_mid = d_start + (d_end - d_start) / 2
            wday = int(d_start.weekday())  # Monday=0 .. Sunday=6
            fill = rgba(dow_rgb[wday], 0.08)  # very subtle
            if not grayscale_safe:
                fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=d_start, x1=d_end,
                    y0=0, y1=1,
                    line=dict(width=0),
                    fillcolor=fill,
                    layer="below",
                )
            # Label at the very top (above traces and grid)
            fig.add_annotation(
                x=x_mid,
                y=1,
                xref="x",
                yref="paper",
                yshift=16,
                text=d_start.strftime("%a"),
                showarrow=False,
                font=dict(size=10),
                align="center"
            )

    # --- Target historical (main panel)
    mask_train = (target_id_df["ds"] >= plot_start) & (target_id_df["ds"] <= plot_end)
    train_grp = target_id_df.loc[mask_train, ["ds", target_col]]

    fig.add_trace(
        go.Scatter(
            x=train_grp["ds"], y=train_grp[target_col],
            name="Target", mode="lines",
            line=dict(width=2, color="black"),
            legendgroup="__target__", showlegend=True
        ),
        row=1, col=1
    )

    # --- Fixed color mapping per model
    if grayscale_safe:
        palette = ["#111111", "#B3B2B2"]
        ordered_models = ['placeholder'] + ordered_models  # shift by 1
    else:
        palette = px.colors.qualitative.Set2
    step = max(1, len(palette) // max(1, len(ordered_models)))
    color_map = {
        m: palette[(i * step) % len(palette)]
        for i, m in enumerate(ordered_models)
    }


    # --- Forecasts (all cutoffs, consistent colors)
    unique_cutoffs = pd.to_datetime(cut_df["cutoff"].unique())
    first_cutoff = unique_cutoffs.min() if len(unique_cutoffs) else None

    for cutoff_val in sorted(unique_cutoffs):
        f_grp = cut_df[cut_df["cutoff"] == cutoff_val]
        for model in ordered_models:
            if model not in f_grp.columns:
                continue
            c = color_map[model]
            fig.add_trace(
                go.Scatter(
                    x=f_grp["ds"], y=f_grp[model],
                    name=model,
                    mode="lines",
                    line=dict(width=2, color=c),
                    opacity=alpha,
                    legendgroup=model,
                    showlegend=(cutoff_val == first_cutoff)
                ),
                row=1, col=1
            )
            # Prediction intervals
            for lv in levels:
                lo_col = f"{model}-lo-{lv}"
                hi_col = f"{model}-hi-{lv}"
                if lo_col in f_grp.columns and hi_col in f_grp.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=f_grp["ds"], y=f_grp[lo_col],
                            mode="lines",
                            line=dict(width=0, color=c),
                            hoverinfo="skip",
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=f_grp["ds"], y=f_grp[hi_col],
                            mode="lines",
                            line=dict(width=0, color=c),
                            fill="tonexty",
                            opacity=0.2,
                            name=f"{model} {lv}% PI",
                            hoverinfo="skip",
                            legendgroup=f"{model}-pi-{lv}",
                            showlegend=(cutoff_val == first_cutoff)
                        ),
                        row=1, col=1
                    )

    # --- Exogenous panels
    exog_vars_to_name = {
        'temperature': 'Temperature (°C)',
        'humidity': 'Humidity (%)',
        'wind_speed': 'Wind Speed (m/s)',
    }
    line_styles = ["solid", "dot", "dash", "longdash", "dashdot"]
    if aux_df is not None and n_exog > 0:
        missing_exog = set(exog_vars) - set(aux_df.columns)
        if missing_exog:
            raise ValueError(f"Missing exogenous vars: {', '.join(missing_exog)}")
        aux_grp = aux_df[
            (aux_df["unique_id"] == id) &
            (aux_df["ds"] >= plot_start) &
            (aux_df["ds"] <= plot_end)
        ].copy()
        for i, exog in enumerate(exog_vars, start=1):
            fig.add_trace(
                go.Scatter(
                    x=aux_grp["ds"], y=aux_grp[exog],
                    name=exog.capitalize(),
                    mode="lines",
                    showlegend=grayscale_safe,
                    line=dict(
                        color="gray" if grayscale_safe else "blue",
                        dash=line_styles[i % len(line_styles)]
                    ),
                ),
                row=1+i, col=1
            )
            fig.update_yaxes(title_text=exog_vars_to_name.get(exog, ""), row=1+i, col=1)

    # --- Axes & layout
    total_hours = max(1, int((plot_end - plot_start) / pd.Timedelta(hours=1)))
    total_days = total_hours / 24.0
    fig_width = int(min(40000, max(1, total_days * width_per_day)))

    for r in range(1, rows+1):
        fig.update_xaxes(showgrid=True, tickformat=r"%Y-%m-%d", row=r, col=1)
        fig.update_yaxes(showgrid=True, zeroline=False, row=r, col=1)

    fig.update_layout(
        height=base_height_per_panel * rows,
        width=fig_width,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=30, t=20, b=40)
    )
    if title is not None:
        fig.update_layout(        
            title=dict(
                text=title,
                x=0.5,               # center horizontally
                xanchor="center",
                y=0.98,              # slightly higher placement
                yanchor="top",
                font=dict(size=18)   # bigger, cleaner title
            ),
            margin=dict(t=80)  # extra top margin for title
        )
    fig.update_xaxes(title_text="Date Time", row=rows, col=1)
    fig.update_yaxes(title_text="Heat Demand (kW/h)", row=1, col=1)
    if grayscale_safe:
        fig.update_layout(template="plotly_white")
        fig.update_xaxes(showgrid=True, gridcolor="rgb(235,235,235)")
        fig.update_yaxes(showgrid=True, gridcolor="rgb(235,235,235)")
    return fig

def custom_plot_results(
    target_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    start_offset: int,
    end_offset: int,
    *,
    levels: list = [],
    target_col: str = 'y',
    ids: Optional[list] = None,
    highlight_dayofweek: bool = False,
    with_naive: bool = False, # Whether to include the naive model in the plot
    target_train_df: Optional[pd.DataFrame] = None,  # Training data for the naive model
    order_of_models: Optional[List[str]] = None,  # Order of models to plot: the included models will be plotted as first and in this order
    alpha: float = 0.9,  # Transparency of the forecast lines
    return_fig: bool = False
) -> plt.Figure:
    """
    Plot historical data and model forecasts for a set of series.
    """

    if not isinstance(target_df, pd.DataFrame) or not isinstance(forecast_df, pd.DataFrame):
        raise TypeError("target_df and forecast_df must be pandas DataFrames.")
    if not all(col in target_df.columns for col in ['unique_id', 'ds', target_col]):
        raise ValueError(f"target_df must contain 'unique_id', 'ds', and '{target_col}' columns.")
    if not all(col in forecast_df.columns for col in ['unique_id', 'ds']):
        raise ValueError("forecast_df must contain 'unique_id' and 'ds' columns.")
    if not isinstance(levels, list) or not all(isinstance(lv, int) for lv in levels):
        raise TypeError("levels must be a list of integers.")
    
    # Validate ids
    if ids is None or len(ids) == 0:
        ids = target_df['unique_id'].unique().tolist()
    else:
        if not isinstance(ids, Iterable) or not all(isinstance(i, str) for i in ids):
            raise TypeError("ids must be an Iterable of strings.")

        unknown = set(ids) - set(target_df['unique_id'].unique())
        if unknown:
            raise ValueError(f"Unknown unique_id values: {', '.join(unknown)}")

    if with_naive:
        h = len(forecast_df['ds'].unique())  # Number of hours to forecast
        
        if target_train_df is None:
            raise ValueError("target_train_df must be provided when with_naive is True.")
        target_train_df = target_train_df.copy()
        
        # Select only the ids we want to plot
        target_train_df = target_train_df[target_train_df['unique_id'].isin(ids)].reset_index(drop=True)
        
        # Compute forecasts using the naive method
        naive_model24 = SeasonalNaive(season_length=24, alias='Naive24h')
        naive_forecast_df = StatsForecast(
            models=[naive_model24], 
            freq='h'
        ).forecast(h, target_train_df, level=levels)

        # Merge the forecasts into a single DataFrame
        forecast_df = (
            forecast_df
            .merge(naive_forecast_df, on=['unique_id', 'ds'], how='left')
        )
        
    # set the frequency of the validation index (e.g. 'H', 'D', …)
    freq = 'h'   

    # Create a date range for the x-axis
    td_start = pd.Timedelta(start_offset, unit=freq)
    td_end   = pd.Timedelta(end_offset,   unit=freq)
    start_dstemp = forecast_df['ds'].min() - td_start
    end_dstemp   = forecast_df['ds'].max() + td_end
    
    fig, axes = plt.subplots(nrows=len(ids), ncols=1, figsize=(12, 4*len(ids)))
    axes = axes.flatten() if len(ids) > 1 else [axes]
    for ax, uid in zip(axes, ids):
        # Plot training data
        mask_target = (
            (target_df['ds'] >= start_dstemp) &
            (target_df['ds'] <= end_dstemp) &
            (target_df['unique_id'] == uid)
        )
        target_grp = target_df[mask_target]
        ax.plot(target_grp['ds'], target_grp[target_col], label='Target', color='black')

        if highlight_dayofweek:
            pallette_wkdays = sns.color_palette(palette='YlOrBr', n_colors=5)  # Use a color palette for weekdays
            pallette_weekends = sns.color_palette(palette='YlGn', n_colors=2)  # Use a color palette for weekends
            days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            pal = pallette_wkdays + pallette_weekends  # Combine palettes for weekdays and weekends
            for i, day in enumerate(days):
                mask = target_grp['ds'].dt.day_name() == day
                ax.scatter(
                    target_grp.loc[mask, 'ds'], 
                    target_grp.loc[mask, target_col], 
                    color=pal[i], 
                    label=f'{day}', 
                )
        
        # Plot forecasted data
        models = [c for c in forecast_df.columns
                  if c != 'ds' and c != 'unique_id' and '-lo-' not in c and '-hi-' not in c]
        if order_of_models is not None:
            if not set(order_of_models).issubset(set(models)):
                raise ValueError("order_of_models must be a subset of the models in forecast_df.")
            models = order_of_models + [m for m in models if m not in order_of_models]
        if len(models) == 1:
            colors = ['blue']
        else:
            colors = sns.color_palette("tab10", len(models))  # Use a color palette for models
        forecast_grp = forecast_df.query("unique_id == @uid")
        for i, model in enumerate(models):
            ax.plot(forecast_grp['ds'], forecast_grp[model], label=model, color=colors[i], alpha=alpha)
            for lv in levels:
                low_col = f'{model}-lo-{lv}'
                high_col = f'{model}-hi-{lv}'
                if low_col in forecast_grp and high_col in forecast_grp:
                    min_alpha = 0.1  # fix alpha to avoid transparency issues
                    max_alpha = 0.9
                    alpha_lvl = max_alpha - (float(lv) / 100) * (max_alpha - min_alpha)
                    ax.fill_between(
                        forecast_grp['ds'], 
                        forecast_grp[f'{model}-lo-{lv}'], 
                        forecast_grp[f'{model}-hi-{lv}'], 
                        color=colors[i], alpha=alpha_lvl, label=f'{lv}% PI'
                    )
        ax.set_title(f'Series {uid}')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
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
               bbox_to_anchor=(1, 1))

    fig.suptitle('Forecast Results', y=0.98, fontsize=16)       # push it up a bit
    fig.tight_layout(rect=[0, 0, 1, 0.97])                      # leave the top 3% for the suptitle

    if return_fig:
        return fig

def custom_plot_exog(
    aux_df: pd.DataFrame,
    forecast_ds: pd.Series,
    start_offset: int,
    end_offset: int,
    exog_list: List[str] = ['dew_point', 'pressure', 'temperature', 'humidity', 'wind_speed'],
)-> plt.Figure:
    """
    Plot a window of exogenous variables surrounding a forecast horizon.
    """
    # Create a date range for the x-axis
    freq='h'
    td_start = pd.Timedelta(start_offset, unit=freq)
    td_end   = pd.Timedelta(end_offset,   unit=freq)
    start_dstemp = forecast_ds.min() - td_start
    end_dstemp   = forecast_ds.max() + td_end

    # Define units of measurement for each exogenous variable
    exog_units = {
        'dew_point': '°C',
        'pressure': 'hPa',
        'temperature': '°C',
        'humidity': '%',
        'wind_speed': 'm/s'
    }
    fig, axes = plt.subplots(nrows=len(exog_list), ncols=1, figsize=(12, 4*len(exog_list)))
    axes = axes.flatten() if len(exog_list) > 1 else [axes]
    aux_palette = sns.color_palette("tab10", aux_df['unique_id'].nunique())  # Use a color palette for unique_ids
    for ax, exog in zip(axes, exog_list):
        # filter & pivot 
        mask = aux_df['ds'].between(start_dstemp, end_dstemp)
        pivot_df = aux_df.loc[mask, ['ds', 'unique_id', exog]] \
                        .pivot(index='ds', columns='unique_id', values=exog)
        # I avoided using seborn to control more easily the legend
        for i, uid in enumerate(pivot_df.columns):
            ax.plot(
                pivot_df.index, 
                pivot_df[uid], 
                label=uid, 
                alpha=0.7, 
                color=aux_palette[i] 
            )
            ax.set_title(f'{exog} ({exog_units[exog]})')
    
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
               bbox_to_anchor=(1, 1))

    fig.suptitle('Exogenous variables', y=0.98, fontsize=16)    # push it up a bit
    fig.tight_layout(rect=[0, 0, 1, 0.97])                      # leave the top 3% for the suptitle
    return fig

def plot_daily_seasonality(
    target_df: pd.DataFrame,
    *,
    only_cold_months: bool = False,
    make_is_winter: Callable[[str], Callable[[pd.Timestamp], bool]] = make_is_winter,
    transform: str = "none",
    lambda_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ids: Optional[Iterable[str]] = None,
    plot_range: Optional[pd.DatetimeIndex] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Plot daily seasonality of heat demand after an optional transformation.

    Returns
    -------
    dict {unique_id: lambda}.  Empty if no Box-Cox applied.
    """
    # Convert ids argument to list and validate
    available_ids = set(target_df["unique_id"].unique())
    ids = list(ids) if ids is not None else sorted(available_ids)
    missing = set(ids) - available_ids
    if missing:
        raise ValueError(
            f"The following ids are not in target_df: {sorted(missing)}. "
            f"Available ids: {sorted(available_ids)}"
        )
    target_df = target_df[target_df["unique_id"].isin(ids)].copy()

    # Estimate Box-Cox lambdas if requested
    lambdas: Dict[str, float] = {}
    if transform.startswith("boxcox"):
        if lambda_window is None:
            raise ValueError("lambda_window must be provided for Box-Cox transforms.")
        if transform == "boxcox_winter" and not callable(make_is_winter):
            raise TypeError("make_is_winter must be a callable function.")
        start, end = lambda_window
        lambda_df = target_df[(target_df["ds"] >= start) & (target_df["ds"] <= end)]
        if lambda_df.empty: 
            raise ValueError("lambda_window contains no data.")
        lambdas = get_lambdas(
            df=lambda_df,
            method="loglik",
            winter_focus=(transform == "boxcox_winter"),
            make_is_winter=make_is_winter
        )
        transform_name = "boxcox"
        if verbose:
            print("Estimated Box-Cox λ:")
            display(pd.DataFrame(lambdas.items(), columns=["unique_id", "lambda"]))
    else:
        transform_name = transform

    # Apply transform and restrict to plot range
    if plot_range is None: plot_range = target_df["ds"].sort_values().unique()
    plot_df = target_df[target_df["ds"].isin(plot_range)].copy()
    fwd = make_transformer(transform_name, "y", lambdas or None, inv=False)
    plot_df["y_transformed"] = transform_column(plot_df, fwd)

    # Keep only winter rows
    if only_cold_months:
        if not callable(make_is_winter):
            raise TypeError("make_is_winter must be a callable function.")
        is_winter_fn = {uid: make_is_winter(uid) for uid in plot_df["unique_id"].unique()}
        winter_mask = (
            plot_df.groupby("unique_id")["ds"]
            .transform(lambda s: s.map(is_winter_fn[s.name]))    # s.name is the uid
        )
        plot_df = plot_df[winter_mask]

    # Prepare month palette with fixed mapping
    months_sorted = np.arange(1, 13)
    month_palette = dict(zip(months_sorted, sns.color_palette("crest", 12)))
    plot_df["month"] = plot_df["ds"].dt.month
    plot_df["hour"] = plot_df["ds"].dt.hour

    # Create one subplot per id
    n = len(ids)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(9, 5 * n), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, (uid, grp) in zip(axes, plot_df.groupby("unique_id")):
        sns.lineplot(
            data=grp,
            x="hour",
            y="y_transformed",
            hue="month",
            palette=month_palette,
            legend=False,
            ax=ax,
            errorbar=("pi", 80),
        )

        # Annotate last point of each month
        for m, subset in grp.groupby("month"):
            ax.text(
                subset['hour'].iloc[-1],
                subset['y_transformed'].iloc[-1],
                str(m),
                fontsize=15,
                color=month_palette[m],  
                ha="left",
                alpha=0.9,
            )

        ax.set_title(uid)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticks(np.arange(0, 24, 1))

    # Shared labels and formatting
    fig.suptitle('Heat Demand for each month in the given period', fontsize=18)
    fig.supxlabel('Hour of Day', fontsize=14)
    fig.supylabel('Heat Demand [kWh]', fontsize=14)
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])

    return lambdas

def plot_weekly_seasonality(
    target_df: pd.DataFrame,
    *,
    only_cold_months: bool = True,
    make_is_winter: Callable[[str], Callable[[pd.Timestamp], bool]] = make_is_winter,
    transform: str = "none",
    lambda_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ids: Optional[Iterable[str]] = None,
    plot_range: Optional[pd.DatetimeIndex] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Plot the average winter-week load profile (hour-of-week 0-167) for one
    or more unique_id s, colour-coded by calendar month.
    """
    # Validate ids
    available_ids = set(target_df["unique_id"].unique())
    ids = list(ids) if ids is not None else sorted(available_ids)
    missing = set(ids) - available_ids
    if missing:
        raise ValueError(
            f"The following ids are not in target_df: {sorted(missing)}. "
            f"Available ids: {sorted(available_ids)}"
        )
    target_df = target_df[target_df["unique_id"].isin(ids)].copy()

    # Estimate Box-Cox lambdas if requested
    lambdas: Dict[str, float] = {}
    if transform.startswith("boxcox"):
        if lambda_window is None:
            raise ValueError("lambda_window must be provided for Box-Cox transforms.")
        if transform == "boxcox_winter" and not callable(make_is_winter):
            raise TypeError("make_is_winter must be a callable function.")
        start, end = lambda_window
        lambda_df = target_df[(target_df["ds"] >= start) & (target_df["ds"] <= end)]
        if lambda_df.empty: 
            raise ValueError("lambda_window contains no data.")
        lambdas = get_lambdas(
            df=lambda_df,
            method="loglik",
            winter_focus=(transform == "boxcox_winter"),
            make_is_winter=make_is_winter
        )
        transform_name = "boxcox"
        if verbose:
            print("Estimated Box-Cox λ:")
            display(pd.DataFrame(lambdas.items(), columns=["unique_id", "lambda"]))
    else:
        transform_name = transform

    # Apply transform and restrict to plot range
    if plot_range is None: plot_range = target_df["ds"].sort_values().unique()
    plot_df = target_df[target_df["ds"].isin(plot_range)].copy()
    fwd = make_transformer(transform_name, "y", lambdas or None, inv=False)
    plot_df["y_transformed"] = transform_column(plot_df, fwd)

    # Keep only winter rows
    if only_cold_months:
        if not callable(make_is_winter):
            raise TypeError("make_is_winter must be a callable function.")
        is_winter_fn = {uid: make_is_winter(uid) for uid in plot_df["unique_id"].unique()}
        winter_mask = (
            plot_df.groupby("unique_id")["ds"]
            .transform(lambda s: s.map(is_winter_fn[s.name]))    # s.name is the uid
        )
        plot_df = plot_df[winter_mask]

    # Hour-of-week + month columns
    plot_df["hour_of_week"] = (
        plot_df["ds"].dt.dayofweek * 24 + plot_df["ds"].dt.hour
    )
    base_date = pd.Timestamp('2023-01-02')  # Monday
    plot_df["hour_of_week_datetime"] = base_date + pd.to_timedelta(plot_df["hour_of_week"], unit='h')
    plot_df["month"] = plot_df["ds"].dt.month

    # Fixed month palette
    month_palette = dict(zip(range(1, 13), sns.color_palette("crest", 12)))

    # Plot
    n_ids = len(ids)
    fig, axes = plt.subplots(nrows=n_ids, ncols=1, figsize=(11, 3 * n_ids), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, (uid, grp) in zip(axes, plot_df.groupby("unique_id")):
        sns.lineplot(
            data=grp,
            x="hour_of_week",
            y="y_transformed",
            hue="month",
            palette=month_palette,
            legend=False,
            ax=ax,
            estimator="mean",
            errorbar=("pi", 80),
        )

        # Inline month labels at the last point (hour 167)
        for m, sub in grp.groupby("month"):
            last = sub.loc[sub["hour_of_week"].idxmax()]
            ax.text(
                last["hour_of_week"],
                last["y_transformed"],
                str(m),
                ha="left",
                fontsize=15,
                alpha=0.9,
                color=month_palette[m],
            )

        ax.set_title(uid)
        ax.set_ylabel("")
        ax.set_xlabel("")
        day_ticks = np.arange(0, 168 + 24, 24)
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", ""]
        ax.set_xticks(day_ticks)
        ax.set_xticklabels(day_labels)

        # set minor ticks every 2 hours
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.tick_params(axis="x", which="minor", length=4, width=1)

    # Shared labels and formatting
    fig.suptitle('Heat Demand for each month in the given period', fontsize=18)
    fig.supxlabel('Day of Week', fontsize=14)
    fig.supylabel('Heat Demand [kWh]', fontsize=14)
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])

    return lambdas

def scatter_temp_vs_target_hourly(
    target_df: pd.DataFrame,
    aux_df: pd.DataFrame,
    date_range: pd.DatetimeIndex,
    transform: str = "none",
    is_winter: Optional[Callable[[pd.Timestamp], bool]] = None,
    one_only: Optional[str] = None,  # 'Winter', 'Non-Winter', or None for both
    alphas: dict[str, float] = {"Winter": 0.6, "Non-Winter": 0.6},            
    id: Optional[str] = None,
    interactive: bool = True,
) -> pd.DataFrame:
    """
    Plot transformed heat demand against temperature, bucketed **by hour of day**.
    """
    if id is not None:
        if not isinstance(id, str):
            raise TypeError("id must be a string representing the unique_id.")
        if id not in target_df["unique_id"].unique():
            raise ValueError(f"Provided id {id} is not present in target_df.")
        if id not in aux_df["unique_id"].unique():
            raise ValueError(f"Provided id {id} is not present in aux_df.")
        target_df = target_df[target_df["unique_id"]==id].copy()
        aux_df  = aux_df [aux_df ["unique_id"]==id].copy()
    else:
        ids = target_df["unique_id"].unique().tolist()
        ids_aux = aux_df["unique_id"].unique().tolist()
        if len(ids) != 1 or len(ids_aux) != 1:
            raise ValueError("No id provided and target_df or aux_df have multiple unique_ids.")
        if set(ids)!=set(ids_aux):
            raise ValueError("target_df and aux_df have different unique_ids.")
        id = ids[0]  # use the single unique_id from both DataFrames
    if id not in ['F1', 'F2', 'F3', 'F4', 'F5']:
        raise ValueError(f"Invalid id: {id}. Expected one of ['F1', 'F2', 'F3', 'F4', 'F5'].")

    heat_train = target_df.loc[target_df["ds"].isin(date_range)].copy()
    aux_train  = aux_df .loc[aux_df ["ds"].isin(date_range)].copy()

    if is_winter is None:
        is_winter = make_is_winter(id)
    
    if transform not in ["none", "log", "boxcox", "boxcox_winter", "arcsinh", "arcsinh2", "arcsinh10"]:
        raise ValueError(
            f"Invalid transform: {transform}. "
            "Valid options are 'none', 'log', 'boxcox', 'boxcox_winter', "
            "'arcsinh', 'arcsinh2', or 'arcsinh10'."
        )

    if transform.startswith("boxcox"):
        lambdas = get_lambdas(
            heat_train_df=heat_train,
            method="loglik",
            winter_focus=(transform == "boxcox_winter"),
            is_winter=is_winter,
        )
        TRANSFORM = "boxcox"
    else:
        TRANSFORM = transform
        lambdas = None
    fwd = make_transformer(TRANSFORM, "y", lambdas, inv=False)
    heat_train["y_transformed"] = transform_column(heat_train, fwd)
    heat_train.drop(columns=["y"], inplace=True)  # drop the old target column

    heat_train["is_winter"] = heat_train["ds"].apply(is_winter)
    aux_train ["is_winter"] = aux_train ["ds"].apply(is_winter)
    heat_train["hour"] = heat_train["ds"].dt.hour
    aux_train ["hour"] = aux_train ["ds"].dt.hour

    buckets_heat: Mapping[str, pd.Series] = {}
    buckets_aux : Mapping[str, pd.Series] = {}
    buckets_alphas: Mapping[str, float] = {}

    hours = range(24)
    if one_only=='Winter':
        for h in hours:   # 48 buckets
            key = f"Winter/H{h:02d}"
            mask = (heat_train["is_winter"] == True) & (heat_train["hour"] == h)
            buckets_heat[key] = mask
            buckets_aux [key] = (aux_train["is_winter"] == True) & (aux_train ["hour"] == h)
            buckets_alphas[key] = alphas.get('Winter', 0.6)  # default alpha if not specified
    elif one_only=='Non-Winter':
        for h in hours:
            key = f"Non-Winter/H{h:02d}"
            mask = (heat_train["is_winter"] == False) & (heat_train["hour"] == h)
            buckets_heat[key] = mask
            buckets_aux [key] = (aux_train["is_winter"] == False) & (aux_train ["hour"] == h)
            buckets_alphas[key] = alphas.get('Non-Winter', 0.6)
    elif one_only is None:
        for h in hours:   # 48 buckets
            for season, flag in [("Winter", True), ("Non-Winter", False)]:
                key = f"{season}/H{h:02d}"
                mask = (heat_train["is_winter"] == flag) & (heat_train["hour"] == h)
                buckets_heat[key] = mask
                buckets_aux [key] = (aux_train["is_winter"] == flag) & (aux_train ["hour"] == h)
                buckets_alphas[key] = alphas.get(season, 0.6)  # default alpha if not specified
    else:
        raise ValueError("one_only must be 'Winter', 'Non-Winter', or None.")

    cmap = plt.colormaps.get_cmap("hsv")  # cyclical palette
    hour_colours = {h: cmap(h / 24) for h in hours}  # normalize to [0,1] range

    def colour_for(key: str) -> tuple:
        # extract hour substring and map to colour
        h = int(key.split("H")[1])
        return hour_colours[h]

    if interactive: 
        fig = go.Figure()
        correlations: dict[str, float] = {}

        for key in buckets_heat.keys():
            x = aux_train .loc[buckets_aux [key], "temperature"]
            y = heat_train.loc[buckets_heat[key], "y_transformed"]
            alpha = buckets_alphas[key]
            ds = heat_train.loc[buckets_heat[key], "ds"]

            # skip empty buckets
            if x.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    name=key,
                    marker=dict(
                        size=6,
                        color=colors.to_hex(colour_for(key)),         
                        opacity=alpha
                    ),
                    # Pass ds via customdata so we can reference it in hovertemplate
                    customdata=ds.dt.strftime("%Y-%m-%d %H:%M"),
                    hovertemplate=(
                        "Temperature: %{x:.2f}<br>"
                        "Heat (transf.): %{y:.2f}<br>"
                        "ds: %{customdata}<extra></extra>"
                    ),
                )
            )

            correlations[key] = np.corrcoef(x, y)[0, 1]

            if len(x) > 1:
                β1, β0 = np.polyfit(x, y, 1)
                x_grid = np.linspace(x.min(), x.max(), 50)
                y_grid = β1 * x_grid + β0
                fig.add_trace(
                    go.Scatter(
                        x=x_grid,
                        y=y_grid,
                        mode="lines",
                        line=dict(
                            color=colors.to_hex(colour_for(key)),
                            width=2
                        ),
                        opacity=alpha,
                        showlegend=False                    # don’t add a second item to legend
                    )
                )

        fig.update_layout(
            title="Heat Demand vs Temperature — hourly buckets",
            xaxis_title="Temperature",
            yaxis_title="Transformed Heat Demand",
            legend_title="Season / Hour",
            template="plotly_white",
            width=900,
            height=600,
        )

        fig.show() 
        
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        correlations: dict[str, float] = {}

        for key in buckets_heat.keys():
            x = aux_train.loc[buckets_aux[key], "temperature"]
            y = heat_train.loc[buckets_heat[key], "y_transformed"]
            alpha = buckets_alphas[key]
            ds = heat_train.loc[buckets_heat[key], "ds"]

            if x.empty:
                continue

            color = colors.to_hex(colour_for(key))

            # --- Correlation ---
            correlations[key] = np.corrcoef(x, y)[0, 1]

            # --- Regression line ---
            if len(x) > 1:
                β1, β0 = np.polyfit(x, y, 1)
                x_grid = np.linspace(x.min(), x.max(), 50)
                y_grid = β1 * x_grid + β0
                ax.plot(
                    x_grid,
                    y_grid,
                    color=color,
                    linewidth=2,
                    alpha=alpha
                )

            # --- Scatter points ---
            ax.scatter(
                x,
                y,
                label=key,
                color=color,
                alpha=alpha,
                s=30,
                linewidth=0.3
            )

        # --- Aesthetics ---
        ax.set_title("Heat Demand vs Temperature — Hourly Buckets")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Transformed Heat Demand")

        # Get all legend entries
        handles, labels = ax.get_legend_handles_labels()

        # Select one each two
        selected_handles = handles[::2]
        selected_labels = labels[::2]

        # Add filtered legend
        ax.legend(
            selected_handles,
            selected_labels,
            title="Season / Hour",
            fontsize=10,
            title_fontsize=11,
            loc='upper right'
        )

        plt.tight_layout()
        plt.show()

    corr_df = pd.DataFrame(correlations, index=[0]).T.rename(columns={0: "Correlation"})
    corr_df = corr_df.reset_index().rename(columns={'index': 'Key'})
    corr_df['Season'] = corr_df['Key'].str.split('/').str[0]
    corr_df['Hour'] = corr_df['Key'].str.extract(r'H(\d+)').astype(int)
    pivot_corr = corr_df.pivot(index='Season', columns='Hour', values='Correlation')

    return pivot_corr

def add_season_background(fig, start, end):
    season_colors = {
        "Spring": "rgba(144, 238, 144, 0.15)",
        "Summer": "rgba(255, 215, 0, 0.10)",
        "Autumn": "rgba(255, 165, 0, 0.15)",
        "Winter": "rgba(135, 206, 235, 0.15)"
    }

    years = range(start.year, end.year + 1)
    drawn_boundaries = set()

    for year in years:
        seasons = {
            "Spring": (datetime(year, 3, 21), datetime(year, 5, 20, 23, 59)),
            "Summer": (datetime(year, 5, 21), datetime(year, 9, 20, 23, 59)),
            "Autumn": (datetime(year, 9, 21), datetime(year, 12, 20, 23, 59)),
            "Winter": (datetime(year, 12, 21), datetime(year + 1, 3, 20, 23, 59))
        }

        for season, (s, e) in seasons.items():
            s_clipped = max(s, start)
            e_clipped = min(e, end)
            if s_clipped < end and e_clipped > start:
                # background
                fig.add_shape(
                    type="rect",
                    x0=s_clipped, x1=e_clipped,
                    y0=0, y1=1,
                    yref="paper",
                    fillcolor=season_colors[season],
                    line=dict(width=0),
                    layer="below"  # stays below everything
                )
                # season label
                fig.add_annotation(
                    x=s_clipped + (e_clipped - s_clipped)/2,
                    y=1.02,
                    xref="x", yref="paper",
                    text=season,
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    yanchor="bottom"
                )
        for season, (s, e) in seasons.items():
            # dashed separator line
            if start < s <= end and s not in drawn_boundaries:
                fig.add_shape(
                    type="line",
                    x0=s, x1=s, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color="black", width=1, dash="dash"),
                )
                drawn_boundaries.add(s)

    return sorted(drawn_boundaries)


def plotly_daily_seasonality(
    target_df: pd.DataFrame,
    *,
    ids: Optional[Iterable[str]] = None,
    width: int = 900,
    height_per_id: int = 500,
    show_legend: bool = True,
    colors: Optional[Any] = None,
    display_fig: bool = True,
) -> go.Figure:
    """
    Interactive Plotly version of 'plot_daily_seasonality'.

    Rows: one subplot per unique_id.
    X: hour of day [0..23].
    Lines: per-month mean profile (fixed month palette).
    Ribbon: 10th-90th percentile across winter days for each month-hour.

    Returns
    -------
    fig : plotly.graph_objs.Figure
    lambdas : dict {unique_id: lambda}  (empty if no Box-Cox applied)
    """
    # --- Validate and subset ids
    available_ids = set(target_df["unique_id"].unique())
    ids = list(ids) if ids is not None else sorted(available_ids)
    missing = set(ids) - available_ids
    if missing:
        raise ValueError(
            f"The following ids are not in target_df: {sorted(missing)}. "
            f"Available ids: {sorted(available_ids)}"
        )
    plot_df = target_df[target_df["unique_id"].isin(ids)].copy()

    # --- Add month and hour columns
    plot_df["month"] = plot_df["ds"].dt.month
    plot_df["hour"] = plot_df["ds"].dt.hour

    # --- Fixed 12-color month palette (1..12) using a stable Plotly palette
    month_colors = colors or px.colors.qualitative.Dark24  
    n_colors = len(month_colors)
    month_palette = {m: month_colors[(m - 1) % (min(12, n_colors))] for m in range(1, 13)}  # 12 colors

    # --- Aggregate to mean and 10th/90th percentiles per uid-month-hour
    def agg_quantiles(x):
        return pd.Series({
            "mean": x.mean(),
            "q10": np.percentile(x, 10),
            "q90": np.percentile(x, 90),
        })

    agg = (
        plot_df.groupby(["unique_id", "month", "hour"])["y"]
        .apply(agg_quantiles)
        .reset_index()
    )
    # columns: unique_id, month, hour, level_3, [mean|q10|q90]
    agg = agg.pivot_table(
        index=["unique_id", "month", "hour"],
        columns="level_3",
        values="y"
    ).reset_index()  # now has columns unique_id, month, hour, mean, q10, q90

    # --- Build subplot grid: one row per id
    n_rows = len(ids)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=tuple(ids),
    )

    show_leg_first_row = show_legend
    month_name = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    for r, uid in enumerate(ids, start=1):
        uid_grp = agg[agg["unique_id"] == uid].copy()
        if uid_grp.empty:
            continue

        # maintain x ticks 0..23
        hours_all = np.arange(0, 24, 1)

        # For each month: add ribbon (q10-q90) and mean line
        for m in range(1, 13):
            month_grp = uid_grp[uid_grp["month"] == m].sort_values("hour")
            if month_grp.empty:
                continue

            color = month_palette[m]
            def ensure_rgb(c: str) -> str:
                if c.startswith("#"):
                    r = int(c[1:3], 16)
                    g = int(c[3:5], 16)
                    b = int(c[5:7], 16)
                    return f"rgb({r},{g},{b})"
                return c
            def make_opaque(rgba_str: str, alpha: float) -> str:
                color_str = rgba_str.replace("rgba", "rgb")
                vals = color_str.strip("rgb() ").split(",")
                r, g, b = [v.strip() for v in vals]
                return f"rgba({r}, {g}, {b}, {alpha})"
            band_color = make_opaque(ensure_rgb(color), 0.2)

            # Build polygon for the ribbon (q90 upper, q10 lower reversed)
            x_poly = np.concatenate([month_grp["hour"].values, month_grp["hour"].values[::-1]])
            y_poly = np.concatenate([month_grp["q90"].values, month_grp["q10"].values[::-1]])

            # Ribbon
            fig.add_trace(
                go.Scatter(
                    x=x_poly,
                    y=y_poly,
                    fill="toself",
                    fillcolor=band_color,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"Month {m} band",
                ),
                row=r, col=1
            )

            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=month_grp["hour"],
                    y=month_grp["mean"],
                    mode="lines+markers",
                    name=month_name[m],
                    line=dict(width=2, color=color),
                    marker=dict(size=5),
                    legendgroup=f"month-{m}",
                    showlegend=(r == 1) and show_leg_first_row,
                    hovertemplate=(
                        f"ID: {uid}<br>"
                        f"Month: {month_name[m]}<br>"
                        "Hour: %{x}<br>"
                        "Mean: %{y:.4g}<br>"
                        "P10: %{customdata[1]:.4g}<br>"
                        "P90: %{customdata[2]:.4g}<extra></extra>"
                    ),
                    customdata=np.column_stack([
                        np.full(len(month_grp), m),
                        month_grp["q10"].values,
                        month_grp["q90"].values
                    ]),
                ),
                row=r, col=1
            )

            # Month label near the last point
            fig.add_annotation(
                x=month_grp["hour"].iloc[-1],
                y=month_grp["mean"].iloc[-1],
                text=str(m),
                xref=f"x{'' if r == 1 else r}",
                yref=f"y{'' if r == 1 else r}",
                showarrow=False,
                font=dict(size=12, color=color),
                xanchor="left",
                yanchor="middle",
                opacity=0.95,
            )

        # Axes per row
        fig.update_xaxes(
            tickmode="array",
            tickvals=hours_all,
            title_text="Hour of Day" if r == n_rows else "",
            row=r, col=1
        )
        fig.update_yaxes(
            title_text="Heat Demand (kWh)" if r == 1 else "",
            zeroline=False,
            row=r, col=1
        )

    fig.update_layout(
        title="Heat Demand by Month across Hours (80% band, P10-P90)",
        template="plotly_white",
        height=max(320, height_per_id * n_rows),
        width=width,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        margin=dict(l=60, r=30, t=80, b=50),
    )

    if display_fig:
        html = fig.to_html(include_plotlyjs="inline", full_html=False)  # offline, self-contained
        display(HTML(html))

    return fig

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Optional, Iterable, Any, List, Tuple
from IPython.display import HTML, display


def plotly_weekly_seasonality(
    target_df: pd.DataFrame,
    *,
    ids: Optional[Iterable[str]] = None,
    width: int = 900,
    height_per_id: int = 350,
    show_legend: bool = True,
    colors: Optional[Any] = None,
    display_fig: bool = True,
    groups: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    group_labels: Optional[List[str]] = None,
    n_cols: int = 1,
    vertical_spacing: float = 0.12,
    horizontal_spacing: float = 0.08,
    annotate: bool = True,
) -> go.Figure:
    """
    Weekly seasonality Plotly figure.

    Rows/cols: grid of subplots, one per unique_id.
    X: hour_of_week [0..167].
    Grouping:
      - default: by month (1..12)
      - if `groups` given: by custom periods (start_ts, end_ts),
        optionally named with `group_labels`.
    """
    import math
    
    # --- Validate IDs ---
    available_ids = set(target_df["unique_id"].unique())
    ids = list(ids) if ids is not None else sorted(available_ids)
    missing = set(ids) - available_ids
    if missing:
        raise ValueError(f"The following ids are not in target_df: {sorted(missing)}. "
                         f"Available ids: {sorted(available_ids)}")
    plot_df = target_df[target_df["unique_id"].isin(ids)].copy()

    # --- Base time feature ---
    plot_df["hour_of_week"] = plot_df["ds"].dt.dayofweek * 24 + plot_df["ds"].dt.hour

    # --- Grouping logic ---
    if groups is None:
        # group by calendar month
        plot_df["group"] = plot_df["ds"].dt.month
        default_labels = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        mode = "month"
    else:
        # custom periods
        n_groups = len(groups)
        if group_labels is not None and len(group_labels) != n_groups:
            raise ValueError("group_labels must have same length as groups.")

        plot_df["group"] = -1
        for gi, (start_ts, end_ts) in enumerate(groups):
            mask = (plot_df["ds"] >= start_ts) & (plot_df["ds"] <= end_ts)
            plot_df.loc[mask, "group"] = gi

        plot_df = plot_df[plot_df["group"] != -1]
        if plot_df.empty:
            raise ValueError("No data left after applying custom groups.")

        if group_labels is None:
            default_labels = {i: f"G{i+1}" for i in range(n_groups)}
        else:
            default_labels = {i: lab for i, lab in enumerate(group_labels)}
        mode = "custom"

    if plot_df.empty:
        raise ValueError("No data to plot after filtering ids/groups.")

    unique_groups = sorted(plot_df["group"].unique())

    # --- Colors ---
    base_colors = colors or px.colors.qualitative.Dark24
    n_colors = len(base_colors)
    group_palette = {
        g: base_colors[i % n_colors] for i, g in enumerate(unique_groups)
    }

    # --- Aggregation: mean, 10th, 90th percentile ---
    def agg_quantiles(x):
        return pd.Series({
            "mean": x.mean(),
            "q10": np.percentile(x, 10),
            "q90": np.percentile(x, 90),
        })

    agg = (
        plot_df.groupby(["unique_id", "group", "hour_of_week"])["y"]
        .apply(agg_quantiles)
        .reset_index()
    )

    agg = agg.pivot_table(
        index=["unique_id", "group", "hour_of_week"],
        columns="level_3",
        values="y",
    ).reset_index()  # unique_id, group, hour_of_week, mean, q10, q90

    # --- Subplot grid with n_cols ---
    n_ids = len(ids)
    n_rows = math.ceil(n_ids / n_cols)

    # subplot titles padded to rows*cols
    titles = [f"Series {uid}" for uid in ids] + [""] * (n_rows * n_cols - n_ids)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        subplot_titles=tuple(titles),
    )

    # small helpers for color handling
    def ensure_rgb(c: str) -> str:
        if c.startswith("#") and len(c) == 7:
            r = int(c[1:3], 16)
            g = int(c[3:5], 16)
            b = int(c[5:7], 16)
            return f"rgb({r},{g},{b})"
        return c

    def make_opaque(color_str: str, alpha: float) -> str:
        if color_str.startswith("rgba"):
            inner = color_str[color_str.find("(")+1:color_str.find(")")]
            r, g, b, _ = [v.strip() for v in inner.split(",")]
            return f"rgba({r},{g},{b},{alpha})"
        if color_str.startswith("rgb"):
            inner = color_str[color_str.find("(")+1:color_str.find(")")]
            r, g, b = [v.strip() for v in inner.split(",")]
            return f"rgba({r},{g},{b},{alpha})"
        return color_str

    # ticks per day (every 24h)
    day_tick_vals = list(range(0, 169, 24))
    day_tick_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", ""]

    # --- Loop over IDs and place them on grid ---
    for idx, uid in enumerate(ids):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        axis_index = idx + 1  # subplot index in row-major order
        axis_suffix = "" if axis_index == 1 else str(axis_index)

        uid_grp = agg[agg["unique_id"] == uid].copy()
        if uid_grp.empty:
            continue

        for g in unique_groups:
            g_grp = uid_grp[uid_grp["group"] == g].sort_values("hour_of_week")
            if g_grp.empty:
                continue

            label = default_labels.get(g, str(g))
            color = ensure_rgb(group_palette[g])
            band_color = make_opaque(color, 0.2)

            x_vals = g_grp["hour_of_week"].values
            mean_vals = g_grp["mean"].values
            q10_vals = g_grp["q10"].values
            q90_vals = g_grp["q90"].values

            # ribbon polygon
            x_poly = np.concatenate([x_vals, x_vals[::-1]])
            y_poly = np.concatenate([q90_vals, q10_vals[::-1]])

            fig.add_trace(
                go.Scatter(
                    x=x_poly,
                    y=y_poly,
                    fill="toself",
                    fillcolor=band_color,
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{label} band",
                ),
                row=row,
                col=col,
            )

            # mean line
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=mean_vals,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    legendgroup=f"group-{g}",
                    showlegend=(idx == 0 and show_legend),
                ),
                row=row,
                col=col,
            )

            # inline label on last point
            if annotate:
                fig.add_annotation(
                    x=x_vals[-1],
                    y=mean_vals[-1],
                    text=label,
                    xref=f"x{axis_suffix}",
                    yref=f"y{axis_suffix}",
                    showarrow=False,
                    font=dict(size=11, color=color),
                    xanchor="left",
                    yanchor="middle",
                    opacity=0.95,
                )

        # axis formatting for this subplot
        fig.update_xaxes(
            tickmode="array",
            tickvals=day_tick_vals,
            ticktext=day_tick_labels,
            title_text="Day of Week" if row == n_rows else "",
            linewidth=1,
            linecolor="lightgrey",
            mirror=True,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            linewidth=1,
            linecolor="lightgrey",
            mirror=True,
            row=row,
            col=col,
            dtick=200
        )

    row = math.ceil(n_rows / 2)
    fig.update_yaxes(
        title_text="Heat Demand (kWh)" if (col == 1) else "",
        zeroline=False,
        row=row,
        col=col,
    )

    title_suffix = " (grouped by month)" if mode == "month" else " (grouped by custom periods)"
    fig.update_layout(
        height=max(320, height_per_id * n_rows),
        width=width,
        template="plotly_white",
        title="Heat Demand across Hours of Week" + title_suffix,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=60, r=30, t=80, b=50),
    )

    if display_fig:
        html = fig.to_html(include_plotlyjs="inline", full_html=False)
        display(HTML(html))

    return fig

def plot_acf_diagnostics(
    df: pd.DataFrame,
    cols_to_plot: Iterable[str],
    acf_max_lag=168,
    acf_opacity=0.7,
    num_bins=30,
    time_col='ds',
    compute_residuals: bool = True,
    ground_truth_col: str = 'y',
    base_title: str = "Diagnostic plots",
    alias_for_value: str = "Value",
) -> Tuple[Dict[str, Any], go.Figure]:
    # Validate cols
    cols = []
    for m in cols_to_plot:
        if m not in df.columns:
            logging.warning(f"Column '{m}' not found in DataFrame. It will be skipped.")
            continue
        if df[m].isnull().all():
            logging.warning(f"Column '{m}' contains only NaN values. It will be skipped.")
            continue
        cols.append(m)
    if not cols:
        raise ValueError("No valid columns to plot.")
    
    if time_col not in df.columns:
        raise ValueError(f"DataFrame must contain a '{time_col}' column for time indexing.")
    
    # Prep
    df = df.copy()
    N = df[time_col].nunique()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).set_index(time_col)

    if compute_residuals and ground_truth_col not in df.columns:
        raise ValueError(f"DataFrame must contain a '{ground_truth_col}' column (actuals).")

    # Precompute per model
    payload = {}
    for m in cols:
        if compute_residuals:
            resid = (df[ground_truth_col] - df[m]).astype(float)
        else:
            resid = df[m].astype(float)
        resid = resid.replace([np.inf, -np.inf], np.nan).dropna()
        if resid.empty:
            continue

        samples = resid.values
        # KDE grid
        xmin, xmax = np.nanpercentile(samples, [0.5, 99.5])
        if xmin == xmax:
            xmin, xmax = xmin - 1e-6, xmax + 1e-6
        kde = gaussian_kde(samples)
        kde_x = np.linspace(xmin, xmax, 400)
        kde_y = kde(kde_x)  # density

        # ACF + CI
        acf_vals = acf(samples, nlags=acf_max_lag, fft=True, missing='drop')
        acf_lags = np.arange(len(acf_vals))

        payload[m] = dict(
            time=resid.index, resid=samples,
            hist_x=samples, kde_x=kde_x, kde_y=kde_y,
            acf_lags=acf_lags, acf_vals=acf_vals
        )

    if not payload:
        raise ValueError("None of the specified models are present or residuals are empty after cleaning.")

    # ---- Figure: 2 rows x 2 cols, top spans both cols ----
    fig = make_subplots(
        rows=2, cols=2, vertical_spacing=0.12, horizontal_spacing=0.08,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=(
            "Lineplot over time",
            "ACF with 95% CI",
            "Histogram (density) + KDE"
        )
    )

    all_cols = list(payload.keys())

    TRACES_PER_COL = 4

    models_colors = px.colors.qualitative.Dark24
    models_palette = {m: models_colors[i % len(all_cols)] for i, m in enumerate(all_cols)}

    for m in all_cols:
        d = payload[m]

        # Residuals line (spanning row1 col1)
        fig.add_trace(
            go.Scatter(
                x=d['time'], y=d['resid'], mode='lines',
                name=m, legendgroup=m,
                hovertemplate="ds=%{x}<br>value=%{y:.4f}<extra></extra>",
                line=dict(color=models_palette[m])
            ),
            row=1, col=1
        )

        # ACF bars (row2 col1)
        fig.add_trace(
            go.Bar(
                x=d['acf_lags'], y=d['acf_vals'], name=f"{m} ACF", legendgroup=m,
                hovertemplate="lag=%{x}<br>acf=%{y:.4f}<extra></extra>",
                marker=dict(color=models_palette[m]),
            ),
            row=2, col=1
        )

        # Histogram (density) (row2 col2)
        fig.add_trace(
            go.Histogram(
                x=d['hist_x'], nbinsx=num_bins, histnorm='probability density',
                name=f"{m} histogram (density)", legendgroup=m,
                hovertemplate="density=%{y:.4f}<br>bin=%{x}<extra></extra>",
                marker=dict(color=models_palette[m]),
            ),
            row=2, col=2
        )

        # KDE (row2 col2)
        fig.add_trace(
            go.Scatter(
                x=d['kde_x'], y=d['kde_y'], mode='lines',
                name=f"{m} KDE", legendgroup=m,
                hovertemplate="x=%{x:.4f}<br>density=%{y:.4f}<extra></extra>",
                line=dict(color=models_palette[m])
            ),
            row=2, col=2
        )

    # Initial visibility: first model only
    vis = [True] * (len(all_cols) * TRACES_PER_COL)
    #vis[:TRACES_PER_COL] = [True] * TRACES_PER_COL
    for t, v in zip(fig.data, vis):
        t.visible = v

    buttons = []
    for i, m in enumerate(all_cols):
        v = [False] * (len(all_cols) * TRACES_PER_COL)
        start = i * TRACES_PER_COL
        v[start:start+TRACES_PER_COL] = [True] * TRACES_PER_COL
        buttons.append(dict(
            label=m,
            method="update",
            args=[
                {"visible": v},
                {"title": dict(
                    text=f"{base_title} - {all_cols[0]}",
                    y=0.97,              # moves the title slightly down (default ≈ 1.0)
                    x=0.08,              # center the title
                    xanchor='left',
                    yanchor='top',
                )}
            ],
        ))
    buttons.append(dict(
        label="All models",
        method="update",
        args=[
            {"visible": [True] * (len(all_cols) * TRACES_PER_COL)},
            {"title": dict(
                text=f"{base_title} - All",
                y=0.97,              # moves the title slightly down (default ≈ 1.0)
                x=0.08,              # center the title
                xanchor='left',
                yanchor='top',
            )}
        ],
    ))
    buttons = buttons[::-1]  # last button on top

    ci_shared = 1.96 / np.sqrt(N)
    acf_x0, acf_x1 = 0, acf_max_lag

    fig.update_layout(
        title=dict(
            text=f"{base_title} - {all_cols[0]}",
            y=0.97,              # moves the title slightly down (default ≈ 1.0)
            x=0.08,               # center the title
            xanchor='left',
            yanchor='top',
        ),
        margin=dict(t=200),  # extra top margin to accommodate title
        updatemenus=[dict(
            type="dropdown", x=0.01, y=1.08, xanchor="left",
            buttons=buttons, showactive=True,
        )],
        shapes=[
            dict(  # CI rectangle on the ACF subplot (row=2, col=1)
                type="rect",
                xref="x2", yref="y2",
                x0=acf_x0, x1=acf_x1,
                y0=-ci_shared, y1=ci_shared,
                fillcolor="rgba(0,0,0,0.12)",
                line=dict(width=0),
                layer="above"  
            )
        ],
        height=900,
        barmode='overlay',
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="left", x=0.0, tracegroupgap=12),
    )

    # Cosmetics
    # Make hist semi-transparent
    for tr in fig.data:
        if isinstance(tr, go.Histogram):
            tr.opacity = 0.55
        if isinstance(tr, go.Bar):
            tr.marker.line.width = 0.5
            tr.marker.opacity = acf_opacity

    # Axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text=alias_for_value, row=1, col=1)

    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_yaxes(title_text="ACF", row=2, col=1)

    fig.update_xaxes(title_text=alias_for_value, row=2, col=2)
    fig.update_yaxes(title_text="Density", row=2, col=2)

    # Zero line on ACF
    fig.add_hline(y=0, line_dash="dash", row=2, col=1)

    html = fig.to_html(include_plotlyjs="inline", full_html=False)  # offline, self-contained
    display(HTML(html))

    with open("plot.html", "w", encoding="utf-8") as f:
        f.write(html)

    return payload, fig

def extract_subplot(fig_subplots: go.Figure, row: int, col: int) -> go.Figure:
    """
    Build a new go.Figure from the subplot at (row, col), copying:
      - All traces in that cell (incl. secondary y)
      - Axis styles (x and y, plus secondary y if present)
      - Shapes, annotations, images that reference those axes
    The new figure's subplot fills the canvas (domains = [0, 1]).
    """
    # Utility: remove " domain" suffix from refs like "x domain"
    def _strip_domain(ref):
        if isinstance(ref, str):
            return ref.replace(" domain", "")
        return ref

    traces = list(fig_subplots.select_traces(row=row, col=col))
    if not traces:
        # Even if no traces, we can still try to bring over axes and decorations
        pass

    xaxes = list(fig_subplots.select_xaxes(row=row, col=col))
    yaxes = list(fig_subplots.select_yaxes(row=row, col=col))

    if not xaxes or not yaxes:
        # If we can’t resolve axes for the cell, bail out to a minimal figure with traces only
        return go.Figure(data=traces)

    # Primary x/y are the first; detect any secondary y that overlays the primary
    x_primary = xaxes[0]
    y_primary = yaxes[0]
    y_secondaries = [ya for ya in yaxes[1:] if getattr(ya, "overlaying", None)]

    def _axis_ids_by_domain(fig, axis_type, domain_tuple):
        ids = []
        # Search xaxis, xaxis2, xaxis3, ... (or yaxis...)
        for idx in range(1, 100):  # generous upper bound
            name = f"{axis_type}axis" + ("" if idx == 1 else str(idx))
            ax = getattr(fig.layout, name, None)
            if ax and getattr(ax, "domain", None) == domain_tuple:
                ids.append(name)  # e.g., 'xaxis2'
        return ids

    def _axis_name_to_ref(name):
        # 'xaxis'  -> 'x'
        # 'xaxis2' -> 'x2'
        if name.endswith("axis"):
            return name[0]
        else:
            return name[0] + name[len(name[0] + "axis"):]

    x_ids_fullnames = _axis_ids_by_domain(fig_subplots, "x", tuple(x_primary.domain))
    y_ids_fullnames = _axis_ids_by_domain(fig_subplots, "y", tuple(y_primary.domain))

    # Map to 'x', 'x2', ...
    x_refs_in_cell = {_axis_name_to_ref(n) for n in x_ids_fullnames}
    y_refs_in_cell = {_axis_name_to_ref(n) for n in y_ids_fullnames}

    # Try to identify which yref belongs to the secondary axis(es)
    # Heuristic: any y-axis in the same domain whose 'overlaying' points to another y-axis is secondary.
    # Build name->object map
    name_to_yaxis = {n: getattr(fig_subplots.layout, n) for n in y_ids_fullnames}
    y_primary_name = None
    for n, ax in name_to_yaxis.items():
        if not getattr(ax, "overlaying", None):
            y_primary_name = n
            break
    if y_primary_name is None:
        # fallback: treat the first as primary
        y_primary_name = y_ids_fullnames[0]

    y_secondary_names = [n for n in y_ids_fullnames if n != y_primary_name]
    y_primary_ref = _axis_name_to_ref(y_primary_name)
    y_secondary_refs = {_axis_name_to_ref(n) for n in y_secondary_names}

    new_traces = []
    for tr in traces:
        trc = copy.deepcopy(tr)
        # Normalize xaxis to 'x'
        if getattr(trc, "xaxis", None) in x_refs_in_cell or getattr(trc, "xaxis", None) is None:
            trc.xaxis = "x"
        # Normalize yaxis to 'y' or 'y2' depending on original
        current_yref = getattr(trc, "yaxis", None) or y_primary_ref
        trc.yaxis = "y2" if current_yref in y_secondary_refs else "y"
        new_traces.append(trc)

    new_fig = go.Figure(new_traces)

    # Primary X
    x_json = x_primary.to_plotly_json()
    x_json["domain"] = [0, 1]
    x_json["anchor"] = "y"
    new_fig.update_layout(xaxis=x_json)

    # Primary Y
    y_json = y_primary.to_plotly_json()
    y_json["domain"] = [0, 1]
    y_json["anchor"] = "x"
    new_fig.update_layout(yaxis=y_json)

    # Secondary Y (if any) -> yaxis2 overlaying 'y'
    if y_secondaries:
        y2_json = y_secondaries[0].to_plotly_json()
        y2_json["domain"] = [0, 1]
        y2_json["overlaying"] = "y"
        # Keep side if it existed, default to 'right'
        y2_json["side"] = y2_json.get("side", "right")
        new_fig.update_layout(yaxis2=y2_json)

    # Copy shapes / annotations / images bound to this subplot’s axes
    def _belongs_to_cell(xref, yref):
        # Accept either exact axis ('x2') or domain refs ('x2 domain')
        def _strip_domain(r):
            return r.replace(" domain", "") if isinstance(r, str) else r
        xr = _strip_domain(xref)
        yr = _strip_domain(yref)
        return (xr in x_refs_in_cell) and (yr in y_refs_in_cell)

    # Shapes
    if getattr(fig_subplots.layout, "shapes", None):
        new_shapes = []
        for shp in fig_subplots.layout.shapes:
            if _belongs_to_cell(shp.xref, shp.yref):
                shp2 = shp.to_plotly_json()
                # Remap xref/yref into the new figure
                shp2["xref"] = "x" + (" domain" if isinstance(shp.xref, str) and "domain" in shp.xref else "")
                # Secondary y?
                source_yref = shp.yref if isinstance(shp.yref, str) else None
                target_y = "y2" if (source_yref and _strip_domain(source_yref) in y_secondary_refs) else "y"
                shp2["yref"] = target_y + (" domain" if isinstance(shp.yref, str) and "domain" in shp.yref else "")
                new_shapes.append(shp2)
        if new_shapes:
            new_fig.update_layout(shapes=new_shapes)

    # Annotations
    if getattr(fig_subplots.layout, "annotations", None):
        new_ann = []
        for ann in fig_subplots.layout.annotations:
            # Only axis-bound annotations (skip 'paper'-anchored globals)
            if isinstance(ann.xref, str) and isinstance(ann.yref, str) and _belongs_to_cell(ann.xref, ann.yref):
                a2 = ann.to_plotly_json()
                a2["xref"] = "x" + (" domain" if "domain" in ann.xref else "")
                source_yref = ann.yref
                target_y = "y2" if _strip_domain(source_yref) in y_secondary_refs else "y"
                a2["yref"] = target_y + (" domain" if "domain" in ann.yref else "")
                new_ann.append(a2)
        if new_ann:
            new_fig.update_layout(annotations=new_ann)

    # Images
    if getattr(fig_subplots.layout, "images", None):
        new_imgs = []
        for im in fig_subplots.layout.images:
            if isinstance(im.xref, str) and isinstance(im.yref, str) and _belongs_to_cell(im.xref, im.yref):
                im2 = im.to_plotly_json()
                im2["xref"] = "x" + (" domain" if "domain" in im.xref else "")
                source_yref = im.yref
                target_y = "y2" if _strip_domain(source_yref) in y_secondary_refs else "y"
                im2["yref"] = target_y + (" domain" if "domain" in im.yref else "")
                new_imgs.append(im2)
        if new_imgs:
            new_fig.update_layout(images=new_imgs)

    if getattr(fig_subplots.layout, "legend", None):
        new_fig.update_layout(legend=fig_subplots.layout.legend)
    if getattr(fig_subplots.layout, "template", None):
        new_fig.update_layout(template=fig_subplots.layout.template)
    if getattr(fig_subplots.layout, "font", None):
        new_fig.update_layout(font=fig_subplots.layout.font)

    return new_fig