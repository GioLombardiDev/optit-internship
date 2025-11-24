import pandas as pd
import numpy as np
import datetime 
import logging
import warnings
from typing import Optional, Dict, Any
import re
import math
_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())

# -------------------------------------------------------------------------------
# FUNCTIONS FOR CROSS-VALIDATION 
# -------------------------------------------------------------------------------

def _assert_hour_aligned(ts: pd.Timestamp, label: str) -> None:
    """
    Raise if *ts* is not aligned to the top of the hour.
    Works for any pandas.Timestamp.
    """
    if ts.minute or ts.second or ts.microsecond or ts.nanosecond:
        raise ValueError(
            f"{label} must be aligned to the hour — got {ts.strftime('%Y-%m-%d %H:%M:%S.%f')}"
        )

def get_cv_params(
    start_test_cv: pd.Timestamp,
    end_test_cv: pd.Timestamp,
    n_windows: int,
    horizon_hours: int = 24 * 7,
    start_forecasts_at_midnight: bool = True, # If true, each forecast is forced to start at midnight
    logger: Optional[logging.Logger] = None
) -> tuple[int, int, pd.Timestamp]:
    """
    Compute rolling-window cross-validation parameters.

    The helper returns:
    * step_size: spacing between successive cut-off times 
        (expressed in whole hours);  
    * test_hours_actual: total length of the evaluated test span
        after adjusting the last cut-off;  
    * end_test_actual: timestamp of the final observation used
        in cross-validation.

    Parameters
    ----------
    start_test_cv : pandas.Timestamp
        First timestamp **included** in the test span.  Must be aligned to
        the hour and, when *start_forecasts_at_midnight* is True, equal to
        00:00:00.
    end_test_cv : pandas.Timestamp
        Last timestamp **included** in the test span (hour-aligned).
    n_windows : int
        Number of rolling windows (cut-offs).  Must be at least 1.
    horizon_hours : int, default ``24 * 7``
        Forecast horizon in hours.
    start_forecasts_at_midnight : bool, default ``True``
        If True, force every forecast window to start at 00:00; this makes
        *step_size* a multiple of 24 h.
    logger : logging.Logger, optional
        Logger to use for logging the parameters. If None, uses local logger.
    """
    _assert_hour_aligned(start_test_cv, "start_test_cv")
    _assert_hour_aligned(end_test_cv, "end_test_cv")

    logger = logger or _LOGGER

    if start_forecasts_at_midnight and start_test_cv.time() != datetime.time(0):
        raise ValueError("start_test_cv must be exactly 00:00:00 when start_forecasts_at_midnight=True.")
    
    test_range = end_test_cv - start_test_cv + pd.Timedelta(hours=1)  # +1 to include the start hour in the test span
    if test_range <= pd.Timedelta(0):
        raise ValueError("end_test_cv must be after start_test_cv.")
    
    test_hours = int(test_range / pd.Timedelta(hours=1))
    if test_hours < horizon_hours:
        raise ValueError("Test span shorter than forecast horizon.")
    
    if n_windows < 1:
        raise ValueError("n_windows must be ≥ 1.")
    elif n_windows == 1:
        step_size = 1 # Else we get a division by zero error when running cv
    else:
        step_size = (test_hours - horizon_hours) // (n_windows - 1) # The -1 is because the first cutoff is before start train, and the last is (close to) the end
        if step_size < 1:
            raise ValueError("Too many windows for the chosen span & horizon.")
        if start_forecasts_at_midnight: 
            step_size = (step_size // 24) * 24  # Ensure step_size is a multiple of 24 hours if starting at midnight
            if step_size < 24:
                raise ValueError("step_size became <24 h after midnight-rounding; relax the midnight constraint or use fewer windows.")

    first_cutoff = start_test_cv - pd.Timedelta(hours=1)
    last_cutoff = first_cutoff + pd.Timedelta(hours=step_size * (n_windows - 1))
    end_test_actual = last_cutoff + pd.Timedelta(hours=horizon_hours)
    test_hours_actual = int((last_cutoff - first_cutoff) / pd.Timedelta(hours=1)) + horizon_hours

    step_size_days = step_size // 24
    step_size_remaining_hours = step_size % 24
    logger.info(
        "CV params: %d window%s, first cutoff %s, last cutoff %s%s.",
        n_windows,
        "" if n_windows == 1 else "s",
        first_cutoff,
        last_cutoff,
        "" if n_windows == 1 else f", step size {step_size_days}d {step_size_remaining_hours}h"
    )
    return step_size, test_hours_actual, end_test_actual

def get_cv_params_v2(
    start_test_cv: pd.Timestamp,
    end_test_cv: pd.Timestamp,
    n_windows: int | None = None,
    step_size: int | None = None,
    max_n_fits: int | None = None,
    horizon_hours: int = 24 * 7,
    start_forecasts_at_midnight: bool = True, # If true, each forecast is forced to start at midnight
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Compute cross-validation windowing and refit schedule for time-series backtesting.

    Given a desired inclusive test span [start_test_cv, end_test_cv], a forecast
    horizon (in hours), and either the number of windows or a stride between cutoffs,
    this function derives the actual step size, realized coverage, and (optionally)
    a model refitting schedule capped by max_n_fits.
    """
    _assert_hour_aligned(start_test_cv, "start_test_cv")
    _assert_hour_aligned(end_test_cv, "end_test_cv")

    logger = logger or _LOGGER

    if not isinstance(horizon_hours, int) or horizon_hours < 1:
        raise ValueError("horizon_hours must be an integer ≥ 1.")
    if start_forecasts_at_midnight and start_test_cv.time() != datetime.time(0):
        raise ValueError("start_test_cv must be exactly 00:00:00 when start_forecasts_at_midnight=True.")
    
    test_range = end_test_cv - start_test_cv + pd.Timedelta(hours=1)  # +1 to include the start hour in the test span
    if test_range <= pd.Timedelta(0):
        raise ValueError("end_test_cv must be after start_test_cv.")
    
    test_hours = int(test_range / pd.Timedelta(hours=1))
    if test_hours < horizon_hours:
        raise ValueError("Test span shorter than forecast horizon.")

    if (step_size is None) == (n_windows is None):
        raise ValueError("Exactly one of step_size or n_windows must be provided.")

    if n_windows is not None: # n_windows is given
        if (not isinstance(n_windows, int)) or n_windows < 1:
            raise ValueError("n_windows must be an integer ≥ 1.")
        elif n_windows == 1:
            # step_size is irrelevant for placement; pick a dummy that won't trip validations
            step_size = 24 if start_forecasts_at_midnight else 1
        else:
            step_size = (test_hours - horizon_hours) // (n_windows - 1) # The -1 is because the first cutoff is before start train, and the last is (close to) the end
            if step_size < 1:
                raise ValueError("Too many windows for the chosen span & horizon.")
            if start_forecasts_at_midnight: 
                step_size = (step_size // 24) * 24  # Ensure step_size is a multiple of 24 hours if starting at midnight
                if step_size < 24:
                    raise ValueError("step_size became <24 h after midnight-rounding; relax the midnight constraint or use fewer windows.")
    
    else: # step_size is given
        if (not isinstance(step_size, int)) or step_size < 1:
            raise ValueError("step_size must be an integer ≥ 1.")
        n_windows = (test_hours - horizon_hours) // step_size + 1
        if start_forecasts_at_midnight and step_size % 24 != 0 and n_windows > 1:
            raise ValueError("step_size must be a multiple of 24 hours when start_forecasts_at_midnight=True.")
        if n_windows == 1:
            logger.warning("Only one window fits in the chosen span with the given step_size & horizon.")

    first_cutoff = start_test_cv - pd.Timedelta(hours=1)
    last_cutoff = first_cutoff + pd.Timedelta(hours=step_size * (n_windows - 1))
    end_test_actual = last_cutoff + pd.Timedelta(hours=horizon_hours)
    test_hours_actual = int((last_cutoff - first_cutoff) / pd.Timedelta(hours=1)) + horizon_hours

    if max_n_fits is not None:
        if (not isinstance(max_n_fits, int)) or max_n_fits < 1:
            raise ValueError("max_n_fits must be an integer ≥ 1.")
        elif max_n_fits == 1:
            refit = False
            n_fits_actual = 1
        elif max_n_fits > n_windows:
            refit = 1
            n_fits_actual = n_windows
        else:
            refit = math.ceil(n_windows / max_n_fits)
            n_fits_actual = math.ceil(n_windows / refit)
    else:
        refit = None
        n_fits_actual = n_windows

    step_size_days = step_size // 24
    step_size_remaining_hours = step_size % 24
    refit_str = f"" if refit is None else \
        ", no refitting" if not refit else \
        ", refit every window" if refit ==1 else \
        f", refit every {refit} windows" if refit > 1 else ""
    n_fits_str = "" if n_fits_actual is None else \
        f", total number of fits: {n_fits_actual}"
    logger.info(
        "CV params: %d window%s, first cutoff %s, last cutoff %s%s%s%s.",
        n_windows,
        "" if n_windows == 1 else "s",
        first_cutoff,
        last_cutoff,
        "" if n_windows == 1 else f", step size {step_size_days}d {step_size_remaining_hours}h",
        refit_str,
        n_fits_str
    )

    out = {
        "step_size": step_size,
        "test_hours": test_hours_actual,
        "end_test_actual": end_test_actual,
        "n_windows": n_windows,
        "refit": refit,
        "n_fits": n_fits_actual
    }
    return out

def display_info_cv(
        cv_df: pd.DataFrame,
        logger: Optional[logging.Logger] = None
    ) -> None:
    """
    Print a one-screen summary of a rolling-origin cross-validation
    DataFrame and return the forecasting horizon length.
    """
    logger = logger or _LOGGER

    # Detect model-forecast columns (exclude PI bounds)
    model_cols = cv_df.columns.difference(['unique_id', 'ds', 'cutoff', 'y'])
    models = [c for c in model_cols if '-lo-' not in c and '-hi-' not in c]

    # Basic CV geometry
    cutoffs = np.sort(cv_df['cutoff'].unique())
    first_cutoff = cutoffs[0]
    last_cutoff = cutoffs[-1]

    if len(cutoffs) > 1:
        step_deltas = np.diff(cutoffs) # array of Timedelta
        step_days = np.array([d / pd.Timedelta(days=1) for d in step_deltas])
        unique_steps = np.unique(step_days)

        if unique_steps.size == 1:                           
            step_size = int(unique_steps[0]) if unique_steps[0].is_integer() else unique_steps[0]
        else:                                                
            step_size = list(step_days)                     
            warnings.warn(
                f"Cut-off windows are not equally spaced: {unique_steps}. Using the full list for reporting.",
                UserWarning, stacklevel=2
            )
    else:
        step_size = None

    horizon_lengths = (
        cv_df.groupby(['cutoff', 'unique_id'], sort=False)['ds'].nunique()
        .to_numpy()
    )
    unique_horizons = np.unique(horizon_lengths)

    if len(unique_horizons) == 1:
        horizon_return = int(unique_horizons[0])
    else:
        horizon_return = horizon_lengths.tolist()
        logger.warning(
            f"Windows have different horizon lengths: {unique_horizons}. Using the full list for reporting.",
            UserWarning, stacklevel=2
        )

    summary_data = {
        'Models': models,
        'Unique IDs': cv_df['unique_id'].unique().tolist(),
        'Horizon length (hours)': horizon_return,
        'Windows': len(cutoffs),
        'First cutoff': first_cutoff,
        'Last cutoff': last_cutoff,
        'Step size (days)': step_size,
    } 

    def _formatter(value):
        if value is None:
            return "–"
        elif isinstance(value, (list, tuple, set)):
            return ", ".join(map(str, value))
        elif isinstance(value, (datetime.datetime, pd.Timestamp)):
            return value.strftime("%Y-%m-%d %H:%M")
        elif isinstance(value, np.datetime64):
            return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
        return str(value)

    log_lines = []
    for key, value in summary_data.items():
        value_str = _formatter(value)
        log_lines.append(f"{key:<24}: {value_str}")
    max_length = max(len(line) for line in log_lines)
    log_lines = [f"{'━' * max_length}"] + ["Cross-validation summary"] + \
        [f"{'─' * max_length}"] + log_lines + [f"{'━' * max_length}"]

    logger.info("\n" + "\n".join(log_lines))


def get_cv_params_for_test(
        horizon_type: str,  # 'week' or 'day'
        use_deprecated: bool = False, # If True, use the old CV plan 
        logger: Optional[logging.Logger] = None,
        unique_id: Optional[str] = None, # 'F1', 'F2', 'F3', 'F4', 'F5', only needed if use_deprecated=True
    ) -> tuple[int, int, pd.Timestamp, int]:
    """
    Convenience wrapper that selects a pre-defined CV plan for a given series
    and horizon type, then calls :func:`get_cv_params_v2`.

    """
    logger = logger or _LOGGER

    # Validate
    if unique_id is not None:
        if use_deprecated and unique_id not in ['F1', 'F2', 'F3', 'F4', 'F5']:
            raise ValueError(f"Unknown unique_id: {unique_id}. Known IDs are: F1, ..., F5")
        if not use_deprecated:
            logger.warning("unique_id is ignored when use_deprecated=False")
    if horizon_type not in ['week', 'day']:
        raise ValueError(f"Unknown horizon_type: {horizon_type}. Known types are: 'week', 'day'.")
    
    if use_deprecated:
        for_get_cv_params = {
            'F1': {
                'week': { # step_size = 9d
                    'n_windows': 21,
                    'start_test_cv': pd.to_datetime('2024-10-20'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
                'day': { # step_size = 5d
                    'n_windows': 37,
                    'start_test_cv': pd.to_datetime('2024-10-20'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
            },
            'F2': {
                'week': { # step_size = 9d
                    'n_windows': 21,
                    'start_test_cv': pd.to_datetime('2024-10-20'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
                'day': { # step_size = 5d
                    'n_windows': 37,
                    'start_test_cv': pd.to_datetime('2024-10-20'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
            },
            'F3': {
                'week': { # step_size = 9d
                    'n_windows': 19,
                    'start_test_cv': pd.to_datetime('2024-10-28'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
                'day': { # step_size = 5d
                    'n_windows': 35,
                    'start_test_cv': pd.to_datetime('2024-10-28'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
            },
            'F4': {
                'week': { # step_size = 9d
                    'n_windows': 18,
                    'start_test_cv': pd.to_datetime('2024-11-10'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
                'day': { # step_size = 5d
                    'n_windows': 32,
                    'start_test_cv': pd.to_datetime('2024-11-10'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
            },
            'F5': {
                'week': { # step_size = 9d
                    'n_windows': 18,
                    'start_test_cv': pd.to_datetime('2024-11-10'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
                'day': { # step_size = 5d
                    'n_windows': 33,
                    'start_test_cv': pd.to_datetime('2024-11-10'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
            },
        },
        params = for_get_cv_params[unique_id][horizon_type]
    else:
        for_get_cv_params = {
            'day': {
                'start_test_cv': pd.to_datetime('2024-05-20'),
                'end_test_cv': pd.to_datetime('2025-05-20'),
                'step_size': 24,
                'max_n_fits': 53,
            },
            'week': {
                'start_test_cv': pd.to_datetime('2024-05-20'),
                'end_test_cv': pd.to_datetime('2025-05-20'),
                'step_size': 24,
                'max_n_fits': 52,
            }
        }
        params = for_get_cv_params[horizon_type]
    out = get_cv_params_v2(
            start_test_cv=params['start_test_cv'],
            end_test_cv=params['end_test_cv'],
            n_windows=params.get('n_windows', None),
            step_size=params.get('step_size', None),
            max_n_fits=params.get('max_n_fits', None),
            horizon_hours=168 if horizon_type == 'week' else 24,
            logger=logger
        )
    return out

def sanity_cv_df(
        cv_df: pd.DataFrame, 
        metadata: dict, 
        *,
        positive_forecasts: bool = True,
        logger: Optional[logging.Logger] = None
    ) -> dict:
    """
    Validate that cv_df respects metadata['for_cv'] (except for 'refit' and 'n_fits') 
    and that each timestamp (row) has a valid forecast value (non-NaN, finite, 
    and > 0 if positive_forecasts=True).
    """
    errs = []

    # ---- unpack for_cv ----
    mcv = (metadata or {}).get("for_cv", {})
    for key in ("step_size", "test_hours", "n_windows"):
        if key not in mcv:
            errs.append(f"metadata['for_cv'][{key!r}] is required.")
    end_test_key = "end_test_cv" if "end_test_cv" in mcv else "end_test_actual"
    if end_test_key not in mcv:
        errs.append("metadata['for_cv'] must include 'end_test_cv' or 'end_test_actual'.")
    if errs:
        raise AssertionError("sanity_cv_test_df FAILED:\n- " + "\n- ".join(errs))

    step_size   = int(mcv["step_size"])
    test_hours  = int(mcv["test_hours"])
    n_windows_exp = int(mcv["n_windows"])
    end_test    = pd.Timestamp(mcv[end_test_key])

    # ---- required columns ----
    required = {"unique_id", "ds", "y", "cutoff"}
    missing = required - set(cv_df.columns)
    if missing:
        errs.append(f"Missing required columns: {sorted(missing)}")

    if errs:
        raise AssertionError("sanity_cv_test_df FAILED:\n- " + "\n- ".join(errs))

    # ---- dtypes & nulls ----
    for c in ("ds", "cutoff"):
        if not np.issubdtype(cv_df[c].dtype, np.datetime64):
            errs.append(f"Column '{c}' must be datetime-like (got {cv_df[c].dtype}).")
        if cv_df[c].isna().any():
            errs.append(f"Column '{c}' contains NaT/NaNs.")

    # ---- horizon per window & contiguity ----
    h_counts = cv_df.groupby("cutoff", observed=True)["ds"].nunique()
    if h_counts.empty:
        errs.append("No cutoff groups found (is the DataFrame empty?).")
        h = None
    else:
        if h_counts.nunique() != 1:
            errs.append(f"Inconsistent horizon per cutoff: {sorted(h_counts.unique())}")
        h = int(h_counts.iloc[0])

    if h is not None:
        # verify each group's ds is exactly (cutoff+1h ... cutoff+h) hourly
        # sample up to 50 groups if very large
        groups = h_counts.index
        if len(groups) > 50:
            groups = groups.sort_values()[::max(1, len(groups)//50)]
        for c in groups:
            ds_g = cv_df.loc[cv_df["cutoff"] == c, "ds"].sort_values()
            expected = pd.date_range(c + pd.Timedelta(hours=1), periods=h, freq="h")
            if not ds_g.reset_index(drop=True).equals(expected.to_series().reset_index(drop=True)):
                errs.append(f"Non-contiguous hourly ds for cutoff {c}.")
                break

    # ---- number of windows & spacing ----
    cutoffs = cv_df["cutoff"].drop_duplicates().sort_values()
    if len(cutoffs) != n_windows_exp:
        errs.append(f"n_windows mismatch: expected {n_windows_exp}, got {len(cutoffs)}")
    if len(cutoffs) >= 2:
        deltas_h = np.diff(cutoffs.values).astype("timedelta64[h]").astype(int)
        if not np.all(deltas_h == step_size):
            errs.append(f"Cutoff spacing must be {step_size}h; found {sorted(set(deltas_h))}h.")

    # ---- coverage ----
    ds_min, ds_max = cv_df["ds"].min(), cv_df["ds"].max()
    exp_min = end_test - pd.Timedelta(hours=test_hours) + pd.Timedelta(hours=1)
    if ds_max != end_test:
        errs.append(f"Max ds != end_test ({ds_max} vs {end_test}).")
    if ds_min != exp_min:
        errs.append(f"Min ds != end_test - test_hours + 1h ({ds_min} vs {exp_min}).")

    # last cutoff alignment
    if (h is not None) and (len(cutoffs) > 0):
        if cutoffs.iloc[-1] + pd.Timedelta(hours=h) != end_test:
            errs.append(f"last_cutoff + h != end_test "
                        f"({cutoffs.iloc[-1] + pd.Timedelta(hours=h)} vs {end_test}).")

    # ---- find the main forecast column and validate values ----
    # Heuristic: first numeric column that's not a core col and not an interval (lo/hi)
    core = {"unique_id", "ds", "y", "cutoff"}
    def _is_interval(col: str) -> bool:
        return bool(re.search(r"(^lo-\d+$)|(^hi-\d+$)|(-lo-\d+$)|(-hi-\d+$)", col))

    forecast_cols = [c for c in cv_df.columns
                     if c not in core
                     and pd.api.types.is_numeric_dtype(cv_df[c])
                     and not _is_interval(c)]
    if not forecast_cols:
        errs.append("Could not locate a forecast column (e.g., model alias).")
    else:
        alias = forecast_cols[0]
        vals = cv_df[alias].to_numpy()
        if np.isnan(vals).any():
            errs.append(f"Forecast column '{alias}' contains NaNs.")
        if not np.isfinite(vals).all():
            errs.append(f"Forecast column '{alias}' contains non-finite values.")
        if positive_forecasts and not (vals > 0).all():
            errs.append(f"Forecast column '{alias}' contains non-positive values, "
                        f"but positive_forecasts=True.")

    if errs:
        raise AssertionError("sanity_cv_test_df FAILED:\n- " + "\n- ".join(errs))

    logger = logger or _LOGGER
    logger.info("✓ All sanity checks passed. CV DataFrame has the expected structure.")

    # ---- summary for logs ----
    return {
        "alias": forecast_cols[0] if forecast_cols else None,
        "n_rows": int(len(cv_df)),
        "n_windows": int(len(cutoffs)),
        "horizon_h": int(h),
        "step_size_h": int(step_size),
        "coverage": {"ds_min": str(ds_min), "ds_max": str(ds_max)},
        "end_test": str(end_test),
        "test_hours": int(test_hours),
        "unique_ids": sorted(map(str, cv_df["unique_id"].unique())),
        "positive_forecasts_enforced": bool(positive_forecasts),
    }

