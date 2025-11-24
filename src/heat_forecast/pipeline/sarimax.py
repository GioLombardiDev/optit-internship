from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Mapping, Optional, Sequence, List, Union, Callable, Tuple, Iterable
import logging
from tqdm.notebook import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from contextlib import nullcontext

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from statsforecast import StatsForecast  
from statsforecast.models import ARIMA, AutoARIMA 
from statsforecast.arima import arima_string
from statsmodels.tsa.seasonal import MSTL 
from utilsforecast.feature_engineering import fourier, pipeline
from functools import partial
from coreforecast.scalers import boxcox, boxcox_lambda

from ..utils.transforms import make_transformer, transform_column, get_lambdas, make_is_winter
from ..utils.datasplit import generate_sets
from ..utils.decomposition import decompose_annual_seasonality, remove_annual_component, add_annual_component
from ..utils.plotting import configure_time_axes

_LOGGER = logging.getLogger(__name__)

def piecewise_sigmoid(x, mid, lower, upper, k_left=0.5, k_right=None):
    """Two logistic curves joined at x=mid. Continuous; derivative continuous if k_right=None."""
    if not (lower < mid < upper):
        raise ValueError("The parameters for temp_transform must satisfy lower < mid < upper")

    x = np.asarray(x)
    
    # pendenze: k_left scelto liberamente; k_right per continuità del 1° ordine
    if k_right is None:
        k_right = k_left * (mid - lower) / (upper - mid)

    # Left piece: for x <= mid
    left_piece = lower + 2*(mid - lower) / (1 + np.exp(-k_left * (x - mid)))
    
    # Right piece: for x > mid  
    right_piece = upper - 2*(upper - mid) / (1 + np.exp(k_right * (x - mid)))
    
    # Use numpy.where to select appropriate piece for each element
    result = np.where(x <= mid, left_piece, right_piece)
    
    return result.item() if result.ndim == 0 else result


@dataclass
class SARIMAXConfig:

    # target transformation 
    transform: Optional[str] = "boxcox_cold"
    lam_method: Optional[str] = "loglik"

    # How much data to use for training by default
    input_size: Optional[int] = 365 * 24  # Number of hours to consider for each training

    # for climate-based variables
    with_exog: bool = True
    exog_vars: Iterable[str] = field(default_factory=lambda: ['temperature'])
    lags_exog: Optional[Iterable[int]] = None
    days_averages: Optional[Iterable[int]] = field(default_factory=lambda: [1])
    hours_averages: Optional[Iterable[int]] = None
    max_lag_daily_agg: Optional[Iterable[int]] = None
    max_lag_hourly_agg: Optional[Iterable[int]] = None
    drop_hourly_exog: bool = False
    drop_warm: bool = True

    # fourier variables 
    k_week: int = 0
    k_week_only_when_cold: bool = False
    k_day: int = 0

    # temperature transformation & cold/warm separation 
    temp_transform: bool = False
    lower_asymptote: float = 0
    upper_asymptote: float = 24
    k: float = 0.15
    k_right: Optional[float] = None
    threshold: float = 13.5
    only_cold_data: bool = False  # If True, only use data where temperature <= threshold for training

    # peak hours separation 
    use_peak_hours: bool = False
    peak_hours: Optional[Iterable[int]] = None

    # for SARIMAX 
    sarima_kwargs: Mapping[str, Any] = field(default_factory=lambda: 
                                            {
                                                'order': (1, 0, 0),
                                                'season_length': 24,
                                                'seasonal_order': (1, 1, 1)
                                            })

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)
    
    def __post_init__(self):
        """Post-initialization checks and transformations."""
        if (self.transform is not None) and (not isinstance(self.transform, str)):
            raise ValueError("transform must be a string (or None).")
        if self.transform is None:
            self.transform = "none"
        if not isinstance(self.sarima_kwargs, Mapping):
            raise ValueError("sarima_kwargs must be a Mapping (e.g., dict).")
        if not ['order', 'season_length', 'seasonal_order'] \
                == sorted(self.sarima_kwargs.keys()):
            raise ValueError("sarima_kwargs must contain 'order', 'season_length', and 'seasonal_order' keys.")
        if not isinstance(self.sarima_kwargs['order'], tuple):
            self.sarima_kwargs['order'] = tuple(self.sarima_kwargs['order'])
        if not isinstance(self.sarima_kwargs['seasonal_order'], tuple):
            self.sarima_kwargs['seasonal_order'] = tuple(self.sarima_kwargs['seasonal_order'])
        if self.input_size is not None and ((not isinstance(self.input_size, int)) or self.input_size <= 0):
            raise ValueError("input_size must be a positive integer (or None), representing the size of the training window in hours.")

class SARIMAXPipeline:
    def __init__(
            self, 
            target_df: pd.DataFrame, 
            config: SARIMAXConfig,
            aux_df: Optional[pd.DataFrame] = None, 
            logger: Optional[logging.Logger] = None
        ):
        self._target_df = target_df.copy()
        self._aux_df = aux_df.copy() if aux_df is not None else None
        self.config = config
        self._logger = logger or _LOGGER

        # internal attributs initialized during data preparation
        self._data_start_ds = None
        self._data_end_ds = None
        self._data = None
        self._lambdas = None

        # internal attributes initialized during fitting
        self._sf = None
        self._start_train = None
        self._end_train = None
        self.alias = None

        # internal attributes initialized during test
        self._start_test = None
        self._end_test = None
        self._forecast_df = None

        # internal attributes initialized during optimal sarima search
        self._last_search_result = None

        # internal attributes initialized during cross validation
        self._last_cv_metadata = None

        # ------------ Check coherence of params ------------
        # NB: although most of the data preparation steps can handle multiple unique_ids, the 
        # target transformation part requires a single unique_id.

        # Make sure a single id is used
        if not target_df['unique_id'].nunique() == 1:
            raise ValueError("The target_df must contain a single unique_id. Found: {}".format(target_df['unique_id'].unique()))
        
        # Make sure aux_df is provided if with_exog is True
        if self.config.with_exog and aux_df is None:
            raise ValueError("aux_df must be provided when with_exog is True.")
        
        # Make sure the aux_df has the same unique_id
        if aux_df is not None:
            if not aux_df['unique_id'].nunique() == 1:
                raise ValueError("The aux_df must contain a single unique_id. Found: {}".format(aux_df['unique_id'].unique()))
            if not aux_df['unique_id'].unique()[0] == target_df['unique_id'].unique()[0]:
                raise ValueError("The unique_id in aux_df must match the one in target_df. Found: {} and {}".format(
                    aux_df['unique_id'].unique()[0], target_df['unique_id'].unique()[0]
                ))
        id = target_df['unique_id'].unique()[0]
        if not id in ['F1', 'F2', 'F3', 'F4', 'F5']:
            raise ValueError(f"Invalid unique_id: {id}. Expected one of ['F1', 'F2', 'F3', 'F4', 'F5'].")
        
        # Save unique_id and target merged with aux
        self._unique_id = id
        if aux_df is not None:
            self._target_plus_aux_df = target_df.merge(
                aux_df, 
                on=['unique_id', 'ds'], 
                how='inner', 
            )
        else:
            self._target_plus_aux_df = target_df.copy()

        
        if self.config.with_exog:
            # When using climate exog, temperature must be included
            if 'temperature' not in self.config.exog_vars:
                raise ValueError("exog_vars must include 'temperature' when with_exog is True.")

            # Make sure exog vars exist in aux_df
            missing_vars = [var for var in self.config.exog_vars if var not in aux_df.columns]
            if missing_vars:
                raise ValueError(f"The following exogenous variables are missing in aux_df: {missing_vars}")

    def prepare_data(
            self, 
            *,
            data_start_ds: Optional[pd.Timestamp] = None,
            data_end_ds: Optional[pd.Timestamp] = None,
        ):
        
        # ------------ Validate split dates ------------ 
        self._logger.debug("Validating split dates: data_start_ds=%s, data_end_ds=%s", data_start_ds, data_end_ds)
        if data_start_ds is None:
            data_start_ds = self._target_plus_aux_df['ds'].min().ceil('D')
        if data_end_ds is None:
            last_available_at_T2300 = (self._target_plus_aux_df['ds'].max() - pd.Timedelta(hours=23)).floor('D') + pd.Timedelta(hours=23)
            data_end_ds = last_available_at_T2300
        if not isinstance(data_start_ds, pd.Timestamp) or not isinstance(data_end_ds, pd.Timestamp):
            raise ValueError("data_start_ds and data_end_ds must be pd.Timestamp objects.")
        if data_start_ds >= data_end_ds:
            raise ValueError("data_start_ds must be before data_end_ds.")
        if not data_start_ds in self._target_plus_aux_df['ds'].tolist():
            raise ValueError(f"data_start_ds ({str(data_start_ds)}) must be in the target_df['ds'] column and aux_df['ds'] column (if aux_df is provided).")
        if not data_end_ds in self._target_plus_aux_df['ds'].tolist():
            raise ValueError(f"data_end_ds ({str(data_end_ds)}) must be in the target_df['ds'] column and aux_df['ds'] column (if aux_df is provided).")
        self._data_start_ds = data_start_ds
        self._data_end_ds = data_end_ds
        data_raw_df = self._target_plus_aux_df[
            (self._target_plus_aux_df['ds'] >= self._data_start_ds) & 
            (self._target_plus_aux_df['ds'] <= self._data_end_ds)
        ]

        # ------------ Add the specified climate exog variables ------------ 
        self._logger.debug("Selecting climate exogenous variables.")
        if self.config.with_exog:

            # Select the climate exog
            data_raw_df = data_raw_df[['unique_id', 'ds', 'y'] + self.config.exog_vars].copy()

        # ------------ Add 'is_cold' variable ------------
            self._logger.debug("Performing seasonal split based on temperature.")
            
            # Add columns 'temp_fewdays_avg' containing rolling window temperature averages
            data_raw_df = self._add_temperature_rolling(data_raw_df)
            data_raw_df['is_cold'] = data_raw_df['temp_fewdays_avg'].apply(lambda x: True if x <= self.config.threshold else False)
            data_raw_df = data_raw_df.drop(columns=['temp_fewdays_avg'])

        # ------------ Temperature transformation ------------
            self._logger.debug("Applying temperature transformation if configured.")
            # Transform temperature using the provided function
            if self.config.temp_transform:
                piecewise_sigmoid_kwgs = {
                    'mid': self.config.threshold,
                    'lower': self.config.lower_asymptote,
                    'upper': self.config.upper_asymptote,
                    'k_left': self.config.k,
                    'k_right': self.config.k_right
                }
                data_raw_df['temperature'] = piecewise_sigmoid(data_raw_df['temperature'].to_numpy(), **piecewise_sigmoid_kwgs)

        # ------------ Add climate exog averages ------------
            self._logger.debug("Adding climate exog averages (days and hours).")
            # add days_averages
            if self.config.days_averages is not None:

                # Make sure days_averages is an iterable of positive integers
                if not isinstance(self.config.days_averages, Iterable) or not all(isinstance(d, int) and d>0 for d in self.config.days_averages):
                    raise ValueError("days_averages must be an iterable of positive integers (or None).")

                for d in self.config.days_averages:
                    for col in self.config.exog_vars:
                        # Calculate the rolling average for the specified number of days
                        data_raw_df[f'{col}_{d}days_avg'] = (
                            data_raw_df
                            .groupby('unique_id')[col]
                            .rolling(window=24*d, min_periods=1)
                            .mean()
                            .reset_index(0, drop=True)
                        )
            
            # add hours_averages
            if self.config.hours_averages is not None:

                # Make sure hours_averages is an iterable of positive integers
                if not isinstance(self.config.hours_averages, Iterable) or not all(isinstance(h, int) and h>0 for h in self.config.hours_averages):
                    raise ValueError("hours_averages must be an iterable of positive integers (or None).")
                
                for hours in self.config.hours_averages:
                    for col in self.config.exog_vars:
                        # Calculate the rolling average for the specified number of hours
                        data_raw_df[f'{col}_{hours}hours_avg'] = (
                            data_raw_df
                            .groupby('unique_id')[col]
                            .rolling(window=hours, min_periods=1)
                            .mean()
                            .reset_index(0, drop=True)
                        )
            
        # ------------ Drop raw climate exog if requested ------------
            self._logger.debug("Dropping raw climate exog variables.")
            if self.config.drop_hourly_exog:
                data_raw_df = data_raw_df.drop(columns=self.config.exog_vars)

        # ------------ Add lagged climate exog ------------
            self._logger.debug("Adding lagged climate exog variables.")
            # Add lags of the hourly climate exog
            if (not self.config.drop_hourly_exog) and (self.config.lags_exog is not None):
                
                # Make sure lags_exog is an iterable of positive integers
                if not isinstance(self.config.lags_exog, Iterable) or not all(isinstance(lag, int) and lag > 0 for lag in self.config.lags_exog):
                    raise ValueError("lags_exog must be an iterable of positive integers (or None).")
                
                # Build the lagged columns in two dicts
                lagged = {}

                for col in self.config.exog_vars:
                    g_data = data_raw_df.groupby("unique_id")[col]

                    for lag in self.config.lags_exog:
                        lag_name = f"{col}_lag{lag}"
                        lagged[lag_name] = g_data.shift(lag)

                # Concatenate once → one new block, no fragmentation
                data_raw_df = pd.concat(
                    [data_raw_df, pd.DataFrame(lagged, index=data_raw_df.index)],
                    axis=1
                )
            
            # Add lags for the days-aggregated exogenous variables
            self._logger.debug("Adding lags for the days-aggregated exogenous variables.")
            if (self.config.days_averages is not None) and (self.config.max_lag_daily_agg is not None): 

                # Make sure max_lag_daily_agg is an iterable of non-negative integers
                if not isinstance(self.config.max_lag_daily_agg, Iterable) or not all(isinstance(lag, int) and lag >= 0 for lag in self.config.max_lag_daily_agg):
                    raise ValueError("max_lag_daily_agg must be an iterable of non-negative integers (or None).")
                # Make sure days_averages and max_lag_daily_agg have the same length
                if len(self.config.days_averages) != len(self.config.max_lag_daily_agg):
                    raise ValueError("days_averages and max_lag_daily_agg must have the same length.")

                # Build the lagged columns in two dicts
                lagged = {}

                for d, lag in zip(self.config.days_averages, self.config.max_lag_daily_agg):
                    for col in self.config.exog_vars:
                        g_data = data_raw_df.groupby("unique_id")[f"{col}_{d}days_avg"]

                        for lag_i in range(1, lag + 1):
                            lag_name = f"{col}_{d}days_avg_lag{lag_i}"
                            lagged[lag_name] = g_data.shift(lag_i*24)
                
                # Concatenate once → one new block, no fragmentation
                data_raw_df = pd.concat(
                    [data_raw_df, pd.DataFrame(lagged, index=data_raw_df.index)],
                    axis=1
                )
            
            # Add lags for the hours-aggregated exogenous variables
            self._logger.debug("Adding lags for the hours-aggregated exogenous variables.")
            if (self.config.hours_averages is not None) and (self.config.max_lag_hourly_agg is not None): 

                # Make sure max_lag_hourly_agg is an iterable of non-negative integers
                if not isinstance(self.config.max_lag_hourly_agg, Iterable) or not all(isinstance(lag, int) and lag >= 0 for lag in self.config.max_lag_hourly_agg):
                    raise ValueError("max_lag_hourly_agg must be an iterable of non-negative integers (or None).")
                # Make sure hours_averages and max_lag_hourly_agg have the same length
                if len(self.config.hours_averages) != len(self.config.max_lag_hourly_agg):
                    raise ValueError("hours_averages and max_lag_hourly_agg must have the same length.")

                # Build the lagged columns
                lagged = {}

                for hours, lag in zip(self.config.hours_averages, self.config.max_lag_hourly_agg):
                    for col in self.config.exog_vars:
                        g_data = data_raw_df.groupby("unique_id")[f"{col}_{hours}hours_avg"]

                        for lag_i in range(1, lag + 1):
                            lag_name = f"{col}_{hours}hours_avg_lag{lag_i}"
                            lagged[lag_name] = g_data.shift(lag_i)
                
                # Concatenate once → one new block, no fragmentation
                data_raw_df = pd.concat(
                    [data_raw_df, pd.DataFrame(lagged, index=data_raw_df.index)],
                    axis=1
                )

            # Clean up rows made incomplete by shifting
            data_raw_df.dropna(inplace=True)

            # Force block consolidation
            data_raw_df = data_raw_df.copy()


        else:
            
            # Do nothing, keep data_raw_df as it was 
            self._logger.debug("No exogenous variables are used, keeping data_raw_df as it is.")

        # ------------ Add fourier variables ------------ 
        self._logger.debug("Adding Fourier features to the training and validation sets.")
        features = [
            partial(fourier, season_length=168,  k=self.config.k_week),
            partial(fourier, season_length=24,  k=self.config.k_day),
        ]
        data_raw_df, _ = pipeline(
            df=data_raw_df,
            features=features,
            freq="h",
        )
        
        # ------------ Separate all variables based on cold weather ------------
        self._logger.debug("Separating exogenous and fourier variables based on cold weather.")
        fourier_features = (
            data_raw_df.filter(like="sin").columns.tolist()
            + data_raw_df.filter(like="cos").columns.tolist()
        )
        climate_features = [col for col in data_raw_df.columns if col.startswith(tuple(self.config.exog_vars))]

        if self.config.with_exog:
            cols_to_winterize = fourier_features + climate_features # Basically, select all columns (except unique_id, ds, y)

            # Build new columns in dicts – one pass, no in-place adds
            cold, warm = {}, {}

            is_cold = data_raw_df["is_cold"].to_numpy(bool)

            for col in cols_to_winterize:
                col_data = data_raw_df[col].to_numpy()

                cold[f"{col}_cold"] = np.where(is_cold, col_data, 0)
                warm[f"{col}_warm"] = np.where(~is_cold, col_data, 0)

            if self.config.drop_warm:
                warm = {k: v for k, v in warm.items() 
                        if k in ['temperature_1days_avg_warm', 'temperature_warm']}

            # Concatenate once per DataFrame
            data_raw_df = pd.concat(
                [
                    data_raw_df.drop(columns=cols_to_winterize),
                    pd.DataFrame(cold, index=data_raw_df.index),
                    pd.DataFrame(warm, index=data_raw_df.index),
                ],
                axis=1,
            )

            # Delete warm weekly Fourier terms if k_week_only_when_cold is True
            if self.config.k_week_only_when_cold and self.config.k_week > 0:
                fourier_weekly_warm = [
                    c for c in data_raw_df.columns
                    if c.startswith(('sin_', 'cos_')) and '_168_' in c and c.endswith('_warm')
                ]
                data_raw_df = data_raw_df.drop(columns=fourier_weekly_warm)
            
            # Convert is_cold to int
            data_raw_df['is_cold'] = data_raw_df['is_cold'].astype(int)

            # Force block consolidation
            data_raw_df = data_raw_df.copy()

        # ------------ Separate climate-based, cold and high-frequency variables based on peak hours ------------
            self._logger.debug("Separating climate-based and high-frequency variables based on peak hours.")
            # create is_peak columns
            if self.config.use_peak_hours:
                # Add is_peak column based on peak_hours mapping and weekday condition
                peak_hours = self._generate_peak_hours()

                data_raw_df['is_peak'] = data_raw_df.apply(
                    lambda row: row['ds'].hour in peak_hours and row['ds'].dayofweek < 5,  # Only weekdays
                    axis=1
                )

                # Create a list with all columns to separate between peak/non-peak hours
                climate_cols = [col for col in data_raw_df.columns if col.startswith(tuple(self.config.exog_vars))] # all climate-based variables
                high_freq_climate_cold_cols = [col for col in climate_cols if ('days_avg' not in col) and ('_warm' not in col)]  

                # Build new columns in dicts – one pass, no in-place adds
                cold_peak, cold_other = {}, {}

                mask = data_raw_df["is_peak"].to_numpy(bool)

                for col in high_freq_climate_cold_cols:
                    col_data = data_raw_df[col].to_numpy()

                    cold_peak[f"{col}_peak"]     = np.where(mask,  col_data, 0)
                    cold_other[f"{col}_offpeak"] = np.where(~mask, col_data, 0)

                # Concatenate once per DataFrame
                data_raw_df = pd.concat(
                    [
                        data_raw_df.drop(columns=high_freq_climate_cold_cols),
                        pd.DataFrame(cold_peak, index=data_raw_df.index),
                        pd.DataFrame(cold_other, index=data_raw_df.index),
                    ],
                    axis=1,
                )

                # Force block consolidation 
                data_raw_df = data_raw_df.copy()

                # Convert is_peak to int
                data_raw_df['is_peak'] = data_raw_df['is_peak'].astype(int)

        # ------------ transform the target variable in train_df ------------
        self._logger.debug("Transforming the target variable.") 
        
        self._data = data_raw_df # Need to save a first version of the prepared data: it's used in _transform_target
        data_raw_df = self._transform_target(data_raw_df, forward=True)

        self._data = data_raw_df

    def describe_prepared_data(self) -> Dict[str, Any]:
        """Return a description of the prepared data."""
        if self._data is None:
            raise ValueError("Data has not been prepared yet. Call prepare_data() first.")
        
        def _select_climate_based_columns(df: pd.DataFrame) -> List[str]:
            """Select climate-based columns from the DataFrame."""
            return [col for col in df.columns if col.startswith(tuple(self.config.exog_vars))] \
                + [col for col in df.columns if col in ['is_cold', 'is_peak']]
        
        data_descr = {
            'Unique id': self._unique_id,
            'Using target transform': self.config.transform ,
            'Using a temperature transform?': self.config.temp_transform,
            'data': {
                'shape': self._data.shape,
                'columns': {
                    'basic': [e for e in self._data.columns.tolist() if e in ['unique_id', 'ds', 'y']],
                    'climate-based': _select_climate_based_columns(self._data),
                    'fourier': [e for e in self._data.columns.tolist() if e.startswith('sin') or e.startswith('cos')],
                },
                'start_date': str(self._data['ds'].min()),
                'end_date': str(self._data['ds'].max()),
            },
        }

        import yaml

        class ListInlineDumper(yaml.SafeDumper):
            pass

        # tuple‐inlining
        def _repr_tuple_flow(dumper, data):
            text = '(' + ', '.join(str(x) for x in data) + ')'
            # emit as a plain scalar (unquoted, if YAML allows)
            return dumper.represent_scalar('tag:yaml.org,2002:str', text, style='')
        ListInlineDumper.add_representer(tuple, _repr_tuple_flow)

        yaml_str = yaml.dump(
            data_descr,
            Dumper=ListInlineDumper,
            sort_keys=False,
            indent=4,
        )
        # log at INFO level (or DEBUG if you prefer)
        self._logger.info("Prepared data description:\n%s", yaml_str)

    def fit(
        self,
        *,
        end_train: pd.Timestamp,
        start_train: pd.Timestamp | None = None,
        n_jobs: int = 2,
        silent: bool = False,
        alias: str = 'SARIMAX',
    ) -> None:
        """Fit the SARIMAX model."""
        self.alias = alias
        
        # ------------ Generate train set ------------
        self._train_from_data(end_train, start_train=start_train)
        self._logger.debug("Fitting SARIMAX model: start_train=%s, end_train=%s, n_jobs=%d", self._start_train, self._end_train, n_jobs)
        if not silent:
            self._logger.info("Fitting SARIMAX model: start_train=%s, end_train=%s", self._start_train, self._end_train)

        # ------------ Initialize SARIMA model and fit ------------
        sarimax_model = ARIMA(
            order=self.config.sarima_kwargs.get('order'),
            season_length=self.config.sarima_kwargs.get('season_length'),
            seasonal_order=self.config.sarima_kwargs.get('seasonal_order'),
            alias=alias,
        )
        sf = StatsForecast(
            models=[sarimax_model],
            freq="h",
            n_jobs=n_jobs, # Use all available CPU cores
            verbose=not silent,  # Print progress messages
        )
        sf.fit(df=self._train_df)
        self._sf = sf

        if not silent:
            self._logger.info("✓ SARIMAX model fitted successfully.")
    
    def _train_from_data(
            self, 
            end_train: pd.Timestamp,
            start_train: pd.Timestamp | None = None,
            plot_cold_data: bool = False
        ):
        """Build self._train_df between start_train and end_train."""

        # --- derive window ---
        if start_train is None:
            if self.config.input_size is None:
                start_train = self._data['ds'].min().ceil('D')
            else:
                start_train = end_train - pd.Timedelta(hours=self.config.input_size) + pd.Timedelta(hours=1)
        self._start_train = start_train
        self._end_train = end_train

        # --- basic validation ---
        if not isinstance(start_train, pd.Timestamp) or not isinstance(end_train, pd.Timestamp):
            raise ValueError("start_train and end_train must be pd.Timestamp objects.")
        if start_train >= end_train:
            raise ValueError("start_train must be before end_train.")
        if self._data is None:
            raise ValueError("Data has not been prepared yet. Call prepare_data() first.")
        data = self._data.sort_values('ds').set_index('ds')
        if start_train not in data.index or end_train not in data.index:
            raise ValueError(
                "start_train and end_train must align with prepared hourly timestamps "
                f"(got start_train={start_train}, end_train={end_train})."
            )
        
        if self.config.only_cold_data:
            # --------- Filter the training data to include only cold days ---------
            # Filter only cold days within the specified range
            window = data.loc[start_train:end_train].reset_index()
            cold_df = window.loc[window['is_cold'] == 1].copy()
            if cold_df.empty:
                raise ValueError(
                    "No cold hours found within the specified training window. "
                    "Increase the window or relax the cold threshold."
                )

            # Plot original data
            if plot_cold_data:
                expected = pd.date_range(start=cold_df['ds'].min(), end=cold_df['ds'].max(), freq='h')
                actual = pd.Series(1, index=cold_df['ds'])
                missing = expected.difference(actual.index)
                plt.figure(figsize=(12, 4))
                plt.plot(cold_df['ds'], cold_df['y'], label='Cold Hours')
                for ts in missing: plt.axvline(ts, linestyle='--', alpha=0.4)
                plt.title('Cold Hours with Missing Timestamps')
                plt.legend(); plt.tight_layout(); plt.show()

            # --- weekly grouping: full cold weeks only (168h) ---
            cold_df = cold_df.sort_values('ds')
            cold_df['week_start'] = cold_df['ds'].dt.to_period('W-SUN').apply(lambda r: r.start_time) # Round to week start (Monday 00:00)

            valid_weeks: list[tuple[pd.Timestamp, pd.DataFrame]] = []
            for wk, grp in cold_df.groupby('week_start'):
                if len(grp) != 168:
                    continue
                expected_hours = pd.date_range(start=wk, periods=168, freq='h')
                actual_hours = grp['ds'].sort_values().reset_index(drop=True)
                if actual_hours.equals(expected_hours):
                    valid_weeks.append((wk, grp.sort_values('ds')))

            if not valid_weeks:
                raise ValueError(
                    "No full (168h) cold weeks found in the training window. "
                    "Try a different window or threshold."
                )

            # --- stitch contiguous week blocks (>= min_weeks weeks) ---
            min_weeks = 3
            valid_weeks.sort(key=lambda x: x[0])
            blocks, cur = [], []
            for wk, grp in valid_weeks:
                if not cur: cur = [(wk, grp)]; continue
                prev_wk, _ = cur[-1]
                if wk == prev_wk + pd.Timedelta(weeks=1):
                    cur.append((wk, grp))
                else:
                    if len(cur) >= min_weeks: blocks.append(cur)
                    cur = [(wk, grp)]
            if len(cur) >= min_weeks: # Final block check
                blocks.append(cur)

            if not blocks:
                raise ValueError(
                    "Found full cold weeks but not ≥3 contiguous weeks. "
                    "Widen the window to include longer cold stretches."
                )
            
            # --- concatenate accepted blocks ---
            self._train_df = (
                pd.concat([grp for block in blocks for (_, grp) in block], ignore_index=True)
                .sort_values('ds')
                .reset_index(drop=True)
            )

            # --- reindex ds to continuous hourly grid (keep weekday alignment) ---
            original_start_dow = int(self._train_df['ds'].iloc[0].dayofweek)  # Mon=0
            base_start = pd.Timestamp('2000-01-01 00:00')                      # Sat
            days_to_shift = (original_start_dow - base_start.dayofweek) % 7
            aligned_start = base_start + pd.Timedelta(days=days_to_shift)
            self._train_df['ds'] = pd.date_range(start=aligned_start, periods=len(self._train_df), freq='h')
            self._train_df.drop(columns=['week_start'], inplace=True)

        else:
            # regular (cold+warm) training window
            self._train_df = data.loc[start_train:end_train].reset_index()

    def predict(
            self, 
            h: int, 
            levels: List[int] | None = None
        ) -> pd.DataFrame:
        """Make predictions using the fitted SARIMAX model."""
        if self._sf is None:
            raise ValueError("Model has not been fitted yet.")

        # Validate forecast horizon
        if not isinstance(h, int) or h <= 0:
            raise ValueError("h must be a positive integer representing the forecast horizon in hours.")
        self._start_test = self._end_train + pd.Timedelta(hours=1)
        self._end_test = self._end_train + pd.Timedelta(hours=h)
        if (self._start_test not in self._data['ds'].tolist()) or (self._end_test not in self._data['ds'].tolist()):
            raise ValueError("Forecast horizon does not fall within the prepared data's 'ds' column. " \
                             f"Test ends at {self._end_test}, but the last available date is {self._data['ds'].max()}."
                             "You can reprepare the data with a larger end date, then .predict() again (without refitting).")
        
        test_df = self._data[
            (self._data['ds'] >= self._start_test) & (self._data['ds'] <= self._end_test)
        ]

        self._X_df = test_df.drop(columns=['y'])

        if self.config.only_cold_data:
            # In this case, we need to call .forward() because we trained on a different dataset
            return self.forward(
                context_end=self._end_train,
                h=h,
                context_start=self._start_train,
                levels=levels
            )
        
        forecasts_raw = self._sf.predict(h=h, X_df=self._X_df, level=levels)

        forecasts = self._transform_target(forecasts_raw, forward=False)
        self._forecast_df = forecasts
        return self._forecast_df

    def forward(
        self,
        context_end: pd.Timestamp,
        h: int,
        context_start: pd.Timestamp | None = None,
        levels: Optional[List[int]] = None,
    ):
        """
        Generate forecasts using the fitted SARIMAX model,
        using a specified context window but without refitting the model.

        Parameters
        ----------
        context_end : pd.Timestamp
            End of the input context window (last available observation).
        h : int
            Forecast horizon (same frequency as the data).
        context_start : pd.Timestamp | None, optional
            Start of the input context window. If None, defaults to the first available date in the prepared data.
        levels : List[int], optional
            Confidence levels for forecast intervals.
        """
        if self._sf is None:
            raise ValueError("Model has not been fitted yet.")
        if context_start is None:
            context_start = self._data['ds'].min().ceil('D')
        
        # Validate inputs
        if not isinstance(context_start, pd.Timestamp) or not isinstance(context_end, pd.Timestamp):
            raise ValueError("context_start and context_end must be pd.Timestamp objects.")
        if context_start >= context_end:
            raise ValueError("context_start must be before context_end.")
        if self._data is None:
            self._logger.warning("Data has not been prepared yet. Preparing data now.")
            self.prepare_data()
        if not context_start in self._data['ds'].tolist():
            raise ValueError("context_start must be in the prepared data's 'ds' column.")
        if not context_end in self._data['ds'].tolist():
            raise ValueError("context_end must be in the prepared data's 'ds' column.")
        if not isinstance(h, int) or h <= 0:
            raise ValueError("h must be a positive integer representing the forecast horizon in hours.")
        if not context_end + pd.Timedelta(hours=h) in self._data['ds'].tolist():
            raise ValueError("Forecast horizon does not fall within the prepared data's 'ds' column.")
        if levels is not None and (not isinstance(levels, Iterable) or not all(isinstance(l, int) for l in levels)):
            raise ValueError("levels must be an iterable of integers representing confidence levels.")

        # Prepare input df
        input_df = self._data[
            (self._data['ds'] >= context_start) & (self._data['ds'] <= context_end)
        ]

        # Prepare input series and input exogenous variables
        y = input_df['y'].to_numpy()
        X_df = input_df.drop(columns=['unique_id', 'ds', 'y']) if self.config.with_exog else None
        X = X_df.to_numpy() if X_df is not None else None

        # Prepare output df
        output_df = self._data[
            (self._data['ds'] > context_end) & (self._data['ds'] <= context_end + pd.Timedelta(hours=h))
        ].copy()

        # Prepare output exog variables
        X_feature_df = output_df.drop(columns=['unique_id', 'ds', 'y']) if self.config.with_exog else None
        X_feature = X_feature_df.to_numpy() if X_feature_df is not None else None

        sarima_model = self._sf.fitted_[0, 0] # first model for the first (and only) unique_id
        fc_dict = sarima_model.forward(
            y=y,
            h=h,
            X=X,
            X_future=X_feature,
            level=levels,
        )

        # Turn dict into a DataFrame
        fc_df = pd.DataFrame(fc_dict)
        
        # Add unique_id and ds columns
        if not hasattr(self, 'alias'):
            raise ValueError("Alias not set. Make sure to call .fit() before .forward().")
        alias = self.alias
        fc_df['unique_id'] = self._unique_id
        fc_df.rename(columns={'mean': alias}, inplace=True)
        if levels:
            for L in levels:
                lo, hi = f'lo-{L}', f'hi-{L}'
                if lo in fc_df.columns:
                    fc_df.rename(columns={lo: f'{alias}-lo-{L}'}, inplace=True)
                if hi in fc_df.columns:
                    fc_df.rename(columns={hi: f'{alias}-hi-{L}'}, inplace=True)
        fc_df['ds'] = output_df['ds'].reset_index(drop=True)

        forecasts = self._transform_target(fc_df, forward=False)
        self._forecast_df = forecasts
        return self._forecast_df

    def cross_validation(
        self,
        *,
        h: int,
        test_size: int,
        end_test: Optional[pd.Timestamp] = None,
        step_size: int = 1,
        input_size: Optional[int] = None,
        levels: List[int] = None,
        refit: Union[bool, int] = True,
        n_jobs: int = 2,
        verbose: bool = True,
        alias: str = 'SARIMAX',
    ) -> pd.DataFrame:
        """
        Rolling-window cross-validation using this SARIMAX pipeline.

        Returns one forecast-ground-truth pair per (unique_id, window),
        with columns [unique_id, ds, y, <alias>, (<alias>-lo-<L>, <alias>-hi-<L>…), cutoff].
        """
        levels = list(levels or [])

        # sanity check
        if (test_size - h) % step_size != 0:
            raise ValueError("`test_size - h` must be a multiple of `step_size`")
        
        # Check if data has been prepared
        if self._data is None:
            if verbose:
                self._logger.warning("Data has not been prepared yet but `.cross_validation` was called. Prepared data now.")
            self.prepare_data()

        # Check if an input size was given in the Config, and if it was given now
        if input_size is not None:
            if not isinstance(input_size, int) or input_size <= 0:
                raise ValueError("input_size must be a positive integer representing the size of the training window in hours.")
            if self.config.input_size is not None:
                self._logger.warning(
                    "Both `input_size` and `self.config.input_size` are set. "
                    "Updating the config's input_size to the provided value."
                )
                self.config.input_size = input_size

        # figure out the final timestamp
        if end_test is None:
            # last available at T23:00 in self._data
            end_test = (self._data["ds"].max() - pd.Timedelta(hours=23)).floor('D') + pd.Timedelta(hours=23)
        else:
            if not isinstance(end_test, pd.Timestamp):
                raise ValueError("end_test must be a pd.Timestamp object.")
            if end_test not in self._data["ds"].tolist():
                raise ValueError(f"end_test ({end_test}) must be in the prepared data's 'ds' column.")

        # relative offsets for each window
        steps = list(range(-test_size, -h + 1, step_size))

        # Check that the train starts inside the prepared data range
        if self.config.input_size is not None:
            first_cutoff = end_test + pd.Timedelta(hours=steps[0])
            first_start_train = first_cutoff - pd.Timedelta(hours=self.config.input_size) + pd.Timedelta(hours=1)
            if first_start_train < self._data["ds"].min().ceil('D'):
                raise ValueError(
                    f"input_size ({self.config.input_size} hours) is too large for the given end test ({end_test}). "
                    f"Training would start at {first_start_train}, but the earliest available data in .prepared_data is {self._data['ds'].min().ceil('D')}."
                )

        all_results = []
        prev_pipeline = None
        t0 = pd.Timestamp.now()

        iterator = tqdm(steps, disable=not verbose, desc="CV windows", leave=True)
        for i, offset in enumerate(iterator):
            cutoff = end_test + pd.Timedelta(hours=offset)

            # build training window
            if self.config.input_size is not None:
                start_train = cutoff - pd.Timedelta(hours=self.config.input_size) + pd.Timedelta(hours=1)
            else:
                # Use all prepared data up to the cutoff
                start_train = self._data["ds"].min().ceil('D')
            end_train = cutoff

            # decide whether to refit
            do_fit = (
                i == 0
                or (isinstance(refit, int) and not isinstance(refit, bool) and i % refit == 0)
                or (refit is True)
            )

            if do_fit or prev_pipeline is None:
                # new pipeline for this window
                pipeline = SARIMAXPipeline(
                    target_df=self._target_df,
                    aux_df=self._aux_df,
                    config=self.config,
                    logger=self._logger,
                )

                # Use the already preparated data: redefine all internals changed in prepare_data()
                pipeline._data = self._data
                pipeline._lambdas = self._lambdas
                pipeline._data_start_ds = self._data_start_ds
                pipeline._data_end_ds = self._data_end_ds

                # Fit the pipeline
                pipeline.fit(
                    start_train=start_train,
                    end_train=end_train,
                    silent=True,  # no logging during fit
                    n_jobs=n_jobs, 
                    alias=alias, 
                )
                prev_pipeline = pipeline
            else:
                # reuse last fitted
                pipeline = prev_pipeline

            # forecast
            fc = pipeline.forward(
                context_start=start_train,   # or None to start from prepared min
                context_end=cutoff,
                h=h,
                levels=levels
            )
            # Sanity checks
            assert fc['ds'].min() == cutoff + pd.Timedelta(hours=1)
            assert fc['ds'].max() == cutoff + pd.Timedelta(hours=h)

            # build validation set for exactly h hours after cutoff
            mask = (
                (self._target_df["ds"] > cutoff)
                & (self._target_df["ds"] <= cutoff + pd.Timedelta(hours=h))
            )
            val_df = self._target_df.loc[mask, ["unique_id", "ds", "y"]]

            # merge forecast & truth
            lo_cols = [f"{alias}-lo-{L}" for L in levels]
            hi_cols = [f"{alias}-hi-{L}" for L in levels]
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
                "target_df_uids": self._target_df["unique_id"].unique().tolist(),
                "aux_df_uids": self._aux_df["unique_id"].unique().tolist() if self._aux_df is not None else [],
            },
            "pipeline_class": self.__class__.__name__,
            "cv_params": {
                "h": h,
                "test_size": test_size,
                "end_test": str(end_test or self._target_df["ds"].max()),
                "step_size": step_size,
                "input_size": self.config.input_size,
                "levels": levels,
                "refit": refit,
                "n_jobs": n_jobs,
                "alias": alias,
            },
            "run_timestamp": pd.Timestamp.now().isoformat(),
            "elapsed_seconds": (pd.Timestamp.now() - t0).total_seconds(),
        }

        return pd.concat(all_results, ignore_index=True)
    
    # ------------------------------------------------------------------
    # SEARCH OPTIMAL SARIMA MODEL WITH AUTO-ARIMA
    # ------------------------------------------------------------------
    def search_optimal_sarima_model(
        self,
        *,
        end_train: pd.Timestamp,
        start_train: pd.Timestamp | None = None,
        n_models: int = 60,
        n_jobs: int = 2,
        verbose: bool = True,
        compare_with_current_model: bool = True
    ) -> None:
        """
        Search for the optimal SARIMA model using AutoARIMA.

        Fits AutoARIMA to training data between `start_train` and `end_train`.
        Searches best SARIMA parameters based on AICc using stepwise search.

        After finding the best model, compares metric with the previous model 
        (stored in the pipeline), fitting it to the data if not already done.
        """

        self._logger.debug("Starting SARIMA search: start_train=%s, end_train=%s, n_jobs=%d", start_train, end_train, n_jobs)
        if verbose:
            self._logger.info("Starting search of optimal SARIMA with AutoARIMA...")
        
        if self._data is None:
            self._logger.warning("Data has not been prepared yet but `.search_optimal_sarima_model` was called. Preparing now...")
            self.prepare_data()

        self._train_from_data(end_train, start_train=start_train)

        sf_for_search = StatsForecast(
            models=[AutoARIMA(
                approximation=False, 
                season_length=self.config.sarima_kwargs.get('season_length', 24),
                start_p=1,
                start_q=0,
                start_P=1,
                start_Q=1,
                alias='AutoARIMA', 
                nmodels=n_models
            )],
            freq="h",
            n_jobs=n_jobs, 
            verbose=verbose,  # Print progress messages
        )

        sf_for_search.fit(
            df=self._train_df,
        )
        logging.info("✓ AutoARIMA fitting completed.")
        best_model = sf_for_search.fitted_[0, 0].model_
        if verbose:
            self._logger.info("✓ SARIMA model search completed.")
        self._logger.info("Best model found: %s", arima_string(best_model))

        # Collect metrics for both models
        index = ['log-likelihood', 'AIC', 'AICc', 'BIC']
        comparison = {
            'Best model': [
                best_model['loglik'],
                best_model['aic'],
                best_model['aicc'],
                best_model['bic'],
            ]
        }

        if compare_with_current_model:
            if verbose:
                self._logger.info("Fitting the previously selected model...")

            self.fit(
                start_train=start_train,
                end_train=end_train,
                n_jobs=n_jobs,
                silent=True,
                alias='SARIMAX'
            )
            previous_model = self._sf.fitted_[0, 0].model_

            comparison['Previous model'] = [
                previous_model['loglik'],
                previous_model['aic'],
                previous_model['aicc'],
                previous_model['bic'],
            ]

        df = pd.DataFrame(comparison, index=index)
        title = "Goodness of fit metrics for the best and previous models: " if compare_with_current_model \
            else "Goodness of fit metrics for the best found model: "
        self._logger.info("%s\n%s", title, df.to_string())

        result = {
            'best_model': arima_string(best_model),
            'comparison_table': df
        }
        self._last_search_result = result
        return result

    @property
    def prepared_data(self) -> pd.DataFrame:
        return self._data

    @property
    def lambdas(self) -> Dict[str, float]:
        return self._lambdas
    
    @property
    def forecast(self) -> pd.DataFrame:
        """Return the last computed forecast."""
        if self._forecast_df is None:
            raise AttributeError("No forecast available; call `predict()` or `forward()` first.")
        return self._forecast_df
    
    @property
    def last_cv_metadata(self) -> Dict[str, Any]:
        """Return the metadata for the last computed cv."""
        if self._last_cv_metadata is None:
            raise AttributeError("No metadata available; call `cross_validation()` first.")
        return self._last_cv_metadata
    
    @property
    def fitted_(self):
        """Return the fitted SARIMA model object."""
        series_index = 0 # using only one series
        model_index = 0 # using only one model
        if self._sf is None:
            raise AttributeError("Model has not been fitted yet. Call `fit()` first.")
        model_obj = self._sf.fitted_[series_index][model_index].model_
        return model_obj
    
    def _transform_target(self, df: pd.DataFrame, forward: bool = True) -> pd.DataFrame:
        """Transform the target variable in the DataFrame df."""

        if self.config.transform.startswith("boxcox"):

            # Estimate lambdas if not already done
            if self._lambdas is None:
                heat_lambdas_df = df[['ds','unique_id','y']].copy()

                if self.config.transform == "boxcox_winter":
                    is_winter = make_is_winter(self._unique_id)
                    winter_mask = heat_lambdas_df['ds'].apply(is_winter)
                    heat_lambdas_df = heat_lambdas_df[winter_mask]

                elif self.config.transform == "boxcox_cold":
                    if not self.config.with_exog:
                        raise ValueError("with_exog must be True when transform is 'boxcox_cold'.")
                    heat_lambdas_df = heat_lambdas_df[self._data['is_cold'].to_numpy(bool)]

                self._lambdas = (
                    heat_lambdas_df
                    .groupby("unique_id")["y"]
                    .apply(lambda y: boxcox_lambda(y.to_numpy(), method=self.config.lam_method, season_length=365))
                    .to_dict()
                )

            _TRANSFORM = "boxcox"
        
        else:
            _TRANSFORM = self.config.transform

        if forward:
            to_transform = ['y']
        else:
            if not hasattr(self, 'alias'):
                raise ValueError("Alias not set. Make sure to call .fit() before .predict()/.forward().")
            to_transform = [col for col in df.columns if self.alias in col]

        # Apply the transformation
        for c in to_transform:
            fwd = make_transformer(_TRANSFORM, c, self._lambdas, inv=not forward)
            df[c] = transform_column(df, fwd)

        return df

    def _generate_peak_hours(self) -> List[int]:
        """Create a mapping list of integers representing peak hours, based on unique_id."""
        if not self.config.use_peak_hours:
            self._logger.debug("Peak hours are not used but _generate_peak_hours was called, returning an empty list.")
            return []

        if self.config.peak_hours is None:
            peaks = {
                'F1': [5, 6],
                'F2': [6],
                'F3': [7],
                'F4': [5],
                'F5': [6, 7],
            }
            return peaks.get(self._unique_id, [])
    
        if not isinstance(self.config.peak_hours, Iterable) or not all(isinstance(h, int) and 0 <= h < 24 for h in self.config.peak_hours):
            raise ValueError("peak_hours must be an iterable of integers between 0 and 23 (or None).")
                
        return self.config.peak_hours

    def _add_temperature_rolling(
            self,
            raw_df: pd.DataFrame,
            n_days: int = 4
    ):
        """Add rolling average of temperature to raw_df for the last few days."""
        ta_df = self._target_plus_aux_df
        window_hours = 24 * n_days
        start_ds = raw_df['ds'].min()
        end_ds = raw_df['ds'].max()
        slice = ta_df[(ta_df['ds'] >= start_ds - pd.Timedelta(hours=window_hours)) & (ta_df['ds'] <= end_ds)]

        rolling = (
            slice
            .set_index('ds')
            .groupby('unique_id')['temperature']
            .rolling(window=f'{window_hours}h', min_periods=1)
            .mean()
            .reset_index()
            .rename(columns={'temperature': 'temp_fewdays_avg'})
        )
        new_df = raw_df.merge(
            rolling, on=['unique_id', 'ds'], how='left'
        )

        return new_df