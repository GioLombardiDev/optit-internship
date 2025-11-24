from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.nonparametric.smoothers_lowess import lowess


# -------------------------------------------------------------------------------
# FUNCTIONS TO HANDLE ANNUAL SEASONALITY
# -------------------------------------------------------------------------------

def decompose_annual_seasonality(
    decomposition_method: str,  # “CD” for classical decomposition or “STL” for seasonal–trend–loess
    df: pd.DataFrame,           # DataFrame with columns 'ds', target_col, and 'unique_id' (daily frequency)
    target_col: str,            # Name of the column to decompose
    robust: bool = False,       # If True and using STL, uses a robust fitting
    seasonal_STL: int = 11,     # Seasonal window length for STL (must be odd)
    smooth_loess: bool = False, # Whether to apply LOESS smoothing to the seasonal component
    window_loess: int = 60,     # Approximate span (in points) for LOESS smoothing
    data_in_dcmp: bool = False  # Whether to include the original data in the output DataFrame
) -> Dict[Any, pd.DataFrame]:
    """
    Decompose each time series in the DataFrame into trend, seasonal, and residual components.
    Supports either classical additive decomposition ("CD") or STL decomposition ("STL"),
    with optional LOESS smoothing of the seasonal component.

    Parameters
    ----------
    decomposition_method : {'CD', 'STL'}
        Decomposition method to use: either 'CD' for classical decomposition or 'STL' for seasonal-trend-loess.
    df : pandas.DataFrame
        Daily-frequency data containing:
        - 'ds'        : date or datetime index
        - target_col  : numeric values to decompose
        - 'unique_id' : identifier for each separate time series
    target_col : str
        Column name in `df` whose values will be decomposed.
    robust : bool, default=False
        Only for STL: if True, uses robust fitting to reduce the influence of outliers.
    seasonal_STL : int, default=11
        Only for STL: length of the seasonal LOESS window. Must be odd; higher values produce smoother seasonals.
    smooth_loess : bool, default=False
        If True, apply an additional LOESS smoothing pass to the extracted seasonal component.
    window_loess : int, default=60
        Number of points (as a fraction of series length) used in LOESS when `smooth_loess=True`.
    """

    # --------- Set parameters for decomposition ---------
    if decomposition_method == "CD":
        period_CD = 365          # Period for seasonal decomposition
        extrapolate_CD = 'freq'  # Extrapolate trend to fill gaps
    elif decomposition_method == "STL":
        period_STL = 365         # Period for STL decomposition
        robust_STL = robust      # Whether to use robust STL
    elif decomposition_method != "none":
        raise ValueError("Unsupported decomposition method. Use 'CD' or 'STL' or 'none'.")

    # --------- Perform decomposition to each series separately ---------
    results = {}
    for uid, grp in df.groupby('unique_id'):
        # Sort & reset index
        grp = grp.sort_values('ds').reset_index(drop=True)
        
        if decomposition_method == "none":
            dcmp = pd.DataFrame({
                'ds':       grp['ds'],
                'data':     grp[target_col],
                'trend':    grp[target_col],
                'seasonal': np.zeros(len(grp)),
                'remainder':np.zeros(len(grp))
            })
            results[uid] = dcmp
            continue  # Skip decomposition if method is "none"

        if decomposition_method == "CD":
            # Fit classical decomposition
            res = seasonal_decompose(x=grp[target_col], period=period_CD, extrapolate_trend=extrapolate_CD)
        else:
            # Fit STL
            stl = STL(grp[target_col], period=period_STL, robust=robust_STL, seasonal=seasonal_STL)
            res = stl.fit()
    
        # Collect into a DataFrame
        dcmp = pd.DataFrame({
            'ds':       grp['ds'],
            'data':     grp[target_col],
            'trend':    res.trend,
            'seasonal': res.seasonal,
            'remainder':res.resid
        })
        results[uid] = dcmp
    
    # --------- Smoothen if requested ---------
    if smooth_loess and decomposition_method!='none':
        # Smooth the seasonal component using LOESS
        for uid, dcmp in results.items():
            frac = window_loess / len(dcmp) 
            raw_seasonal = dcmp['seasonal']
            # Smooth the seasonal component using LOESS
            loess_smoothed = lowess(raw_seasonal.values,
                                    np.arange(len(raw_seasonal)), 
                                    frac=frac,
                                    return_sorted=False)
            dcmp['seasonal'] = loess_smoothed
            dcmp['remainder'] = dcmp['data'] - dcmp['trend'] - dcmp['seasonal']
            results[uid] = dcmp
    
    # --------- Drop 'data' columns ---------
    if not data_in_dcmp:
        for uid, dcmp in results.items():
            dcmp.drop(columns=['data'], inplace=True, errors='ignore')

    return results

def remove_annual_component( 
    df: pd.DataFrame,                   # DataFrame containing the hourly data to adjust (tipically training data), must contain 'unique_id', 'ds', and targel_col (the column to adjust)
    season_df: pd.DataFrame,            # DataFrame containing the annual seasonal component, must contain 'unique_id', 'ds' (which must have daily frequency), and 'seasonal' (the seasonal component)
    target_col: str,                    # Column to adjust, typically 'y_transformed' or 'y'
    target_col_deseason: str = None,    # Column to store the deseasonalized values. If None, it will be set to target_col+'_deseason'
) -> pd.DataFrame:
    """
    Subtract a daily-frequency annual seasonal component from an hourly time series.

    Parameters
    ----------
    df : pandas.DataFrame
        Hourly-frequency data containing:
        - 'unique_id' : identifier for each time series
        - 'ds'        : timestamp column (hourly granularity)
        - target_col  : column to be deseasonalized
    season_df : pandas.DataFrame
        Daily-frequency seasonal component containing:
        - 'unique_id' : same identifier as in `df`
        - 'ds'        : timestamp column (daily granularity)
        - 'seasonal'  : seasonal value for that day
    target_col : str
        Name of the column in `df` whose values are to be adjusted.
    target_col_deseason : str, optional
        Name for the new column that will hold deseasonalized values.
        If None, defaults to `target_col + '_deseason'`.
    """

    # Work on copies so we don't mutate inputs
    df = df.copy()
    season_df = season_df.copy()

    # Add a date column to merge by date
    season_df['day'] = season_df['ds'].dt.floor('D')
    df['day'] = df['ds'].dt.floor('D')

    # Left‐merge on (unique_id, day)
    df = df.merge(
        season_df[['unique_id','day','seasonal']], 
        on=['unique_id','day'], 
        how='left'
    )

    # Compute the deseasonalized series
    if target_col_deseason is None:
        target_col_deseason = target_col + '_deseason'
    df[target_col_deseason] = df[target_col] - df['seasonal']

    # Drop the 'day' column used for merging
    df.drop(columns=['day', 'seasonal'], inplace=True)
    return df

def add_annual_component(
    forecast_df: pd.DataFrame,              # DataFrame containing the hourly forecasts to adjust, must contain 'unique_id', 'ds', and numeric columns to adjust
    season_df: pd.DataFrame,                # DataFrame containing the annual seasonal component, must contain 'unique_id', 'ds' (which must have daily frequency), and 'seasonal' (the seasonal component)
    target_cols: Optional[list] = None,     # Columns to adjust. If None, all numeric columns will be adjusted
) -> pd.DataFrame:
    """
    Re-apply an annual daily-frequency seasonal profile to an hourly forecast series.
    """

    # Build the 'day_shifted' column to merge with the seasonal component
    forecast_df['day_shifted'] = (
        forecast_df['ds']
        .dt.floor('D')
        .subtract(pd.DateOffset(years=1))
    )
    
    # Add the 'day' column to season_df for merging
    season_df['day'] = season_df['ds'].dt.floor('D')

    # Merge the seasonal component in
    forecast_df = forecast_df.merge(
        season_df[['unique_id','day','seasonal']],
        left_on=['unique_id','day_shifted'],
        right_on=['unique_id','day'],
        how='left'
    )

    # Select only the numeric forecast columns to adjust
    if target_cols is None:
        target_cols = (
            forecast_df
            .select_dtypes(include='number') # all numeric columns
            .columns
            .difference(['day_shifted','seasonal','day']) # drop our merge‐keys / seasonal
        )

    # Adjust those columns by adding the seasonal component
    forecast_df[target_cols] = (
        forecast_df[target_cols]
        .add(forecast_df['seasonal'], axis=0)
    )

    # Drop the columns used for merging
    forecast_df.drop(columns=['day_shifted', 'seasonal', 'day'], inplace=True)
    return forecast_df
