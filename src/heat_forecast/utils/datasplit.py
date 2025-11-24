import pandas as pd
from typing import Optional
import logging
_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())

# -------------------------------------------------------------------------------
# FOR TRAIN / VALIDATION / TEST SPLIT
# -------------------------------------------------------------------------------

def _split(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df['ds'] >= start) & (df['ds'] <= end)].copy()

def generate_sets(
    target_df:    pd.DataFrame, 
    start_train: pd.Timestamp,
    end_train:  pd.Timestamp,
    end_test:   pd.Timestamp,
    aux_flag:   bool = False, # Whether to include auxiliary data in the splits
    aux_df:     Optional[pd.DataFrame] = None, # Auxiliary data DataFrame
    max_lag:    int = 0, # Maximum lag
    verbose:   bool = False
):
    """
    Generate train and test (or validation) sets for heat and auxiliary data. start_train and end_train define the training set,
    and are inclusive. end_test defines the end of the test/validation set, and is inclusive. 
    """
    if not (start_train < end_train < end_test): 
        raise ValueError("Dates must satisfy start_train < end_train < end_test")
    if aux_flag and aux_df is None:
        raise ValueError("aux_df must be provided when aux_flag=True")
    
    if verbose:
        _LOGGER.info(f"Generating train/test sets...")
        _LOGGER.info(f"TRAIN                                     | TEST                  ")
        _LOGGER.info("%s → %s | → %s ",
                    start_train, end_train, end_test)

    lag_delay = pd.Timedelta(hours=max_lag)
    split_periods = [(start_train-lag_delay,                       end_train), # Keep lag_delay hours more, that will be dropped after lagging
                     (end_train-lag_delay+pd.Timedelta(hours=1),   end_test)]
    # heat splits
    h_tr, h_te = (_split(target_df, s, e).reset_index(drop=True) for s, e in split_periods)
    target_sets = (h_tr, h_te)

    if not aux_flag:
        return target_sets, None

    # aux splits
    a_tr, a_te = (_split(aux_df, s, e).reset_index(drop=True) for s, e in split_periods)
    aux_sets = (a_tr, a_te)
    
    return target_sets, aux_sets
