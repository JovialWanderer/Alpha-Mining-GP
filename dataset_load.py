from hyperparam import *
import pandas as pd
import re

def load(
    df: pd.DataFrame,
    rng: np.random.Generator,
    do_shift: bool = True,
    do_shuffle: bool = True,
    shift_amt: int = 1,
) -> pd.DataFrame:
    """
    Loads and preprocesses the DataFrame by shifting, shuffling, and renaming columns.
    
    This function works on a copy of the input DataFrame to ensure reproducibility.
    """
    df = df.copy()
    start_col=config['execution']['start_col']
    # --- Extract OHLCV and Indicator columns ---
    ohlcv_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    ohlcv_cols_set = set(ohlcv_cols)
    
    if not ohlcv_cols_set.issubset(df.columns):
        raise ValueError(f"DataFrame must contain OHLCV columns: {ohlcv_cols}")

    indicator_cols = [col for col in df.columns if col not in ohlcv_cols_set]
    if not indicator_cols:
        raise ValueError("No indicator columns found in the DataFrame.")

    # --- Shifting ---
    if do_shift:
        df[indicator_cols] = df[indicator_cols].shift(shift_amt)
        df = df.fillna(0)
    
    # --- Shuffling ---
    if do_shuffle:
        rng.shuffle(indicator_cols)
        final_col_order = ohlcv_cols + indicator_cols
        df = df[final_col_order]
        
    # --- Renaming---
    # List of new generic 'f' names
    new_f_names = [f'f{i+1}' for i in range(len(indicator_cols))]
    current_indicator_cols = df.columns[start_col:]
    rename_map = dict(zip(current_indicator_cols, new_f_names))
    df = df.rename(columns=rename_map)
        
    return df