from Alpha101.packages import *


# region Auxiliary functions
def ts_sum(df: pd.DataFrame, window:int =10):
    return df.rolling(window).sum()


def sma(df:pd.DataFrame, window:int =10):
    return df.rolling(window).mean()


def stddev(df:pd.DataFrame, window:int =10):
    return df.rolling(window).std()


def correlation(x:pd.DataFrame, y:pd.DataFrame, window:int=10):
    return x.rolling(window).corr(y)


def covariance(x:pd.DataFrame, y:pd.DataFrame, window:int=10):
    return x.rolling(window).cov(y)


def rolling_rank(na: np.ndarray):
    return rankdata(na)[-1]


def ts_rank(df:pd.DataFrame, window:int =10):
    return df.rolling(window).apply(rolling_rank)


def rolling_prod(na:np.ndarray):
    return np.prod(na)


def product(df:pd.DataFrame, window:int =10):
    return df.rolling(window).apply(rolling_prod)


def ts_min(df:pd.DataFrame, window:int =10):
    return df.rolling(window).min()


def ts_max(df:pd.DataFrame, window:int =10):
    return df.rolling(window).max()


def delta(df:pd.DataFrame, period:int=1):
    return df.diff(period)


def delay(df, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)


def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    # return df.rank(axis=1, pct=True)
    return df.rank(pct=True)


def scale(df, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.mul(k).div(np.abs(df).sum())


def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1


def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1


def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :]
    na_series = df.as_matrix()

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index, columns=['CLOSE'])
    # endregion