import vectorbt as vbt
from typing import Literal
from hyperparam import *
import numpy as np
import pandas as pd

# Define a literal type for valid fitness metrics
FitnessMetric = Literal[
    "sharpe", "information_ratio", "max_drawdown", 
    "calmar_ratio", "sortino_ratio", "omega_ratio"
]

class BuyAndHoldBacktest:
    """
    A simple backtest for a buy-and-hold strategy on a given dataset.
    """
    def __init__(self, dataset: pd.DataFrame):
        if 'Close' not in dataset.columns:
            raise ValueError("Dataset must contain a 'Close' column.")

        self.df = dataset.copy()
        self.portfolio = self._create_portfolio()

    def _create_portfolio(self) -> vbt.Portfolio:
        """Initializes and returns a vectorbt portfolio for the buy-and-hold strategy."""
        entries = pd.Series(False, index=self.df.index)
        exits = pd.Series(False, index=self.df.index)
        entries.iloc[0] = True   # Enter on the first day
        exits.iloc[-1] = True  # Exit on the last day

        return vbt.Portfolio.from_signals(
            close=self.df['Close'],
            entries=entries,
            exits=exits,
            init_cash=config['backtest']['initial_cash'],
            fees=config['backtest']['fees'],
            slippage=config['backtest']['slippage'],
            freq=config['backtest']['freq']
        )

    def get_portfolio(self) -> vbt.Portfolio:
        """Returns the generated portfolio object."""
        return self.portfolio

class VectorBacktest:
    """
    Performs a vectorized backtest on a set of trading signals.
    """
    def __init__(self, dataset: pd.DataFrame, terminal_signals: np.ndarray):
        self.df = dataset.copy()

        # Accept lists of signals (common caller behavior) and convert to ndarray
        if isinstance(terminal_signals, list):
            terminal_signals = np.asarray(terminal_signals)

        # Ensure we have a NumPy array
        if not isinstance(terminal_signals, np.ndarray):
            terminal_signals = np.array(terminal_signals)

        # Normalize shape: expect (n_signals, time_length)
        if terminal_signals.ndim == 1:
            # Single signal vector -> make it a single-row 2D array
            terminal_signals = terminal_signals[np.newaxis, :]
        elif terminal_signals.ndim == 2:
            # If the time dimension is in axis 0, transpose
            if terminal_signals.shape[1] != len(self.df) and terminal_signals.shape[0] == len(self.df):
                terminal_signals = terminal_signals.T

        # Ensure signal shape matches dataset length after normalization
        if terminal_signals.shape[1] != len(self.df):
            raise ValueError("Signal array length must match the dataset length.")

        self.signals = terminal_signals
        self._benchmark_portfolio: vbt.Portfolio | None = None
        
        #process signals and create the portfolio
        self.portfolio = self._create_portfolio_from_signals()

    def _process_signals(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes raw signals to prevent lookahead bias and format for vectorbt.
        
        Returns:
            A tuple of (entries_df, exits_df).
        """
        #Shift signals by 1 to prevent lookahead bias (trade on the next bar)
        shifted_signals = np.roll(self.signals, shift=1, axis=1)
        shifted_signals[:, 0] = 0    #Clear first signal from the last day
        shifted_signals[:, -1] = -1 #Force an exit on last day for all strategies
        
        signal_df = pd.DataFrame(
            shifted_signals.T,
            index=self.df.index,
            columns=[f'sig{i+1}' for i in range(self.signals.shape[0])]
        )
        
        entries_df = signal_df == 1
        exits_df = signal_df == -1
        return entries_df, exits_df

    def _create_portfolio_from_signals(self) -> vbt.Portfolio:
        """Builds the vectorbt Portfolio object from processed signals."""
        entries_df, exits_df = self._process_signals()
        price_df = self.df[['Close']].copy()

        return vbt.Portfolio.from_signals(
            close=price_df['Close'],
            entries=entries_df,
            exits=exits_df,
            init_cash=config['backtest']['initial_cash'],
            slippage=config['backtest']['slippage'],
            fees=config['backtest']['fees'],
            freq=config['backtest']['frequency']
        )
        
    @property
    def benchmark_returns(self) -> pd.Series:
        """
        Loads and returns the benchmark (buy-and-hold) returns.
        """
        if self._benchmark_portfolio is None:
            self._benchmark_portfolio = BuyAndHoldBacktest(self.df).get_portfolio()
        return self._benchmark_portfolio.returns()

    def get_portfolio(self) -> vbt.Portfolio:
        """Returns the generated portfolio object for all strategies."""
        return self.portfolio

    def fitness(self, metric: FitnessMetric = "sharpe") -> pd.Series | float:
        """
        Calculates a specific performance metric for the portfolio.
        
        Args:
            metric: The name of the fitness metric to calculate.
        """
        metric_mappers = {
            "sharpe": self.portfolio.sharpe_ratio,
            "calmar_ratio": self.portfolio.calmar_ratio,
            "sortino_ratio": self.portfolio.sortino_ratio,
            "omega_ratio": self.portfolio.omega_ratio,
            "max_drawdown": lambda: -self.portfolio.max_drawdown, #Negative for maximization
            "information_ratio": lambda: self.portfolio.information_ratio(self.benchmark_returns)
        }
        
        if metric not in metric_mappers:
            raise ValueError(f"Invalid metric '{metric}'. Choose from {list(metric_mappers.keys())}")

        return metric_mappers[metric]()