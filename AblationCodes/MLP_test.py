#MLP ablation module: Provides a lightweight MLP-based signal generator and backtesting.

import logging
from typing import Any, Dict, List, Optional, Tuple
from packages import *
import torch
import torch.nn as nn
import torch.optim as optim
from BacktestFolder.backtest import VectorBacktest
from hyperparam import config

# --- Logging configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s", force=True)
logger = logging.getLogger(__name__)

# --- Config-driven defaults ---
DATA_PATH = config['basicfeed']['filepath']
SIGNAL_THRESHOLD = config['backtest']['signal_threshold']
TRAIN_LEN = config['execution']['data_window']['fixed_train_length']
TEST_LEN = config['execution']['data_window']['fixed_test_length']
HIDDEN_DIM = config['ablation']['mlp']['hidden_dim']
LR = config['ablation']['mlp']['lr']
EPOCHS = config['ablation']['mlp']['epochs']
N_STRATEGIES = config['ablation']['mlp']['n_strategies']
START_COL = config['execution']['start_col']
FORECAST_HORIZON = config['ablation']['mlp']['horizon']
SLIDE_WINDOW_DAYS = config['execution']['data_window']['sliding_window_days']

def pick_indicator_columns(
    df: pd.DataFrame,
    indicator_prefix: Optional[str] = None,
    start_col: Optional[int] = None,
    num_ind: int = 80,
) -> List[str]:
    """Select a set of indicator columns from the dataset."""
    if indicator_prefix:
        cols = [c for c in df.columns if c.startswith(indicator_prefix)]
        if len(cols) >= num_ind:
            return cols[:num_ind]
        if cols:
            return cols

    if start_col is not None:
        return list(df.columns[start_col:(start_col + num_ind)])

    exclude = {'Date', 'Close', 'High', 'Low', 'Open', 'Volume'}
    cols = [c for c in df.columns if c not in exclude]
    return cols[:num_ind]


def add_forward_return(
    df: pd.DataFrame,
    horizon: int = 1,
    price_col: str = "Close",
    out_col: str = "fwd_return",
) -> pd.DataFrame:
    """Add a forward return column for supervised training.

    The target at index t is the percent return from t -> t+1.
    """

    df = df.copy()
    df[out_col] = df[price_col].pct_change().shift(-horizon)
    return df


def load_dataset(path: Optional[str] = None, start_ind: int = 0,end_ind: int = TRAIN_LEN+TEST_LEN) -> pd.DataFrame:
    """Load dataset and add forward return column."""
    path = path or DATA_PATH
    df = pd.read_csv(path)
    df = df.iloc[start_ind:end_ind].reset_index(drop=True)
    df = add_forward_return(df, horizon=FORECAST_HORIZON)
    df = df.dropna().reset_index(drop=True)
    return df


class MLPTraderBatch:
    """MLP model that predicts forward returns and converts them into trading signals."""

    def __init__(
        self,
        df: pd.DataFrame,
        train_start: int = 0,
        train_len: int = TRAIN_LEN,
        test_len: int = TEST_LEN,
        signal_threshold: float = SIGNAL_THRESHOLD,
        hidden_dim: int = HIDDEN_DIM,
        lr: float = LR,
        epochs: int = EPOCHS,
        n_strategies: int = N_STRATEGIES,
        indicator_cols: Optional[List[str]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.train_df = self.df.iloc[train_start : train_start + train_len].copy().reset_index(drop=True)
        self.test_df = self.df.iloc[train_start + train_len : train_start + train_len + test_len].copy().reset_index(drop=True)

        self.signal_threshold = signal_threshold
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.n_strategies = n_strategies

        if indicator_cols is None:
            indicator_cols = pick_indicator_columns(df, start_col=START_COL)

        self.indicator_cols = indicator_cols
        self.X_train = torch.tensor(self.train_df[indicator_cols].values, dtype=torch.float32)
        self.y_train = torch.tensor(self.train_df['fwd_return'].values, dtype=torch.float32)
        self.X_test = torch.tensor(self.test_df[indicator_cols].values, dtype=torch.float32)
        self.y_test = torch.tensor(self.test_df['fwd_return'].values, dtype=torch.float32)

        self.model = nn.Sequential(
            nn.Linear(self.X_train.shape[1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_strategies),
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _to_signals(self, preds: np.ndarray) -> np.ndarray:
        """Convert regression predictions to discrete trading signals."""
        return np.where(preds > self.signal_threshold, 1, np.where(preds < -self.signal_threshold, -1, 0)).T

    def custom_loss(self, preds: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Custom loss that maximizes correlation between prediction and target."""
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)
        y_true = y_true.unsqueeze(1).expand_as(preds)

        vx = preds - preds.mean(dim=0, keepdim=True)
        vy = y_true - y_true.mean(dim=0, keepdim=True)

        corr = (vx * vy).mean(dim=0) / (vx.std(dim=0) * vy.std(dim=0) + 1e-8)
        return -corr.mean()

    def train(self) -> None:
        """Train the MLP model."""
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train)
            loss = self.custom_loss(outputs, self.y_train)
            loss.backward()
            self.optimizer.step()
            if epoch % 5 == 0:
                logger.info("Epoch %d, Loss %.4f", epoch, loss.item())

        with torch.no_grad():
            grads = self.model[0].weight.grad
            if grads is not None:
                grad_per_feature = grads.abs().sum(dim=0).cpu().numpy()
                logger.debug("Gradient magnitude per feature: %s", grad_per_feature)

    def _backtest_from_preds(self, preds: np.ndarray, df_close: pd.DataFrame) -> VectorBacktest:
        signals = self._to_signals(preds)
        return VectorBacktest(df_close[['Close']], signals)

    def evaluate(self) -> Tuple[np.ndarray, np.ndarray, VectorBacktest]:
        """Evaluate on the test set and return (sharpe, max_drawdown, portfolio)."""
        with torch.no_grad():
            preds = self.model(self.X_test).cpu().numpy()
        bt = self._backtest_from_preds(preds, self.test_df)
        sharpes = bt.fitness("sharpe")
        mdds = bt.fitness("max_drawdown")
        return sharpes, mdds, bt.port_ret()

    def evaluate_oos(self, oos_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, VectorBacktest]:
        X_oos = torch.tensor(oos_df[self.indicator_cols].values, dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X_oos).cpu().numpy()
        bt = self._backtest_from_preds(preds, oos_df)
        sharpes = bt.fitness("sharpe")
        mdds = bt.fitness("max_drawdown")
        return sharpes, mdds, bt.port_ret()


def run_pipeline(
    data_path: Optional[str] = None,
    train_len: int = TRAIN_LEN,
    test_len: int = TEST_LEN,
    n_strategies: int = N_STRATEGIES,
    hidden_dim: int = HIDDEN_DIM,
    lr: float = LR,
    epochs: int = EPOCHS,
    signal_threshold: float = SIGNAL_THRESHOLD,
    top_n: int = 10,
    is_oos: bool = False,
    start_ind: int =0
) -> Dict[str, Any]:
    """Run a full train/test pipeline and return top-performing strategy metrics."""

    df = load_dataset(data_path,start_ind=start_ind,end_ind=start_ind+train_len+test_len)
    trader = MLPTraderBatch(
        df,
        train_len=train_len,
        test_len=test_len,
        signal_threshold=signal_threshold,
        hidden_dim=hidden_dim,
        lr=lr,
        epochs=epochs,
        n_strategies=n_strategies,
    )

    trader.train()
    sharpes, mdds, portfolio = trader.evaluate()

    sharpe_mdd_arr: List[Tuple[float, float, int]] = []
    for i in range(len(sharpes)):
        sharpe = float(sharpes.iloc[i]) if hasattr(sharpes, 'iloc') else float(sharpes[i])
        mdd = float(mdds.iloc[i]) if hasattr(mdds, 'iloc') else float(mdds[i])
        if not np.isinf(sharpe):
            sharpe_mdd_arr.append((round(sharpe, 3), round(-mdd, 3), i))

    sharpe_mdd_arr.sort(key=lambda x: (x[0], x[1]), reverse=True)

    results: Dict[str, Any] = {
        "train_top": [],
        "oos_top": [],
        "portfolio": portfolio,
    }

    for sharpe, mdd, idx in sharpe_mdd_arr[:top_n]:
        ann_ret = round(portfolio.iloc[idx].annualized_return(), 3)
        results["train_top"].append((sharpe, ann_ret, mdd))

    logger.info("Trained top strategies: %s", results["train_top"])

    if is_oos:
        oos_df = df.iloc[train_len : train_len + test_len].reset_index(drop=True)
        sharpes_oos, mdds_oos, portfolio_oos = trader.evaluate_oos(oos_df)
        oos_arr: List[Tuple[float, float, int]] = []
        for i in range(len(sharpes_oos)):
            sharpe = float(sharpes_oos.iloc[i]) if hasattr(sharpes_oos, 'iloc') else float(sharpes_oos[i])
            mdd = float(mdds_oos.iloc[i]) if hasattr(mdds_oos, 'iloc') else float(mdds_oos[i])
            if not np.isinf(sharpe):
                oos_arr.append((round(sharpe, 3), round(-mdd, 3), i))
        oos_arr.sort(key=lambda x: (x[0], x[1]), reverse=True)

        for sharpe, mdd, idx in oos_arr[:top_n]:
            ann_ret = round(portfolio_oos.iloc[idx].annualized_return(), 3)
            results["oos_top"].append((sharpe, ann_ret, mdd))

        results["oos_portfolio"] = portfolio_oos
        logger.info("OOS top strategies: %s", results["oos_top"])

    return results


if __name__ == "__main__":
    # Example standalone execution - runs based on sliding window defined in config
    for st_ind in range(7,8):
        print("="*100)
        logger.info("Running MLP pipeline for Sliding Dataset %d", st_ind)
        out = run_pipeline(is_oos=(st_ind==7), n_strategies=10000, epochs=30,start_ind=st_ind*SLIDE_WINDOW_DAYS)
        logger.info("Final results for start index %d: %s", st_ind*SLIDE_WINDOW_DAYS, out)
