import json
import logging
import random
import itertools
from pathlib import Path
from typing import Dict, List, Any, Tuple, Generator

from packages import *
from hyperparam import config
from BacktestFolder.backtest import VectorBacktest

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from joblib import Parallel, delayed
from tqdm import tqdm

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# Config and constants

DATA_PATH = config["basicfeed"]["filepath"]

HORIZON = config["ablation"]["lgb_xgb"]["horizon"]
INDICATOR_PREFIX = config["ablation"]["lgb_xgb"]["indicator_prefix"]
NUM_INDICATORS = config["ablation"]["lgb_xgb"]["num_ind"]

TRAIN_DAYS = config["execution"]["data_window"]["fixed_train_length"]
TEST_DAYS = config["execution"]["data_window"]["fixed_test_length"]
STEP_DAYS = config["execution"]["data_window"]["sliding_window_days"]

POS_THRESH = config["backtest"]["signal_threshold"]

OUTPUT_DIR = Path(config["ablation"]["lgb_xgb"]["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEARCH_TRIALS = config["lgb_xgb_search"]["random_search_trials"]
FOCUSED_GRID_TOP_K = config["lgb_xgb_search"]["focused_grid_top_k"]
PRUNE_PERCENTILE = config["lgb_xgb_search"]["prune_percentile"]


# Data utilities

def add_forward_return(
    df: pd.DataFrame,
    horizon: int,
    price_col: str = "Close"
) -> pd.DataFrame:

    df = df.copy()
    df[f"{horizon}_d_return"] = df[price_col].pct_change(horizon).shift(-horizon)
    return df


def load_dataset(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)
    df = add_forward_return(df, HORIZON)
    df = df.dropna().reset_index(drop=True)

    return df


# Feature utilities

def pick_indicator_columns(
    df: pd.DataFrame,
    indicator_prefix: str = None,
    start_col: int = None,
    num_ind: int = 80,
) -> List[str]:

    if indicator_prefix:
        cols = [c for c in df.columns if c.startswith(indicator_prefix)]
        return cols[:num_ind]

    if start_col is not None:
        return list(df.columns[start_col:start_col + num_ind])

    exclude = {"Date", "Open", "High", "Low", "Close", "Volume"}
    cols = [c for c in df.columns if c not in exclude]

    return cols[:num_ind]


def prune_features(model: LGBMRegressor|XGBRegressor,feature_names: List[str],percentile: float) -> List[str]:
    importances = np.array(model.feature_importances_)
    cutoff = np.percentile(importances, percentile)
    keep_mask = importances > cutoff
    kept = [c for i, c in enumerate(feature_names) if keep_mask[i]]
    if len(kept) == 0:
        return feature_names
    return kept

# Walk-Forward Window Engine

def generate_walk_forward_windows(data_len: int,train_days: int,test_days: int,
                                  step_days: int) -> Generator[Tuple[int, slice, slice], None, None]:
    window_id = 0
    train_start = 0

    while True:

        train_end = train_start + train_days
        test_start = train_end
        test_end = test_start + test_days

        if test_end > data_len:
            break

        yield (window_id,slice(train_start, train_end),slice(test_start, test_end),)

        train_start += step_days
        window_id += 1


# Backtest utilities

def preds_to_signals(preds: np.ndarray) -> np.ndarray:
    return np.where(preds > POS_THRESH,1,np.where(preds < -POS_THRESH, -1, 0),)


def evaluate_model(model: LGBMRegressor|XGBRegressor, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, test_df: pd.DataFrame) -> float:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    signals = preds_to_signals(preds)
    try:
        bt = VectorBacktest(test_df[["Close"]], signals)
        sharpe = round(bt.fitness("sharpe"),3)
        if np.isnan(sharpe) or np.isinf(sharpe):
            return 0.0
        return sharpe
    except Exception as e:
        logger.warning(f"Backtest failed: {e}")
        return -999.0

# Hyperparameter search

def sample_param_combos(param_grid: Dict[str, List[Any]],n_samples: int) -> List[Dict[str, Any]]:
    combos = list(itertools.product(*param_grid.values()))
    sampled = random.sample(combos, min(n_samples, len(combos)))
    return [dict(zip(param_grid.keys(), c)) for c in sampled]


def evaluate_param_set(model_class,params,X_train,
                       y_train,X_test,test_df):
    model = model_class(**params)
    score = evaluate_model(model, X_train, y_train, X_test, test_df)
    return params, score

# Model Search

def randomized_search(model_class,param_grid,X_train,
                      y_train,X_test,test_df):

    samples = sample_param_combos(param_grid, RANDOM_SEARCH_TRIALS)
    results = Parallel(n_jobs=2)(delayed(evaluate_param_set)(model_class,p,X_train,y_train,
                                                             X_test,test_df,)for p in tqdm(samples))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# Walk-Forward Pipeline

def run_pipeline(data_path: str = DATA_PATH):

    df = load_dataset(data_path)

    indicator_cols = pick_indicator_columns(
        df,
        INDICATOR_PREFIX,
        config["execution"]["start_col"],
        NUM_INDICATORS,
    )

    target_col = f"{HORIZON}_d_return"

    records = []

    for window_id, train_slice, test_slice in generate_walk_forward_windows(len(df),TRAIN_DAYS,TEST_DAYS,STEP_DAYS,):
        logger.info(f"Window {window_id}==train[{train_slice.start}:{train_slice.stop}]==test[{test_slice.start}:{test_slice.stop}]")
        train_df,test_df = df.iloc[train_slice],df.iloc[test_slice]
        X_train = train_df[indicator_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[indicator_cols].values
        # Randomized search

        lgb_results = randomized_search(LGBMRegressor,lgb_params_grid,
                                        X_train,y_train,X_test,test_df,)

        xgb_results = randomized_search(XGBRegressor,xgb_params_grid,
                                        X_train,y_train,X_test,test_df,)

        best_lgb = lgb_results[0]
        best_xgb = xgb_results[0]
        # Choose model based on best score

        if best_lgb[1] >= best_xgb[1]:
            chosen_model = "LGBM"
            best_params = best_lgb[0]
            model = LGBMRegressor(**best_params)
        else:
            chosen_model = "XGB"
            best_params = best_xgb[0]
            model = XGBRegressor(**best_params)
        # Feature pruning
        model.fit(X_train, y_train)
        kept_features = prune_features(model,indicator_cols,PRUNE_PERCENTILE,)

        # retrain on pruned features
        X_train_pruned = train_df[kept_features].values
        X_test_pruned = test_df[kept_features].values
        final_score = evaluate_model(model,X_train_pruned,y_train,
                                     X_test_pruned,test_df,)

        records.append({
            "window": window_id,
            "model": chosen_model,
            "params": best_params,
            "sharpe": final_score,
            "features_used": len(kept_features),
        })
        logger.info(
            f"Window {window_id} best model {chosen_model} Sharpe={final_score:.4f}"
        )
    return records


# Hyperparameter grids

lgb_params_grid = {
    "n_estimators": [200, 400, 800],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 8],
    "num_leaves": [15, 31, 63],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

xgb_params_grid = {
    "n_estimators": [200, 400, 800],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 8],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

if __name__ == "__main__":

    results = run_pipeline(DATA_PATH)
    out_file = OUTPUT_DIR / "walkforward_summary.json"

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Pipeline completed")