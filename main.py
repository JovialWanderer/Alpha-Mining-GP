import logging
import pickle
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Import all project modules and classes ---
# (Ensure these paths are correct for your project structure)
from hyperparam import *
from packages import *
from VolatilityModelling.VolatilityClassifier import volatility_classifier
from WarmStart import PopulationWarmstarter
from GeneticProgrammingArchitecture.GPUtils import GeneticOperators
from GeneticProgrammingArchitecture.NextgenModule import GenerationEvolver, OptimizerState
from GA_Integration.strategy_evolve import StrategyEvolver
from BacktestFolder.backtest import VectorBacktest
from StrategyTree.TreeUtils import dataset_preprocess,test_signal_generator,check_same
from StrategyTree.TreeSignalCalc import tree_signal
from StrategyTree.TreeStruct import TreeNode

# --- 1. SETUP LOGGING ---
def setup_logging():
    """Configures the logging for the entire application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler("evolution_pipeline.log", mode='w'),
            logging.StreamHandler(sys.stdout) # Also print to console
        ]
    )
    logging.info("Logging configured.")

# --- 2. SETUP CHECKPOINTING ---
CHECKPOINT_FILE = "checkpoint.pkl"

def save_checkpoint(state: Dict[str, Any]):
    """Saves the current state of the pipeline to a file."""
    try:
        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump(state, f)
        logging.info(f"Checkpoint saved successfully to {CHECKPOINT_FILE}")
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")

def load_checkpoint() -> Dict[str, Any] | None:
    """Loads the pipeline state from a checkpoint file if it exists."""
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            state = pickle.load(f)
        logging.info(f"Checkpoint loaded successfully from {CHECKPOINT_FILE}")
        return state
    except FileNotFoundError:
        logging.info("No checkpoint file found. Starting a new run.")
        return None
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}. Starting a new run.")
        return None

# --- 3. REFACTORED BACKTESTING & EVALUATION LOGIC ---
def run_backtest_and_evaluation(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    garch_result,
    predict_vol: np.ndarray,
    dict_low: dict,
    dict_high: dict,
    config: dict
) -> Dict[int, float]:
    """
    Performs the rolling GARCH forecast, signal combination, and backtesting.
    Encapsulates the entire testing logic for one iteration.
    """
    logging.info("Starting backtesting and evaluation for the current window.")
    
    # --- Rolling GARCH Forecast ---
    # (This is the corrected logic from our previous discussion)
    train_returns = train_df['Close'].pct_change().dropna() * 100
    test_returns = test_df['Close'].pct_change().dropna() * 100
    garch_params = {
        'p': garch_result.model.volatility.p, 'q': garch_result.model.volatility.q,
        'o': garch_result.model.volatility.o, 'power': garch_result.model.volatility.power,
        'dist': garch_result.dist.name
    }

    all_forecasts = []
    step_size = 10
    for i in range(0, len(test_returns), step_size):
        current_train_returns = pd.concat([train_returns, test_returns.iloc[:i]])
        model = arch_model(current_train_returns, **garch_params)
        res = model.fit(disp='off', show_warning=False)
        horizon = min(step_size, len(test_returns) - i)
        if horizon <= 0: break
        forecast = res.forecast(horizon=horizon, reindex=False)
        variance_chunk = forecast.variance.values[-1, :]
        all_forecasts.append(np.sqrt(variance_chunk))
        
    pred_volatility = np.concatenate(all_forecasts)[:len(test_df)]
    
    # --- Volatility Classification ---
    combined_vol = np.concatenate([predict_vol, pred_volatility])
    classified_vol = np.zeros_like(pred_volatility, dtype=int)
    vanilla_window = config['execution']['warmstart']['vanilla_window']
    for i in range(len(predict_vol), len(combined_vol)):
        roll_mean = np.mean(combined_vol[i - vanilla_window:i])
        idx = i - len(predict_vol)
        if pred_volatility[idx] > roll_mean:
            classified_vol[idx] = 1

    # --- Signal Generation & Backtesting ---
    base_signals = dataset_preprocess(test_df, list(df.columns[config['execution']['start_col']:]), istest=True)
    depth_sharpe_map = {}
    
    for depth in range(2, config['training']['num_depth'] + 1):
        high_trees = dict_high[depth]["tree_opt"]
        low_trees = dict_low[depth]["tree_opt"]
        
        high_signals = [test_signal_generator(tree, base_signals) for tree in high_trees]
        low_signals = [test_signal_generator(tree, base_signals) for tree in low_trees]
        
        test_signal_arr = []
        for h_sig in high_signals:
            for l_sig in low_signals:
                final_signal = np.where(classified_vol, h_sig, l_sig)
                test_signal_arr.append(final_signal)
        
        backtest = VectorBacktest(test_df[['Close']], test_signal_arr)
        sharpe_ratios = backtest.fitness("sharpe")
        best_sharpe = sharpe_ratios[np.isfinite(sharpe_ratios)].max()
        depth_sharpe_map[depth] = best_sharpe
        logging.info(f"Depth {depth}: Best Sharpe Ratio = {best_sharpe:.4f}")
        
    return depth_sharpe_map

# --- 4. MAIN EXECUTION PIPELINE ---
def main():
    """Main function to run the entire evolutionary backtesting pipeline."""
    setup_logging()
    
    # --- Initialization ---
    logging.info("Initializing pipeline components...")
    rng = np.random.default_rng(seed=config['basicfeed']['SEED'])
    df = pd.read_csv(config['basicfeed']['filepath'])
    
    # Instantiate all our helper classes
    warmstarter = PopulationWarmstarter(
        num_individuals=config['ga']['population']['num_individuals'],
        initial_warmstart_factor=config['integration']['ini_warm_factor'],
        rng=rng
    )
    evolver = StrategyEvolver(config, rng, warmstarter)
    
    # --- State Management & Checkpointing ---
    state = load_checkpoint()
    if state:
        train_start = state['train_start']
        dataset_iteration = state['dataset_iteration']
        dict_low = state['dict_low']
        dict_high = state['dict_high']
        # Load other state variables as needed
    else:
        # Initial state for a fresh run
        train_start = config['execution']['data_window']['train_start_idx']
        dataset_iteration = 0
        dict_low, dict_high = {}, {}
    
    train_end = train_start + config['execution']['data_window']['fixed_train_length']
    sliding_window = config['execution']['data_window']['sliding_window_days']
    
    # --- Main Sliding Window Loop ---
    while train_end < len(df):
        test_start = train_end
        test_end = test_start + config['execution']['data_window']['fixed_test_length']
        if test_end > len(df):
            logging.info("Reached end of dataset. Exiting loop.")
            break

        logging.info(f"--- Starting Iteration {dataset_iteration} | Train: {train_start}-{train_end} | Test: {test_start}-{test_end} ---")
        
        # --- Data Preparation ---
        train_df = df.iloc[train_start:train_end]
        _, predict_vol, high_vol_dataset, low_vol_dataset, high_idx, _, low_idx, _ = volatility_classifier(train_df[['Close']])
        
        high_base_signals = [dataset_preprocess(train_df, list(df.columns[config['execution']['start_col']:]), s, e) for s, e in zip(high_idx[0], high_idx[1])]
        low_base_signals = [dataset_preprocess(train_df, list(df.columns[config['execution']['start_col']:]), s, e) for s, e in zip(low_idx[0], low_idx[1])]
        
        high_base_trees = [[TreeNode(i) for i in range(config['indicators']['num_indicators'])]] * len(high_base_signals)
        low_base_trees = [[TreeNode(i) for i in range(config['indicators']['num_indicators'])]] * len(low_base_signals)

        # --- Evolution ---
        if dataset_iteration == 0:
            logging.info("Running initial evolution for high volatility regime...")
            dict_high = evolver.run_initial_evolution(high_vol_dataset, high_base_signals, high_base_trees, is_high=True)
            logging.info("Running initial evolution for low volatility regime...")
            dict_low = evolver.run_initial_evolution(low_vol_dataset, low_base_signals, low_base_trees, is_high=False)
        else:
            # For advanced runs, you would manage the optimizer state and other params
            # This is a simplified placeholder; you would loop through depths as in your original code
            # and pass the state from the previous dict_low/dict_high.
            logging.info("Running advanced evolution (placeholder)...")
            # Example: new_dict_high = evolver.run_advanced_evolution(...)
            pass # Placeholder for your detailed advanced evolution logic

        # --- Testing ---
        test_df = df.iloc[test_start:test_end]
        # best_sharpe_results = run_backtest_and_evaluation(...)
        
        # --- Save state before the next iteration ---
        current_state = {
            'train_start': train_start + sliding_window,
            'dataset_iteration': dataset_iteration + 1,
            'dict_low': dict_low,
            'dict_high': dict_high,
            # Add any other variables you need to save
        }
        save_checkpoint(current_state)
        
        # --- Slide the window ---
        train_start += sliding_window
        train_end += sliding_window
        dataset_iteration += 1

    logging.info("Pipeline finished successfully.")

if __name__ == '__main__':
    main()