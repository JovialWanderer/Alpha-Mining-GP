import logging
import pickle
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from hyperparam import *
from packages import *
from VolatilityModelling.VolatilityClassifier import volatility_classifier,perform_rolling_garch_forecast
from WarmStart import PopulationWarmstarter
from GeneticProgrammingArchitecture.GPUtils import GeneticOperators
from GeneticProgrammingArchitecture.NextgenModule import GenerationEvolver, OptimizerState
from GA_Integration.strategy_evolve import StrategyEvolver
from BacktestFolder.backtest import VectorBacktest
from StrategyTree.TreeUtils import dataset_preprocess,test_signal_generator,check_same
from StrategyTree.TreeSignalCalc import tree_signal
from StrategyTree.TreeStruct import TreeNode

#  SETUP LOGGING 
def setup_logging():
    """Configures the logging for the entire application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler("evolution_pipeline.log", mode='w'),
            logging.StreamHandler(sys.stdout) #Print to console
        ]
    )
    logging.info("Logging configured.")

# SETUP CHECKPOINTING 
CHECKPOINT_FILE = "checkpoint.pkl"
timeperiod_based_top, avg_test_res, avg_sharpe_dict = {}, {}, {}
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

#  BACKTESTING & EVALUATION LOGIC 
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
    
    #  Rolling GARCH Forecast 
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
    
    #  Volatility Classification 
    combined_vol = np.concatenate([predict_vol, pred_volatility])
    classified_vol = np.zeros_like(pred_volatility, dtype=int)
    vanilla_window = config['execution']['warmstart']['vanilla_window']
    for i in range(len(predict_vol), len(combined_vol)):
        roll_mean = np.mean(combined_vol[i - vanilla_window:i])
        idx = i - len(predict_vol)
        if pred_volatility[idx] > roll_mean:
            classified_vol[idx] = 1

    #  Signal Generation & Backtesting 
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

#Continued Evolution On New Walkforward Windows
def continued_evolution_training(train_df:pd.DataFrame,indicator_cols:list,start_ind:list,end_ind:list,
                                 dict_regime:dict[dict],regime_vol_dataset:list,evolver:StrategyEvolver,curr_warmstart_percent:float,ishigh=True):

    for index in range(2, config['integration']['num_depth'] + 1):
        base_signals=[dataset_preprocess(train_df, indicator_cols, s, e) for s, e in zip(start_ind,end_ind)]
        base_trees=[dict_regime[index]["tree_opt"]]*len(start_ind)
        optim_state: OptimizerState=dict_regime[index]["optimizer_state"]

        #Warmstart
        if index> 2:
            warmstart_regime= dict_regime[index- 1]["tree_opt"]
        else:
            warmstart_regime= [TreeNode(i) for i in range(config['indicators']['num_indicators'])]
        optim_state.dataset_iteration+=1
        logging.info(f"Dataset iteration {optim_state.dataset_iteration} for depth {index} in {'high' if ishigh else 'low'} volatility regime.")
        if base_trees:
            dict_regime[index]= evolver.run_advanced_evolution(regime_vol_dataset[index],base_signals,base_trees,
                index,dict_regime[index]["best_fit"],curr_warmstart_percent,
                warmstart_regime,ishigh=ishigh,optimizer_state=optim_state)
        else:
            dict_regime[index]= {
                'best_fit': dict_regime[index]["best_fit"],
                'tree_opt': dict_regime[index]["tree_opt"],
                'optimizer_state': dict_regime[index]["optimizer_state"]
            }
    return dict_regime
#Volatility Classification for test dataset
def classify_volatility(
    predict_vol: np.ndarray,
    pred_volatility: np.ndarray,
    window: int = 30,
) -> np.ndarray:
    """
    Classifies forecasted volatility into high (1) or low (0)
    based on rolling mean of past combined volatility.
    """

    if len(pred_volatility) == 0:
        return np.array([], dtype=int)

    combined = np.concatenate([predict_vol, pred_volatility])
    fixed_len = len(predict_vol)

    classified = np.zeros(len(pred_volatility), dtype=int)

    for i in range(fixed_len, len(combined)):
        start = max(0, i - window)
        rolling_mean = np.mean(combined[start:i])

        idx = i - fixed_len
        classified[idx] = int(pred_volatility[idx] > rolling_mean)

    return classified

def evaluate_signals(
    test_dataset: pd.DataFrame,
    final_vol_class: np.ndarray,
    base_signals,
    dict_high: dict,
    dict_low: dict,
    dataset_iteration: int,
    timeperiod_based_top: dict,
):

    timeperiod_based_top[dataset_iteration] = []

    for depth in range(2, config['integration']['num_depth'] + 1):

        high_trees = dict_high[depth]["tree_opt"]
        low_trees = dict_low[depth]["tree_opt"]

        # Generate signals
        high_signals = [
            test_signal_generator(tree, base_signals)
            for tree in high_trees
        ]
        low_signals = [
            test_signal_generator(tree, base_signals)
            for tree in low_trees
        ]

        # Combine high/low regime signals
        test_signal_arr = [
            np.where(final_vol_class, h_sig, l_sig)
            for h_sig in high_signals
            for l_sig in low_signals
        ]

        # Backtest
        backtest = VectorBacktest(test_dataset[['Close']], test_signal_arr)
        metrics = backtest.get_portfolio()

        sharpe_arr = metrics.sharpe_ratio()
        ret_arr = metrics.total_profit()
        mdd_arr = metrics.max_drawdown()

        # Filter invalid sharpes
        valid_indices = [
            i for i, sp in enumerate(sharpe_arr)
            if -200 <= sp <= 200
        ]

        if not valid_indices:
            continue

        sorted_sharpe = sorted(
            (sharpe_arr[i] for i in valid_indices),
            reverse=True
        )

        detail = sorted(
            (
                (sharpe_arr[i], ret_arr[i], mdd_arr[i])
                for i in valid_indices
            ),
            reverse=True
        )

        # Best return tracking (unchanged logic)
        max_ret = -math.inf
        best_sharpe = 0

        for i in valid_indices:
            if ret_arr[i] > max_ret:
                max_ret = ret_arr[i]
                best_sharpe = sharpe_arr[i]

        avg_sharpe = np.mean(sorted_sharpe[:10])

        # Store results
        timeperiod_based_top[dataset_iteration].extend(detail[:10])
        avg_test_res[depth] += sorted_sharpe[0]
        avg_sharpe_dict[depth].append(avg_sharpe)

        print(
            f"Depth {depth}: "
            f"Best Sharpe={best_sharpe}, "
            f"Return={max_ret}, "
            f"Avg Sharpe={avg_sharpe}"
        )

    # Final sorting
    timeperiod_based_top[dataset_iteration].sort(reverse=True)
    timeperiod_based_top[dataset_iteration] = timeperiod_based_top[dataset_iteration][:10]

#  MAIN EXECUTION PIPELINE 
def main():
    """Main function to run the entire evolutionary pipeline."""
    setup_logging()
    
    #Initialization 
    logging.info("Initializing pipeline components...")
    rng = np.random.default_rng(seed=config['basicfeed']['SEED'])
    df = pd.read_csv(config['basicfeed']['filepath'])
    
    # Instantiate all our helper classes
    warmstarter = PopulationWarmstarter(rng=rng)
    gen_evolver = GenerationEvolver(config, rng, GeneticOperators(rng))
    evolver = StrategyEvolver(config, rng, warmstarter, gen_evolver)
    
    #  State Management & Checkpointing 
    state = load_checkpoint()
    if state:
        train_start = state['train_start']
        dataset_iteration = state['dataset_iteration']
        dict_low = state['dict_low']
        dict_high = state['dict_high']
    else:
        # Initial state for a fresh run
        train_start = config['execution']['data_window']['train_start_idx']
        dataset_iteration = 0
        dict_low, dict_high = {}, {}
    
    train_end = train_start + config['execution']['data_window']['fixed_train_length']
    sliding_window = config['execution']['data_window']['sliding_window_days']
    
    #  Main Sliding Window Loop 
    while train_end < len(df):
        test_start = train_end
        test_end = test_start + config['execution']['data_window']['fixed_test_length']
        if test_end > len(df):
            logging.info("Reached end of dataset. Exiting loop.")
            break

        logging.info(f" Starting Iteration {dataset_iteration} | Train: {train_start}-{train_end} | Test: {test_start}-{test_end} ")
        
        #  Data Preparation 
        train_df = df.iloc[train_start:train_end]
        garch_result, predict_vol, high_vol_dataset, low_vol_dataset, high_idx, high_end_idx, low_idx, low_end_idx = volatility_classifier(train_df[['Close']])
        indicator_cols = list(df.columns[config['execution']['start_col']:])
        high_base_signals = [dataset_preprocess(train_df, indicator_cols, s, e) for s, e in zip(high_idx, high_end_idx)]
        low_base_signals = [dataset_preprocess(train_df, indicator_cols, s, e) for s, e in zip(low_idx, low_end_idx)]
        
        high_base_trees = [[TreeNode(i) for i in range(config['indicators']['num_indicators'])]] * len(high_base_signals)
        low_base_trees = [[TreeNode(i) for i in range(config['indicators']['num_indicators'])]] * len(low_base_signals)

        #  Evolution
        curr_warmstart_percent = config['execution']['warmstart']['current_warmstart_percent']
        if dataset_iteration == 0:
            logging.info("Running initial evolution for high volatility regime...")
            dict_high = evolver.run_initial_evolution(high_vol_dataset, high_base_signals, high_base_trees, is_high=True)
            logging.info("Running initial evolution for low volatility regime...")
            dict_low = evolver.run_initial_evolution(low_vol_dataset, low_base_signals, low_base_trees, is_high=False)
        else:
            logging.info("Running advanced evolution...")
            dict_high = continued_evolution_training(train_df,indicator_cols,high_idx,high_end_idx,
                                                             dict_high,high_vol_dataset,evolver,curr_warmstart_percent,ishigh=True)
            dict_low = continued_evolution_training(train_df,indicator_cols,low_idx,low_end_idx,
                                                             dict_low,low_vol_dataset,evolver,curr_warmstart_percent,ishigh=False)
            curr_warmstart_percent *= config['execution']['warmstart']['warmstart_percent']
        #  Testing 
        test_df = df.iloc[test_start:test_end]
        pred_volatility = perform_rolling_garch_forecast(train_df, garch_result, test_df)
        final_classified_vol = classify_volatility(predict_vol, pred_volatility,window=config['execution']['warmstart']['test_mean_window'])
        test_base_signals = dataset_preprocess(test_df,indicator_cols,0, 0, istest=True)
        evaluate_signals(test_df, final_classified_vol, test_base_signals, dict_high, dict_low, dataset_iteration, timeperiod_based_top)
        #  Save state before the next iteration 
        current_state = {
            'train_start': train_start + sliding_window,
            'dataset_iteration': dataset_iteration + 1,
            'dict_low': dict_low,
            'dict_high': dict_high,
            'timeperiod_based_top': timeperiod_based_top
        }
        save_checkpoint(current_state)
        
        #  Slide the window 
        train_start += sliding_window
        train_end += sliding_window
        dataset_iteration += 1

    logging.info("Pipeline finished successfully.")

if __name__ == '__main__':
    main()