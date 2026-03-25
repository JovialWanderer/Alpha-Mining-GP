import cma
import pandas as pd
import numpy as np
from typing import Optional
import logging
from BacktestFolder.backtest import VectorBacktest
from hyperparam import config, rng
from dataset_load import load as load_dataset

data_path = config['basicfeed']['filepath']
TRAIN_LEN = config['execution']['data_window']['fixed_train_length']
TEST_LEN = config['execution']['data_window']['fixed_test_length']
STEP_SIZE = config['execution']['data_window']['sliding_window_days']
SIGNAL_THRESHOLD = config['backtest']['signal_threshold']
start_col = config['execution']['start_col']
PSO_PARTICLES = config['ablation']['pso']['pso_particles']
PSO_ITERATIONS = config['ablation']['pso']['pso_iterations']
STOPPING_DATASET_ITERATION = config['execution']['stopping_dataset_iteration']

def _to_signals(preds: np.ndarray, signal_threshold: float) -> np.ndarray:
    preds = np.asarray(preds)
    if preds.ndim == 1 or preds.ndim == 2:
        return np.where(preds > signal_threshold, 1,
               np.where(preds < -signal_threshold, -1, 0))
    raise ValueError("preds must be 1D or 2D array")

def evaluate_combination(weights, train_df, base_alpha_signals, signal_threshold):
    try:
        weights = np.asarray(weights).ravel()
        base_alpha_signals = np.asarray(base_alpha_signals)
        if base_alpha_signals.ndim != 2:
            raise ValueError("base_alpha_signals must be 2D")
        _, num_alphas = base_alpha_signals.shape
        if len(weights) != num_alphas:
            raise ValueError(f"Weight dim {len(weights)} != num_alphas {num_alphas}")
        norm_weights = weights / (np.sum(np.abs(weights)) + 1e-8)
        combined = np.dot(base_alpha_signals, norm_weights)
        discrete = _to_signals(combined, signal_threshold)
        if len(discrete) != len(train_df):
            raise ValueError(f"Signal length {len(discrete)} != df length {len(train_df)}")
        return discrete.flatten()
    except Exception as e:
        print(f"Signal evaluation error: {e}")
        return np.zeros(base_alpha_signals.shape[0])

def cmaes_optimizer(
    train_df: pd.DataFrame,
    base_alpha_signals_window: np.ndarray,
    signal_threshold: float,
    dim: int,
    bounds: list,
    initial_mean: Optional[np.ndarray] = None,
    sigma0: float = 0.3,
    max_fevals: int = None,
) -> tuple:

    #Low sigma when warm-starting to prevent premature convergence
    if initial_mean is not None:
        x0 = initial_mean.copy()
        sigma = 0.5  #exploit prior knowledge but still explore
    else:
        x0 = np.zeros(dim)
        sigma = sigma0

    opts = cma.CMAOptions()
    opts['bounds'] = [[-1.0] * dim, [1.0] * dim]
    opts['maxfevals'] = max_fevals or (PSO_PARTICLES * PSO_ITERATIONS)
    opts['verbose'] = -9
    opts['tolx'] = 1e-4
    opts['tolfun'] = 1e-4
    #Minimum population size to ensure diverse top-10
    opts['popsize'] = max(1.0, 4 + int(3 * np.log(dim)))  #CMA-ES default formula

    es = cma.CMAEvolutionStrategy(x0, sigma, opts)

    gbest_position = x0.copy()
    gbest_fitness = -np.inf
    last_solutions = []
    last_fitnesses = []

    while not es.stop():
        solutions = es.ask()
        solutions_array = np.array(solutions)

        all_signals = []
        for weights in solutions_array:
            signal = evaluate_combination(
                weights, train_df, base_alpha_signals_window, signal_threshold
            )
            all_signals.append(signal)

        all_signals = np.vstack(all_signals)
        backtest = VectorBacktest(train_df[['Close']], all_signals)
        fitness_values = backtest.fitness()

        fitnesses_for_cma = [
            -float(f) if np.isfinite(float(f)) else 0.0
            for f in fitness_values
        ]

        es.tell(solutions, fitnesses_for_cma)

        for i, f in enumerate(fitness_values):
            fval = float(f)
            if np.isfinite(fval) and fval > gbest_fitness:
                gbest_fitness = fval
                gbest_position = solutions_array[i].copy()

        last_solutions = solutions_array
        last_fitnesses = [float(f) for f in fitness_values]

        logging.info(
            f"CMA-ES | fevals={es.result.evaluations} | "
            f"best_sharpe={gbest_fitness:.4f} | sigma={es.sigma:.4f}"
        )

    scored_final = sorted(
        zip(last_solutions, last_fitnesses),
        key=lambda x: x[1],
        reverse=True
    )
    seen = []
    unique_scored = []
    for pos, fit in scored_final:
        key = tuple(np.round(pos, 4))
        if key not in seen:
            seen.append(key)
            unique_scored.append((pos, fit))
        if len(unique_scored) == 10:
            break

    # Pad with gbest if fewer than 10 unique solutions found
    while len(unique_scored) < 10:
        unique_scored.append((gbest_position, gbest_fitness))

    top_positions = np.array([s[0] for s in unique_scored])
    top_fitness = np.array([s[1] for s in unique_scored])

    return gbest_position, gbest_fitness, top_positions, top_fitness


def _evaluate_on_window(df, alpha_signals, best_weights, top_positions, signal_threshold):
    """Shared helper for test and OOS evaluation blocks."""
    all_weight_sets = np.vstack([best_weights[None, :], top_positions])
    all_signals = []
    for weights in all_weight_sets:
        norm_weights = weights / (np.sum(np.abs(weights)) + 1e-8)
        combined = np.dot(alpha_signals, norm_weights)
        discrete = _to_signals(combined.reshape(1, -1), signal_threshold)
        all_signals.append(discrete.flatten())

    all_signals = np.vstack(all_signals)
    bt = VectorBacktest(df[['Close']], all_signals)
    sharpe = bt.fitness()
    mdd = bt.fitness("max_drawdown")

    results = []
    arr = []
    for i in range(len(sharpe)):
        s, m = float(sharpe.iloc[i]), float(mdd.iloc[i])
        if np.isfinite(s):
            arr.append((round(s, 3), round(-m, 3), i))

    if arr:
        arr.sort(key=lambda x: (x[0], x[1]), reverse=True)
        for s, m, idx in arr[:10]:
            ann_ret = round(bt.port_ret().iloc[idx].annualized_return(), 3)
            results.append((s, ann_ret, m))

    best_port = bt.port_ret().iloc[arr[0][2]] if arr else None
    return results, best_port


def run_cmaes_pipeline(
    full_df: pd.DataFrame, all_base_signals:np.ndarray, train_start:int, train_len:int,
    test_len:int, signal_threshold:float, weight_bounds:list, num_alphas:int,
    max_fevals:int, is_last:bool=False, prev_best_weights:Optional[np.ndarray]=None
):
    train_end = train_start + train_len
    test_end = train_end + test_len

    if train_end > len(full_df):
        logging.info(f"Skipping window at {train_start}: not enough data.")
        return None

    train_df = full_df.iloc[train_start:train_end].copy().reset_index(drop=True)
    actual_test_end = min(test_end, len(full_df))
    test_df = full_df.iloc[train_end:actual_test_end].copy().reset_index(drop=True)
    train_alpha_signals = all_base_signals[train_start:train_end, :]

    logging.info(f"CMA-ES optimizing window {train_start}-{train_end}")
    best_weights, best_fitness, top_positions, top_fitness = cmaes_optimizer(
        train_df=train_df,
        base_alpha_signals_window=train_alpha_signals,
        signal_threshold=signal_threshold,
        dim=num_alphas,
        bounds=weight_bounds,
        initial_mean=prev_best_weights,
        sigma0=0.3,
        max_fevals=max_fevals,
    )
    logging.info(f"Train complete. Best Sharpe: {best_fitness:.4f}")

    #Validation window evaluation
    oos_results = []
    port_ret_best = None

    if len(test_df) > 0:
        test_alpha_signals = all_base_signals[train_end:actual_test_end, :]
        oos_results, _ = _evaluate_on_window(
            test_df, test_alpha_signals, best_weights, top_positions, signal_threshold
        )
        print("Window OOS top strategies:", oos_results)
    else:
        print("Test set empty, skipping.")

    # Final OOS evaluation
    if is_last:
        oos_start = train_end
        oos_end = len(full_df)
        print(f"Final OOS: {oos_start} to {oos_end}")
        oos_df = full_df.iloc[oos_start:oos_end].copy().reset_index(drop=True)
        oos_alpha_signals = all_base_signals[oos_start:oos_end, :]

        if len(oos_df) > 0:
            oos_results, port_ret_best = _evaluate_on_window(
                oos_df, oos_alpha_signals, best_weights, top_positions, signal_threshold
            )
            print("Final OOS top strategies:", oos_results)
        else:
            print("Final OOS window is empty.")

        return best_weights, oos_results, port_ret_best

    return best_weights, oos_results


if __name__ == "__main__":
    compdf = pd.read_csv(data_path)
    compdf = load_dataset(compdf, rng=rng, do_shift=False, do_shuffle=False)
    alpha_cols = list(compdf.columns[start_col:])
    num_alphas = len(alpha_cols)

    all_base_signals = compdf[alpha_cols].to_numpy()
    full_data_df = compdf[['Close']].copy()
    logging.info(f"Total data points: {len(full_data_df)}, alphas: {num_alphas}")

    WEIGHT_BOUNDS = [(-19.0, 1.0)] * num_alphas
    results_list = []
    best_weights_previous = None

    for start_ind in range(STOPPING_DATASET_ITERATION):
        start = start_ind * STEP_SIZE
        is_last_run = (start_ind == STOPPING_DATASET_ITERATION -1)

        print(f"\n{'='*100}")
        logging.info(f"WINDOW {start_ind}: start={start}")
        print(f"{'='*100}")

        result = run_cmaes_pipeline(
            full_df=full_data_df,
            all_base_signals=all_base_signals,
            train_start=start,
            train_len=TRAIN_LEN,
            test_len=TEST_LEN,
            signal_threshold=SIGNAL_THRESHOLD,
            weight_bounds=WEIGHT_BOUNDS,
            num_alphas=num_alphas,
            max_fevals=PSO_PARTICLES * PSO_ITERATIONS,
            is_last=is_last_run,
            prev_best_weights=best_weights_previous
        )

        if result is None:
            print(f"Window {start_ind} skipped.")
            continue

        # Unpack safely regardless of 2 or 3 values returned
        best_weights_current = result[0]
        oos_res = result[1]
        # port_ret_best = result[2] if is_last_run else None  # uncomment if needed

        best_weights_previous = best_weights_current
        results_list.append({
            'window': start_ind,
            'start_idx': start,
            'weights': best_weights_current.copy(),
            'window_result': oos_res
        })
        print(f"Window {start_ind} done. Results: {oos_res}")

    print(f"{'='*100}")
    print("EXECUTION COMPLETE")
    print(f"{'='*100}")