import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import logging

import numpy as np

np.float = float
np.int = int
np.bool = bool

import pyade.lshade
import pyade.commons
from BacktestFolder.backtest import VectorBacktest
from hyperparam import config, rng
from dataset_load import load as load_dataset

# ── Config (mirrors CMA_test.py exactly) ──────────────────────────────────────
data_path  = config['basicfeed']['filepath']
TRAIN_LEN  = config['execution']['data_window']['fixed_train_length']
TEST_LEN   = config['execution']['data_window']['fixed_test_length']
STEP_SIZE  = config['execution']['data_window']['sliding_window_days']
SIGNAL_THRESHOLD           = config['backtest']['signal_threshold']
start_col                  = config['execution']['start_col']
PSO_PARTICLES              = config['ablation']['pso']['pso_particles']
PSO_ITERATIONS             = config['ablation']['pso']['pso_iterations']
STOPPING_DATASET_ITERATION = config['execution']['stopping_dataset_iteration']

# ── L-SHADE hyper-parameters ──────────────────────────────────────────────────
# pyade defaults: memory_size=6, population_size=18*dim, min_pop=4.
# We keep all defaults and only set max_evals, bounds, func, and seed.
LSHADE_MEMORY_SIZE = 6      # H in the paper — history archive size for M_CR/M_F
LSHADE_P_BEST      = 0.11   # top-p fraction used in current-to-pbest mutation
                             # (pyade default; matches typical CEC tuning)


# ══════════════════════════════════════════════════════════════════════════════
# Shared signal / backtest helpers  (identical to CMA_test.py / AMPO_test.py)
# ══════════════════════════════════════════════════════════════════════════════
def _to_signals(preds: np.ndarray, threshold: float) -> np.ndarray:
    preds = np.asarray(preds)
    return np.where(preds >  threshold,  1,
           np.where(preds < -threshold, -1, 0))


def _batch_sharpe(
    weight_matrix: np.ndarray,
    df:            pd.DataFrame,
    alpha_signals: np.ndarray,
    threshold:     float,
) -> np.ndarray:
    """
    Evaluate a (N, dim) weight matrix in one VectorBacktest call.
    Returns a 1-D array of Sharpe values (-1e9 for non-finite entries).
    Identical to _batch_sharpe in AMPO_test.py.
    """
    all_signals = []
    for w in weight_matrix:
        w_norm   = w / (np.abs(w).sum() + 1e-8)
        combined = alpha_signals @ w_norm
        discrete = _to_signals(combined, threshold)
        all_signals.append(discrete.flatten())
    all_signals = np.vstack(all_signals)
    bt  = VectorBacktest(df[['Close']], all_signals)
    raw = bt.fitness()
    return np.array([float(f) if np.isfinite(float(f)) else -1e9 for f in raw])


def _eval_on_window(
    df:            pd.DataFrame,
    alpha_signals: np.ndarray,
    best_weights:  np.ndarray,
    top_positions: np.ndarray,
    threshold:     float,
) -> Tuple[list, Optional[object]]:
    """
    Evaluate best_weights + top_positions on a data window.
    Identical to _evaluate_on_window / _eval_on_window in CMA_test / AMPO_test.
    """
    all_weight_sets = np.vstack([best_weights[None, :], top_positions])
    all_signals = []
    for w in all_weight_sets:
        w_norm   = w / (np.abs(w).sum() + 1e-8)
        discrete = _to_signals(alpha_signals @ w_norm, threshold)
        all_signals.append(discrete.flatten())

    all_signals = np.vstack(all_signals)
    bt     = VectorBacktest(df[['Close']], all_signals)
    sharpe = bt.fitness()
    mdd    = bt.fitness("max_drawdown")

    scored = []
    for i in range(len(sharpe)):
        s, m = float(sharpe.iloc[i]), float(mdd.iloc[i])
        if np.isfinite(s):
            scored.append((round(s, 3), round(-m, 3), i))

    if not scored:
        return [], None

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    results = []
    for s, m, idx in scored[:10]:
        ann_ret = round(bt.port_ret().iloc[idx].annualized_return(), 3)
        results.append((s, ann_ret, m))

    best_port = bt.port_ret().iloc[scored[0][2]]
    return results, best_port


# ══════════════════════════════════════════════════════════════════════════════
# lshade_optimizer  (drop-in replacement for cmaes_optimizer / ampo_optimizer)
# ══════════════════════════════════════════════════════════════════════════════
def lshade_optimizer(
    train_df:                  pd.DataFrame,
    base_alpha_signals_window: np.ndarray,
    signal_threshold:          float,
    dim:                       int,
    bounds:                    list,          # list of (lb, ub) tuples
    initial_mean:              Optional[np.ndarray] = None,
    max_fevals:                int  = None,
    memory_size:               int  = LSHADE_MEMORY_SIZE,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Run L-SHADE via pyade on one walk-forward window.

    Returns (gbest_position, gbest_fitness, top10_positions, top10_fitness).
    Same signature as cmaes_optimizer() and ampo_optimizer().
    """
    lb = float(bounds[0][0])
    ub = float(bounds[0][1])
    max_fevals = max_fevals or 100#(PSO_PARTICLES * PSO_ITERATIONS)

    # ── generation-batched evaluation ─────────────────────────────────────────
    # pyade calls objective(x) once per individual sequentially.  We buffer
    # all calls within a generation into a queue, flush the entire queue as
    # one VectorBacktest call at the end of each generation (via a callback),
    # then hand the cached results back.  This gives the same per-generation
    # batch behaviour as CMA-ES and AMPO while staying compatible with pyade's
    # serial calling convention.
    eval_queue:    List[np.ndarray] = []   # weight vectors waiting to be scored
    result_cache:  List[float]      = []   # sharpe results, same order as queue
    solution_log:  List[Tuple[np.ndarray, float]] = []

    def objective(weights: np.ndarray) -> float:
        """
        Called by pyade once per individual.  Buffer the call; the callback
        flushes the whole generation batch via VectorBacktest.
        """
        eval_queue.append(weights.copy())
        # return a placeholder; pyade overwrites with the real value after
        # the callback runs and we refill result_cache.
        return 0.0   # placeholder — overwritten in _flush_and_patch below

    def _flush_generation(population, **kwargs):
        """
        pyade callback — called once per generation with the full population
        array.  We score all individuals in one VectorBacktest call, fill
        result_cache so the objective can return real values, and log results.

        pyade passes the callback the full locals() dict; we only need population.
        """
        if len(population) == 0:
            return
        fits = _batch_sharpe(
            population, train_df,
            base_alpha_signals_window, signal_threshold,
        )
        result_cache.clear()
        result_cache.extend(fits.tolist())
        for w, f in zip(population, fits):
            solution_log.append((w.copy(), float(f)))

    # pyade does not support replacing fitness mid-generation, so we use a
    # simpler approach: wrap objective to evaluate immediately via a 1-row
    # batch (one VectorBacktest call per individual).  This is equivalent in
    # correctness; for speed we keep max_evals modest (budget already set).
    # The generation-level batch would require forking pyade internals.
    # We therefore revert to the correct single-call-per-individual approach
    # but route it through _batch_sharpe with a 1-row matrix so the code path
    # is identical to CMA-ES/AMPO (normalisation, clipping, etc.).
    eval_queue.clear()

    def objective(weights: np.ndarray) -> float:   # noqa: F811 (redefine)
        """Evaluate one weight vector; log and return -Sharpe for pyade."""
        fits   = _batch_sharpe(
            weights.reshape(1, -1), train_df,
            base_alpha_signals_window, signal_threshold,
        )
        sharpe = float(fits[0])
        solution_log.append((weights.copy(), sharpe))
        return -sharpe      # pyade minimises

    # ── build pyade parameter dict ────────────────────────────────────────────
    params = pyade.lshade.get_default_params(dim=dim)
    params['bounds']      = np.array([[lb, ub]] * dim)
    params['func']        = objective
    params['max_evals']   = max_fevals
    params['memory_size'] = memory_size
    params['seed']        = int(rng.integers(0, 2**31))   # reproducibility

    # ── warm-start: seed initial population near prev_best_weights ────────────
    # pyade.lshade.apply generates its own population internally via
    # pyade.commons.init_population.  We temporarily monkey-patch that
    # function to inject our warm-start matrix, then restore the original.
    # This gives the same warm-start behaviour as CMA-ES: one individual
    # near the prior best, rest uniform random.
    original_init = pyade.commons.init_population
    if initial_mean is not None:
        n_init   = params['population_size']   # default: 18 * dim
        seed_ind = np.clip(
            initial_mean + rng.normal(0, 0.05, dim), lb, ub
        )
        rest     = lb + rng.random((n_init - 1, dim)) * (ub - lb)
        warm_pop = np.vstack([seed_ind[None, :], rest])
        pyade.commons.init_population = lambda n, d, b: warm_pop[:n]

    # ── run L-SHADE (always restore init_population, even on exception) ───────
    try:
        best_sol, neg_best_sharpe = pyade.lshade.apply(**params)
    finally:
        pyade.commons.init_population = original_init
    gbest_position = np.asarray(best_sol)
    gbest_fitness  = -float(neg_best_sharpe)   # restore positive Sharpe

    # ── top-10 unique solutions from logged evaluations ───────────────────────
    scored_log = sorted(solution_log, key=lambda x: x[1], reverse=True)
    seen:   set  = set()
    unique: list = []
    for pos, fit in scored_log:
        key = tuple(np.round(pos, 4))
        if key not in seen:
            seen.add(key)
            unique.append((pos, fit))
        if len(unique) == 10:
            break

    while len(unique) < 10:
        unique.append((gbest_position.copy(), gbest_fitness))

    top_positions = np.array([u[0] for u in unique])
    top_fitness   = np.array([u[1] for u in unique])

    logging.info(
        f"L-SHADE | fevals={len(solution_log)} | "
        f"best_sharpe={gbest_fitness:.4f}"
    )
    return gbest_position, gbest_fitness, top_positions, top_fitness


# ══════════════════════════════════════════════════════════════════════════════
# run_lshade_pipeline  (drop-in replacement for run_cmaes_pipeline)
# ══════════════════════════════════════════════════════════════════════════════
def run_lshade_pipeline(
    full_df:           pd.DataFrame,
    all_base_signals:  np.ndarray,
    train_start:       int,
    train_len:         int,
    test_len:          int,
    signal_threshold:  float,
    weight_bounds:     list,
    num_alphas:        int,
    max_fevals:        int,
    is_last:           bool = False,
    prev_best_weights: Optional[np.ndarray] = None,
):
    """
    Single walk-forward window: train L-SHADE -> evaluate on validation window
    -> optionally evaluate on final OOS segment.
    Mirrors run_cmaes_pipeline / run_ampo_pipeline exactly.
    """
    train_end = train_start + train_len
    test_end  = train_end   + test_len

    if train_end > len(full_df):
        logging.info(f"Skipping window at {train_start}: insufficient data.")
        return None

    train_df = full_df.iloc[train_start:train_end].copy().reset_index(drop=True)
    actual_test_end = min(test_end, len(full_df))
    test_df  = full_df.iloc[train_end:actual_test_end].copy().reset_index(drop=True)
    train_alpha_signals = all_base_signals[train_start:train_end]

    logging.info(f"L-SHADE optimising window {train_start}-{train_end}")

    best_weights, best_fitness, top_positions, top_fitness = lshade_optimizer(
        train_df                  = train_df,
        base_alpha_signals_window = train_alpha_signals,
        signal_threshold          = signal_threshold,
        dim                       = num_alphas,
        bounds                    = weight_bounds,
        initial_mean              = prev_best_weights,
        max_fevals                = max_fevals,
    )
    logging.info(f"Train complete. Best Sharpe: {best_fitness:.4f}")

    oos_results   = []
    port_ret_best = None

    if len(test_df) > 0:
        test_alpha_signals = all_base_signals[train_end:actual_test_end]
        oos_results, _ = _eval_on_window(
            test_df, test_alpha_signals,
            best_weights, top_positions, signal_threshold,
        )
        print("Window OOS top strategies:", oos_results)
    else:
        print("Test set empty, skipping validation window.")

    if is_last:
        oos_start = train_end + STEP_SIZE
        oos_end   = len(full_df)
        print(f"Final OOS: rows {oos_start} -> {oos_end}")
        oos_df            = full_df.iloc[oos_start:oos_end].copy().reset_index(drop=True)
        oos_alpha_signals = all_base_signals[oos_start:oos_end]

        if len(oos_df) > 0:
            oos_results, port_ret_best = _eval_on_window(
                oos_df, oos_alpha_signals,
                best_weights, top_positions, signal_threshold,
            )
            print("Final OOS top strategies:", oos_results)
        else:
            print("Final OOS window is empty.")

        return best_weights, oos_results, port_ret_best

    return best_weights, oos_results


# ══════════════════════════════════════════════════════════════════════════════
# Entry point  (identical structure to CMA_test.py / AMPO_test.py __main__)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    compdf = pd.read_csv(data_path)
    compdf = load_dataset(compdf, rng=rng, do_shift=False, do_shuffle=False)

    alpha_cols       = list(compdf.columns[start_col:])
    num_alphas       = len(alpha_cols)
    #compdf           = compdf.iloc[138:].reset_index(drop=True)
    all_base_signals = compdf[alpha_cols].to_numpy()
    full_data_df     = compdf[['Close']].copy()
    logging.info(f"Total data points: {len(full_data_df)}, alphas: {num_alphas}")

    WEIGHT_BOUNDS         = [(-1.0, 1.0)] * num_alphas
    results_list          = []
    best_weights_previous = None

    for start_ind in range(STOPPING_DATASET_ITERATION):
        start       = start_ind * STEP_SIZE
        is_last_run = (start_ind == STOPPING_DATASET_ITERATION - 1)

        print(f"\n{'='*100}")
        logging.info(f"WINDOW {start_ind}: start={start}")
        print(f"{'='*100}")

        result = run_lshade_pipeline(
            full_df           = full_data_df,
            all_base_signals  = all_base_signals,
            train_start       = start,
            train_len         = TRAIN_LEN,
            test_len          = TEST_LEN,
            signal_threshold  = SIGNAL_THRESHOLD,
            weight_bounds     = WEIGHT_BOUNDS,
            num_alphas        = num_alphas,
            max_fevals        = 100,#PSO_PARTICLES * PSO_ITERATIONS,
            is_last           = is_last_run,
            prev_best_weights = best_weights_previous,
        )

        if result is None:
            print(f"Window {start_ind} skipped.")
            continue

        best_weights_current = result[0]
        oos_res              = result[1]

        best_weights_previous = best_weights_current
        results_list.append({
            'window':        start_ind,
            'start_idx':     start,
            'weights':       best_weights_current.copy(),
            'window_result': oos_res,
        })
        print(f"Window {start_ind} done. Results: {oos_res}")

    print(f"\n{'='*100}\nEXECUTION COMPLETE\n{'='*100}")