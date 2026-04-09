import numpy as np
import pandas as pd
from typing import Optional
import logging

from BacktestFolder.backtest import VectorBacktest
from hyperparam import config, rng
from dataset_load import load as load_dataset

# ── config ─────────────────────────────────────────────────────────────────────
data_path  = config['basicfeed']['filepath']
TRAIN_LEN  = config['execution']['data_window']['fixed_train_length']
TEST_LEN   = config['execution']['data_window']['fixed_test_length']
STEP_SIZE  = config['execution']['data_window']['sliding_window_days']
SIGNAL_THRESHOLD           = config['backtest']['signal_threshold']
start_col                  = config['execution']['start_col']
PSO_PARTICLES              = config['ablation']['pso']['pso_particles']
PSO_ITERATIONS             = config['ablation']['pso']['pso_iterations']
STOPPING_DATASET_ITERATION = config['execution']['stopping_dataset_iteration']

# ── AMPO default hyper-parameters (paper Section III-B) ───────────────────────
AMPO_POP    = 100
AMPO_PR     = 0.6
AMPO_W      = 0.1
AMPO_R      = 0.9   # gamma in the paper (decay rate for local-search sigma)
AMPO_PLD_LS = 0.8
AMPO_PLS_LS = 0.8


# ══════════════════════════════════════════════════════════════════════════════
# _Solution  (mirrors solution.py from the repo)
# ══════════════════════════════════════════════════════════════════════════════
class _Solution:
    """
    Holds a weight vector and all per-individual control factors.
    bound = [lb, ub]  (uniform scalar bounds, matching the repo's API).
    """

    def __init__(self, dim, bound, w, r, seed_position=None):
        lb, ub = bound
        self.dim   = dim
        self.bound = bound
        self.w     = w
        self.r     = r

        # control factors initialised per eqs. 12-13
        self.sigma = rng.uniform(0.1, 1.0)
        self.global_search_step_size = rng.uniform(lb, ub, size=dim) / 10.0
        self.local_search_step_size  = np.zeros(dim)

        if seed_position is not None:
            self.solution = np.clip(seed_position.copy(), lb, ub)
        else:
            self.solution = lb + rng.random(dim) * (ub - lb)

    def update_control_factors(self, trans_type, gbest):
        if trans_type == 'local_search':
            self.sigma *= self.r                                       # eq. 5
            self.local_search_step_size = (                            # eq. 6
                rng.normal(0, max(self.sigma, 1e-10)) * self.solution
            )
        elif trans_type == 'global_search':
            self.global_search_step_size = (                           # eq. 3
                self.w * self.global_search_step_size
                + rng.random() * (gbest - self.solution)
            )

    def update(self, trans_type):
        lb, ub = self.bound
        if trans_type == 'leader':
            pass                                                       # eq. 9
        elif trans_type == 'local_search':
            self.solution = self.solution + self.local_search_step_size    # eq. 7
        elif trans_type == 'global_search':
            self.solution = self.solution + self.global_search_step_size   # eq. 4
        self.solution = np.clip(self.solution, lb, ub)                # eq. 14

    def random_update(self):
        lb, ub = self.bound
        self.solution = rng.uniform(lb, ub, size=self.dim)             # eq. 2


# ══════════════════════════════════════════════════════════════════════════════
# _Individual  (mirrors individual.py from the repo)
# ══════════════════════════════════════════════════════════════════════════════
class _Individual:
    """
    source_ind=False  ->  random-search group (not yet a source individual)
    source_ind=True   ->  global_search / local_search / leader
    """

    def __init__(self, dim, bound, w, r, seed_position=None):
        self.dim        = dim
        self.bound      = bound
        self.w          = w
        self.r          = r
        self.source_ind = False
        self.type       = 'random_search'
        self.fitness    = -np.inf           # we MAXIMISE; -inf = worst
        self.solution   = _Solution(dim, bound, w, r, seed_position)

    def recover(self):
        """Algorithm 5: downgrade local->global, global->random; re-init position."""
        old_type = self.type
        self.__init__(self.dim, self.bound, self.w, self.r)   # re-init
        if old_type == 'local_search':
            self.type       = 'global_search'
            self.source_ind = True
        elif old_type == 'global_search':
            self.type       = 'random_search'
            self.source_ind = False

    def transform(self, source_ind, trans_type):
        """Algorithm 3 / Figure 1: solution cloning mechanism."""
        if not self.source_ind:
            self.source_ind = True
            self.type       = trans_type
            if trans_type == 'local_search':
                # Fig. 1b: copy entire solution
                self.solution.solution = source_ind.solution.solution.copy()
                self.solution.sigma    = source_ind.solution.sigma
            else:
                # Fig. 1a: single-side crossover (p=0.5 per dimension)
                cross = rng.integers(0, 2, size=self.dim).astype(bool)
                self.solution.solution = np.where(
                    cross,
                    source_ind.solution.solution,
                    self.solution.solution,
                )

    def update(self, gbest):
        """Algorithm 2: call update_control_factors then update position."""
        if self.source_ind:
            self.solution.update_control_factors(self.type, gbest)
            self.solution.update(self.type)
        else:
            self.solution.random_update()


# ══════════════════════════════════════════════════════════════════════════════
# Shared signal / backtest helpers  (identical to CMA_test.py)
# ══════════════════════════════════════════════════════════════════════════════
def _to_signals(preds, signal_threshold):
    preds = np.asarray(preds)
    if preds.ndim in (1, 2):
        return np.where(preds >  signal_threshold,  1,
               np.where(preds < -signal_threshold, -1, 0))
    raise ValueError("preds must be 1-D or 2-D array")


def evaluate_combination(weights, train_df, base_alpha_signals, signal_threshold):
    try:
        weights = np.asarray(weights).ravel()
        base_alpha_signals = np.asarray(base_alpha_signals)
        if base_alpha_signals.ndim != 2:
            raise ValueError("base_alpha_signals must be 2-D")
        _, num_alphas = base_alpha_signals.shape
        if len(weights) != num_alphas:
            raise ValueError(f"Weight dim {len(weights)} != num_alphas {num_alphas}")
        norm_w   = weights / (np.sum(np.abs(weights)) + 1e-8)
        combined = np.dot(base_alpha_signals, norm_w)
        discrete = _to_signals(combined, signal_threshold)
        if len(discrete) != len(train_df):
            raise ValueError(f"Signal length {len(discrete)} != df length {len(train_df)}")
        return discrete.flatten()
    except Exception as e:
        print(f"Signal evaluation error: {e}")
        return np.zeros(base_alpha_signals.shape[0])


def _batch_sharpe(weight_matrix, train_df, alpha_signals, signal_threshold):
    """Evaluate a (N, dim) weight matrix in one VectorBacktest call."""
    all_signals = []
    for w in weight_matrix:
        sig = evaluate_combination(w, train_df, alpha_signals, signal_threshold)
        all_signals.append(sig)
    all_signals = np.vstack(all_signals)
    bt  = VectorBacktest(train_df[['Close']], all_signals)
    raw = bt.fitness()
    return np.array([float(f) if np.isfinite(float(f)) else -1e9 for f in raw])


# ══════════════════════════════════════════════════════════════════════════════
# ampo_optimizer  (drop-in replacement for cmaes_optimizer)
# ══════════════════════════════════════════════════════════════════════════════
def ampo_optimizer(
    train_df,
    base_alpha_signals_window,
    signal_threshold,
    dim,
    bounds,                        # list of (lb, ub) tuples, one per alpha
    initial_mean=None,
    max_fevals=None,
    pop=AMPO_POP,
    pr=AMPO_PR,
    w=AMPO_W,
    r=AMPO_R,
    p_ld_ls=AMPO_PLD_LS,
    p_ls_ls=AMPO_PLS_LS,
):
    """
    Returns (gbest_position, gbest_fitness, top10_positions, top10_fitness).
    Same signature as cmaes_optimizer().
    """
    lb = float(bounds[0][0])
    ub = float(bounds[0][1])
    scalar_bound = [lb, ub]

    max_fevals = max_fevals or (PSO_PARTICLES * PSO_ITERATIONS)
    max_iters  = 100#max(1, max_fevals // pop)

    # Transformation probability table (directly from repo's ampo.py).
    # Note: the repo checks `if rand <= prob_global_search` first.
    trans_probs = {
        'leader':        {'global_search': 1.0 - p_ld_ls,  'local_search': p_ld_ls},
        'local_search':  {'global_search': 1.0 - p_ls_ls,  'local_search': p_ls_ls},
        'global_search': {'global_search': 1.0,             'local_search': 0.0},
    }

    main_pop_size      = int(pop * pr)
    migrating_pop_size = pop - main_pop_size

    # ── initialisation ───────────────────────────────────────────────────────
    individuals = []
    for i in range(main_pop_size):
        seed = None
        if initial_mean is not None and i == 0:
            # warm-start: perturb prior best with small noise
            seed = np.clip(initial_mean + rng.normal(0, 0.05, dim), lb, ub)
        individuals.append(_Individual(dim, scalar_bound, w, r, seed))

    # migrating group: vectorised numpy array (matches repo design)
    migration_solutions = lb + rng.random((migrating_pop_size, dim)) * (ub - lb)

    gbest_solution           = individuals[0].solution.solution.copy()
    gbest_fitness            = -np.inf
    migration_gbest_solution = migration_solutions[0].copy()
    migration_gbest_fitness  = -np.inf

    # full history for top-10 extraction
    history = []   # list of (np.ndarray position, float fitness)

    # ── batch fitness helpers ────────────────────────────────────────────────
    def _eval_inds(inds):
        wm = np.vstack([ind.solution.solution for ind in inds])
        return _batch_sharpe(wm, train_df, base_alpha_signals_window, signal_threshold)

    def _eval_mig(sols):
        return _batch_sharpe(sols, train_df, base_alpha_signals_window, signal_threshold)

    # ── vectorised DE/rand/1/bin for migrating group ─────────────────────────
    def _migrating_step(sols):
        n = len(sols)
        # mutation
        idx = np.stack([rng.choice(n, 3, replace=False) for _ in range(n)])
        r1, r2, r3 = idx[:, 0], idx[:, 1], idx[:, 2]
        V = sols[r1] + 0.5 * (sols[r2] - sols[r3])
        rand_fill = rng.uniform(lb, ub, size=(n, dim))
        V = np.where(V < lb, rand_fill, V)
        V = np.where(V > ub, rand_fill, V)
        # crossover (CR=0.3 from repo)
        mask = rng.random((n, dim)) < 0.3
        U    = np.where(mask, V, sols)
        # evaluate X and U in ONE batch call to avoid doubling VectorBacktest
        # memory (stacking both prevents MemoryError on large alpha dimensions)
        combined_batch = np.vstack([sols, U])               # shape: (2n, dim)
        combined_fits  = _eval_mig(combined_batch)
        f_X = combined_fits[:n]
        f_U = combined_fits[n:]
        updated   = np.where((f_U >= f_X).reshape(-1, 1), U, sols)
        all_fits  = np.where(f_U >= f_X, f_U, f_X)
        best_idx  = int(np.argmax(all_fits))
        return updated, updated[best_idx].copy(), float(all_fits[best_idx])

    # ════════════════════════════════════════════════════════════════════════
    # Main loop  (mirrors AMPO.run() in the repo, adapted for maximisation)
    # ════════════════════════════════════════════════════════════════════════
    for iteration in range(max_iters):

        # ── function evaluation ──────────────────────────────────────────────
        fitnesses = _eval_inds(individuals)
        for ind, f in zip(individuals, fitnesses):
            ind.fitness = f
            history.append((ind.solution.solution.copy(), float(f)))

        # ── selection ────────────────────────────────────────────────────────
        # pbest = best individual this iteration
        pbest_ind = max(individuals, key=lambda x: x.fitness)
        if pbest_ind.fitness > gbest_fitness:
            # demote any existing leader to local_search
            for ind in individuals:
                if ind.type == 'leader':
                    ind.type       = 'local_search'
                    ind.source_ind = True
            gbest_fitness              = pbest_ind.fitness
            gbest_solution             = pbest_ind.solution.solution.copy()
            pbest_ind.source_ind       = True
            pbest_ind.type             = 'leader'

        # ── transformation ───────────────────────────────────────────────────
        # Sort ASCENDING so best individuals (high Sharpe) appear at the end.
        # This mirrors the repo's descending sort for minimisation:
        #   repo descending -> [worst, ..., best]; source = best (tail)
        #   our ascending   -> [worst, ..., best]; source = best (tail)
        # source_individuals[::-1] gives best-source-first in both cases.
        individuals = sorted(individuals, key=lambda x: x.fitness)

        source_inds = [ind for ind in individuals if ind.source_ind]
        random_inds = [ind for ind in individuals if not ind.source_ind]

        # pair best source with worst random target
        for src in source_inds[::-1]:
            if not random_inds:
                break
            tgt  = random_inds.pop(0)
            prob = rng.random()
            if prob <= trans_probs[src.type]['global_search']:
                tgt.transform(src, 'global_search')
            else:
                tgt.transform(src, 'local_search')

        # ── migration ────────────────────────────────────────────────────────
        migration_solutions, mig_best_sol, mig_best_fit = _migrating_step(
            migration_solutions
        )
        history.append((mig_best_sol.copy(), mig_best_fit))

        if mig_best_fit > migration_gbest_fitness:
            migration_gbest_fitness  = mig_best_fit
            migration_gbest_solution = mig_best_sol.copy()

        # migrate to leader (Algorithm 4)
        if rng.random() <= 0.5 * iteration / max(max_iters, 1):
            if migration_gbest_fitness > gbest_fitness:
                gbest_fitness  = migration_gbest_fitness
                gbest_solution = migration_gbest_solution.copy()
                for ind in individuals:
                    if ind.type == 'leader':
                        ind.solution.solution = gbest_solution.copy()
                        ind.fitness           = gbest_fitness

        # ── update ───────────────────────────────────────────────────────────
        for ind in individuals:
            ind.update(gbest_solution)

        # ── recovery (Algorithm 5) ────────────────────────────────────────────
        # Trigger when ALL main-pop individuals have become source individuals
        source_now = [ind for ind in individuals if ind.source_ind]
        if len(source_now) == main_pop_size:
            rev_pct = rng.uniform(0.1, 0.9)
            n_reset = int(len(source_now) * rev_pct)
            # recover worst source individuals (sorted ascending -> first = worst)
            for ind in source_now[:n_reset]:
                ind.recover()

        logging.info(
            f"AMPO | iter={iteration+1}/{max_iters} | "
            f"best_sharpe={gbest_fitness:.4f}"
        )
        

    # ── top-10 unique solutions from full history ─────────────────────────────
    history.sort(key=lambda x: x[1], reverse=True)
    seen:   set  = set()
    unique: list = []
    for pos, fit in history:
        key = tuple(np.round(pos, 4))
        if key not in seen:
            seen.add(key)
            unique.append((pos, fit))
        if len(unique) == 10:
            break
    while len(unique) < 10:
        unique.append((gbest_solution.copy(), gbest_fitness))

    top_positions = np.array([u[0] for u in unique])
    top_fitness   = np.array([u[1] for u in unique])

    return gbest_solution, gbest_fitness, top_positions, top_fitness


# ══════════════════════════════════════════════════════════════════════════════
# Shared OOS evaluation helper  (identical to CMA_test.py)
# ══════════════════════════════════════════════════════════════════════════════
def _evaluate_on_window(df, alpha_signals, best_weights, top_positions, signal_threshold):
    all_weight_sets = np.vstack([best_weights[None, :], top_positions])
    all_signals = []
    for weights in all_weight_sets:
        norm_w   = weights / (np.sum(np.abs(weights)) + 1e-8)
        combined = np.dot(alpha_signals, norm_w)
        discrete = _to_signals(combined.reshape(1, -1), signal_threshold)
        all_signals.append(discrete.flatten())

    all_signals = np.vstack(all_signals)
    bt     = VectorBacktest(df[['Close']], all_signals)
    sharpe = bt.fitness()
    mdd    = bt.fitness("max_drawdown")

    results   = []
    arr       = []
    for i in range(len(sharpe)):
        s, m = float(sharpe.iloc[i]), float(mdd.iloc[i])
        if np.isfinite(s):
            arr.append((round(s, 3), round(-m, 3), i))

    best_port = None
    if arr:
        arr.sort(key=lambda x: (x[0], x[1]), reverse=True)
        for s, m, idx in arr[:10]:
            ann_ret = round(bt.port_ret().iloc[idx].annualized_return(), 3)
            results.append((s, ann_ret, m))
        best_port = bt.port_ret().iloc[arr[0][2]]

    return results, best_port


# ══════════════════════════════════════════════════════════════════════════════
# run_ampo_pipeline  (drop-in replacement for run_cmaes_pipeline)
# ══════════════════════════════════════════════════════════════════════════════
def run_ampo_pipeline(
    full_df,
    all_base_signals,
    train_start,
    train_len,
    test_len,
    signal_threshold,
    weight_bounds,
    num_alphas,
    max_fevals,
    is_last=False,
    prev_best_weights=None,
):
    train_end = train_start + train_len
    test_end  = train_end   + test_len

    if train_end > len(full_df):
        logging.info(f"Skipping window at {train_start}: not enough data.")
        return None

    train_df = full_df.iloc[train_start:train_end].copy().reset_index(drop=True)
    actual_test_end = min(test_end, len(full_df))
    test_df  = full_df.iloc[train_end:actual_test_end].copy().reset_index(drop=True)
    train_alpha_signals = all_base_signals[train_start:train_end, :]

    logging.info(f"AMPO optimising window {train_start}-{train_end}")
    best_weights, best_fitness, top_positions, top_fitness = ampo_optimizer(
        train_df=train_df,
        base_alpha_signals_window=train_alpha_signals,
        signal_threshold=signal_threshold,
        dim=num_alphas,
        bounds=weight_bounds,
        initial_mean=prev_best_weights,
        max_fevals=max_fevals,
    )
    logging.info(f"Train complete. Best Sharpe: {best_fitness:.4f}")

    oos_results   = []
    port_ret_best = None

    if len(test_df) > 0:
        test_alpha_signals = all_base_signals[train_end:actual_test_end, :]
        oos_results, _ = _evaluate_on_window(
            test_df, test_alpha_signals, best_weights, top_positions, signal_threshold
        )
        print("Window OOS top strategies:", oos_results)
    else:
        print("Test set empty, skipping.")

    if is_last:
        oos_start = train_end+STEP_SIZE
        oos_end = len(full_df)
        print(f"Final OOS: {oos_start} to {oos_end}")
        oos_df            = full_df.iloc[oos_start:oos_end].copy().reset_index(drop=True)
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


# ══════════════════════════════════════════════════════════════════════════════
# Entry point  (identical structure to CMA_test.py __main__)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    compdf = pd.read_csv(data_path)
    compdf = load_dataset(compdf, rng=rng, do_shift=False, do_shuffle=False)
    alpha_cols = list(compdf.columns[start_col:])
    num_alphas = len(alpha_cols)
    #compdf     = compdf.iloc[138:].reset_index(drop=True)
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

        result = run_ampo_pipeline(
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
            prev_best_weights=best_weights_previous,
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

    print(f"{'='*100}")
    print("EXECUTION COMPLETE")
    print(f"{'='*100}")