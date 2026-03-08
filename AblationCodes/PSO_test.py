from typing import Optional
from packages import *
from BacktestFolder.backtest import VectorBacktest
from dataset_load import load as load_dataset
from hyperparam import config, rng
import logging


data_path = config['basicfeed']['filepath']
TRAIN_LEN = config['execution']['data_window']['fixed_train_length']
TEST_LEN = config['execution']['data_window']['fixed_test_length']
STEP_SIZE = config['execution']['data_window']['sliding_window_days']
SIGNAL_THRESHOLD = config['backtest']['signal_threshold']
start_col = config['execution']['start_col']
PSO_PARTICLES = config['ablation']['pso']['pso_particles']
PSO_ITERATIONS = config['ablation']['pso']['pso_iterations']
PSO_W = config['ablation']['pso']['w']
PSO_C1 = config['ablation']['pso']['c1']
PSO_C2 = config['ablation']['pso']['c2']

def _to_signals(preds: np.ndarray, signal_threshold: float) -> np.ndarray:
    """ Convert continuous predictions to discrete {-1, 0, 1} signals. """
    preds=np.asarray(preds)
    if preds.ndim == 1 or preds.ndim==2:
        signals = np.where(preds > signal_threshold, 1,
                   np.where(preds < -signal_threshold, -1, 0))
        return signals
    else:
        raise ValueError("preds must be 1D or 2D array")

def evaluate_combination(weights: np.ndarray,
                         train_df: pd.DataFrame,
                         base_alpha_signals: np.ndarray,
                         signal_threshold: float):
    """Evaluates weights by combining a 2D base_alpha_signals array and running a backtest."""

    try:
        weights = np.asarray(weights).ravel()
        base_alpha_signals = np.asarray(base_alpha_signals)
        if base_alpha_signals.ndim != 2:
            raise ValueError("base_alpha_signals must be 2D array (num_alphas, num_samples)")
        _,num_alphas = base_alpha_signals.shape
        if len(weights) != num_alphas:
            raise ValueError(f"Weight vector dim ({len(weights)}) != num alphas ({num_alphas})")

        #Normalize weights
        norm_weights = weights / (np.sum(np.abs(weights)) + 1e-8)

        #Combine weighted signals across alphas
        combined_signal_continuous = np.dot(base_alpha_signals,norm_weights)

        #Discretize the combined signal
        discrete_signal = _to_signals(combined_signal_continuous, signal_threshold)
        if len(discrete_signal) != len(train_df):
            raise ValueError(f"Discrete signal length {len(discrete_signal)} != train_df length {len(train_df)}")
        return discrete_signal.flatten()

    except Exception as e:
        print(f"Signal evaluation error: {e}")
        return np.zeros(base_alpha_signals.shape[1])

#Particle Class
class Particle:
    def __init__(self, dim: int, bounds: list):
        self.position = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        vel_val=abs(bounds[0][1]-bounds[0][0])*0.1
        self.velocity = np.random.uniform(-vel_val,vel_val, dim)
        self.fitness = -np.inf
        self.pbest_position = self.position.copy()
        self.pbest_fitness = -np.inf
        self.bounds = bounds
    def update_velocity(self, gbest_position: np.ndarray, w: float, c1: float, c2: float):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive_velocity = c1 * r1 * (self.pbest_position - self.position)
        social_velocity = c2 * r2 * (gbest_position - self.position)
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity
        max_vel = np.array([abs(b[1]-b[0])*0.5 for b in self.bounds])
        self.velocity = np.clip(self.velocity, -max_vel, max_vel)
    def update_position(self):
        self.position += self.velocity
        for i in range(len(self.position)):
            self.position[i] = np.clip(self.position[i], self.bounds[i][0], self.bounds[i][1])

#PSO Optimizer Function
def pso_optimizer(
    train_df: pd.DataFrame,
    base_alpha_signals_window:np.ndarray,
    signal_threshold: float,
    dim: int,
    bounds: list,
    num_particles: int = PSO_PARTICLES,
    max_iterations: int = PSO_ITERATIONS,
    w: float = PSO_W, c1: float = PSO_C1, c2: float = PSO_C2,
    initial_swarm_positions= None
) -> tuple:

    swarm = [Particle(dim, bounds) for _ in range(num_particles)]
    if initial_swarm_positions is not None and len(initial_swarm_positions) > 0:
        num_seed = min(len(swarm), len(initial_swarm_positions))
        for i in range(num_seed):
            noise = np.random.normal(0, np.mean([abs(b[1]-b[0]) for b in bounds]) * 0.05, dim)
            swarm[i].position = np.clip(initial_swarm_positions[i] + noise,
                                        [b[0] for b in bounds], [b[1] for b in bounds])
            swarm[i].pbest_position = swarm[i].position.copy()

    gbest_position = None
    gbest_fitness = -np.inf
    for iteration in range(max_iterations):
        #Generate signals for all particles
        all_signals = []
        for particle in swarm:
            signal = evaluate_combination(particle.position,train_df,base_alpha_signals_window, signal_threshold)
            all_signals.append(signal)
        all_signals = np.vstack(all_signals)  #shape: (num_particles, n_samples)

        #Run backtest across all signals
        backtest = VectorBacktest(train_df[['Close']], all_signals)
        fitness_values = backtest.fitness()
        #Update particles
        for i, particle in enumerate(swarm):
            particle.fitness = fitness_values.iloc[i] if np.isfinite(fitness_values.iloc[i]) else -np.inf

            if particle.fitness > particle.pbest_fitness:
                particle.pbest_fitness = particle.fitness
                particle.pbest_position = particle.position.copy()

            if particle.fitness > gbest_fitness:
                gbest_fitness = particle.fitness
                gbest_position = particle.position.copy()

            particle.update_velocity(gbest_position, w, c1, c2)
            particle.update_position()
        logging.info(f"Iter {iteration+1}/{max_iterations} | gbest_fitness (Train Sharpe): {gbest_fitness:.4f}")

    sorted_particles = sorted(swarm, key=lambda p: p.fitness, reverse=True)
    top_particles = sorted_particles[:10]
    top_positions = np.array([p.position for p in top_particles])
    top_fitness = np.array([p.fitness for p in top_particles])
    return gbest_position, gbest_fitness, top_positions, top_fitness


#Helper function to safely extract scalar values
def extract_scalar(value):
    """Safely extract scalar from pandas Series or return the value itself."""
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    return float(value)

#PSO pipeline function
def run_pso_pipeline(full_df: pd.DataFrame,
                     all_base_signals: np.ndarray,
                     train_start: int, train_len: int, test_len: int,
                     signal_threshold: float,
                     weight_bounds: list,
                     num_alphas: int,
                     pso_particles: int,
                     pso_iterations: int,
                     is_last: bool = False,
                     prev_best_weights: Optional[np.ndarray] = None
                    ) -> Optional[tuple]:
    train_end = train_start + train_len
    test_end = train_end + test_len
    if train_end > len(full_df):
        logging.info(f"Skipping window starting at {train_start}: Not enough data for training.")
        return None
    if test_end > len(full_df) and not is_last:
        logging.info(f"Skipping window starting at {train_start}: Not enough data for testing.")
        return None

    train_df = full_df.iloc[train_start:train_end].copy().reset_index(drop=True)
    actual_test_end = min(test_end, len(full_df))
    test_df = full_df.iloc[train_end:actual_test_end].copy().reset_index(drop=True)
    train_alpha_signals = all_base_signals[train_start:train_end, :]
    logging.info(f"Optimizing PSO for Window (Train Indices {train_start}-{train_end})")

    best_weights_train, best_fitness_train, top_positions, top_fitness = pso_optimizer(
        train_df=train_df,
        base_alpha_signals_window=train_alpha_signals,
        signal_threshold=signal_threshold,
        dim=num_alphas,
        bounds=weight_bounds,
        num_particles=pso_particles,
        max_iterations=pso_iterations,
        initial_swarm_positions=[prev_best_weights] if prev_best_weights is not None else None
    )
    logging.info(f"Training Complete. Best Train Sharpe: {best_fitness_train:.4f}. Now evaluating Best Weights on Test Window (Indices {train_end}-{actual_test_end})")

    oos_results = []
    port_ret_best = None
    if len(test_df) > 0:
        test_alpha_signals = all_base_signals[train_end:actual_test_end, :]
        all_weight_sets = np.vstack([best_weights_train[None, :], top_positions])
        all_signals_test = []
        for weights in all_weight_sets:
            norm_weights = weights / (np.sum(np.abs(weights)) + 1e-8)
            combined_cont = np.dot(test_alpha_signals, norm_weights)
            discrete_signal = _to_signals(combined_cont.reshape(1, -1), signal_threshold)
            all_signals_test.append(discrete_signal.flatten())

        all_signals_test = np.vstack(all_signals_test)
        test_backtest = VectorBacktest(test_df[['Close']], all_signals_test)
        fitnesses = test_backtest.fitness()

        sharpe_test = fitnesses
        mdd_test = test_backtest.fitness("max_drawdown")

        oos_arr = []
        for i in range(len(sharpe_test)):
            if not np.isinf(sharpe_test.iloc[i]):
                oos_arr.append((round(sharpe_test.iloc[i], 3),
                                round(-mdd_test.iloc[i], 3), i))

        if len(oos_arr):
            oos_arr.sort(key=lambda x: (x[0], x[1]), reverse=True)
            oos_results = []
            for sharpe, mdd, idx in oos_arr[:10]:
                ann_ret = round(test_backtest.port_ret().iloc[idx].annualized_return(), 3)
                oos_results.append((sharpe, ann_ret, mdd))

            print("Window OOS top strategies:", oos_results)
        else:
            print("Test set is empty, skipping evaluation.")
    else:
        print("Test set is empty, skipping evaluation.")

    if is_last:
        oos_start = train_end
        oos_end=len(full_df)-465
        print(f"The start and end are {oos_start},{oos_end}")
        oos_df = full_df.iloc[oos_start:oos_end].copy().reset_index(drop=True)
        oos_alpha_signals = all_base_signals[oos_start:, :]

        all_weight_sets = np.vstack([best_weights_train[None, :], top_positions])
        all_signals_oos = []
        for weights in all_weight_sets:
            norm_weights = weights / (np.sum(np.abs(weights)) + 1e-9)
            combined_cont = np.dot(oos_alpha_signals, norm_weights)
            discrete_signal = _to_signals(combined_cont.reshape(1, -1), signal_threshold)
            all_signals_oos.append(discrete_signal.flatten())

        all_signals_oos = np.vstack(all_signals_oos)
        oos_backtest = VectorBacktest(oos_df[['Close']], all_signals_oos)
        fitnesses = oos_backtest.fitness()

        sharpe_test = fitnesses
        mdd_test = oos_backtest.fitness("max_drawdown")

        oos_arr = []
        for i in range(len(sharpe_test)):
            if not np.isinf(sharpe_test.iloc[i]):
                oos_arr.append((round(sharpe_test.iloc[i], 3),
                                round(-mdd_test.iloc[i], 3), i))

        if len(oos_arr):
            oos_arr.sort(key=lambda x: (x[0], x[1]), reverse=True)

            oos_results = []
            best_idx = oos_arr[0][2]
            port_ret_best = oos_backtest.port_ret().iloc[best_idx]

            for sharpe, mdd, idx in oos_arr[:10]:
                ann_ret = round(oos_backtest.port_ret().iloc[idx].annualized_return(), 3)
                oos_results.append((sharpe, ann_ret, mdd))

            print("OOS top strategies:", oos_results)
        else:
            print("OOS Test set is empty, skipping evaluation.")
    else:
        print("OOS Test set is empty, skipping evaluation.+++++++++++++++++")
    if is_last:
        return best_weights_train, oos_results, port_ret_best
    else:
        return best_weights_train, oos_results


if __name__ == "__main__":
    compdf = pd.read_csv(data_path)
    # Keep the indicator layout stable for PSO (no shift / no shuffle)
    compdf = load_dataset(compdf, rng=rng, do_shift=False, do_shuffle=False)
    alpha_cols = list(compdf.columns[start_col:])
    num_alphas = len(alpha_cols)
    logging.info(f"Using {num_alphas} alpha signals: {alpha_cols[:5]}..." if len(alpha_cols) > 5 else f"Using {num_alphas} alpha signals")

    all_base_signals = compdf[alpha_cols].to_numpy()
    full_data_df = compdf[['Close']].copy()
    logging.info(f"Total data points: {len(full_data_df)}")

    WEIGHT_BOUNDS = [(-1.0, 1.0)] * num_alphas
    results_list = []
    best_weights_previous = None

    for start_ind in range(21):
        start = start_ind * STEP_SIZE
        end=start+TRAIN_LEN
        is_last_run = (start_ind==20)

        print(f"\n{'='*100}")
        logging.info(f"WINDOW {start_ind}: Start Index = {start}")
        logging.info(f"Train Dataset: {start} to {start + TRAIN_LEN} & Test Dataset: {start + TRAIN_LEN} to {start+TRAIN_LEN+TEST_LEN}")
        print(f"{'='*100}")

        best_weights_current,oos_res,best_portfolio = run_pso_pipeline(
            full_df=full_data_df,
            all_base_signals=all_base_signals,
            train_start=start,
            train_len=TRAIN_LEN,
            test_len=TEST_LEN,
            signal_threshold=SIGNAL_THRESHOLD,
            weight_bounds=WEIGHT_BOUNDS,
            num_alphas=num_alphas,
            pso_particles=PSO_PARTICLES,
            pso_iterations=PSO_ITERATIONS,
            is_last=is_last_run,
            prev_best_weights=best_weights_previous
        )

        if best_weights_current is not None:
            best_weights_previous = best_weights_current
            results_list.append({
                'window': start_ind,
                'start_idx': start,
                'weights': best_weights_current.copy(),
                'window_result': oos_res
            })

            print(f"############################# Window {start_ind} Summary #############################")
            logging.info(f"{results_list[-1]['window_result']}, {len(results_list[-1]['window_result'])}")
        else:
            print(f"Window {start_ind} (starting at {start}) failed or skipped.")

    print(f"{'='*100}")
    print("EXECUTION COMPLETE")
    print(f"{'='*100}")