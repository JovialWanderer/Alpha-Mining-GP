from kwargs_dataclass import *
from StrategyTree.TreeStruct import TreeNode
from StrategyTree.TreeSignalCalc import tree_signal
from BacktestFolder.backtest import VectorBacktest
from GeneticProgrammingArchitecture.SimilarityScore import *
from WarmStart import PopulationWarmstarter
from GeneticProgrammingArchitecture.NextgenModule import *
from hyperparam import *
class StrategyEvolver:
    """
    Manages the evolutionary optimization process for trading strategies.
    
    This class handles the entire pipeline, from warm-starts to iteratively
    evolving strategies across different depths and datasets, including fitness
    evaluation and diversity management.
    """
    def __init__(self, config: dict, rng: np.random.Generator, warmstarter: PopulationWarmstarter):
        self.config = config
        self.rng = rng
        self.warmstarter = warmstarter
        self.num_depth = config['training']['num_depth']
        self.ga_params = config['ga']
        self.adam_params = config['adam']
        
        # Trackers are now instance attributes, not globals
        self.tree_tracker_high: Dict[int, List[TreeNode]] = {}
        self.tree_tracker_low: Dict[int, List[TreeNode]] = {}
        self.count_tracker_high: Dict[int, List[int]] = {}
        self.count_tracker_low: Dict[int, List[int]] = {}

    def _get_initial_optimizer_state(self) -> OptimizerState:
        """Creates the initial state for the optimizer from the config."""
        return OptimizerState(
            prev_cross_rate=self.ga_params['ini_prev_cross'], # ... and so on for all params
            # ... (fill in the rest of the optimizer state parameters from config)
        )

    def _evaluate_population(
        self, population: List[TreeNode], dataset: pd.DataFrame, base_signals: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates a population, calculates fitness, and applies diversity penalties.
        This is a reusable helper that replaces duplicated code.
        """
        # 1. Generate signals for the entire population
        signals_to_test = []
        for tree in population:
            signal = tree_signal(base_signals, tree) # Assumes tree_signal is available
            final_signal = np.where(signal > self.config['execution']['signal_threshold'], 1,
                                    np.where(signal < -self.config['execution']['signal_threshold'], -1, 0))
            signals_to_test.append(final_signal)
        
        # 2. Run backtest
        backtest = VectorBacktest(dataset, np.array(signals_to_test))
        metrics = backtest.get_portfolio()
        
        # 3. Calculate raw fitness and PnL arrays
        raw_fitness = np.array(backtest.fitness(metric="sharpe"))
        pnl_array = metrics.value().to_numpy().T

        # 4. Apply diversity penalty
        # Assuming these are available utility functions
        similarity_matrix = calculate_similarity_matrix_np(pnl_array)
        _, counts = analyze_similarity(similarity_matrix)
        
        penalized_fitness = np.array([
            fit / (1.0 + counts[i]) for i, fit in enumerate(raw_fitness)
        ])
        
        return penalized_fitness, pnl_array

    def _run_evolution_loop(
        self, initial_pop: List[TreeNode], dataset_chunks: list, signal_chunks: list, 
        optimizer_state: OptimizerState, depth: int, is_high: bool
    ) -> Tuple[List[TreeNode], float, OptimizerState]:
        """A private helper to run the main generational evolution loop."""
        current_pop = initial_pop
        
        # Initial fitness evaluation for the starting population
        fitness_arr, pnl_arr = self._evaluate_population(current_pop, dataset_chunks[0], signal_chunks[0])

        best_fitness = -np.inf
        best_strategy_pop = current_pop
        
        # Get top 10% for early stopping comparison
        num_elite = self.config['ga']['num_elite']
        sorted_fitness = np.sort(fitness_arr)[::-1]
        prev_avg_fit = np.mean(sorted_fitness[:num_elite])
        
        stop_counter = 0
        
        # Distributed Evolution Loop
        total_len = sum(len(d) for d in dataset_chunks)
        num_generations = self.config['ga']['num_generations']
        
        for j, (data_chunk, signal_chunk) in enumerate(zip(dataset_chunks, signal_chunks)):
            gens_for_chunk = max(1, (num_generations * len(data_chunk)) // total_len)
            
            for gen in range(gens_for_chunk):
                # This function contains your core GA logic (selection, crossover, mutation)
                current_pop, next_avg_fit, fitness_arr, pnl_arr, optimizer_state = \
                    simulated_next_generation(
                        signal_chunk, current_pop, data_chunk, gen, gens_for_chunk,
                        fitness_arr, depth, is_high, **optimizer_state.__dict__
                    )

                if next_avg_fit > best_fitness:
                    best_fitness = next_avg_fit
                    best_strategy_pop = current_pop.copy()

                # Early stopping logic
                if abs(next_avg_fit - prev_avg_fit) <= self.config['training']['stop_threshold']:
                    stop_counter += 1
                    if stop_counter > self.config['training']['stopping_generation']:
                        return best_strategy_pop, best_fitness, optimizer_state
                else:
                    stop_counter = 1
                prev_avg_fit = next_avg_fit
        
        return best_strategy_pop, best_fitness, optimizer_state

    def run_initial_evolution(self, dataset: list, base_signals: list, base_trees: list, is_high: bool) -> dict:
        """
        Runs the full, initial evolutionary process for a new volatility regime.
        This replaces the old `integrator` and `best_strategy` functions.
        """
        optimizer_state = self._get_initial_optimizer_state()
        strategy_population = base_trees[0]
        depth_results = {}

        for d in range(2, self.num_depth + 1):
            warm_pop = self.warmstarter.begin(strategy_population)
            
            # Update trackers
            tracker = self.count_tracker_high if is_high else self.count_tracker_low
            tree_dict = self.tree_tracker_high if is_high else self.tree_tracker_low
            tree_dict[d] = warm_pop
            if d not in tracker: tracker[d] = []
            tracker[d].append(len(warm_pop))

            # Run the main evolution loop
            best_pop, best_fit, final_state = self._run_evolution_loop(
                warm_pop, dataset, base_signals, optimizer_state, d, is_high
            )
            
            depth_results[d] = {
                'best_fit': best_fit,
                'tree_opt': best_pop,
                'optimizer_state': final_state
            }
            strategy_population = best_pop  # The best from this depth becomes the base for the next
            print(f"##*****************DEPTH {d} has BEST_FITNESS of {best_fit}***********************##")
        
        return depth_results

    # The 'run_advanced_evolution' method would be structured similarly,
    # calling warmstarter.advance() and the same _run_evolution_loop helper.