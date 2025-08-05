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

    def _get_initial_optimizer_state(self) -> OptimizerState:
        """Creates the initial state for the optimizer from the config."""
        return OptimizerState(
            prev_cross_rate=config['integration']['crossover_rates']['initial_prev_cross'],
            curr_cross_rate=config['integration']['crossover_rates']['initial_current_cross'],
            prev_cross_mom=config['integration']['crossover_rates']['initial_prev_cross_momentum'],
            prev_cross_vel= config['integration']['crossover_rates']['initial_prev_cross_velocity'],
            prev_mut_rate=config['integration']['mutation_rates']['initial_prev_mut'],
            curr_mut_rate=config['integration']['mutation_rates']['initial_current_mut'],
            prev_mut_mom=config['integration']['mutation_rates']['initial_prev_mut_momentum'],
            prev_mut_vel=config['integration']['mutation_rates']['initial_prev_mut_velocity'],
            beta1=config['evolutionary_algorithm']['crossover_mutation_params']['beta1'],
            beta2=config['evolutionary_algorithm']['crossover_mutation_params']['beta2'],
            eta=config['evolutionary_algorithm']['crossover_mutation_params']['eta'],
            dataset_iteration=0
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
            signal = tree_signal(base_signals, tree)
            final_signal = np.where(signal > config['backtest']['signal_threshold'], 1,
                                    np.where(signal < -self.config['backtest']['signal_threshold'], -1, 0))
            signals_to_test.append(final_signal)
        
        # 2. Run backtest
        backtest = VectorBacktest(dataset, np.array(signals_to_test))
        metrics = backtest.get_portfolio()
        
        # 3. Calculate raw fitness and PnL arrays
        raw_fitness = np.array(backtest.fitness(metric="sharpe"))
        pnl_array = [metrics.value()[col].values for col in metrics.value().columns]

        # 4. Apply diversity penalty
        # Assuming these are available utility functions
        similarity_matrix = calculate_similarity_matrix_np(pnl_array)
        _, counts = analyze_similarity(similarity_matrix)
        
        penalized_fitness = np.array([
            (fit / (1.0 + counts[i]),i) for i, fit in enumerate(raw_fitness)
        ])        
        return penalized_fitness

    def _run_evolution_loop(
        self, initial_pop: List[TreeNode], dataset_chunks: list, signal_chunks: list, 
        optimizer_state: OptimizerState, depth: int, is_high: bool
    ) -> Tuple[List[TreeNode], float, OptimizerState]:
        """A private helper to run the main generational evolution loop."""
        current_pop = initial_pop
        
        # Initial fitness evaluation for the starting population
        fitness_arr= self._evaluate_population(current_pop, dataset_chunks[0], signal_chunks[0])

        best_fitness = -np.inf
        best_strategy_pop = current_pop
        # Get top 10% for early stopping comparison
        sorted_fitness = sorted(fitness_arr, key=lambda x: x[0], reverse=True)
        prev_avg_fit = np.mean([x[0] for x in sorted_fitness[:num_elite]])
        
        stop_counter = 1
        
        # Distributed Evolution Loop
        total_len = sum(len(d) for d in dataset_chunks)
        num_generations = config['integration']['num_generations']
        
        for j, (data_chunk, signal_chunk) in enumerate(zip(dataset_chunks, signal_chunks)):
            gens_for_chunk = max(1, (num_generations * len(data_chunk)) // total_len)
            
            for gen in range(gens_for_chunk):
                simulated_next_gen=GenerationEvolver(
                    config=config,
                    rng=self.rng,
                    ga_ops=GeneticOperators(self.rng),
                )
                current_pop, next_avg_fit, fitness_arr, pnl_arr, optimizer_state = \
                    simulated_next_gen.evolve(
                    current_pop=current_pop,
                    fitness_arr_with_indices=fitness_arr,
                    dataset=data_chunk,
                    base_signals=signal_chunk,
                    curr_gen=gen+1,
                    tot_gen=gens_for_chunk+1,
                    **optimizer_state.__dict__
                )

                if next_avg_fit > best_fitness:
                    best_fitness = next_avg_fit
                    best_strategy_pop = current_pop.copy()

                # Early stopping logic
                if abs(next_avg_fit - prev_avg_fit) <= config['integration']['stop_threshold']:
                    stop_counter += 1
                    if stop_counter > config['integration']['stopping_generation']:
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