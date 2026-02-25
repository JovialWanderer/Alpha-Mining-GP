from GA_Integration.kwargs_dataclass import *
from StrategyTree.TreeStruct import TreeNode
from StrategyTree.TreeSignalCalc import tree_signal
from BacktestFolder.backtest import VectorBacktest
from GeneticProgrammingArchitecture.SimilarityScore import *
from WarmStart import PopulationWarmstarter
from GeneticProgrammingArchitecture.NextgenModule import *
from hyperparam import *
import logging
class StrategyEvolver:
    """
    Manages the evolutionary optimization process for trading strategies.
    
    This class handles the entire pipeline, from warm-starts to iteratively
    evolving strategies across different depths and datasets, including fitness
    evaluation and diversity management.
    """
    def __init__(self, config: dict, rng: np.random.Generator, warmstarter: PopulationWarmstarter, gen_evolver: GenerationEvolver):
        self.rng = rng
        self.warmstarter = warmstarter
        self.num_depth = config['integration']['num_depth']
        self.gen_evolver = gen_evolver

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
        """
        raw_fitness, metrics = self.gen_evolver._calculate_fitness(population, dataset, base_signals, return_pnl=True)
        pnl_array = [metrics.value()[col].values for col in metrics.value().columns]
        penalized_fitness = self.gen_evolver._suppress_fitness_by_similarity(pnl_array, raw_fitness, 0, config['integration']['num_generations'])
        
        return penalized_fitness

    def _run_evolution_loop(
        self, initial_pop: List[TreeNode], dataset_chunks: list, signal_chunks: list, 
        optimizer_state: OptimizerState, depth: int, is_high: bool
    ) -> Tuple[List[TreeNode], float, OptimizerState]:
        """A private helper to run the main generational evolution loop."""
        current_pop = initial_pop.copy()
        
        # Initial fitness evaluation for the starting population
        fitness_arr= self._evaluate_population(current_pop, dataset_chunks[0], signal_chunks[0])
        print(f"Total dataset chunks: {len(dataset_chunks)}. Starting evolution loop with initial population of size {len(current_pop)}. Length of fitness array: {len(fitness_arr)}. ")
        best_fitness = -np.inf
        best_strategy_pop = current_pop.copy()
        #Get top 10% for early stopping comparison
        sorted_fitness = sorted(fitness_arr, key=lambda x: x[0], reverse=True)
        num_elite = max(1, len(sorted_fitness) // 10)
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
                current_pop, next_avg_fit, fitness_arr,_, optimizer_state = simulated_next_gen.evolve(
                    current_pop=current_pop,
                    fitness_arr_with_indices=fitness_arr,
                    dataset=data_chunk,
                    base_signals=signal_chunk,
                    optimizer_state=optimizer_state,
                    curr_gen=gen+1,
                    tot_gen=gens_for_chunk+1,
                )
                logging.info(f"For chunk {j+1}/{len(dataset_chunks)}, generation {gen+1}/{gens_for_chunk} and length of current population {len(current_pop)} .")
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
        """
        optimizer_state = self._get_initial_optimizer_state()
        strategy_population = base_trees[0]
        depth_results = {}

        for d in range(2, self.num_depth + 1):
            warm_pop = self.warmstarter.begin(strategy_population)
            print(f"Running evolution for depth {d} with population size {len(warm_pop)}...")
            # Run the main evolution loop
            best_pop, best_fit, final_state = self._run_evolution_loop(
                warm_pop.copy(), dataset, base_signals, optimizer_state, d, is_high
            )
            print(f"Completed evolution for depth {d} and no of individuals {len(best_pop)}")
            depth_results[d] = {
                'best_fit': best_fit,
                'tree_opt': best_pop,
                'optimizer_state': final_state
            }
            strategy_population = best_pop.copy()  # The best from this depth becomes the base for the next
            print(f"##*****************DEPTH {d} has BEST_FITNESS of {best_fit}***********************##")
        return depth_results
    
    def run_advanced_evolution(self, dataset: list, base_signals: list, new_trees: list,
                               depth:int,best_fit:float,warmstart_percent:float,warmstart_trees:list[TreeNode],
                               ishigh: bool,optimizer_state: OptimizerState) -> dict:
        """
        Runs the full, advanced evolutionary process for a new volatility regime.
        """
        strategy_population = new_trees[0]
        warm_pop = self.warmstarter.advance(prev_trees=strategy_population.copy(),new_base_trees=warmstart_trees.copy(),factor=warmstart_percent)

        # Run the main evolution loop
        strategy_population, best_fit, final_state = self._run_evolution_loop(
            warm_pop.copy(), dataset, base_signals, optimizer_state, depth, ishigh
        )
        
        depth_results= {
            'best_fit': best_fit,
            'tree_opt': strategy_population,
            'optimizer_state': final_state
        }
        print(f"##*****************ADVANCED DEPTH {depth} has BEST_FITNESS of {best_fit}***********************##")
        return depth_results
    
