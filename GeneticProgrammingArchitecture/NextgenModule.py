from dataclasses import dataclass
import copy
@dataclass
class OptimizerState:
    """A structured container for the evolutionary optimizer's state."""
    prev_cross_rate: float
    curr_cross_rate: float
    prev_cross_mom: float
    prev_cross_vel: float
    prev_mut_rate: float
    curr_mut_rate: float
    prev_mut_mom: float
    prev_mut_vel: float
    beta1: float
    beta2: float
    eta: float
    dataset_iteration: int = 0
from hyperparam import *
from SimilarityScore import *
from StrategyTree.TreeStruct import TreeNode
from StrategyTree.TreeSignalCalc import tree_signal
from BacktestFolder.backtest import VectorBacktest
import numpy as np
import random
import numpy as np
import pandas as pd
from typing import List, Tuple

class GenerationEvolver:
    """
    Manages the process of evolving one generation of strategies to the next.
    
    This class encapsulates fitness calculation, diversity management, elitism,
    crossover, mutation, and adaptive rate updates.
    """
    def __init__(self, config: dict, rng: np.random.Generator, ga_ops: 'GeneticOperators'):
        """
        Initializes the GenerationEvolver.

        Args:
            config: The main project configuration dictionary.
            rng: A NumPy random number generator.
            ga_ops: An instance of the GeneticOperators class.
        """
        self.config = config
        self.rng = rng
        self.ga_ops = ga_ops
        
        # Extract key parameters for easier access
        self.frac_elite = config['evolutionary_algorithm']['population']['frac_elite']
        self.num_elite = int(config['evolutionary_algorithm']['population']['num_individuals'] * self.frac_elite)
        self.is_fixed_rate = config['ablation']['fixed_rates']['is_fixed_rate']
        self.is_simulated_annealing = config['ablation']['is_simulated_annealing']
        self.signal_threshold = config['backtest']['signal_threshold']

    def _calculate_fitness(
        self, population: List[TreeNode], dataset: pd.DataFrame, base_signals: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates raw fitness and PnL arrays for a population."""
        signals_to_test = []
        for tree in population:
            signal = tree_signal(base_signals, tree)
            final_signal = np.where(signal > self.signal_threshold, 1, np.where(signal < -self.signal_threshold, -1, 0))
            signals_to_test.append(final_signal)
            
        backtest = VectorBacktest(dataset, signals_to_test)
        
        raw_fitness = np.array(backtest.fitness(metric="sharpe"))
        # Clip fitness values to a reasonable range
        raw_fitness = np.clip(raw_fitness, -200.0, 200.0)
        
        pnl_array = backtest.get_portfolio().value().to_numpy().T
        return raw_fitness, pnl_array

    def _suppress_fitness_by_similarity(
        self, pnl_array: np.ndarray, fitness_array: np.ndarray, curr_gen: int, tot_gen: int
    ) -> np.ndarray:
        """Applies a diversity penalty to fitness scores based on PnL similarity."""
        # This check is from your original code, it seems to disable the penalty in later generations
        if self.is_simulated_annealing:
            multiplicative_factor = np.exp(-curr_gen / tot_gen)
            if multiplicative_factor <= 1.0 / (1.0 + 1): # Heuristic check
                return fitness_array

        similarity_matrix = calculate_similarity_matrix_np(pnl_array)
        _, counts = analyze_similarity(similarity_matrix)
        
        penalized_fitness = np.array([
            fit / (1.0 + counts[i]) for i, fit in enumerate(fitness_array)
        ])
        return penalized_fitness

    def _update_adaptive_rates(
        self, optimizer_state: OptimizerState, curr_gen: int, prev_fitness: np.ndarray, curr_fitness: np.ndarray
    ) -> OptimizerState:
        """Updates crossover and mutation rates using the ADAM optimizer logic."""
        # Create a mutable copy to update
        new_state = copy.deepcopy(optimizer_state)

        # Update Crossover Rate
        # ... (Assuming adam_rate_controller is a function you have defined)
        # new_state.curr_cross_rate, new_state.prev_cross_mom, new_state.prev_cross_vel = adam_rate_controller(...)
        
        # Update Mutation Rate
        # new_state.curr_mut_rate, new_state.prev_mut_mom, new_state.prev_mut_vel = adam_rate_controller(...)
        
        return new_state

    def evolve(
        self,
        current_pop: List[TreeNode],
        fitness_arr_with_indices: List[Tuple[float, int]],
        dataset: pd.DataFrame,
        base_signals: list,
        optimizer_state: OptimizerState,
        curr_gen: int,
        tot_gen: int
    ) -> Tuple[List[TreeNode], float, List[Tuple[float, int]], np.ndarray, OptimizerState]:
        """
        Runs a single generation of the genetic algorithm.
        This method replaces the old `simulated_next_generation` function.
        """
        pop_size = len(current_pop)
        
        # --- 1. Elitism: Preserve the best individuals ---
        sorted_indices = sorted(range(pop_size), key=lambda i: fitness_arr_with_indices[i][0], reverse=True)
        elite_indices = sorted_indices[:self.num_elite]
        next_gen_pop = [current_pop[i] for i in elite_indices]
        
        # --- 2. Crossover: Create the rest of the new generation ---
        num_children_needed = pop_size - self.num_elite
        fitness_scores_only = np.array([f[0] for f in fitness_arr_with_indices])
        
        for _ in range(num_children_needed // 2):
            # Select parents using tournament selection from the GeneticOperators class
            id1 = self.ga_ops.selection(fitness_scores_only, k=3)
            id2 = self.ga_ops.selection(fitness_scores_only, k=3)
            while id1 == id2:
                id2 = self.ga_ops.selection(fitness_scores_only, k=3)

            parent1, parent2 = current_pop[id1], current_pop[id2]
            
            # Perform crossover using the GeneticOperators class
            if self.rng.random() < optimizer_state.curr_cross_rate:
                child1, child2 = self.ga_ops.crossover(parent1, parent2)
                next_gen_pop.extend([child1, child2])
            else:
                next_gen_pop.extend([copy.deepcopy(parent1), copy.deepcopy(parent2)])

        # --- 3. Mutation ---
        num_mutations = int(len(next_gen_pop) * optimizer_state.curr_mut_rate)
        if num_mutations > 0:
            indices_to_mutate = self.rng.choice(len(next_gen_pop), size=num_mutations, replace=False)
            for i in indices_to_mutate:
                # Perform mutation using the GeneticOperators class
                self.ga_ops.mutation(next_gen_pop[i])

        # --- 4. Evaluate the new generation ---
        new_raw_fitness, new_pnl_array = self._calculate_fitness(next_gen_pop, dataset, base_signals)
        
        # Apply diversity penalty
        new_penalized_fitness = self._suppress_fitness_by_similarity(new_pnl_array, new_raw_fitness, curr_gen, tot_gen)
        
        new_fitness_with_indices = list(zip(new_penalized_fitness, range(len(next_gen_pop))))
        
        # --- 5. Update adaptive rates and return ---
        if not self.is_fixed_rate:
            # Assuming you have an unpenalized version of the previous fitness array
            # optimizer_state = self._update_adaptive_rates(...)
            pass

        avg_new_fitness = np.mean(sorted(new_penalized_fitness, reverse=True)[:self.num_elite])

        return next_gen_pop, avg_new_fitness, new_fitness_with_indices, new_pnl_array, optimizer_state