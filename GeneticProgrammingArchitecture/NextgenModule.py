from dataclasses import dataclass
from hyperparam import *
from SimilarityScore import *
from StrategyTree.TreeStruct import TreeNode
from StrategyTree.TreeSignalCalc import tree_signal
from BacktestFolder.backtest import VectorBacktest
from GPUtils import GeneticOperators
import numpy as np
import random
import numpy as np
import pandas as pd
from typing import List, Tuple
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



class GenerationEvolver:
    """
    Manages the process of evolving one generation of strategies to the next.
    
    This class encapsulates fitness calculation, diversity management, elitism,
    crossover, mutation, and adaptive rate updates.
    """
    def __init__(self, config: dict, rng: np.random.Generator, ga_ops: GeneticOperators):
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
        self, population: List[TreeNode], dataset: pd.DataFrame, base_signals: list,return_pnl: bool = False
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
        if(return_pnl):
            return raw_fitness,backtest.get_portfolio()
        return raw_fitness#, pnl_array
    
    @staticmethod
    def _calculate_elite_mean_fitness(fitness_array: np.ndarray) -> float:
        """Calculates the mean fitness of the top elite_perc percent of individuals."""
        if fitness_array.size == 0:
            return 0.0
        try:
            fit_values = np.array([t[0] if isinstance(t, tuple) else t for t in fitness_array], dtype=float)
            #Handles potential NaNs or Infs introduced
            fit_values = fit_values[np.isfinite(fit_values)]
            if fit_values.size == 0:
                return 0.0
        except (TypeError, ValueError) as e:
            warnings.warn(f"Could not process fitness array: {fitness_array}. Error: {e}. Returning 0.0")
            return 0.0
        elite_perc = np.clip(config['evolutionary_algorithm']['population']['perc_elite'], 0.0, 100.0)
        if elite_perc == 0.0:
            return 0.0
        
        num_elite = max(1, int(np.floor(len(fitness_array) * elite_perc)))
        
        #Sort descending and take the mean of the top elite
        sorted_fitness = np.sort(fit_values)[::-1]
        return np.mean(sorted_fitness[:num_elite])
    
    def _update_adaptive_rates(
        self, optimizer_state: OptimizerState, curr_gen: int, 
        prev_fitness: np.ndarray, curr_fitness: np.ndarray
    ) -> OptimizerState:
        """
        Orchestrates the update of both crossover and mutation rates using Adam logic.
        This replaces the old `update_rates` function.
        """
        # Create a mutable copy to update
        new_state = copy.deepcopy(optimizer_state)
        
        cross_bounds = (config['evolutionary_algorithm']['crossover']['min'], config['evolutionary_algorithm']['crossover']['max'])
        next_cross_rate, mom_cross, vel_cross = self._adam_controller(
            prev_rate=new_state.prev_cross_rate, curr_rate=new_state.curr_cross_rate,
            prev_momentum=new_state.prev_cross_mom, prev_velocity=new_state.prev_cross_vel,
            prev_fitness_arr=prev_fitness, curr_fitness_arr=curr_fitness,
            rate_bounds=cross_bounds, curr_gen=curr_gen
        )
        new_state.prev_cross_rate, new_state.curr_cross_rate = new_state.curr_cross_rate, next_cross_rate
        new_state.prev_cross_mom, new_state.prev_cross_vel = mom_cross, vel_cross

        mut_bounds = (config['evolutionary_algorithm']['mutation']['min'], config['evolutionary_algorithm']['mutation']['max'])
        next_mut_rate, mom_mut, vel_mut = self._adam_controller(
            prev_rate=new_state.prev_mut_rate, curr_rate=new_state.curr_mut_rate,
            prev_momentum=new_state.prev_mut_mom, prev_velocity=new_state.prev_mut_vel,
            prev_fitness_arr=prev_fitness, curr_fitness_arr=curr_fitness,
            rate_bounds=mut_bounds, curr_gen=curr_gen
        )
        new_state.prev_mut_rate, new_state.curr_mut_rate = new_state.curr_mut_rate, next_mut_rate
        new_state.prev_mut_mom, new_state.prev_mut_vel = mom_mut, vel_mut
        
        return new_state

    def _adam_controller(
        self, prev_rate: float, curr_rate: float, prev_momentum: float, prev_velocity: float,
        prev_fitness_arr: np.ndarray, curr_fitness_arr: np.ndarray,
        rate_bounds: Tuple[float, float], curr_gen: int, epsilon: float = 1e-8
    ) -> Tuple[float, float, float]:
        """
        Dynamically adjusts a single GA operator rate using an Adam-like optimizer.
        This is the core logic from the old `adam_rate_controller` function.
        """
        #Calculate change in elite fitness
        prev_top_mean = self._calculate_elite_mean_fitness(prev_fitness_arr)
        curr_top_mean = self._calculate_elite_mean_fitness(curr_fitness_arr)
        delta_fitness = curr_top_mean - prev_top_mean
        delta_rate = curr_rate - prev_rate
        robust_delta_rate = np.sign(delta_rate) * max(abs(delta_rate), epsilon)

        #Estimate the gradient
        gradient = delta_fitness / robust_delta_rate

        #Update Momentum and Velocity
        beta1 = config['evolutionary_algorithm']['crossover_mutation_params']['beta1']
        beta2 = config['evolutionary_algorithm']['crossover_mutation_params']['beta2']
        
        curr_momentum = beta1 * prev_momentum + (1 - beta1) * gradient
        curr_velocity = beta2 * prev_velocity + (1 - beta2) * (gradient ** 2)
        
        #Apply bias correction
        momentum_hat = curr_momentum / (1 - beta1 ** (curr_gen))
        velocity_hat = curr_velocity / (1 - beta2 ** (curr_gen))

        #Update using Adam rule
        eta = config['evolutionary_algorithm']['crossover_mutation_params']['eta']
        next_rate = curr_rate + eta * momentum_hat / (np.sqrt(velocity_hat) + epsilon)

        #Clip the rate to bounds
        return np.clip(next_rate, rate_bounds[0], rate_bounds[1]), curr_momentum, curr_velocity

    def _suppress_fitness_by_similarity(
        self, pnl_array: np.ndarray, fitness_array: np.ndarray, curr_gen: int, tot_gen: int
    ) -> np.ndarray:
        """Applies a diversity penalty to fitness scores based on PnL similarity."""
        if self.is_simulated_annealing:
            multiplicative_factor = np.exp(-curr_gen / tot_gen)

            similarity_matrix = calculate_similarity_matrix_np(pnl_array)
            _, counts = analyze_similarity(similarity_matrix)
            penalized_fitness = np.array([
                (fit / ((1.0 + counts[index]) * multiplicative_factor)
                 if ((1 + counts[index]) * multiplicative_factor) > 1
                 else fit,
                 index)
                 for _, (fit, index) in enumerate(fitness_array)])

        else:
            penalized_fitness=fitness_array
        return penalized_fitness

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
        
        #Elitism: Preserve the best individuals

        org_fitness_arr = self._calculate_fitness(current_pop, dataset, base_signals)
        sorted_fitness_arr=sorted(fitness_arr_with_indices, key=lambda x: x[0], reverse=True)
        next_gen_pop=[current_pop[ind] for i,(_,ind) in sorted_fitness_arr[:self.num_elite]]
        
        
        #Crossover: Create the rest of the new generation
        num_children_needed = pop_size - self.num_elite+1# 1 to always have a pair of children when num_children is odd
        fitness_scores_only = np.array([f[0] for f in fitness_arr_with_indices])
        
        parent_trees,children_trees,parent_fit_arr=[],[],[]
        for _ in range(num_children_needed // 2):
            id1 = self.ga_ops.selection(fitness_scores_only, k=3)
            id2 = self.ga_ops.selection(fitness_scores_only, k=3)
            while id1 == id2:
                id2 = self.ga_ops.selection(fitness_scores_only, k=3)

            parent1, parent2 = current_pop[id1], current_pop[id2]
            
            #Perform crossover using the GeneticOperators class
            if self.rng.random() < optimizer_state.curr_cross_rate:
                child1, child2 = self.ga_ops.crossover(parent1, parent2)
                parent_trees.extend([parent1, parent2])
                children_trees.extend([child1, child2])
                parent_fit_arr.extend([org_fitness_arr[id1], org_fitness_arr[id2]])
            else:
                next_gen_pop.extend([copy.deepcopy(parent1), copy.deepcopy(parent2)])
        
        # Level 2 selection: Choose the best among parents and children
        if children_trees:
            children_fit_arr = self._calculate_fitness(children_trees, dataset, base_signals)
            for i in range(len(children_trees),2):
                if (max(children_fit_arr[i],children_fit_arr[i+1])> min(parent_fit_arr[i],parent_fit_arr[i+1])):
                    next_gen_pop.extend([children_trees[i], children_trees[i+1]])
                else:
                    next_gen_pop.extend([parent_trees[i], parent_trees[i+1]])
        #Mutation
        num_mutations = int(len(next_gen_pop) * optimizer_state.curr_mut_rate)
        if num_mutations > 0:
            indices_to_mutate = self.rng.choice(len(next_gen_pop), size=num_mutations, replace=False)
            for i in indices_to_mutate:
                self.ga_ops.mutation(next_gen_pop[i])

        #Evaluate the new generation
        new_raw_fitness, metrics = self._calculate_fitness(next_gen_pop, dataset, base_signals, return_pnl=True)
        signal_names=metrics.value().columns
        new_pnl_array=[]
        for signal_name in signal_names:
            new_pnl_array.append(metrics.value()[signal_name].values)
        new_penalized_fitness = self._suppress_fitness_by_similarity(new_pnl_array, new_raw_fitness, curr_gen, tot_gen)
        
        new_fitness_with_indices = list(zip(new_penalized_fitness, range(len(next_gen_pop))))
        
        #Update adaptive rates and return
        if not self.is_fixed_rate:
            # Assuming you have an unpenalized version of the previous fitness array
            optimizer_state = self._update_adaptive_rates(
                optimizer_state, curr_gen, 
                np.array([f[0] for f in fitness_arr_with_indices]), 
                np.array([f[0] for f in new_fitness_with_indices])
            )

        avg_new_fitness = np.mean(sorted(new_penalized_fitness, reverse=True)[:10])

        return next_gen_pop, avg_new_fitness, new_fitness_with_indices, new_pnl_array, optimizer_state