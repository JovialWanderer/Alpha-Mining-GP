import numpy as np
import pandas as pd
from typing import List
from hyperparam import *
from StrategyTree.TreeStruct import TreeNode
from StrategyTree.TreeUtils import add_depth_binary, add_depth_unary

class PopulationWarmstarter:
    """
    Manages the warm-start process for generating populations of strategy trees.
    This class tackles the initiation and sequential deprecating warm-starts for evolving populations.
    """
    def __init__(self,rng: np.random.Generator):
        """
        Initializes the Warmstarter with core configuration parameters.

        Args:
            rng: A NumPy random number generator for all stochastic operations.
        """
        self.num_individuals = config['evolutionary_algorithm']['population']['num_individuals']
        if self.num_individuals <= 0:
            raise ValueError("Number of individuals must be positive.")
        
        self.initial_warmstart_factor = config['intergration']['ini_warm_factor']
        self.rng = rng

    def _create_initial_population(self, base_trees: List[TreeNode]) -> List[TreeNode]:
        """
        Generates a mix of unary and binary rooted trees for the first warmstart.
        This is a private helper method.
        """
        count = len(base_trees)
        target_size = int(self.num_individuals * self.initial_warmstart_factor)
        
        # Determine the split between unary and binary trees
        num_rng = self.rng.integers(low=0, high=count + 1)
        num_unary = min(num_rng, target_size//5)
        num_binary = target_size - num_unary
        
        # Ensure counts are not negative
        if num_binary < 0:
            num_binary = 0
            num_unary = target_size

        binary_signals = add_depth_binary(base_trees, num_binary, self.rng)
        unary_signals = add_depth_unary(base_trees, num_unary, self.rng)
        return binary_signals + unary_signals

    def begin(self, base_trees: List[TreeNode]) -> List[TreeNode]:
        """
        Performs the initial warmstart.        
        Generates a larger-than-needed population and randomly samples from it to create the first generation of strategies.
        """
        warm_trees = self._create_initial_population(base_trees)
        
        population_size = min(self.num_individuals, len(warm_trees))
        chosen_indices = self.rng.choice(len(warm_trees), size=population_size, replace=False)
        
        final_strategy_trees = [warm_trees[i] for i in chosen_indices]
        return final_strategy_trees

    def _create_advanced_population(self, base_trees: List[TreeNode], factor: float) -> List[TreeNode]:
        """
        Generates new unary and binary trees for subsequent warmstarts.
        """
        count = int(self.num_individuals * factor)
        if count == 0 or not base_trees:
            return []

        #The number of new unary trees relatively small
        num_unary = self.rng.integers(low=0, high=max(2, count // 5))
        num_binary = count - num_unary
        
        print(f"Number of new binary & unary trees: {num_binary},{num_unary} (Total: {count})")
        
        binary_signals = add_depth_binary(base_trees, num_binary, self.rng)
        unary_signals = add_depth_unary(base_trees, num_unary, self.rng)

        return binary_signals + unary_signals

    def advance(self, prev_trees: List[TreeNode], new_base_trees: List[TreeNode], factor: float) -> List[TreeNode]:
        """
        Performs an advanced warmstart for subsequent generations.

        Combines the previous generation with newly generated trees and then
        samples to create the next generation.
        """
        if factor <= 0.01:
            return prev_trees

        new_warm_trees = self._create_advanced_population(new_base_trees, factor)
        
        #Combine the new trees with the best from the previous generation
        combined_pool = new_warm_trees + prev_trees
        
        #Sample from the combined pool to create the final population
        population_size = min(self.num_individuals, len(combined_pool))
        chosen_indices = self.rng.choice(len(combined_pool), size=population_size, replace=False)

        final_strategy_trees = [combined_pool[i] for i in chosen_indices]
        return final_strategy_trees