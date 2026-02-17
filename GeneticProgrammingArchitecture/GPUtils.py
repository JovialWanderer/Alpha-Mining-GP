import copy
from hyperparam import *
from StrategyTree.TreeUtils import *

import copy
import itertools
from typing import List, Tuple

import numpy as np

# Assume these are imported from your project
from hyperparam import config
from StrategyTree.TreeStruct import TreeNode, get_height, bfs

class GeneticOperators:
    """
    A class to encapsulate all genetic algorithm operators for strategy trees.
    
    This includes selection, crossover, and mutation operations, using a shared
    configuration and random number generator.
    """
    def __init__(self, rng: np.random.Generator):
        """
        Initializes the GeneticOperators with a random number generator and config.

        Args:
            rng: A NumPy random number generator instance.
        """
        self.rng = rng
        # Extract constants from the config for clean and easy access
        self.num_indicators = config['indicators']['num_indicators']
        self.num_binary_ops = config['operators']['num_binary_operators']
        self.num_total_ops = config['operators']['num_operators']

    def _unary_rootswap(self, root_a: TreeNode, root_b: TreeNode):
        """Swaps a child of a unary node with a child of a binary node."""
        if root_a.left and root_b.left:
            # Choose whether to swap with the binary node's left or right child
            if self.rng.integers(low=0, high=2):
                root_a.left, root_b.right = root_b.right, root_a.left
            else:
                root_a.left, root_b.left = root_b.left, root_a.left

    def _binary_rootswap(self, node_a: TreeNode, node_b: TreeNode):
        """Performs a full swap of two nodes (values and all children)."""
        node_a.val, node_b.val = node_b.val, node_a.val
        node_a.left, node_b.left = node_b.left, node_a.left
        node_a.right, node_b.right = node_b.right, node_a.right

    def selection(self, fitness_arr: np.ndarray, k: int = 3) -> int:
        """
        Performs tournament selection to choose the index of a single parent.

        Args:
            fitness_arr: A 1D NumPy array of fitness scores for all individuals.
            k: The number of individuals in the tournament.

        Returns:
            The index of the winning individual from the original population.
        """
        n_individuals = len(fitness_arr)
        if n_individuals == 0:
            raise ValueError("Cannot perform selection on an empty fitness array.")
        
        # Sample indices for the tournament without replacement
        competitor_indices = self.rng.choice(
            n_individuals, size=min(k, n_individuals), replace=False
        )
        
        # Find the winner within the tournament and return its original index
        tournament_fitness = fitness_arr[competitor_indices]
        winner_local_index = np.argmax(tournament_fitness)
        return competitor_indices[winner_local_index]

    def crossover(self, tree1: TreeNode, tree2: TreeNode) -> Tuple[TreeNode, TreeNode]:
        """
        Performs crossover between two strategy trees at randomly chosen subtrees.
        """
        height1 = tree1.height if hasattr(tree1, 'height') else get_height(tree1)
        height2 = tree2.height if hasattr(tree2, 'height') else get_height(tree2)

        if height1 <= 1 or height2 <= 1:
            return copy.deepcopy(tree1), copy.deepcopy(tree2)

        depth1 = self.rng.integers(low=0, high=height1)
        depth2 = self.rng.integers(low=0, high=height2)

        child1, child2 = copy.deepcopy(tree1), copy.deepcopy(tree2)
        node1 = bfs(child1, depth1, self.rng)
        node2 = bfs(child2, depth2, self.rng)

        if node1 and node2:
            is_node1_unary = node1.val >= self.num_binary_ops
            is_node2_unary = node2.val >= self.num_binary_ops

            if is_node1_unary and is_node2_unary:
                node1.left, node2.left = node2.left, node1.left
            elif is_node1_unary:
                self._unary_rootswap(node1, node2)
            elif is_node2_unary:
                self._unary_rootswap(node2, node1)
            else:  # Both are binary
                self._binary_rootswap(node1, node2)

        return child1, child2

    def mutation(self, root: TreeNode) -> TreeNode:
        """
        Performs point mutation on a strategy tree by modifying a random node's value.
        The tree is modified in-place.
        """
        if not root:
            return None

        tree_height = root.height if hasattr(root, 'height') else get_height(root)
        if tree_height == 0: return root

        random_depth = self.rng.integers(low=0, high=tree_height)
        node_to_mutate = bfs(root, random_depth, self.rng)

        if not node_to_mutate:
            return root

        is_leaf = node_to_mutate.left is None and node_to_mutate.right is None
        
        if is_leaf:
            # Create a list of all possible values except the current one
            possible_values = [i for i in range(self.num_indicators) if i != node_to_mutate.val]
            if possible_values:
                node_to_mutate.val = self.rng.choice(possible_values)
        else:  # It's an operator node
            is_unary = node_to_mutate.right is None
            if is_unary:
                possible_values = [i for i in range(self.num_binary_ops, self.num_total_ops) if i != node_to_mutate.val]
            else:  # Is binary
                possible_values = [i for i in range(self.num_binary_ops) if i != node_to_mutate.val]
            
            if possible_values:
                node_to_mutate.val = self.rng.choice(possible_values)
        
        return root