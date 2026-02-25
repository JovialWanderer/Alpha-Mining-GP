import copy
from hyperparam import *
from StrategyTree.TreeUtils import *

import copy
import itertools
from typing import List, Tuple

import numpy as np
import logging

# Assume these are imported from your project
from hyperparam import config
from StrategyTree.TreeStruct import TreeNode

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
        self.num_binary_ops = config['indicators']['num_binary_operators']
        self.num_total_ops = config['indicators']['num_operators']

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

    def selection(self, fitness_arr: List[Tuple[float, int]], k: int = 3) -> int:
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

        depth1 = self.rng.integers(low=0, high=height1-1)
        depth2 = self.rng.integers(low=0, high=height2-1)

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
                swap_choice = self.rng.choice(['left_left', 'right_right', 'left_right', 'right_left'])
                if swap_choice == 'left_left' and node1.left and node2.left:
                    self._binary_rootswap(node1.left, node2.left)
                elif swap_choice == 'left_right' and node1.left and node2.right:
                    self._binary_rootswap(node1.left, node2.right)
                elif swap_choice == 'right_left' and node1.right and node2.left:
                    self._binary_rootswap(node1.right, node2.left)
                elif swap_choice == 'right_right' and node1.right and node2.right:
                    self._binary_rootswap(node1.right, node2.right)

            # end if node1 and node2

        # Sanitize children before returning to prevent invalid operator IDs
        try:
            self._sanitize_tree(child1)
            self._sanitize_tree(child2)
        except Exception as e:
            logging.info("Failed to sanitize children after crossover: %s", e)

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
        
        # Sanitize mutated tree before returning
        try:
            self._sanitize_tree(root)
        except Exception as e:
            logging.info("Failed to sanitize tree after mutation: %s", e)
        return root

    def _sanitize_tree(self, root: TreeNode):
        """
        Walk the tree and ensure node.val values are valid for the node's arity.
        Repairs invalid operator IDs by replacing them with a random valid ID
        based on whether the node is a leaf, unary operator, or binary operator.
        """
        if root is None:
            return

        stack = [root]
        while stack:
            node = stack.pop()
            # Determine arity from children
            is_leaf = (node.left is None and node.right is None)
            is_unary = (node.right is None and node.left is not None)
            is_binary = (node.left is not None and node.right is not None)

            try:
                if is_leaf:
                    # Leaves must reference a base indicator
                    if not (0 <= node.val < self.num_indicators):
                        old = node.val
                        node.val = int(self.rng.integers(low=0, high=self.num_indicators))
                        logging.info("Sanitized leaf node val %s -> %s", old, node.val)
                elif is_unary:
                    if not (self.num_binary_ops <= node.val < self.num_total_ops):
                        old = node.val
                        node.val = int(self.rng.integers(low=self.num_binary_ops, high=self.num_total_ops))
                        # Ensure right child is None for unary nodes
                        node.right = None
                        logging.info("Sanitized unary node val %s -> %s", old, node.val)
                elif is_binary:
                    if not (0 <= node.val < self.num_binary_ops):
                        old = node.val
                        node.val = int(self.rng.integers(low=0, high=self.num_binary_ops))
                        logging.info("Sanitized binary node val %s -> %s", old, node.val)
            except Exception as e:
                # Defensive: log and continue
                logging.info("Error while sanitizing node %s: %s", getattr(node, 'val', None), e)

            # Push children
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)