import copy
from hyperparam import *
from StrategyTree.TreeUtils import *

def unary_rootswap(root_a:TreeNode, root_b:TreeNode,rng:np.random.Generator):
  """
  Function to swap subtree when the root node of one
  subtree is a unary operator and the other is binary.
  Args:
    root_a: Unary operator rooted subtree
    root_b:Binary operator rooted subtree
  """
  if root_a.left and root_b.left:
    left_subtree= root_a.left
    r = rng.randint(0, 2)
    if r:
      root_a.left, root_b.right = root_b.right, left_subtree
    else:
      root_a.left, root_b.left = root_b.left, left_subtree


def binary_rootswap(node_a:TreeNode, node_b:TreeNode):
  """
  Function to swap subtree when the root node of both
  subtrees are binary operators.
  Args:
    node_a: Binary operator rooted subtree
    node_b:Binary operator rooted subtree
  """
  node_a.val, node_b.val = node_b.val, node_a.val  # Swap values
  #Swap left subtree
  node_a.left, node_b.left = node_b.left, node_a.left

  #Swap right subtree
  node_a.right, node_b.right = node_b.right, node_a.right


#This is the crossover between 2 buy or 2 sell signals at any random node at a particular depth d.
def crossover(tree1:TreeNode, tree2:TreeNode,rng:np.random.Generator):
  """
  Perform crossover between two strategy trees at random depths.
  Args:
    tree1 (TreeNode): First strategy tree.
    tree2 (TreeNode): Second strategy tree.

  Returns:
     tuple: Two child trees resulting from the crossover.
  """
  height1 = get_height(tree1)
  height2 = get_height(tree2)
  if (height1<= 1 or height2<=1):
    return copy.deepcopy(tree1), copy.deepcopy(tree2)  # Return copies if tree too shallow

  depth1 = rng.randint(0, height1 - 1)  # range = [0, h-2]
  depth2 = rng.randint(0, height2 - 1)

  child1, child2 = copy.deepcopy(tree1), copy.deepcopy(tree2)
  root1, root2 = bfs(child1, depth1), bfs(child2, depth2)

  if root1 and root2:
    # If both nodes are unary operators
    if root1.val >= config['indicators']['operators']['num_binary_operators'] and root2.val >= config['indicators']['operators']['num_binary_operators']:
      root1.left, root2.left = root2.left, root1.left

    # If only one is 'abs'; apply unary swap
    elif root1.val >= config['indicators']['operators']['num_binary_operators']:
      unary_rootswap(root1, root2)
    elif root2.val >= config['indicators']['operators']['num_binary_operators']:
      unary_rootswap(root2, root1)

    # Otherwise, perform standard binary swap
    else:
      swap_choice = rng.choice(['left_left', 'right_right', 'left_right', 'right_left'])
      if swap_choice == 'left_left' and root1.left and root2.left:
        binary_rootswap(root1.left, root2.left)
      elif swap_choice == 'left_right' and root1.left and root2.right:
        binary_rootswap(root1.left, root2.right)
      elif swap_choice == 'right_left' and root1.right and root2.left:
        binary_rootswap(root1.right, root2.left)
      elif swap_choice == 'right_right' and root1.right and root2.right:
        binary_rootswap(root1.right, root2.right)

  return child1, child2


def mutation(root:TreeNode,rng:np.random.Generator,num_base: int=config['indicators']['num_indicators']):
  """
  Perform (point)mutation on a strategy tree by modifying a random node's value.
  Args:
    root (TreeNode): Root of the strategy tree to mutate.
    num_base(int) : Number of base signals
  Returns:
    The tree is modified in-place.
  """
  if root is None:
    return root  # Handle edge case where tree is empty

  tree_height = get_height(root)
  random_depth = rng.randint(0, tree_height)  # Pick a random depth
  node = bfs(root, random_depth)  # Get a random node at that depth
  node_height = get_height(node)  # Get the height of the node
  if node is None:
    return root  # No valid node found, return tree unchanged

  def set_value(beg,end):
    new_val = rng.randint(beg,end)
    while new_val == node.val:  # Ensure mutation actually changes the value
      new_val = rng.randint(beg,end)
    node.val = new_val
  if node_height == 1:  # If it's a leaf node
    set_value(0,num_base)

  else:  # If it's an internal node
    if node.val >= config['indicators']['operators']['num_binary_operators']:  # If it's a NOT operator (assuming 5 represents NOT)
      set_value(config['indicators']['operators']['num_binary_operators'],config['indicators']['num_operators'])
    else:
      set_value(0,config['indicators']['operators']['num_binary_operators'])
  return root

# tournament selection to determine the parents
def tournament_selection(fitness_arr:list[float],rng:np.random.Generator,k:int = 3):
    """
    Perform tournament selection to choose a parent based on fitness.
    Args:
        fitness_arr (list): List of fitness scores for individuals.
        k (int, optional): Number of individuals to select for the tournament (default: 3).
    Returns:
        int: Index of the selected individual.
    """
    tournament = rng.sample(list(enumerate(fitness_arr)), k)
    winner = max(tournament, key=lambda x: float(x[1][0]))
    id = winner[1][1]
    return id

from typing import List, Tuple


NUM_INDICATORS = config['indicators']['num_indicators']
NUM_BINARY_OPS = config['operators']['operators']['num_binary_operators']
NUM_TOTAL_OPS = config['operators']['num_operators']


#Helper Functions for Crossover

def unary_rootswap(root_a: TreeNode, root_b: TreeNode, rng: np.random.Generator):
    """Swaps a child of a unary node with a child of a binary node."""
    if root_a.left and root_b.left:
        if rng.integers(low=0, high=2):
            root_a.left, root_b.right = root_b.right, root_a.left
        else:
            root_a.left, root_b.left = root_b.left, root_a.left


def binary_rootswap(node_a: TreeNode, node_b: TreeNode):
    """Performs a full swap of two nodes (values and all children)."""
    node_a.val, node_b.val = node_b.val, node_a.val
    node_a.left, node_b.left = node_b.left, node_a.left
    node_a.right, node_b.right = node_b.right, node_a.right


#Core Genetic Operator Functions

def crossover(
    tree1: TreeNode, tree2: TreeNode, rng: np.random.Generator
) -> Tuple[TreeNode, TreeNode]:
    """
    Performs crossover between two strategy trees at randomly chosen subtrees.
    """
    height1 = tree1.height if hasattr(tree1, 'height') else get_height(tree1)
    height2 = tree2.height if hasattr(tree2, 'height') else get_height(tree2)

    if height1 <= 1 or height2 <= 1:
        return copy.deepcopy(tree1), copy.deepcopy(tree2)

    depth1 = rng.integers(low=0, high=height1)
    depth2 = rng.integers(low=0, high=height2)

    child1, child2 = copy.deepcopy(tree1), copy.deepcopy(tree2)
    node1 = bfs(child1, depth1, rng)
    node2 = bfs(child2, depth2, rng)

    if node1 and node2:
        is_node1_unary = node1.val >= NUM_BINARY_OPS
        is_node2_unary = node2.val >= NUM_BINARY_OPS

        if is_node1_unary and is_node2_unary:
            node1.left, node2.left = node2.left, node1.left
        elif is_node1_unary:
            unary_rootswap(node1, node2, rng)
        elif is_node2_unary:
            unary_rootswap(node2, node1, rng)
        else:  # Both are binary
            binary_rootswap(node1, node2)

    return child1, child2


def mutation(root: TreeNode, rng: np.random.Generator) -> TreeNode:
    """
    Performs point mutation on a strategy tree by modifying a random node's value.
    The tree is modified in-place and returned.
    """
    if root is None:
        return None

    tree_height = root.height if hasattr(root, 'height') else get_height(root)
    if tree_height == 0: return root

    # Correctly use rng.integers to select depth
    random_depth = rng.integers(low=0, high=tree_height)
    node_to_mutate = bfs(root, random_depth, rng)

    if node_to_mutate is None:
        return root  # No valid node found

    # --- Robustly select a new, different value ---
    is_leaf = node_to_mutate.left is None and node_to_mutate.right is None
    
    if is_leaf:
        possible_values = [i for i in range(NUM_INDICATORS) if i != node_to_mutate.val]
        if possible_values:
            node_to_mutate.val = rng.choice(possible_values)
    else: # It's an operator node
        is_unary = node_to_mutate.right is None
        if is_unary:
            possible_values = [i for i in range(NUM_BINARY_OPS, NUM_TOTAL_OPS) if i != node_to_mutate.val]
        else: # Is binary
            possible_values = [i for i in range(NUM_BINARY_OPS) if i != node_to_mutate.val]
        
        if possible_values:
            node_to_mutate.val = rng.choice(possible_values)
            
    return root


def tournament_selection(fitness_arr: np.ndarray, rng: np.random.Generator, k: int = 3) -> int:
    """
    Performs tournament selection to choose the index of a parent based on fitness.

    Args:
        fitness_arr: A 1D NumPy array of fitness scores for all individuals.
        rng: A NumPy random number generator.
        k: The number of individuals in the tournament.

    Returns:
        The index of the winning individual.
    """
    n_individuals = len(fitness_arr)
    if n_individuals == 0:
        raise ValueError("Cannot perform selection on an empty fitness array.")
    
    # 1. Correctly sample *indices* for the tournament without replacement
    competitor_indices = rng.choice(n_individuals, size=min(k, n_individuals), replace=False)
    
    # 2. Get the fitness scores of the competitors
    tournament_fitness = fitness_arr[competitor_indices]
    
    # 3. Find the index of the winner *within the tournament*
    winner_local_index = np.argmax(tournament_fitness)
    
    # 4. Return the winner's *original* index from the full population
    winner_global_index = competitor_indices[winner_local_index]
    
    return winner_global_index