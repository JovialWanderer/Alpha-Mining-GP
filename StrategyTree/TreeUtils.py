from collections import deque
import itertools
from hyperparam import *
from TreeStruct import *
from TreeSignalCalc import tree_signal

def get_height(root: TreeNode) -> int:
    """Gets the height of the tree recursively."""
    if root is None:
        return 0
    return 1 + max(get_height(root.left), get_height(root.right))


def bfs(root: TreeNode, depth: int, rng: np.random.Generator) -> TreeNode | None:
    """
    Performs a breadth-first search (BFS) to find a random node at a given depth.
    """
    if depth < 0 or not root:
        return None
    if depth == 0:
        return root

    queue = deque([root])
    current_depth = 0

    while queue and current_depth < depth:
        level_size = len(queue)
        if level_size == 0: return None # Depth is greater than tree height
        
        for _ in range(level_size):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        current_depth += 1

    return rng.choice(list(queue)) if queue else None


def unary_create_tree(tree: TreeNode, op_node: TreeNode, val: int) -> TreeNode:
    """
    This function increases the depth of the tree using a UNARY operator.
    """
    op_node.val = val
    op_node.left = tree
    op_node.height = 1 + tree.height 
    return op_node

def binary_create_tree(tree1: TreeNode, tree2: TreeNode, op_node: TreeNode, val: int) -> TreeNode:
    """
    This function increases the depth of the tree using a BINARY operator.
    """
    op_node.val = val
    op_node.left = tree1
    op_node.right = tree2
    op_node.height = 1 + max(tree1.height, tree2.height)
    return op_node

def add_depth_binary(base_pop: list[TreeNode], n: int, rng: np.random.Generator) -> list[TreeNode]:
    """
    Creates n new trees by combining pairs of trees from the base population
    with a random binary operator.
    """
    if len(base_pop) < 2:
        return []

    #---Pair Selection---
    #Generate all possible unique pairs of indices
    all_pairs_indices = list(itertools.combinations(range(len(base_pop)), 2))
    
    #Sample with replacement
    num_pairs_to_sample = min(n, len(all_pairs_indices))
    replace = n > len(all_pairs_indices)
    
    chosen_pair_indices = rng.choice(len(all_pairs_indices), size=num_pairs_to_sample, replace=replace)

    base_pop_new = []
    num_binary_ops = config['indicators']['operators']['num_binary_operators']
    
    for pair_idx in chosen_pair_indices:
        left_id, right_id = all_pairs_indices[pair_idx]
        
        root = TreeNode()
        op_val = rng.integers(low=0, high=num_binary_ops)        
        new_tree = binary_create_tree(base_pop[left_id], base_pop[right_id], root, op_val)
        
        base_pop_new.append(new_tree)

    return base_pop_new

def add_depth_unary(base_pop: list[TreeNode], n: int, rng: np.random.Generator) -> list[TreeNode]:
    """
    Creates n new trees by applying a random unary operator to trees from the base population."""
    if not base_pop:
        return []

    population_size = len(base_pop)

    if n <= population_size:
        chosen_indices = rng.choice(population_size, size=n, replace=False)
    else:
        all_unique_indices = np.arange(population_size)
        rng.shuffle(all_unique_indices)
        remaining_needed = n - population_size
        
        #Sample remainder with replacement from full population.
        remaining_indices = rng.choice(population_size, size=remaining_needed, replace=True)
        chosen_indices = np.concatenate([all_unique_indices, remaining_indices])

    base_pop_new = []
    num_binary_ops = config['operators']['num_binary_operators']
    num_total_ops = config['operators']['num_operators']

    for i in chosen_indices:
        root = TreeNode()
        op_val = rng.integers(low=num_binary_ops, high=num_total_ops)
        new_tree = unary_create_tree(base_pop[i], root, op_val)
        
        base_pop_new.append(new_tree)
        
    return base_pop_new

def check_same(tree1:TreeNode, tree2:TreeNode) -> bool:
    """ Checks if two trees are structurally the same"""
    if tree1 is None and tree2 is None:
        return True
    if (tree1 is None or tree2 is None) or (tree1.val != tree2.val):
        return False

    # Check for commutative operators
    if tree1.val in [0,2,3,4]:  # Assuming 0 = +, 2 = *,3=max,4=min
        return ((check_same(tree1.left, tree2.left) and check_same(tree1.right, tree2.right)) or
                (check_same(tree1.left, tree2.right) and check_same(tree1.right, tree2.left)))

    # All other operators compared normally
    return check_same(tree1.left, tree2.left) and check_same(tree1.right, tree2.right)

def test_signal_generator(optim_tree,base_signals):
    final_signal=np.array([tree_signal(base_signals,optim_tree)][0])
    # Apply thresholding to classify signals
    final_signal = np.where(final_signal >config['backtest']['signal_threshold'], 1, np.where(final_signal < -config['backtest']['signal_threshold'], -1, 0))
    return final_signal


def dataset_preprocess(df,indicator_cols,start_idx,end_idx,isfirst=False,istest=False):
    if istest:
       base_signals = df[indicator_cols].values.T
       return base_signals
    base_signals=(df[indicator_cols][start_idx:end_idx].values.T)
    if isfirst:
        base_trees = [TreeNode(i) for i in range(len(indicator_cols))]
        return base_trees,base_signals
    return base_signals