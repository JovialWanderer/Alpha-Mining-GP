from hyperparam import *
from TreeStruct import *

def tree_signal(base_signals: list[np.ndarray], node: TreeNode) -> np.ndarray:
    """
    Recursively evaluates a signal tree.
    """
    if node is None:
        return np.zeros(base_signals[0].shape, dtype=np.float16)

    # Leaf node: Return the corresponding base signal
    if node.left is None and node.right is None:
        return base_signals[node.val].astype(np.float16)

    #---Operator Evaluation using Dictionary Lookup---
    result = np.zeros(len(base_signals[0]), dtype=np.float16)
    try:
        op_func = OPERATOR_DISPATCH[node.val]
    except KeyError:
        raise ValueError(f"Invalid or unknown operator ID: {node.val}")

    # Check if the operator is binary or unary
    if node.val < NUM_BINARY_OPERATORS:
        # Binary operator
        left = tree_signal(base_signals, node.left)
        right = tree_signal(base_signals, node.right)
        result = op_func(left, right)
    else:
        # Unary operator
        child_signal = tree_signal(base_signals, node.left)
        result = op_func(child_signal)

    return result.astype(np.float16)