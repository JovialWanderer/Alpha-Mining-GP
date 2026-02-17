import numpy as np
from hyperparam import *

class TreeNode:
  def __init__(self, val=0, height=1, ismut=False, left=None, right=None):
    '''
    Tree structure used for the genetic algorithm (GA).
    Internal nodes are operators, and leaves are base strategies.
    '''
    self.val = val
    self.left = left
    self.right = right
    self.height = height
    self.ismut = ismut

  def __repr__(self):
    return f"TreeNode({self.val})"



OPERATOR_DISPATCH = {
    # --- Binary Operators ---
    0: np.add,
    1: np.subtract,
    2: np.multiply,
    3: np.maximum,
    4: np.minimum,
    # 5: lambda left, right: np.where(right != 0, left / right, 1.0), # Example for your commented-out division

    # --- Unary Operators ---
    5: np.abs,
    6: np.cos,
    7: np.sin,
    8: np.tan,
    9: np.exp,
    10: lambda signal: np.log(np.maximum(signal, 1e-8)), # Safe log
    # Add entries for 11, 12, etc. here
}
NUM_BINARY_OPERATORS = config['indicators']['num_binary_operators']
NUM_TOTAL_OPERATORS = config['indicators']['num_operators']