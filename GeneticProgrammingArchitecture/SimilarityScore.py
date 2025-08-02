from hyperparam import *
import warnings
def calculate_similarity_matrix_np(data_matrix):
    """
    Calculates the Pearson similarity (correlation) matrix for a 2D NumPy array.

    Assumes rows are variables (e.g., PnL arrays) and columns are observations (e.g., time points).

    Args:
        data_matrix: A 2D NumPy array where each row is a data series (e.g., shape (num_individual, num_days)).

    Returns:
        A 2D NumPy array representing the pairwise Pearson correlation matrix
        (e.g., shape (num_individual,num_individual)). Returns NaN for correlations involving
        constant rows (zero standard deviation) or rows with insufficient data points.
    """
    # --- Input Validation---
    if not isinstance(data_matrix, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if data_matrix.ndim != 2:
        raise ValueError(f"Input must be a 2D array, but got {data_matrix.ndim} dimensions.")

    num_arrays = data_matrix.shape[0]
    num_samples = data_matrix.shape[1]

    if num_arrays == 0:
        print("Warning: Input array has 0 rows. Returning empty matrix.")
        return np.array([]) # Return empty 2D array

    if num_arrays == 1:
        print("Warning: Input array has only 1 row. Correlation with itself is 1.")
        return np.array([[1.0]]) # Correlation of an array with itself

    if num_samples < 2:
        print(f"Warning: Number of samples ({num_samples}) is less than 2. Cannot calculate correlation.")
        return np.full((num_arrays, num_arrays), np.nan) # Return NaN matrix

    # --- Calculate Correlation Matrix ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Ensure input is float for correlation calculation
        correlation_matrix = np.corrcoef(data_matrix.astype(float, copy=False))

    # Safeguard for unexpected scalar output (should be handled by checks above)
    if correlation_matrix.shape == ():
         return np.full((num_arrays, num_arrays), np.nan)

    # Ensure the output shape is correct
    if correlation_matrix.shape != (num_arrays, num_arrays):
         print(f"Warning: Unexpected output shape {correlation_matrix.shape}. Expected ({num_arrays}, {num_arrays}). Returning NaN matrix.")
         return np.full((num_arrays, num_arrays), np.nan)

    return correlation_matrix


def analyze_similarity(similarity_matrix, threshold=config['evolutionary_algorithm']['similarity']['sim_threshold']):
    """
    Analyzes a similarity matrix: replaces NaNs with 0 and counts items
    above a threshold for each row (excluding self-similarity).

    Args:
        similarity_matrix: The 2D NumPy array correlation matrix (potentially containing NaNs).
        threshold: The float correlation threshold for counting neighbors.

    Returns:
        A tuple containing:
        - cleaned_matrix: The similarity matrix with NaNs replaced by 0.
        - counts_above_threshold: A 1D NumPy array where counts_above_threshold[i]
                                  is the number of items j (j!=i) such that
                                  cleaned_matrix[i, j] > threshold.
    """
    if not isinstance(similarity_matrix, np.ndarray) or similarity_matrix.ndim != 2:
         raise ValueError("Input similarity_matrix must be a 2D NumPy array.")
    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
         raise ValueError("Input similarity_matrix must be square.")

    # 1. Replace NaNs with 0
    cleaned_matrix = np.nan_to_num(similarity_matrix, nan=0.0)

    # 2. Count elements above threshold (excluding diagonal)
    num_items = cleaned_matrix.shape[0]
    if num_items == 0:
        return cleaned_matrix, np.array([], dtype=int) # Return empty counts for empty input

    # Create boolean matrix where condition (value > threshold) is met
    above_threshold_matrix = cleaned_matrix > threshold

    # The diagonal is False to exclude self-similarity (corr(i, i)),so we only count j != i
    np.fill_diagonal(above_threshold_matrix, False)

    # Sum the True values along each row (axis=1) to get the count for each item
    counts_above_threshold = np.sum(above_threshold_matrix, axis=1, dtype=int)

    return cleaned_matrix, counts_above_threshold