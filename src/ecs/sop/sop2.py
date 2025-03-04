import numpy as np

# Function: Gaussian Random Walk with Clipping
def gaussian_random_walk(best_solution, current_solution, t, lower_bound=None, upper_bound=None):
    """
    Implements the Gaussian random walk for replacing Levy flights in Cuckoo Search,
    with boundary constraints to ensure solutions remain within the valid range.

    Parameters:
    - best_solution (numpy.ndarray): Best solution in the population.
    - current_solution (numpy.ndarray): Current cuckoo's position.
    - t (int): Current iteration (used for step size control).
    - lower_bound (float): Lower bound for the search space (default: -500).
    - upper_bound (float): Upper bound for the search space (default: 500).

    Returns:
    - numpy.ndarray: New solution after applying Gaussian perturbation, clipped within the search space.
    """
    r1, r2 = np.random.rand(), np.random.rand()  # Random factors
    sigma = (np.log(t + 2) / (t + 1)) * np.linalg.norm(best_solution - current_solution)  # Compute σ
    
    gaussian_noise = np.random.normal(best_solution, sigma, size=len(best_solution))  # Generate noise
    direction_term = r1 * best_solution - r2 * current_solution  # Adjust direction
    
    new_solution = gaussian_noise + direction_term  # New solution

    # Clip the values to ensure they remain within the bounds
    new_solution = np.clip(new_solution, lower_bound, upper_bound)

    return new_solution
