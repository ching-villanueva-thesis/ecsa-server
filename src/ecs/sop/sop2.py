import numpy as np

def gaussian_random_walk(best_solution, current_solution, t, lower_bound=None, upper_bound=None):
    r1, r2 = np.random.rand(), np.random.rand()  # Random factors
    sigma = (np.log(t + 2) / (t + 1)) * np.linalg.norm(best_solution - current_solution)  # Compute σ
    
    gaussian_noise = np.random.normal(current_solution, sigma, size=len(current_solution))  # Generate noise
    direction_term = r1 * best_solution - r2 * current_solution  # Adjust direction
    
    new_solution = gaussian_noise + direction_term  # New solution

    # Clip the values to ensure they remain within the bounds
    new_solution = np.clip(new_solution, lower_bound, upper_bound)

    return new_solution
