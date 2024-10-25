import numpy  as np
from scipy.stats.qmc import Sobol

class SobolInitialization:
    def __init__(self, seed=None):
       # Seed for reproducible Sobol sequence generation
        self.seed = seed
    
    def sobol_initialization(self, size, min_values, max_values, target_function, start_init=None):
        """
        Returns population array with points and their fitness values
        """
        dim = len(min_values)
        min_values = np.array(min_values)
        max_values = np.array(max_values)
        
        if not all(max_values > min_values):
            raise ValueError("All max_values must be greater than min_values")
        
        if start_init is not None:
            start_init = np.atleast_2d(start_init)
            if start_init.shape[1] != dim:
                raise ValueError(f"start_init must have {dim} dimensions")
            
            n_rows = size - start_init.shape[0]
            
            if n_rows > 0:
                # Combine provided initial points with additional Sobol points
                sampler = Sobol(d=dim, scramble=True, seed=self.seed)
                sobol_points = sampler.random(n=n_rows)
                scaled_points = min_values + (max_values - min_values) * sobol_points
                population = np.vstack((start_init[:, :dim], scaled_points))
            else:
                population = start_init[:size, :dim]
        else:
            sampler = Sobol(d=dim, scramble=True, seed=self.seed)
            sobol_points = sampler.random(n=size)
            population = min_values + (max_values - min_values) * sobol_points
        
        # Handle both vectorized and non-vectorized target functions
        try:
            fitness_values = target_function(population) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, population)
            fitness_values = np.atleast_1d(fitness_values)
            if fitness_values.ndim > 2:
                raise ValueError("target_function must return a 1D or 2D array")
        except Exception as e:
            raise RuntimeError(f"Error calculating fitness values: {str(e)}")
            
        population = np.hstack((population, fitness_values[:, np.newaxis] if fitness_values.ndim == 1 else fitness_values))
        
        return population