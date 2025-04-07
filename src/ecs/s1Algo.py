import numpy  as np
from src.ecs.sop import CosineAnnealingWithWarmRestarts
from src.ecs.sop import SobolInitialization
from src.ecs.sop import gaussian_random_walk

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_variables(size, min_values, max_values, target_function, start_init = None):
    dim = len(min_values)
    if (start_init is not None):
        start_init = np.atleast_2d(start_init)
        n_rows     = size - start_init.shape[0]
        if (n_rows > 0):
            rows       = np.random.uniform(min_values, max_values, (n_rows, dim))
            start_init = np.vstack((start_init[:, :dim], rows))
        else:
            start_init = start_init[:size, :dim]
        fitness_values = target_function(start_init) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, start_init)
        population     = np.hstack((start_init, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    else:
        population     = np.random.uniform(min_values, max_values, (size, dim))
        fitness_values = target_function(population) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, population)
        population     = np.hstack((population, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    return population

############################################################################

# Function: Levy Distribution
def levy_flight(mean):
    u1 = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)
    u2 = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)
    v  = np.random.uniform(0.0, 1.0)
    x1 = np.sin((mean - 1.0) * u1) / np.power(np.cos(u1), 1.0 / (mean - 1.0))
    x2 = np.power(np.cos((2.0 - mean) * u2) / (-np.log(v)), (2.0 - mean) / (mean - 1.0))
    return x1 * x2

# Function: Replace Bird
def replace_bird(position, alpha_value, lambda_value, min_values, max_values, target_function, best_solution, t, walk_type = None):
    random_bird  = np.random.randint(position.shape[0])

    new_solution = np.copy(position[random_bird, :-1])
    rand_factors = np.random.rand(len(min_values))

    """
        Hybrid
        walk_type 0: Levy
        walky_type 1: Gaussian
    """
    if walk_type == 0 or walk_type is None:
        levy_values = levy_flight(lambda_value)
        new_solution = np.clip(new_solution + alpha_value * levy_values * new_solution * rand_factors, min_values, max_values)
    if walk_type == 1:
        new_solution = gaussian_random_walk(best_solution=best_solution, current_solution=new_solution, t=t, upper_bound=max_values, lower_bound=min_values)

    new_fitness  = target_function(new_solution)
    new_fitness  = target_function(new_solution)
    if (new_fitness < position[random_bird, -1]):
        position[random_bird,:-1] = new_solution
        position[random_bird, -1] = new_fitness
    return position

# Function: Update Positions
def update_positions(position, discovery_rate, min_values, max_values, target_function):
    updated_position = np.copy(position)
    abandoned_nests  = int(np.ceil(discovery_rate * position.shape[0])) + 1
    fitness_values   = position[:, -1]
    nest_list        = np.argsort(fitness_values)[-abandoned_nests:]
    random_birds     = np.random.choice(position.shape[0], size=2, replace=False)
    bird_j, bird_k   = random_birds
    for i in nest_list:
        rand = np.random.rand(updated_position.shape[1] - 1)
        if np.random.rand() > discovery_rate:
            updated_position[i, :-1] = np.clip(updated_position[i, :-1] + rand * (updated_position[bird_j, :-1] - updated_position[bird_k, :-1]), min_values, max_values)
    updated_position[:, -1] = np.apply_along_axis(target_function, 1, updated_position[:, :-1])
    updated_position = np.vstack([updated_position, position])
    updated_position = updated_position[updated_position[:, -1].argsort()]
    updated_position = updated_position[:position.shape[0], :]
    return updated_position

############################################################################

# Function: CS
def enhanced_cuckoo_search(birds = 3, discovery_rate = 0.25, alpha_value = 0.01, lambda_value = 1.5, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function, verbose = True, start_init = None, target_value = None, simulator = False, init_population = None): 
    """
    Enhanced Cuckoo Search algorithm using Sobol sequence initialization
    """
    if not simulator:
        # Create instance of SobolInitialization
        sobol_initializer = SobolInitialization()
        # Use the sobol_initialization method
        position = sobol_initializer.sobol_initialization(
            size=birds,
            min_values=min_values,
            max_values=max_values,
            target_function=target_function,
            start_init=start_init
        )
    else:
        position = init_population

    best_ind = np.copy(position[position[:,-1].argsort()][0,:])
    count    = 0
    
    hasReachedTarget = False
    _iter = 0
    fmin = []

    return best_ind, position, hasReachedTarget, _iter, fmin

############################################################################