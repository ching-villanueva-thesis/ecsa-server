import numpy as np
import math
from scipy.special import gamma
from scipy.stats.qmc import Sobol
from scipy.spatial.distance import hamming

from .sop import CosineAnnealingWithWarmRestarts

from src.dataset import getDataFromDataset
from functions.distance import Distance

def target_function(sol):
    return np.random.rand()

def sigmoid(new_solution):
    return 1 / (1 + np.exp(-new_solution))

def initialize_nests(size, dimensions, target_function):
    # building/barangay (bb), evacuation space (es)
    bb, es = dimensions
    dim = bb * es

    # Compute smallest m such that 2^m >= size
    m = math.ceil(math.log2(size))
    sobol_sampler = Sobol(d=dim, scramble=True)
    sobol_samples = sobol_sampler.random_base2(m=m)  # Generates 2^m samples

    # Truncate to desired size
    sobol_samples = sobol_samples[:size]

    # Transform to binary using thresholding
    binary_population = np.where(sobol_samples < np.mean(sobol_samples, axis=0), 0, 1)

    fitness_values = np.array([target_function(sol) for sol in binary_population])

    return binary_population, fitness_values

def levy_flight(lambda_value, n):
    # Calculate Lévy flight step
    # Using Mantegna's algorithm
    sigma_u = (gamma(1 + lambda_value) * np.sin(np.pi * lambda_value / 2) / 
              (gamma((1 + lambda_value) / 2) * lambda_value * 2**((lambda_value - 1) / 2)))**(1 / lambda_value)
    sigma_v = 1
    
    u = np.random.normal(0, sigma_u, n)
    v = np.random.normal(0, sigma_v, n)
    
    step = u / (np.abs(v)**(1 / lambda_value))

    return step

def gaussian_random_walk(best_solution, current_solution, t, lower_bound=None, upper_bound=None):
    r1, r2 = np.random.rand(), np.random.rand()  # Random factors
    sigma = (np.log(t + 2) / (t + 1)) * np.linalg.norm(best_solution - current_solution)  # Compute σ

    sigma = sigma * (1 - np.exp(-0.1 * t))
    
    gaussian_noise = np.random.normal(current_solution, sigma, size=len(current_solution))  # Generate noise
    direction_term = r1 * best_solution - r2 * current_solution  # Adjust direction
    
    new_solution = gaussian_noise + direction_term  # New solution

    # Clip the values to ensure they remain within the bounds
    new_solution = np.clip(new_solution, lower_bound, upper_bound)

    return new_solution

def get_cuckoo(nests, best_solution, lambda_value, alpha_value, walk_type, t):
    random_nest = np.random.randint(nests.shape[0])
    n = nests[random_nest].shape[0]
    
    new_solution = None

    if walk_type == 0:
        step = levy_flight(lambda_value, n)
        step_size = alpha_value * step
        new_solution = nests[random_nest] + step_size
    else:
        new_solution = gaussian_random_walk(best_solution, nests[random_nest], t, lower_bound=0, upper_bound=1)
    
    prob = sigmoid(new_solution)
    binary_solution = (np.random.rand(*prob.shape) < prob).astype(int)
    
    return binary_solution, random_nest

def abandon_nest(nests, f_values, discovery_rate):
    new_nests = np.copy(nests)
    new_f_values = np.copy(f_values)
    abandoned_nest = int(np.ceil(discovery_rate * nests.shape[0]))
    nest_list = np.argsort(f_values)[-abandoned_nest:]

    random_birds = np.random.choice(nests.shape[0], size=2, replace=False)
    bird_j, bird_k = random_birds

    for i in nest_list:
        if np.random.rand() > discovery_rate:
            rand = np.random.rand(nests[i].shape[0])
            
            new_sol = nests[i] + rand * (new_nests[bird_j] - new_nests[bird_k])

            prob = sigmoid(new_sol)
            binary_solution = (np.random.rand(*prob.shape) < prob).astype(int)
            
            new_nests[i] = binary_solution
            new_f_values[i] = target_function(new_nests[i])

    return new_nests, new_f_values


def discrete_enhanced_cuckoo_search_algorithm(size = 3, discovery_rate = 0.25, alpha_value = 0.01, lambda_value = 1.5, dimensions = (5,5), iterations = 50, target_function = target_function):
    nests, f_values = initialize_nests(size, dimensions, target_function)
    best_ind = f_values.argsort()[0]
    count = 0

    d_max, d_min = discovery_rate
    d_cos_annealing = CosineAnnealingWithWarmRestarts(n_max=d_max, n_min=d_min, t_max=iterations, c_max=4)

    a_max, a_min = alpha_value
    a_cos_annealing = CosineAnnealingWithWarmRestarts(n_max=a_max, n_min=a_min, t_max=iterations, c_max=4)

    fmin = []
    while(count <= iterations):
        # adaptive step size
        a_val = a_cos_annealing.step() if a_cos_annealing.walk_type == 0 else a_cos_annealing.step()

        for _ in range(0, nests.shape[0]):
            new_sol, old_sol = get_cuckoo(nests=nests, lambda_value=lambda_value, alpha_value=a_val, walk_type=a_cos_annealing.walk_type, t=count, best_solution=nests[best_ind])

            new_fitness = target_function(new_sol)
            
            if(new_fitness < f_values[old_sol]):
                nests[old_sol] = new_sol
                f_values[old_sol] = new_fitness

        # adaptive discovery rate
        d_rate = d_cos_annealing.step() if d_cos_annealing.walk_type == 0 else d_cos_annealing.step()
        
        n_nests, n_f_values = abandon_nest(nests=nests, f_values=f_values, discovery_rate=d_rate)

        nests = n_nests
        f_values = n_f_values

        value_ind = f_values.argsort()[0]

        if f_values[best_ind] > f_values[value_ind]:
            best_ind = value_ind

        fmin.append(f_values[f_values.argsort()[0]])
        count += 1

    best = nests[best_ind]
    best_fev = f_values[f_values.argsort()[0]]
    return nests, f_values, fmin, best, best_fev

