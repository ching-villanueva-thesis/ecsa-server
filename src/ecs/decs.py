import numpy as np
from scipy.special import gamma
from scipy.stats.qmc import Sobol

from .sop import SobolInitialization, CosineAnnealingWithWarmRestarts

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

    # Generate Sobol sequence samples
    sobol_sampler = Sobol(d=dim, scramble=True)  # Scramble adds randomness while maintaining low discrepancy
    sobol_samples = sobol_sampler.random(n=size)

    # Convert to binary: 1 if the value is >= 0.5, else 0
    binary_vector = (sobol_samples >= 0.5).astype(int)

    fitness_values = np.array([target_function(solution) for solution in binary_vector])

    return binary_vector, fitness_values

def get_cuckoo(nests, lambda_value, alpha_value):
    random_nest = np.random.randint(nests.shape[0])
    n = nests[random_nest].shape[0]

    # Calculate Lévy flight step
    # Using Mantegna's algorithm
    sigma_u = (gamma(1 + lambda_value) * np.sin(np.pi * lambda_value / 2) / 
              (gamma((1 + lambda_value) / 2) * lambda_value * 2**((lambda_value - 1) / 2)))**(1 / lambda_value)
    sigma_v = 1
    
    u = np.random.normal(0, sigma_u, n)
    v = np.random.normal(0, sigma_v, n)
    
    step = u / (np.abs(v)**(1 / lambda_value))
    
    # Generate new solution
    step_size = alpha_value * step
    new_solution = nests[random_nest] + step_size
    
    # Convert to probability (Sigmoid)
    probability = sigmoid(new_solution)
    
    # Convert to binary using probability
    random_values = np.random.random(n)
    binary_solution = (probability > random_values).astype(int)

    return binary_solution, random_nest

def abandon_nest(nests, f_values, discovery_rate):
    new_nests = np.copy(nests)
    new_f_values = np.copy(f_values)
    abandoned_nest = int(np.ceil(discovery_rate * nests.shape[0])) + 1
    nest_list = np.argsort(f_values)[-abandoned_nest:]

    random_birds = np.random.choice(nests.shape[0], size=2, replace=False)
    bird_j, bird_k = random_birds

    for i in nest_list:
        if np.random.rand() > discovery_rate:
            rand = np.random.rand(nests[i].shape[0])
            
            new_sol = nests[i] + rand * (new_nests[bird_j] - new_nests[bird_k])

            probability = sigmoid(new_sol)

            new_nests[i] = (probability > np.random.rand(nests[i].shape[0])).astype(int)


            new_f_values[i] = target_function(new_nests[i])

    return new_nests, new_f_values


def discrete_enhanced_cuckoo_search_algorithm(size = 3, discovery_rate = 0.25, alpha_value = 0.01, lambda_value = 1.5, dimensions = (5,5), iterations = 50, target_function = target_function):
    nests, f_values = initialize_nests(size, dimensions, target_function)
    best_ind = f_values.argsort()[0]
    count = 0

    d_max, d_min = discovery_rate
    d_cos_annealing = CosineAnnealingWithWarmRestarts(n_max=d_max, n_min=d_min, t_max=iterations/5)

    a_max, a_min = alpha_value
    a_cos_annealing = CosineAnnealingWithWarmRestarts(n_max=a_max, n_min=a_min, t_max=iterations/5)

    fmin = []
    while(count <= iterations):
        # adaptive step size
        a_val = a_cos_annealing.step()

        for _ in range(0, nests.shape[0]):
            new_sol, old_sol = get_cuckoo(nests=nests, lambda_value=lambda_value, alpha_value=a_val)

            new_fitness = target_function(new_sol)
            
            if(new_fitness < f_values[old_sol]):
                nests[old_sol] = new_sol
                f_values[old_sol] = new_fitness

        # adaptive discovery rate
        d_rate = d_cos_annealing.step()
            
        # print("Alpha val: ", a_val)
        # print("Discovery rate: ", d_rate)
        
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

# db_coordinates, es_coordinates = getDataFromDataset()

# d = Distance(db_coordinates=db_coordinates, es_coordinates=es_coordinates)

# nests, f_values, best, best_fev = discrete_enhanced_cuckoo_search_algorithm(size=50, dimensions=(len(db_coordinates), len(es_coordinates)), iterations=500, discovery_rate=(0.75, 0.25), alpha_value=(0.01, 0.05), target_function=d.fitness)

# print(best)
# print(best_fev)