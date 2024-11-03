import numpy as np
from scipy.special import gamma

def target_function(sol):
    return np.random.rand()

def sigmoid(new_solution):
    return 1 / (1 + np.exp(-new_solution))

def initialize_nests(size, dimensions):
    # building/barangay (bb), evacuation space (es)
    bb, es = dimensions

    # Generate a vector with values uniformly distributed between 0 and 1
    uniform_samples = np.random.uniform(0, 1, (size, bb, es))

    # Convert to binary: 1 if the value is >= 0.5, else 0
    binary_vector = (uniform_samples >= 0.5).astype(int)
    fitness_values = np.random.rand(size)

    return binary_vector, fitness_values

def get_cuckoo(nests, lambda_value, alpha_value):
    random_nest = np.random.randint(nests.shape[0])
    bb,es = nests[random_nest].shape

    # Calculate Lévy flight step
    # Using Mantegna's algorithm
    sigma_u = (gamma(1 + lambda_value) * np.sin(np.pi * lambda_value / 2) / 
              (gamma((1 + lambda_value) / 2) * lambda_value * 2**((lambda_value - 1) / 2)))**(1 / lambda_value)
    sigma_v = 1
    
    u = np.random.normal(0, sigma_u, (bb,es))
    v = np.random.normal(0, sigma_v, (bb,es))
    
    step = u / (np.abs(v)**(1 / lambda_value))
    
    # Generate new solution
    step_size = alpha_value * step
    new_solution = nests[random_nest] + step_size
    
    # Convert to probability (Sigmoid)
    probability = sigmoid(new_solution)
    
    # Convert to binary using probability
    random_values = np.random.random((bb, es))
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
            bb,es = nests[i].shape
            rand = np.random.rand(bb,es)
            
            new_sol = nests[i] + rand * (new_nests[bird_j] - new_nests[bird_k])

            probability = sigmoid(new_sol)

            new_nests[i] = (probability > np.random.rand(bb,es)).astype(int)

            new_f_values[i] = target_function(new_nests[i])

    return new_nests, new_f_values


def discrete_cuckoo_search_algorithm(size = 3, discovery_rate = 0.25, alpha_value = 0.01, lambda_value = 1.5, dimensions = (5,5), iterations = 50, target_function = target_function):
    nests, f_values = initialize_nests(size, dimensions)
    best_ind = f_values.argsort()[0]
    count = 0

    while(count <= iterations):
      for _ in range(0, nests.shape[0]):
          new_sol, old_sol = get_cuckoo(nests=nests, lambda_value=lambda_value, alpha_value=alpha_value)

          new_fitness = target_function(new_sol)
          print("Previous Sol\n", nests[old_sol])
          print("New Sol\n", new_sol)
          
          if(new_fitness < f_values[old_sol]):
               nests[old_sol] = new_sol
               f_values[old_sol] = new_fitness
          
      n_nests, n_f_values = abandon_nest(nests=nests, f_values=f_values, discovery_rate=discovery_rate)

      nests = n_nests
      f_values = n_f_values

      value_ind = f_values.argsort()[0]

      if f_values[best_ind] > f_values[value_ind]:
          best_ind = value_ind

      count += 1

    best = nests[best_ind]
    return nests, f_values, best

nests, f_values, best = discrete_cuckoo_search_algorithm(size=15, dimensions=(10, 5), iterations=10)