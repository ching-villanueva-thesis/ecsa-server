import numpy  as np

# Function
def target_function():
    return

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
def replace_bird(position, alpha_value, lambda_value, min_values, max_values, target_function):
    """
        STEPS
        1. Get Cuckoo by Levy flights
        2. Evaluate its quality/fitness
        3. Choose a nest among n randomly
        4. Compare new fitness and old fitness, replace j by the new solution if Fi > Fj
    """
    random_bird  = np.random.randint(position.shape[0]) # chose nest randomly

    levy_values  = levy_flight(lambda_value) # generate cuckoo via levy flights
    new_solution = np.copy(position[random_bird, :-1])
    rand_factors = np.random.rand(len(min_values))
    new_solution = np.clip(new_solution + alpha_value * levy_values * new_solution * rand_factors, min_values, max_values)

    new_fitness  = target_function(new_solution) # get fitness of new solution
    if (new_fitness < position[random_bird, -1]):
        position[random_bird,:-1] = new_solution
        position[random_bird, -1] = new_fitness
    return position

# Function: Update Positions
def update_positions(position, discovery_rate, min_values, max_values, target_function):
    """
        STEPS
        1. A fraction of worse nests are abandoned and new ones are built
    """
    # print(position.shape[1])
    updated_position = np.copy(position)
    abandoned_nests  = int(np.ceil(discovery_rate * position.shape[0])) + 1
    # print(abandoned_nests)
    fitness_values   = position[:, -1]
    # print(fitness_values)
    nest_list        = np.argsort(fitness_values)[-abandoned_nests:]
    # print(nest_list)
    random_birds     = np.random.choice(position.shape[0], size=2, replace=False)
    # print(random_birds)
    bird_j, bird_k   = random_birds

    for i in nest_list:
        rand = np.random.rand(updated_position.shape[1] - 1)
        # based on randomness
        # print(f"dr = {discovery_rate}")
        if np.random.rand() > discovery_rate:
            # print(f"Worse nest: {i} got updated")
            updated_position[i, :-1] = np.clip(updated_position[i, :-1] + rand * (updated_position[bird_j, :-1] - updated_position[bird_k, :-1]), min_values, max_values)
    updated_position[:, -1] = np.apply_along_axis(target_function, 1, updated_position[:, :-1])
    updated_position = np.vstack([updated_position, position])
    updated_position = updated_position[updated_position[:, -1].argsort()]
    updated_position = updated_position[:position.shape[0], :]

    return updated_position

def enhanced_cuckoo_search(nests, birds = 3, discovery_rate = 0.25, alpha_value = 0.01, lambda_value = 1.5, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function, verbose = True, start_init = None, target_value = None): 
    fitness_history = []
    position = nests   
    best_ind = np.copy(position[position[:,-1].argsort()][0,:])
    count    = 0

    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', best_ind[-1])
    
        for i in range(0, position.shape[0]):
            position = replace_bird(position, alpha_value, lambda_value, min_values, max_values, target_function)

        
        position = update_positions(position, discovery_rate, min_values, max_values, target_function)
        value    = np.copy(position[position[:,-1].argsort()][0,:])

        if (best_ind[-1] > value[-1]):
            best_ind = np.copy(position[position[:,-1].argsort()][0,:])     
            
        if (target_value is not None):
            if (best_ind[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1
            # print(neighbors)
            fitness_history.append(best_ind[-1])

    return best_ind, fitness_history, position