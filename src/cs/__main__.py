import matplotlib.pyplot as plt
from src.cs.algo import cuckoo_search, initial_variables
from functions.test import ackley, griewangk_8

from src.util.plot_population import plot_population
from src.util.standard_deviation import sdv

if __name__ == '__main__':
    CS_PARAMS = {
    'birds': 300,
    'iterations': 100,
    'discovery_rate': 0.25,
    'alpha_value': 0.01,
    'lambda_value': 1.5,
    'target_value': 0,
    'min_values': (-32,-32),
    'max_values': (32,32),
    'verbose': False
    }

    # Generate and plot initial population
    initial_pop = initial_variables(
        size=CS_PARAMS['birds'],
        min_values=CS_PARAMS['min_values'],
        max_values=CS_PARAMS['max_values'],
        target_function=ackley
    )
    
    # Visualize initial population
    plot_population(
        initial_pop, 
        CS_PARAMS['min_values'], 
        CS_PARAMS['max_values']
    )

    converged = 0
    total = 0
    fitness_values = []
    for i in range(30):
        print(f"====== Running CS 100 times. Trial {i + 1} ======")
        best, hasReachedTarget, _iter = cuckoo_search(target_function=ackley, **CS_PARAMS)

        total += _iter
        fitness_values.append(best[-1])
        if hasReachedTarget:
            converged += 1
            
        print(best)

    
    print("RESULTS")
    print(f"CS has converged {converged} times.")
    print(f"Avg convergence {total / 30}")

    _min = min(fitness_values)
    _max = max(fitness_values)
    avg = sum(fitness_values) / len(fitness_values)
    standard_deviation = sdv(fitness_values)

    print(f"Minimum Value: {_min}")
    print(f"Maximum Value: {_max}")
    print(f"Average: {avg}")
    print(f"Standard Deviation: {standard_deviation}")