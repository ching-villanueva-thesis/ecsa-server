import csv
import pathlib

from datetime import datetime

from src.cs.algo import cuckoo_search
from functions.unimodal import sphere

from src.util.standard_deviation import sdv

if __name__ == '__main__':
    # D=15, n=50
    CS_PARAMS = {
    'birds': 50,
    'iterations': 500,
    'discovery_rate': 0.25,
    'alpha_value': 0.01,
    'lambda_value': 1.5,
    'target_value': 0,
    'min_values': tuple([-100 for _ in range(15)]),
    'max_values': tuple([100 for _ in range(15)]),
    'verbose': False
    }

    converged = 0
    total = 0
    fitness_values = []

    for i in range(30):
        print(f"====== Running CS 30 times. Trial {i + 1} ======")
        best, hasReachedTarget, _iter, fmin = cuckoo_search(target_function=sphere, **CS_PARAMS)

        total += _iter
        fitness_values.append(best[-1])
        if hasReachedTarget:
            converged += 1
    
    print("RESULTS")
    print(f"CS has converged {converged} times.")

    _min = min(fitness_values)
    _max = max(fitness_values)
    avg = sum(fitness_values) / len(fitness_values)
    standard_deviation = sdv(fitness_values)

    print(f"Minimum Value: {_min}")
    print(f"Maximum Value: {_max}")
    print(f"Average: {avg}")
    print(f"Standard Deviation: {standard_deviation}")

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'f1.csv'
    file_path =  pathlib.Path(__file__).cwd() / 'src' / 'cs' /'results' / file_name

    _data = [
        ["Min", "Max", "Mean", "Standard Deviation"],
        [_min, _max, avg, standard_deviation],
        ["Fitness Values"],
        *[[x] for x in fitness_values]
    ]

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(_data)