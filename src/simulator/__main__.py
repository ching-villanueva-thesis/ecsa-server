from src.ecs.algo import enhanced_cuckoo_search
from src.cs.algo import cuckoo_search
from functions.test import ackley, griewangk_8
from scipy.stats import ranksums

from src.util.standard_deviation import sdv
from src.simulator.population import initial_variables
from src.simulator.writer import writer

if __name__ == '__main__':
    CS_PARAMS = {
    'birds': 15,
    'iterations': 100,
    'lambda_value': 1.5,
    'target_value': 0,
    'min_values': (-32,-32),
    'max_values': (32,32),
    'verbose': False
    # discovery_rate
    }

    runs = 30
    cs_converged = 0
    ecs_converged = 0
    cs_fitness = []
    ecs_fitness = []

    data = []

    print(f"Running CS vs ECS Simulator {runs} times")
    for i in range(runs):
        print(f"====== Trial {i + 1} ======")

        # init_population = initial_variables(size=CS_PARAMS['birds'], min_values=CS_PARAMS['min_values'], max_values=CS_PARAMS['max_values'], target_function=ackley, start_init=None)

        cs_best, cs_hasReachedTarget, cs_iter = cuckoo_search(target_function=ackley, discovery_rate=0.25, alpha_value=0.01, **CS_PARAMS)

        ecs_best, ecs_hasReachedTarget, ecs_iter = enhanced_cuckoo_search(target_function=ackley, discovery_rate=[0.5, 0.25], alpha_value=[0.01, 0.05], **CS_PARAMS)

        cs_fitness.append(cs_best[-1])
        ecs_fitness.append(ecs_best[-1])

        if cs_hasReachedTarget:
            cs_converged += 1
        
        if ecs_hasReachedTarget:
            ecs_converged += 1

    print(f"CS has converged {cs_converged} times.")
    print(f"ECS has converged {ecs_converged} times.")

    data.append(['Cuckoo Search', min(cs_fitness), max(cs_fitness), sum(cs_fitness) / len(cs_fitness), sdv(cs_fitness)])
    data.append(['Enhanced Cuckoo Search', min(ecs_fitness), max(ecs_fitness), sum(ecs_fitness) / len(ecs_fitness), sdv(ecs_fitness)])

    file_path = writer(data=data)
    print(f"Results has been saved in {file_path}")

    

