from src.ecs.algo import enhanced_cuckoo_search
from functions.test import ackley, griewangk_8

if __name__ == '__main__':
    CS_PARAMS = {
    'birds': 15,
    'iterations': 100,
    'discovery_rate': [0.9, 0.1],
    'alpha_value': 0.01,
    'lambda_value': 1.5,
    'target_value': 0,
    'min_values': (-32,-32),
    'max_values': (32,32),
    'verbose': False
    }

    converged = 0
    for i in range(100):
        print(f"====== Running ECS 100 times. Trial {i + 1} ======")
        best, hasReachedTarget = enhanced_cuckoo_search(target_function=griewangk_8, **CS_PARAMS)
        
        if hasReachedTarget:
            converged += 1
            
        print(best)

    print(f"ECS has converged {converged} times")