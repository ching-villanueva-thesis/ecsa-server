from src.ecs.algo import enhanced_cuckoo_search
from functions.test import ackley, griewangk_8

from src.util.standard_deviation import sdv
from src.util.plot_parameters import create_line_graph

from src.ecs.sop import CosineAnnealingWithWarmRestarts

if __name__ == '__main__':
    # CS_PARAMS = {
    # 'birds': 15,
    # 'iterations': 100,
    # 'discovery_rate': [0.75, 0.25],
    # 'alpha_value': [0.01, 0.05],
    # 'lambda_value': 1.5,
    # 'target_value': 0,
    # 'min_values': (-32,-32),
    # 'max_values': (32,32),
    # 'verbose': False
    # }

    # converged = 0
    # total = 0
    # fitness_values = []
    # for i in range(30):
    #     print(f"====== Running ECS 100 times. Trial {i + 1} ======")
    #     best, hasReachedTarget, _iter = enhanced_cuckoo_search(target_function=ackley, **CS_PARAMS)

    #     total += _iter
    #     fitness_values.append(best[-1])
    #     if hasReachedTarget:
    #         converged += 1

    #     print(best)

    
    # print("RESULTS")
    # print(f"ECS has converged {converged} times.")
    # print(f"Avg convergence {total / 30}")

    # _min = min(fitness_values)
    # _max = max(fitness_values)
    # avg = sum(fitness_values) / len(fitness_values)
    # standard_deviation = sdv(fitness_values)

    # print(f"Minimum Value: {_min}")
    # print(f"Maximum Value: {_max}")
    # print(f"Average: {avg}")
    # print(f"Standard Deviation: {standard_deviation}")
    # Adaptive Parameter Visualization
    discovery_rate = [0.5, 0.25]
    alpha_value = [0.01, 0.05]

    d_max, d_min = discovery_rate
    d_cos_annealing = CosineAnnealingWithWarmRestarts(n_max=d_max, n_min=d_min, t_max=20)

    a_max, a_min = alpha_value
    a_cos_annealing = CosineAnnealingWithWarmRestarts(n_max=a_max, n_min=a_min, t_max=20)

    d_cos_annealing_values = []
    a_cos_annealing_values = []

    for _ in range(100):
        dr = d_cos_annealing.step()
        av = a_cos_annealing.step()

        d_cos_annealing_values.append(dr)
        a_cos_annealing_values.append(av)

    create_line_graph(
        title="Adaptive Discovery Rate using Cosine Annealing",
        x=[i for i in range(100)],
        y=d_cos_annealing_values,
        xlabel="Iterations",
        ylabel="Discovery Rate"
        )
    create_line_graph(
        title="Adaptive Alpha Value (Step size) using Cosine Annealing",
        x=[i for i in range(100)],
        y=a_cos_annealing_values,
        xlabel="Iterations",
        ylabel="Alpha Values"
        )
