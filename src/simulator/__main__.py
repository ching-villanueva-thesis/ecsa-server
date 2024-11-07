import numpy as np

from src.ecs.algo import enhanced_cuckoo_search
from src.cs.algo import cuckoo_search
from src.util.convergence_curve import show_convergence

from functions.unimodal import sphere,rosenbrocks
from functions.multimodal import ackley, griewank

if __name__ == '__main__':
    # D=15, n=50
    CS_PARAMS = {
    'birds': 50,
    'iterations': 500,
    'lambda_value': 1.5,
    'target_value': 0,
    'min_values': tuple([-32 for _ in range(15)]),
    'max_values': tuple([32 for _ in range(15)]),
    'verbose': False
    }

    print("=============== Cuckoo Search Algorithm ===============")
    csa_best, csa_hasReachedTarget, csa_iter, csa_fmin = cuckoo_search(target_function=sphere, discovery_rate=0.25, alpha_value=0.01, **CS_PARAMS)
    print("Done.\nBest:", csa_best[-1])

    print("=============== Enhanced Cuckoo Search Algorithm ===============")
    ecsa_best, ecsa_hasReachedTarget, ecsa_iter, ecsa_fmin = enhanced_cuckoo_search(target_function=sphere, discovery_rate=[0.5, 0.25], alpha_value=[0.01, 0.05], **CS_PARAMS)
    print("Done.\nBest:", ecsa_best[-1])

    print("\nIs ECSA closer to zero?")
    print(ecsa_best[-1] < csa_best[-1])

    print("\nVisualizing Results...")
    show_convergence(csa_fmin=csa_fmin, ecsa_fmin=ecsa_fmin, csa_best=csa_best[-1], ecsa_best=ecsa_best[-1], function="F3")