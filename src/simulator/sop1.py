import numpy as np
from .writer import std_writer

from src.ecs.s1Algo import enhanced_cuckoo_search
from src.cs.algo import cuckoo_search
from src.util.convergence_curve import show_convergence
from scipy.stats import wilcoxon

from functions.unimodal import unimodal_fns
from functions.multimodal import multimodal_fns

if __name__ == '__main__':
    # D=15, n=50

    CS_PARAMS = {
    'birds': 50,
    'iterations': 100,
    'lambda_value': 1.5,
    'alpha_value': 0.01,
    'discovery_rate': 0.25,
    'verbose': False,
    }

    functions = {
        'F1': {
            **CS_PARAMS,
            'min_values': tuple([-100 for _ in range(15)]),
            'max_values': tuple([100 for _ in range(15)])
        },
        'F2': {
            **CS_PARAMS,
            'min_values': tuple([-10 for _ in range(15)]),
            'max_values': tuple([10 for _ in range(15)])
        },
        'F3': {
            **CS_PARAMS,
            'min_values': tuple([-100 for _ in range(15)]),
            'max_values': tuple([100 for _ in range(15)])
        },
        'F4': {
            **CS_PARAMS,
            'min_values': tuple([-100 for _ in range(15)]),
            'max_values': tuple([100 for _ in range(15)])
        },
        'F5': {
            **CS_PARAMS,
            'min_values': tuple([-30 for _ in range(15)]),
            'max_values': tuple([30 for _ in range(15)])
        },
        'F6': {
            **CS_PARAMS,
            'min_values': tuple([-100 for _ in range(15)]),
            'max_values': tuple([100 for _ in range(15)])
        },
        'F7': {
            **CS_PARAMS,
            'min_values': tuple([-1.28 for _ in range(15)]),
            'max_values': tuple([1.28 for _ in range(15)])
        },
        'F8': {
            **CS_PARAMS,
            'min_values': tuple([-5.12 for _ in range(15)]),
            'max_values': tuple([5.12 for _ in range(15)])
        },
        'F9': {
            **CS_PARAMS,
            'min_values': tuple([-32 for _ in range(15)]),
            'max_values': tuple([32 for _ in range(15)])
        },
        'F10': {
            **CS_PARAMS,
            'min_values': tuple([-600 for _ in range(15)]),
            'max_values': tuple([600 for _ in range(15)])
        },
        'F11': {
            **CS_PARAMS,
            'min_values': tuple([-500 for _ in range(15)]),
            'max_values': tuple([500 for _ in range(15)])
        },
        'F12': {
            **CS_PARAMS,
            'min_values': tuple([-50 for _ in range(15)]),
            'max_values': tuple([50 for _ in range(15)])
        },
        'F13': {
            **CS_PARAMS,
            'min_values': tuple([-50 for _ in range(15)]),
            'max_values': tuple([50 for _ in range(15)])
        },
    }

    _csa_std = []
    _csa_var = []

    _ecsa_std = []
    _ecsa_var = []

    for key in functions.keys():
        target_function = None

        target_function = unimodal_fns[key] if key in unimodal_fns else multimodal_fns[key]

        print(f"=============== Initial Population of Cuckoo Search Algorithm in {key} ===============")
        csa_best, csa_pos, csa_hasReachedTarget, csa_iter, csa_fmin = cuckoo_search(target_function=target_function, **CS_PARAMS)

        csa_fevs = csa_pos[:, -1]

        _csa_std.append(np.std(csa_fevs))
        _csa_var.append(np.var(csa_fevs))

        print(f"=============== Initial Population of Enhanced Cuckoo Search Algorithm in {key} ===============")
        ecsa_best, ecsa_pos, ecsa_hasReachedTarget, ecsa_iter, ecsa_fmin = enhanced_cuckoo_search(target_function=target_function, **CS_PARAMS)
        ecsa_fevs = ecsa_pos[:, -1]

        _ecsa_std.append(np.std(ecsa_fevs))
        _ecsa_var.append(np.var(ecsa_fevs))

    std_writer(data=[
        list(functions.keys()),
        _csa_std,
        _csa_var,
        _ecsa_std,
        _ecsa_var
    ])
    

    # print("=============== Results ===============")
    # print("Runs: 30")
    # print("CSA Mean: ", csa_best_mean)
    # print("CSA Std: ", csa_best_std)

    # print("ECSA Mean: ", ecsa_best_mean)
    # print("ECSA Std: ", ecsa_best_std)

    # print("\nVisualizing Results...")

    # writer(
    #     function=f_no,
    #     data=[
    #     ["csa",csa_best_mean, csa_best_std, w.pvalue],
    #     ["ecsa",ecsa_best_mean, ecsa_best_std],
    #     ]
    #     )
    
    # fmin_writer(
    #     function=f_no,
    #     data=[_csa_best, _ecsa_best]
    # )

    # show_convergence(
    #     function=f_no,
    #     csa_fmin=avg_csa_fmin, ecsa_fmin=avg_ecsa_fmin
    #     )