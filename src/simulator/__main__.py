import numpy as np
from .writer import writer, fmin_writer

from src.ecs.algo import enhanced_cuckoo_search
from src.cs.algo import cuckoo_search
from src.util.convergence_curve import show_convergence
from scipy.stats import wilcoxon

from functions.unimodal import unimodal_fns
from functions.multimodal import multimodal_fns

if __name__ == '__main__':
    # D=15, n=50
    f_no = 'F11'

    CS_PARAMS = {
    'birds': 50,
    'iterations': 100,
    'target_function': multimodal_fns[f_no],
    'min_values': tuple([-500 for _ in range(15)]),
    'max_values': tuple([500 for _ in range(15)]),
    'lambda_value': 1.5,
    'target_value': -418.9829 * 15,
    # 'target_value': 0,
    'verbose': False
    }

    runs = 30
    _csa_fmin = None
    _csa_best = []
    _ecsa_fmin = None
    _ecsa_best = []

    print(f"CSA vs ECSA in {runs} individual runs\n")
    print("Parameters")
    print("CSA: Discovery Rate = 0.25, Alpha Value = 0.01")
    print("ECSA: Discovery Rate = [0.5,0.25], Alpha Value = [0.01,0.05]")

    print("=============== Cuckoo Search Algorithm ===============")
    for i in range(runs):
        csa_best, csa_hasReachedTarget, csa_iter, csa_fmin = cuckoo_search(discovery_rate=0.25, alpha_value=0.01,**CS_PARAMS)
        _csa_fmin = csa_fmin if _csa_fmin is None else np.sum([_csa_fmin, csa_fmin], axis=0)
        print(f"Trial {i+1} Done.\nBest:\n", csa_best[-1])
        _csa_best.append(csa_best[-1])
    
    print("=============== Enhanced Cuckoo Search Algorithm ===============")
    for i in range(runs):
        ecsa_best, ecsa_hasReachedTarget, ecsa_iter, ecsa_fmin = enhanced_cuckoo_search(discovery_rate=[0.5, 0.25], alpha_value=[0.01, 0.05],**CS_PARAMS)
        _ecsa_fmin = ecsa_fmin if _ecsa_fmin is None else np.sum([_ecsa_fmin, ecsa_fmin], axis=0)
        print(f"Trial {i+1} Done.\nBest:\n", ecsa_best[-1])
        _ecsa_best.append(ecsa_best[-1])

    avg_csa_fmin = np.divide(_csa_fmin, runs)
    avg_ecsa_fmin = np.divide(_ecsa_fmin, runs)

    csa_best_mean = np.mean(_csa_best) 
    csa_best_std = np.std(_csa_best)

    ecsa_best_mean = np.mean(_ecsa_best)
    ecsa_best_std = np.std(_ecsa_best)

    w = wilcoxon(x=_csa_best, y=_ecsa_best)

    print("=============== Results ===============")
    print("Runs: 30")
    print("CSA Mean: ", csa_best_mean)
    print("CSA Std: ", csa_best_std)

    print("ECSA Mean: ", ecsa_best_mean)
    print("ECSA Std: ", ecsa_best_std)

    print("\nVisualizing Results...")

    writer(
        function=f_no,
        data=[
        ["csa",csa_best_mean, csa_best_std, w.pvalue],
        ["ecsa",ecsa_best_mean, ecsa_best_std],
        ]
        )
    
    fmin_writer(
        function=f_no,
        data=[_csa_best, _ecsa_best]
    )

    show_convergence(
        function=f_no,
        csa_fmin=avg_csa_fmin, ecsa_fmin=avg_ecsa_fmin
        )