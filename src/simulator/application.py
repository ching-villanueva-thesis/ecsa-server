import numpy as np
from scipy.stats import wilcoxon

from src.cs.dcs import discrete_cuckoo_search_algorithm
from src.ecs.decs import discrete_enhanced_cuckoo_search_algorithm

from src.dataset.get_data import getDataFromDataset
from src.util.convergence_curve import show_convergence

from functions.distance import Distance

from .writer import writer, fmin_writer, solution_writer

runs = 30
_dcsa_fmin = None
_dcsa_best = []
_decsa_fmin = None
_decsa_best = []

_dcsa_best_sol = []
_decsa_best_sol = []

db_coordinates, es_coordinates = getDataFromDataset()
d = Distance(db_coordinates=db_coordinates, es_coordinates=es_coordinates)

print("=============== Discrete Cuckoo Search Algorithm ===============")
for i in range(runs):
    nests, f_values, fmin, best, best_fev = discrete_cuckoo_search_algorithm(size=50, dimensions=(len(db_coordinates), len(es_coordinates)), iterations=500, target_function=d.fitness)

    print(f"Trial {i+1} Done.\nBest:\n", best_fev)

    _dcsa_fmin = fmin if _dcsa_fmin is None else np.sum([_dcsa_fmin, fmin], axis=0)
    _dcsa_best.append(best_fev)
    _dcsa_best_sol.append(best)
    
print("=============== Discrete Enhanced Cuckoo Search Algorithm ===============")
for i in range(runs):
    nests, f_values, fmin, best, best_fev = discrete_enhanced_cuckoo_search_algorithm(size=50, dimensions=(len(db_coordinates), len(es_coordinates)), iterations=500, discovery_rate=(0.5, 0.25), alpha_value=(0.01, 0.05), target_function=d.fitness)

    print(f"Trial {i+1} Done.\nBest:\n", best_fev)

    _decsa_fmin = fmin if _decsa_fmin is None else np.sum([_decsa_fmin, fmin], axis=0)
    _decsa_best.append(best_fev)
    _decsa_best_sol.append(best)

avg_dcsa_fmin = np.divide(_dcsa_fmin, runs)
avg_decsa_fmin = np.divide(_decsa_fmin, runs)

dcsa_best_mean = np.mean(_dcsa_best)
dcsa_best_std = np.std(_dcsa_best)

decsa_best_mean = np.mean(_decsa_best)
decsa_best_std = np.std(_decsa_best)

w = wilcoxon(x=_dcsa_best, y=_decsa_best)

print("=============== Results ===============")
print("Runs: 30")
print("CSA Mean: ", dcsa_best_mean)
print("CSA Std: ", dcsa_best_std)
print("ECSA Mean: ", decsa_best_mean)
print("ECSA Std: ", decsa_best_std)

writer(
    function="Optimal Allocation of Evacuation Areas",
    data=[
    ["dcsa",dcsa_best_mean, dcsa_best_std, w.pvalue],
    ["decsa",decsa_best_mean, decsa_best_std],
    ]
    )

fmin_writer(
    function="Optimal Allocation of Evacuation Areas",
    data=[_dcsa_best, _decsa_best]
)

solution_writer("dcsa", _dcsa_best_sol)
solution_writer("decsa", _decsa_best_sol)

show_convergence(
    function="Optimal Allocation of Evacuation Areas",
    csa_fmin=avg_dcsa_fmin, ecsa_fmin=avg_decsa_fmin
    )