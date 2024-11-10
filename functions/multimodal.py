import numpy as np

def F8(x):
    """Rastrigin function
    Dimensions: 15
    Range: [-5.12, 5.12]
    Optimum: 0
    """
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def F9(x):
    """Ackley function
    Dimensions: 15
    Range: [-32, 32]
    Optimum: 0
    """
    d = len(x)
    sum_sq = np.sum(x**2)
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / d))

    sum_cos = np.sum(np.cos(2*np.pi*x))
    term2 = -np.exp(sum_cos / d)

    return term1 + term2 + 20 + np.exp(1)

def F10(x):
    """Griewank function
    Dimensions: 15
    Range: [-600, 600]
    Optimum: 0
    """
    n = len(x)
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
    return sum_term - prod_term + 1

multimodal_fns = {
    'F8': F8,
    'F9': F9,
    'F10': F10
}