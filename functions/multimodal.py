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

def F11(x):
    return np.sum(x * np.sin(np.sqrt(np.abs(x))))

def F12(x, a=10, k=100, m=4):
    D = len(x)
    y = 1 + (x + 1) / 4

    # Calculate the main function part
    term1 = (np.pi / D) * (10 * np.sin(np.pi * y[0])**2 +
                           sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2)) +
                           (y[-1] - 1)**2)
    
    # Calculate the penalty term
    u = np.where(x > a, k * (x - a)**m,
                 np.where(x < -a, k * (-x - a)**m, 0))
    penalty = np.sum(u)
    
    return term1 + penalty

def f13_u(x, a, k, m):
    return np.where(x > a, k * (x - a) ** m, np.where(x < -a, k * (-x - a) ** m, 0))

def F13(x):
    n = len(x)
    
    # Term 1
    term1 = 0.1 * np.sin(3 * np.pi * x[0]) ** 2
    
    # Term 2
    term2 = np.sum([(x[i] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[i + 1]) ** 2) for i in range(n - 1)])
    
    # Term 3
    term3 = (x[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[-1]) ** 2)
    
    # Term 4 (Penalty function)
    term4 = np.sum(f13_u(x,5,100,4))
    
    # Total function value
    return term1 + term2 + term3 + term4

multimodal_fns = {
    'F8': F8,
    'F9': F9,
    'F10': F10,
    'F11': F11,
    'F12': F12,
    'F13': F13
}