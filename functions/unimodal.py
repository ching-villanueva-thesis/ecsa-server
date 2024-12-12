import numpy as np

def F1(x):
    """Sphere function
    Dimensions: 15
    Range: [-100, 100]
    Optimum: 0
    """
    return np.sum(x**2)

def F2(x):
    """Schwefel's 2. function
    Dimensions: 15
    Range: [-10, 10]
    Optimum: 0
    """
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F3(x):
    """Schwefel's 1.20 function
    Dimensions: 15
    Range: [-100, 100]
    Optimum: 0
    """
    n = len(x)
    result = 0
    for i in range(n):
        inner_sum = sum(x[j] for j in range(i+1))
        result += inner_sum**2
    return result

def F4(x):
    """Schwefel's 2.21 function
    Dimensions: 15
    Range: [-100, 100]
    Optimum: 0
    """
    return np.max(np.abs(x))

def F5(x):
    """Rosenbrock's function
    Dimensions: 15
    Range: [-30, 30]
    Optimum: 0
    """
    x_im1 = x[:-1]  # x_{i-1}
    x_i = x[1:]     # x_i
    return np.sum(100.0 * (x_i - x_im1**2)**2 + (x_im1 - 1)**2)

def F6(x):
    """Step function
    Dimensions: 15
    Range: [-100, 100]
    Optimum: 0
    """
    return np.sum((x + 0.5)**2)

def F7(x):
    """Quartic Noise function
    Dimensions: 15
    Range: [-1.28, 1.28]
    Optimum: 0
    """
    i = np.arange(1, len(x) + 1)
    return np.sum(i * x**4) + np.random.uniform(0, 1)

unimodal_fns = {
    'F1': F1,
    'F2': F2,
    'F3': F3,
    'F4': F4,
    'F5': F5,
    'F6': F6,
    'F7': F7,
}