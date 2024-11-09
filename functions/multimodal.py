import numpy as np

# F6
def ackley(n = [0,0]):
    n = np.array(n)
    return -20 * np.exp(-0.2 * np.sqrt(1/len(n) * np.sum(n**2))) - np.exp(1/len(n) * np.sum(np.cos(2*np.pi*n))) + 20 + np.exp(1)
# F7
def griewank(n = [0,0]):
    n = np.array(n)
    term1 = np.sum(n**2) / 4000
    term2 = np.prod(np.cos(n / np.sqrt(np.arange(1, len(n) + 1))))
    return term1 - term2 + 1
