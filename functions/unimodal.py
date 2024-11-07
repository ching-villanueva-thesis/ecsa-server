import numpy as np

# F1
def sphere(n = [0,0]):
    n = np.array(n)
    func_value = 0
    for i in range(0, len(n)):
        func_value = func_value + n[i]**2
    return func_value

# F2
def rosenbrocks(n = [0,0]):
     n = np.array(n)
     return np.sum(100 * (n[1:] - n[:-1]**2)**2 + (n[:-1] - 1)**2)