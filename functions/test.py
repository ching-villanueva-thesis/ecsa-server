import numpy as np

# Function: Ackley. Solution -> f(x1, x2) = 0; (x1, x2) = (0, 0). Domain -> -5 <= x1, x2 <= 5
def ackley(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = -20*np.exp(-0.2*np.sqrt(0.5*(x1**2 + x2**2))) - np.exp(0.5*(np.cos(2*np.pi*x1) +np.cos(2*np.pi*x2) )) + np.exp(1) + 20
    return func_value

# Function: Griewangk F8. Solution -> f(xi) = 0; xi = 0. Domain -> -600 <= xi <= 600
def griewangk_8(variables_values = [0, 0]):
    fv_0 = 0
    fv_1 = 1
    for i in range(0, len(variables_values)):
        fv_0 = fv_0 + (variables_values[i]**2)/4000
        fv_1 = fv_1 * (np.cos(variables_values[i]/np.sqrt(i+1)))
    func_value = fv_0 - fv_1 + 1
    return func_value