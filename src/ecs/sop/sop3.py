# Cosine Annealing w/ Warm Restarts
import numpy as np
import math

class CosineAnnealingWithWarmRestarts:
    """
        walk_type 0: Levy
        walky_type 1: Gaussian
    """
    def __init__(self, n_min, n_max, t_max, t_cur = 0, c_max = 4):
        self.n_min = n_min
        self.n_max = n_max
        self.t_cur = t_cur
        self.t_max = t_max / c_max
        self.c_max = c_max
        self.mid = c_max / 2
        self.cycle = 1
        self.walk_type = 0

    def step(self):
        eta = self.n_min + (self.n_max - self.n_min) / 2 * (1 + np.cos(np.pi * self.t_cur / self.t_max))

        self.t_cur += 1
        
        if self.t_cur % self.t_max == 0:
            self.t_cur = 0
            self.cycle += 1
            
            if self.walk_type == 0 and self.cycle == self.mid:
                self.walk_type = 1
                print("UPDATED WALK TYPE TO: ", self.walk_type)


        return round(eta, 2)