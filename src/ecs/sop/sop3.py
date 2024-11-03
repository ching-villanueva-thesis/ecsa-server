# Cosine Annealing w/ Warm Restarts
import numpy as np

class CosineAnnealingWithWarmRestarts:
    def __init__(self, n_min, n_max, t_max, t_cur = 0):
        self.n_min = n_min
        self.n_max = n_max
        self.t_cur = t_cur
        self.t_max = t_max

    def step(self):
        eta = self.n_min + (self.n_max - self.n_min) / 2 * (1 + np.cos(np.pi * self.t_cur / self.t_max))

        self.t_cur += 1

        if self.t_cur % self.t_max == 0:
            self.t_cur = 0

        return round(eta, 2)