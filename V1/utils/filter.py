import numpy as np


"""
    State Filter: normalizes the input state
"""
class Filter(object):

    def __init__(self, state_dim):
        self.num = np.zeros(state_dim)
        self.mean = np.zeros(state_dim)
        self.mean_diff = np.zeros(state_dim)
        self.var = np.zeros(state_dim)

    def push(self, x):
        self.num += 1
        old_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.num
        self.mean_diff += (x - old_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.num).clip(min=1e-2)

    def __call__(self, state):
        return (state - self.mean) / np.sqrt(self.var)

