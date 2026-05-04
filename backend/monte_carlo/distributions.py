# Statistical distributions for price modeling

import numpy as np

def normal_distribution(size, mean=0, std=1):
    return np.random.normal(mean, std, size)

def lognormal_distribution(size, mean=0, std=1):
    return np.random.lognormal(mean, std, size)

def exponential_distribution(size, scale=1):
    return np.random.exponential(scale, size)