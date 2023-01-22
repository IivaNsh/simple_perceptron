import numpy as np
import random

def step_func(x): 
    return np.where(x > 0, 1, 0) 
 
def sigmod_func(x):
    return ((1/(1+np.exp(-1*x)))-0.5)*2
