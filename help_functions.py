import numpy as np
import random

#x: result value
#y: target value
def cost(y,x): return 0.5*(y-x)**2
def cost_prime(y,x): return y-x
def sigmoida(x): return 1/(1+np.exp(-1*x))
def sigmoida_prime(x): return sigmoida(x)*(1-sigmoida(x))
