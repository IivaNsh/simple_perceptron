import numpy as np

def generate_points(n, k, c1=(0, 0), c2=(4, 4)):
    np.random.seed(1)
    points_1 = np.random.randn(n, 2) + np.array(c1)
    points_2 = np.random.randn(k, 2) + np.array(c2)
    return points_1, points_2
