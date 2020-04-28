import numpy as np
from numpy.linalg import norm

def init(A, b, T = np.float64):
    x = np.zeros(b.size,T)
    b_norm = norm(b)
    N = b.size
    max_iter = N#* 2
    residual = np.zeros(max_iter + 1,T)
    solution_updates = np.zeros(max_iter + 1, np.int)
    solution_updates[0] = 0
    
    return x, b_norm, N, max_iter, residual, solution_updates