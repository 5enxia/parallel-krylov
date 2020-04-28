import cupy as cp 
from cupy.linalg import norm

def init(A, b, T = cp.float64):
    x = cp.zeros(b.size,T)
    b_norm = norm(b)
    N = b.size
    max_iter = N#* 2
    residual = cp.zeros(max_iter + 1,T)
    solution_updates = cp.zeros(max_iter + 1, cp.int)
    solution_updates[0] = 0
    
    return x, b_norm, N, max_iter, residual, solution_updates
