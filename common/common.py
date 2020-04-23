import time

import numpy as np
from numpy import dot
from numpy.linalg import norm
from numpy.linalg import multi_dot

def init(A, b, T = np.float64):
    x = np.zeros(b.size,T)
    b_norm = norm(b)
    N = b.size
    max_iter = N * 2
    residual = np.zeros(max_iter + 1,T)
    solution_updates = np.zeros(max_iter + 1, np.int)
    solution_updates[0] = 0
    
    return x, b_norm, N, max_iter, residual, solution_updates

def start(method_name = '', k = None):
    print('# ============== INFO ================= #')
    print(f'Method:\t{ method_name }')
    print(f'k:\t{ k }')

    return time.perf_counter()

def end(start_time, isConverged, num_of_iter, residual, residual_index):
    elapsed_time = time.perf_counter() - start_time

    print(f'time:\t{ elapsed_time } s')
    status = 'converged' if isConverged else 'diverged'
    print(f'status:\t{ status }')
    if isConverged:
        print(f'iteration:\t{ num_of_iter } times')
        print(f'residual:\t{residual[residual_index]}')
    print('# ===================================== #')

    return elapsed_time