import time

import numpy as np
from numpy.linalg import norm

from krylov.method._common import _start, _end

def init(A, b, T = np.float64):
    x = np.zeros(b.size,T)
    b_norm = norm(b)
    N = b.size
    max_iter = N#* 2
    residual = np.zeros(max_iter + 1,T)
    solution_updates = np.zeros(max_iter + 1, np.int)
    solution_updates[0] = 0
    
    return x, b_norm, N, max_iter, residual, solution_updates

def start(method_name = '', k = None):
    _start(method_name, k)
    return time.perf_counter()

def end(start_time, isConverged, num_of_iter, residual, residual_index, final_k = None):
    elapsed_time = time.perf_counter() - start_time
    _end(elapsed_time, isConverged, num_of_iter, residual, residual_index, final_k)
    return elapsed_time