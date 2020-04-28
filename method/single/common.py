import time

import numpy as np
from numpy.linalg import norm

from krylov.method._common import _start, _end

def start(method_name = '', k = None):
    _start(method_name, k)
    return time.perf_counter()

def end(start_time, isConverged, num_of_iter, residual, residual_index, final_k = None):
    elapsed_time = time.perf_counter() - start_time
    _end(elapsed_time, isConverged, num_of_iter, residual, residual_index, final_k)
    return elapsed_time