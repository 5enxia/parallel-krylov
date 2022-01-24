import time

import numpy as np

from ..common import _start, _finish


# 計測開始
def start(method_name: str = '', k: int = None) -> float:
    _start(method_name, k)
    return time.perf_counter()


# 計測終了
def finish(start_time: float, isConverged: bool, num_of_iter: int, final_residual: float, final_k: int = None) -> float:
    elapsed_time = time.perf_counter() - start_time
    _finish(elapsed_time, isConverged, num_of_iter, final_residual, final_k)
    return elapsed_time


# パラメータの初期化
def init(b, maxiter=None) -> tuple:
    b_norm = np.linalg.norm(b)
    N = b.size
    if maxiter == None:
        maxiter = N
    residual = np.zeros(maxiter+1, np.float64)
    num_of_solution_updates = np.zeros(maxiter+1, np.int)

    return b_norm, N, maxiter, residual, num_of_solution_updates
