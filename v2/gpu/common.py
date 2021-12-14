import time
import numpy as np
import cupy as cp

from ..common import _start, _finish


# 計測開始
def start(method_name: str = '', k: int = None) -> float:
    """[summary]

    Args:
        method_name (str, optional): [description]. Defaults to ''.
        k (int, optional): [description]. Defaults to None.

    Returns:
        float: [description]
    """
    _start(method_name, k)
    return time.perf_counter()


# 計測終了
def finish(start_time: float, isConverged: bool, num_of_iter: int, final_residual: float, final_k: int = None) -> float:
    """[summary]

    Args:
        start_time (float): [description]
        isConverged (bool): [description]
        num_of_iter (int): [description]
        final_residual (float): [description]
        final_k (int, optional): [description]. Defaults to None.

    Returns:
        float: [description]
    """
    elapsed_time = time.perf_counter() - start_time
    _finish(elapsed_time, isConverged, num_of_iter, final_residual, final_k)
    return elapsed_time


# パラメータの初期化
def init(A: np.ndarray, b: np.ndarray, T) -> tuple:
    x = cp.zeros(b.size, T)  # 初期解
    b_norm = cp.linalg.norm(b)
    N = b.size
    max_iter = N * 2
    residual = cp.zeros(max_iter+1, T)
    num_of_solution_updates = cp.zeros(max_iter+1, np.int)
    num_of_solution_updates[0] = 0

    return cp.asarray(A), cp.asarray(b), x, b_norm, N, max_iter, residual, num_of_solution_updates


# GPUの初期化
def init_gpu():
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)
