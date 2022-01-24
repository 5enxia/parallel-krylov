import time

import numpy as np

from ..common import _start, _end


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


def end(start_time: float, isConverged: bool, num_of_iter: int, final_residual: float, final_k: int = None) -> float:
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
    _end(elapsed_time, isConverged, num_of_iter, final_residual, final_k)
    return elapsed_time


def init(A: np.ndarray, b: np.ndarray, T, pu: str) -> tuple:
    if pu == 'gpu':
        import cupy as cp
        pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
        cp.cuda.set_allocator(pool.malloc)

    x = np.zeros(b.size, T)
    b_norm = np.linalg.norm(b)
    N = b.size
    max_iter = N * 2
    residual = np.zeros(max_iter+1, T)
    num_of_solution_updates = np.zeros(max_iter+1, np.int)
    num_of_solution_updates[0] = 0

    if pu == 'gpu':
        return cp.asarray(A), cp.asarray(b), cp.asarray(x), b_norm, N, max_iter, residual, num_of_solution_updates

    return x, b_norm, N, max_iter, residual, num_of_solution_updates
