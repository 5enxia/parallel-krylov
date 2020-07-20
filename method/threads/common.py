import time

import numpy as np

from ..common import _start, _end


def start(method_name='', k=None):
    """計算開始時処理
    Args:
        method_name (str, optional): 手法名 Defaults to ''.
        k (int, optional): k. Defaults to None.

    Returns:
        float: 計算開始時刻
    """
    _start(method_name, k)
    return time.perf_counter()


def end(start_time, isConverged, num_of_iter, final_residual, final_k=None):
    """計算終了処理

    Args:
        start_time (float): 計算開始時刻
        isConverged (bool): 収束判定
        num_of_iter (int): 反復回数
        final_residual (numpy.ndarray): 最終残差
        residual_index (int): 反復終了時の残差インデックス
        final_k (int, optional): 反復終了時のk Defaults to None.

    Returns:
        float: 経過時間
    """
    elapsed_time = time.perf_counter() - start_time
    _end(elapsed_time, isConverged, num_of_iter, final_residual, final_k)
    return elapsed_time


def init(A, b, T, pu):
    """init_cpu, init_gpuの共通処理

    Args:
        A ([np.ndarray]): [係数行列]
        b ([np.nadrray]): [厳密解]
        T ([dtype], optional): [description]. Defaults to np.float64.

    Returns:
        x [np.ndarray]: [初期解(np.zeros)]
        b_norm [float64]: [bのL2ノルム]
        N [int]: [次元数]
        max_iter [int]: [最大反復回数(N * 2)]
        residual [np.ndarray]: [残差履歴]
        num_of_solution_updates [np.ndarray]: [解の更新回数履歴]
    """
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
