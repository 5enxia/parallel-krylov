import cupy as cp

from ..common import _init


def init(A, b, T=cp.float64):
    """[summary]

    Args:
        A (np.ndarray): [係数行列]
        b (np.ndarray): [厳密解]
        T (np.dtype, optional): [description]. Defaults to cp.float64.

    Returns:
        A (cp.ndarray): [係数行列]
        b (cp.ndarray): [厳密解]
        x [cp.ndarray]: [初期解]
        b_norm [np.float64]: [bのL2ノルム]
        N [int]: [次元数]
        max_iter [int]: [最大反復回数]
        residual [cp.ndarray]: [残差履歴]
        num_of_solution_updates [cp.ndarray]: [解の更新回数履歴]
    """
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    x, b_norm, N, max_iter, residual, num_of_solution_updates = _init(A, b, T)
    return cp.asarray(A), cp.asarray(b), cp.asarray(x), b_norm, N, max_iter, residual, num_of_solution_updates
