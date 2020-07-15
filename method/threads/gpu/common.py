import cupy as cp
from cupy.linalg import norm


def init(A, b, T=cp.float64):
    """[summary]

    Args:
        A (np.ndarray): [係数行列]
        b (np.ndarray): [厳密解]
        T (np.dtype, optional): [description]. Defaults to cp.float64.

    Returns:
        A (cp.ndarray): [係数行列]
        b (cp.ndarray): [厳密解]
        x [cp.zeros]: [初期解]
        b_norm [cp.float64]: [bのL2ノルム]
        N [int]: [次元数]
        max_iter [int]: [最大反復回数]
        residual [cp.zeros]: [残差履歴]
        num_of_solution_updates [cp.zeros]: [解の更新回数履歴]
    """
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    x = cp.zeros(b.size, T)
    b_norm = norm(b)
    N = b.size
    max_iter = N  # * 2
    residual = cp.zeros(max_iter + 1, T)
    num_of_solution_updates = cp.zeros(max_iter + 1, cp.int)
    num_of_solution_updates[0] = 0

    return cp.asarray(A), cp.asarray(b), x, b_norm, N, max_iter, residual, num_of_solution_updates
