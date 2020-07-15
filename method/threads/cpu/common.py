import numpy as np
from numpy.linalg import norm


def init(A, b, T=np.float64):
    """[summary]

    Args:
        A ([type]): [係数行列]
        b ([type]): [厳密解]
        T ([type], optional): [description]. Defaults to np.float64.

    Returns:
        b ([type]): [厳密解]
        x [np.zeros]: [初期解]
        b_norm [float64]: [bのL2ノルム]
        N [int]: [次元数]
        max_iter [int]: [最大反復回数]
        residual [np.zeros]: [残差履歴]
        num_of_solution_updates [np.zeros]: [解の更新回数履歴]
    """
    x = np.zeros(b.size, T)
    b_norm = norm(b)
    N = b.size
    max_iter = N  # * 2
    residual = np.zeros(max_iter + 1, T)
    num_of_solution_updates = np.zeros(max_iter + 1, np.int)
    num_of_solution_updates[0] = 0
    
    return b, x, b_norm, N, max_iter, residual, num_of_solution_updates
