import numpy as np
from numpy import dot
from numpy.linalg import norm

from ..common import start, end
from .common import init


def cg(A: np.ndarray, b: np.ndarray, epsilon: float, T=np.float64):
    """[summary]

    Args:
        A (np.ndarray): [description]
        b (np.ndarray): [description]
        epsilon (float): [description]
        T ([type], optional): [description]. Defaults to np.float64.

    Returns:
        float: 経過時間
        np.ndarray: 残差更新履歴
        np.ndarray: 残差履歴
    """
    # 初期化
    b, x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T)

    # 初期残差
    r = b - dot(A, x)
    p = r.copy()

    # 反復計算
    start_time = start(method_name='CG')
    for i in range(max_iter):
        # 収束判定
        residual[i] = norm(r) / b_norm
        if residual[i] < epsilon:
            isConverged = True
            break

        # 解の更新
        alpha = dot(r, p) / dot(dot(p, A), p)
        x += alpha * p
        old_r = r.copy()
        r -= alpha * dot(A, p)
        beta = dot(r, r) / dot(old_r, old_r)
        p = r + beta * p
        num_of_solution_updates[i+1] = i + 1
        
    else:
        isConverged = False

    num_of_iter = i
    residual_index = i
    elapsed_time = end(start_time, isConverged, num_of_iter, residual, residual_index)

    return elapsed_time, num_of_solution_updates[:num_of_iter+1], residual[:residual_index+1]
