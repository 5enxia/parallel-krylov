import numpy as np
from numpy import dot
from numpy.linalg import norm

from ..common import start, end
from .common import init


def cg(A, b, epsilon, T=np.float64):
    """[summary]

    Args:
        A (np.ndarray): 係数行列A
        b (np.ndarray): bベクトル
        epsilon (float): 収束判定子
        T ([type], optional): 浮動小数精度 Defaults to np.float64.

    Returns:
        float: 経過時間
        np.ndarray: 残差更新履歴
        np.ndarray: 残差履歴
    """
    # 初期化
    x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T)

    # 初期残差
    r = b - dot(A, x)
    p = r.copy()
    rr = dot(r, r)

    # 反復計算
    i = 0
    start_time = start(method_name='CG')
    while i < max_iter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        if residual[i] < epsilon:
            isConverged = True
            break

        # 解の更新
        Ap = dot(A, p)
        alpha = dot(r, p) / dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        old_rr = rr.copy()
        rr = dot(r ,r)
        beta = rr / old_rr
        p = r + beta * p
        i += 1
        num_of_solution_updates[i] = i
    else:
        isConverged = False

    elapsed_time = end(start_time, isConverged, i, residual[i])
    return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
