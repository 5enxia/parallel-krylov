import cupy as cp
from cupy import dot
from cupy.linalg import norm

from ..common import start, end
from .common import init


def cg(A, b, epsilon, T=cp.float64):
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
    A, b, x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T)

    # 初期残差
    r = b - dot(A, x)
    p = r.copy()

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
        alpha = dot(r, p) / dot(dot(p, A), p)
        x += alpha * p
        old_r = r.copy()
        r -= alpha * dot(A, p)
        beta = dot(r, r) / dot(old_r, old_r)
        p = r + beta * p
        i += 1
        num_of_solution_updates[i+1] = i + 1
    else:
        isConverged = False

    elapsed_time = end(start_time, isConverged, i, residual[i]) 
    return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
