from numpy import float64
from cupy import dot
from cupy.linalg import norm

from .common import start, finish, init, MultiGpu


def cg(A, b, x=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None) -> tuple:
    # 初期化
    T = float64
    MultiGpu.init()
    b, x, maxiter, b_norm, N, residual, num_of_solution_updates = init(b, x, maxiter)
    MultiGpu.alloc(A, b, T)

    # 初期残差
    r = b - MultiGpu.dot(A, x)
    p = r.copy()
    gamma = dot(r, r)

    # 反復計算
    i = 0
    start_time = start(method_name='CG + GPU')
    while i < maxiter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        if residual[i] < tol:
            isConverged = True
            break

        # 解の更新
        v = MultiGpu.dot(A, p)
        sigma = dot(p, v)
        alpha = gamma / sigma
        x += alpha * p
        r -= alpha * v
        old_gamma = gamma.copy()
        gamma = dot(r, r)
        beta = gamma / old_gamma
        p = r + beta * p
        i += 1
        num_of_solution_updates[i] = i
    else:
        isConverged = False
        residual[i] = norm(r) / b_norm

    elapsed_time = finish(start_time, isConverged, i, residual[i])
    return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
