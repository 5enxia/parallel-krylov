from numpy import float64, dot
from numpy.linalg import norm

from .common import start, finish, init


def cg(A, b, x=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None) -> tuple:
    # 初期化
    x, maxiter, b_norm, N, residual, num_of_solution_updates = init(b, x, maxiter)

    # 初期残差
    r = b - A.dot(x)
    p = r.copy()
    gamma = dot(r, r)

    # 反復計算
    i = 0
    start_time = start(method_name='CG')
    while i < maxiter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        if residual[i] < tol:
            isConverged = True
            break

        # 解の更新
        v = A.dot(p)
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
    info = {
        'time': elapsed_time,
        'nosl': num_of_solution_updates[:i+1],
        'residual': residual[:i+1],
    }
    return x, info
