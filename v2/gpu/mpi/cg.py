from cupy import dot
from cupy.linalg import norm

from .common import start, finish, init, MultiGpu


def cg(A, b, epsilon, T):
    # 初期化
    MultiGpu.init_gpu(0, 3)
    A, b, x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T, 4)
    MultiGpu.alloc(A, b, T)

    # 初期残差
    r = b - MultiGpu.dot(A, x)
    p = r.copy()
    gamma = dot(r, r)

    # 反復計算
    i = 0
    start_time = start(method_name='CG + GPU')
    while i < max_iter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        if residual[i] < epsilon:
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

    elapsed_time = finish(start_time, isConverged, i, residual[i])
    return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
