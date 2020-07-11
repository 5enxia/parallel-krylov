import sys

import cupy as cp
from cupy import dot
from cupy.linalg import norm

if __name__ == "__main__":
    sys.path.append('../../../../')
    from krylov.method.common import getConditionParams
    from krylov.method.threads.common import start, end
    from krylov.method.threads.gpu.common import init


def cg(A, b, epsilon, T=cp.float64):
    # 初期化
    x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T)

    # 初期残差
    r = b - dot(A, x)
    p = r.copy()

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

    num_of_iter = i + 1
    residual_index = i
    end(start_time, isConverged, num_of_iter, residual, residual_index)


if __name__ == "__main__":
    # GPU Memory Settings
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    A, b, epsilon, k, T = getConditionParams('condition.json')
    A, b = cp.asarray(A), cp.asarray(b)

    cg(A, b, epsilon, T)
