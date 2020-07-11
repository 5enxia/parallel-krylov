import sys

import cupy as cp
from cupy import dot
from cupy.linalg import norm

if __name__ == "__main__":
    sys.path.append('../../../../')
    from krylov.method.common import getConditionParams
    from krylov.method.threads.common import start, end
    from krylov.method.threads.gpu.common import init


def k_skip_cg(A, b, epsilon, k, T=cp.float64):
    # 初期化
    x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T)
    Ar = cp.zeros((k + 2, N), T)
    Ap = cp.zeros((k + 3, N), T)
    a = cp.zeros(2 * k + 2, T)
    f = cp.zeros(2 * k + 4, T)
    c = cp.zeros(2 * k + 2, T)

    # 初期残差
    Ar[0] = b - dot(A, x)
    Ap[0] = Ar[0]

    # 反復計算
    start_time = start(method_name='k-skip CG', k=k)
    for i in range(0, max_iter):
        # 収束判定
        residual[i] = norm(Ar[0]) / b_norm
        if residual[i] < epsilon:
            isConverged = True
            break

        # 事前計算
        for j in range(1, k + 1):
            Ar[j] = dot(A, Ar[j-1])
        for j in range(1, k + 2):
            Ap[j] = dot(A, Ap[j-1])
        for j in range(2 * k + 1):
            jj = j // 2
            a[j] = dot(Ar[jj], Ar[jj])
            a[j+1] = dot(Ar[jj], Ar[jj+1])
        for j in range(0, 2 * k + 3, 2):
            jj = j // 2
            f[j] = dot(Ap[jj], Ap[jj])
            f[j+1] = dot(Ap[jj], Ap[jj+1])
        for j in range(0, 2 * k + 1, 2):
            jj = j // 2
            c[j] = dot(Ar[jj], Ap[jj])
            c[j+1] = dot(Ar[jj], Ap[jj+1])

        # CGでの1反復
        alpha = a[0] / f[1]
        beta = alpha ** 2 * f[2] / a[0] - 1
        x += alpha * Ap[0]
        Ar[0] -= alpha * Ap[1]
        Ap[0] = Ar[0] + beta * Ap[0]
        Ap[1] = dot(A, Ap[0])

        # CGでのk反復
        for j in range(0, k):
            for l in range(0, 2*(k-j)+1):
                a[l] += alpha*(alpha*f[l+2] - 2*c[l+1])
                d = c[l] - alpha*f[l+1]
                c[l] = a[l] + d*beta
                f[l] = c[l] + beta*(d + beta*f[l])

            # 解の更新
            alpha = a[0] / f[1]
            beta = alpha ** 2 * f[2] / a[0] - 1
            x += alpha * Ap[0]
            Ar[0] -= alpha * Ap[1]
            Ap[0] = Ar[0] + beta * Ap[0]
            Ap[1] = dot(A, Ap[0])

        num_of_solution_updates[i+1] = num_of_solution_updates[i] + k + 1

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

    k_skip_cg(A, b, epsilon, k, T)
