from numpy import float64
import cupy as cp
from cupy import dot
from cupy.linalg import norm

from .common import start, finish, init, MultiGpu


def kskipcg(A, b, x=None, tol=1e-05, maxiter=None, k=0, M=None, callback=None, atol=None) -> tuple:
    # 初期化
    T = float64
    MultiGpu.init()
    b, x, maxiter, b_norm, N, residual, num_of_solution_updates = init(b, x, maxiter)
    MultiGpu.alloc(A, b, T)

    # 初期化
    Ar = cp.zeros((k + 2, N), T)
    Ap = cp.zeros((k + 3, N), T)
    a = cp.zeros(2 * k + 2, T)
    f = cp.zeros(2 * k + 4, T)
    c = cp.zeros(2 * k + 2, T)

    # 初期残差
    Ar[0] = b - MultiGpu.dot(A, x)
    Ap[0] = Ar[0]

    # 反復計算
    i = 0
    index = 0
    start_time = start(method_name='k-skip CG + GPU', k=k)
    while i < maxiter:
        # 収束判定
        residual[index] = norm(Ar[0]) / b_norm
        if residual[index] < tol:
            isConverged = True
            break

        # 事前計算
        for j in range(1, k + 1):
            Ar[j] = MultiGpu.dot(A, Ar[j-1])
        for j in range(1, k + 2):
            Ap[j] = MultiGpu.dot(A, Ap[j-1])

        for j in range(2 * k + 1):
            jj = j // 2
            a[j] = dot(Ar[jj], Ar[jj + j % 2])
        for j in range(2 * k + 4):
            jj = j // 2
            f[j] = dot(Ap[jj], Ap[jj + j % 2])
        for j in range(2 * k + 2):
            jj = j // 2
            c[j] = dot(Ar[jj], Ap[jj + j % 2])

        # CGでの1反復
        alpha = a[0] / f[1]
        beta = alpha ** 2 * f[2] / a[0] - 1
        x += alpha * Ap[0]
        Ar[0] -= alpha * Ap[1]
        Ap[0] = Ar[0] + beta * Ap[0]
        Ap[1] = MultiGpu.dot(A, Ap[0])

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
            Ap[1] = MultiGpu.dot(A, Ap[0])

        i += (k + 1)
        index += 1
        num_of_solution_updates[index] = i
    else:
        isConverged = False
        residual[index] = norm(Ar[0]) / b_norm

    elapsed_time = finish(start_time, isConverged, i, residual[index])
    return elapsed_time, num_of_solution_updates[:index+1], residual[:index+1]
