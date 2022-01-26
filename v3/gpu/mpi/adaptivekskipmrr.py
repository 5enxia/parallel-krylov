import numpy as np
import cupy as cp
from cupy import dot
from cupy.linalg import norm
from mpi4py import MPI

from .common import start, finish, init, MultiGpu


def adaptivekskipmrr(comm, local_A, b, x=None, tol=1e-05, maxiter=None, k=0, M=None, callback=None, atol=None) -> tuple:
    # MPI初期化
    rank = comm.Get_rank()
    MultiGpu.joint_mpi(comm)

    # 初期化
    T = np.float64
    ## GPU初期化
    MultiGpu.init()
    b, x, maxiter, b_norm, N, residual, num_of_solution_updates = init(b, x, maxiter)
    MultiGpu.alloc(local_A, b, T)
    Ax = cp.zeros(N, T)
    Ar = cp.zeros((k + 2, N), T)
    Ay = cp.zeros((k + 1, N), T)
    alpha = cp.zeros(2 * k + 3, T)
    beta = cp.zeros(2 * k + 2, T)
    delta = cp.zeros(2 * k + 1, T)

    # 初期残差
    MultiGpu.dot(local_A, x, out=Ax)
    Ar[0] = b - Ax
    residual[0] = norm(Ar[0]) / b_norm

    # kの履歴
    k_history = cp.zeros(maxiter+1, np.int)
    k_history[0] = k

    # 残差現象判定変数
    cur_residual = residual[0]
    pre_residual = None

    # 初期反復
    if rank == 0:
        start_time = start(method_name='adaptive k-skip MrR + gpu + mpi', k=k)
    MultiGpu.dot(local_A, Ar[0], out=Ar[1])
    zeta = dot(Ar[0], Ar[1]) / dot(Ar[1], Ar[1])
    Ay[0] = zeta * Ar[1]
    z = -zeta * Ar[0]
    Ar[0] -= Ay[0]
    x -= z
    num_of_solution_updates[1] = 1
    i = 1
    index = 1

    k_history[1] = k

    # 反復計算
    while i < maxiter:
        pre_residual = cur_residual
        # 収束判定
        residual[index] = norm(Ar[0]) / b_norm
        if residual[index] < tol:
            isConverged = True
            break
        cur_residual = residual[index]

        # 残差減少判定
        if cur_residual > pre_residual:
            # 解と残差を再計算
            x = pre_x

            MultiGpu.dot(local_A, x, out=Ax)
            zeta = dot(Ar[0], Ar[1]) / dot(Ar[1], Ar[1])
            Ay[0] = zeta * Ar[1]
            z = -zeta * Ar[0]
            Ar[0] -= Ay[0]
            x -= z

            i += 1
            index += 1
            num_of_solution_updates[index] = i
            residual[index] = norm(Ar[0]) / b_norm
            
            # kを下げて収束を安定化させる
            if k > 1:
                k -= 1
            k_history[index] = k
        else:
            pre_x = x

        # 基底計算
        for j in range(1, k + 2):
            MultiGpu.dot(local_A, Ar[j-1], out=Ar[j])
        for j in range(1, k + 1):
            MultiGpu.dot(local_A, Ay[j-1], out=Ay[j])

        # 係数計算
        for j in range(2 * k + 3):
            jj = j // 2
            alpha[j] = dot(Ar[jj], Ar[jj + j % 2])
        for j in range(1, 2 * k + 2):
            jj = j//2
            beta[j] = dot(Ay[jj], Ar[jj + j % 2])
        for j in range(2 * k + 1):
            jj = j // 2
            delta[j] = dot(Ay[jj], Ay[jj + j % 2])
        
        # MrRでの1反復(解の更新)
        d = alpha[2] * delta[0] - beta[1] ** 2
        zeta = alpha[1] * delta[0] / d
        eta = -alpha[1] * beta[1] / d
        Ay[0] = eta * Ay[0] + zeta * Ar[1]
        z = eta * z - zeta * Ar[0]
        Ar[0] -= Ay[0]
        MultiGpu.dot(local_A, Ar[0], out=Ar[1])
        x -= z

        # MrRでのk反復
        for j in range(k):
            delta[0] = zeta ** 2 * alpha[2] + eta * zeta * beta[1]
            alpha[0] -= zeta * alpha[1]
            delta[1] = eta ** 2 * delta[1] + 2 * eta * \
                zeta * beta[2] + zeta ** 2 * alpha[3]
            beta[1] = eta * beta[1] + zeta * alpha[2] - delta[1]
            alpha[1] = -beta[1]
            for l in range(2, 2 * (k - j) + 1):
                delta[l] = eta ** 2 * delta[l] + 2 * eta * \
                    zeta * beta[l+1] + zeta ** 2 * alpha[l + 2]
                tau = eta * beta[l] + zeta * alpha[l + 1]
                beta[l] = tau - delta[l]
                alpha[l] -= tau + beta[l]
            # 解の更新
            d = alpha[2] * delta[0] - beta[1] ** 2
            zeta = alpha[1] * delta[0] / d
            eta = -alpha[1] * beta[1] / d
            Ay[0] = eta * Ay[0] + zeta * Ar[1]
            z = eta * z - zeta * Ar[0]
            Ar[0] -= Ay[0]
            MultiGpu.dot(local_A, Ar[0], out=Ar[1])
            x -= z

        i += (k + 1)
        index += 1
        num_of_solution_updates[index] = i
        k_history[index] = k
    else:
        isConverged = False
        residual[index] = norm(Ar[0]) / b_norm

    if rank == 0:
        elapsed_time = finish(start_time, isConverged, i, residual[index], k)
        info = {
            'time': elapsed_time,
            'nosl': num_of_solution_updates[:index+1],
            'residual': residual[:index+1],
            'khistory': k_history[:index+1]
        }
        return x, info
    else:
        exit(0)