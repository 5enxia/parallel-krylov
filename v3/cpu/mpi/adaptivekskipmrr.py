import numpy as np
from numpy import float64, dot
from numpy.linalg import norm

from .common import start, finish, init, MultiCpu


def adaptivekskipmrr(comm, local_A, b, x=None, tol=1e-05, maxiter=None, k=0, M=None, callback=None, atol=None) -> tuple:
    # MPI初期化
    rank = comm.Get_rank()
    MultiCpu.joint_mpi(comm)

    # 初期化
    T = float64
    x, maxiter, b_norm, N, residual, num_of_solution_updates = init(
        b, x, maxiter)
    MultiCpu.alloc(local_A, T)
    Ax = np.zeros(N, T)
    Ar = np.zeros((k + 2, N), T)
    Ay = np.zeros((k + 1, N), T)
    rAr = np.zeros(1, T)
    ArAr = np.zeros(1, T)
    alpha = np.zeros(2*k + 3, T)
    beta = np.zeros(2*k + 2, T)
    delta = np.zeros(2*k + 1, T)

    # kの履歴
    k_history = np.zeros(maxiter+1, np.int)
    k_history[0] = k

    # 初期残差
    MultiCpu.dot(local_A, x, out=Ax)
    Ar[0] = b - Ax
    residual[0] = norm(Ar[0]) / b_norm

    # 残差減少判定変数
    cur_residual = residual[0].copy()
    pre_residual = residual[0].copy()

    # 初期反復
    if rank == 0:
        start_time = start(method_name='Adaptive k-skip MrR + MPI', k=k)
    MultiCpu.dot(local_A, Ar[0], out=Ar[1])
    rAr = dot(Ar[0], Ar[1])
    ArAr = dot(Ar[1], Ar[1])

    zeta = rAr / ArAr
    Ay[0] = zeta * Ar[1]
    z = -zeta * Ar[0]
    Ar[0] -= Ay[0]
    x -= z

    i = 1
    index = 1
    num_of_solution_updates[1] = 1
    k_history[1] = k

    # 反復計算
    while i < maxiter:
        pre_residual = cur_residual
        cur_residual = norm(Ar[0]) / b_norm
        residual[index] = cur_residual

        # 残差減少判定
        if cur_residual > pre_residual:
            # 解と残差を再計算
            x = pre_x.copy()

            MultiCpu.dot(local_A, x, out=Ax)
            MultiCpu.dot(local_A, Ar[0], out=Ar[1])
            rAr = dot(Ar[0], Ar[1])
            ArAr = dot(Ar[1], Ar[1])

            zeta = rAr / ArAr
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
            pre_x = x.copy()

        # 収束判定
        if cur_residual < tol:
            isConverged = True
            break

        # 基底計算
        for j in range(1, k + 2):
            MultiCpu.dot(local_A, Ar[j-1], out=Ar[j])
        for j in range(1, k + 1):
            MultiCpu.dot(local_A, Ay[j-1], out=Ay[j])

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

        # MrRでの1反復(解と残差の更新)
        d = alpha[2] * delta[0] - beta[1] ** 2
        zeta = alpha[1] * delta[0] / d
        eta = -alpha[1] * beta[1] / d
        Ay[0] = eta * Ay[0] + zeta * Ar[1]
        z = eta * z - zeta * Ar[0]
        Ar[0] -= Ay[0]
        MultiCpu.dot(local_A, Ar[0], out=Ar[1])
        x -= z

        # MrRでのk反復
        for j in range(k):
            zz = zeta ** 2
            ee = eta ** 2
            ez = eta * zeta
            delta[0] = zz * alpha[2] + ez * beta[1]
            alpha[0] -= zeta * alpha[1]
            delta[1] = ee * delta[1] + 2 * eta * zeta * beta[2] + zz * alpha[3]
            beta[1] = eta * beta[1] + zeta * alpha[2] - delta[1]
            alpha[1] = -beta[1]
            for l in range(2, 2 * (k - j) + 1):
                delta[l] = ee * delta[l] + 2 * ez * \
                    beta[l+1] + zz * alpha[l + 2]
                tau = eta * beta[l] + zeta * alpha[l + 1]
                beta[l] = tau - delta[l]
                alpha[l] -= tau + beta[l]
            # 解と残差の更新
            d = alpha[2] * delta[0] - beta[1] ** 2
            zeta = alpha[1] * delta[0] / d
            eta = -alpha[1] * beta[1] / d
            MultiCpu.dot(local_A, Ar[0], out=Ar[1])
            Ay[0] = eta * Ay[0] + zeta * Ar[1]
            z = eta * z - zeta * Ar[0]
            Ar[0] -= Ay[0]
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
        return elapsed_time, num_of_solution_updates[:index+1], residual[:index+1], k_history[:index+1]
    else:
        exit(0)
