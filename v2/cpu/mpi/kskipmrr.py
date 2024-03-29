import numpy as np
from numpy import dot
from numpy.linalg import norm


from .common import start, finish, init, init_mpi 


def kskipmrr(A, b, epsilon, k, T):
    # 共通初期化
    comm, rank, num_of_process = init_mpi()
    local_A, b, x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T, rank, num_of_process)
    Ax = np.zeros(N, T)

    # 初期化
    Ar = np.zeros((k + 2, N), T)
    Ay = np.zeros((k + 1, N), T)
    rAr = np.zeros(1, T)
    ArAr = np.zeros(1, T)
    alpha = np.zeros(2*k + 3, T)
    beta = np.zeros(2*k + 2, T)
    delta = np.zeros(2*k + 1, T)

    # 初期残差
    comm.Allgather(local_A.dot(x), Ax)
    Ar[0] = b - Ax
    residual[0] = norm(Ar[0]) / b_norm

    # 初期反復
    if rank == 0:
        start_time = start(method_name='k-skip MrR + MPI', k=k)

    comm.Allgather(local_A.dot(Ar[0]), Ar[1])
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

    # 反復計算
    while i < max_iter:
        # 収束判定
        residual[index] = norm(Ar[0]) / b_norm
        isConverged = residual[index] < epsilon
        if isConverged:
            break

        # 基底計算
        for j in range(1, k + 2):
            comm.Allgather(local_A.dot(Ar[j-1]), Ar[j])
        for j in range(1, k + 1):
            comm.Allgather(local_A.dot(Ay[j-1]), Ay[j])

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
        comm.Allgather(local_A.dot(Ar[0]), Ar[1])
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
                delta[l] = ee * delta[l] + 2 * ez * beta[l+1] + zz * alpha[l + 2]
                tau = eta * beta[l] + zeta * alpha[l + 1]
                beta[l] = tau - delta[l]
                alpha[l] -= tau + beta[l]
            # 解と残差の更新
            d = alpha[2] * delta[0] - beta[1] ** 2
            zeta = alpha[1] * delta[0] / d
            eta = -alpha[1] * beta[1] / d
            Ay[0] = eta * Ay[0] + zeta * Ar[1]
            z = eta * z - zeta * Ar[0]
            Ar[0] -= Ay[0]
            comm.Allgather(local_A.dot(Ar[0]), Ar[1])
            x -= z

        i += (k + 1)
        index += 1
        num_of_solution_updates[index] = i
    else:
        isConverged = False
        residual[index] = norm(Ar[0]) / b_norm

    if rank == 0:
        elapsed_time = finish(start_time, isConverged, i, residual[index])
        return elapsed_time, num_of_solution_updates[:index+1], residual[:index+1]
    else:
        exit(0)
