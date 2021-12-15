import numpy as np
from numpy.linalg import norm

from .common import start, finish, init, init_mpi


def adaptivekskipmrr(A, b, epsilon, k, T):
    # 共通初期化
    comm, rank, num_of_process = init_mpi()
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(
        A, b, T)
    begin, end = rank * local_N, (rank+1) * local_N

    # 初期化
    Ax = np.empty(N, T)
    Ar = np.empty((k + 2, N), T)
    Ay = np.empty((k + 1, N), T)
    rAr = np.zeros(1, T)
    ArAr = np.zeros(1, T)
    alpha = np.zeros(2*k + 3, T)
    beta = np.zeros(2*k + 2, T)
    delta = np.zeros(2*k + 1, T)
    # local
    local_alpha = np.zeros(2*k + 3, T)
    local_beta = np.zeros(2*k + 2, T)
    local_delta = np.zeros(2*k + 1, T)
    # kの履歴
    k_history = np.zeros(max_iter+1, np.int)
    k_history[0] = k

    # 初期残差
    comm.Allgather(A[begin:end].dot(x), Ax)
    Ar[0] = b - Ax
    residual[0] = norm(Ar[0]) / b_norm

    # 残差減少判定変数
    cur_residual = residual[0].copy()
    pre_residual = residual[0].copy()

    # 初期反復
    if rank == 0:
        start_time = start(method_name='Adaptive k-skip MrR + MPI', k=k)
    comm.Allgather(A[begin:end].dot(Ar[0]), Ar[1])
    comm.Allreduce(Ar[0][begin:end].dot(Ar[1][begin:end]), rAr)
    comm.Allreduce(Ar[1][begin:end].dot(Ar[1][begin:end]), ArAr)
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
    while i < max_iter:
        pre_residual = cur_residual
        cur_residual = norm(Ar[0]) / b_norm
        residual[index] = cur_residual

        # 残差減少判定
        if cur_residual > pre_residual:
            # 解と残差を再計算
            x = pre_x.copy()

            comm.Allgather(A[begin:end].dot(x), Ax)
            comm.Allgather(A[begin:end].dot(Ar[0]), Ar[1])
            comm.Allreduce(Ar[0][begin:end].dot(Ar[1][begin:end]), rAr)
            comm.Allreduce(Ar[1][begin:end].dot(Ar[1][begin:end]), ArAr)
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
        isConverged = cur_residual < epsilon
        if isConverged:
            break

        # 基底計算
        for j in range(1, k + 1):
            comm.Allgather(A[begin:end].dot(Ar[j-1]), Ar[j])
            comm.Allgather(A[begin:end].dot(Ay[j-1]), Ay[j])
        comm.Allgather(A[begin:end].dot(Ar[k]), Ar[k+1])

        # 係数計算
        local_alpha[0] = Ar[0][begin:end].dot(Ar[0][begin:end])
        local_delta[0] = Ay[0][begin:end].dot(Ay[0][begin:end])
        for j in range(1, 2*k+1):
            jj = j//2
            local_alpha[j] = Ar[jj][begin:end].dot(Ar[jj + j % 2][begin:end])
            local_beta[j] = Ay[jj][begin:end].dot(Ar[jj + j % 2][begin:end])
            local_delta[j] = Ay[jj][begin:end].dot(Ay[jj + j % 2][begin:end])
        local_alpha[2*k+1] = Ar[k][begin:end].dot(Ar[k+1][begin:end])
        local_beta[2*k+1] = Ay[k][begin:end].dot(Ar[k+1][begin:end])
        local_alpha[2*k+2] = Ar[k+1][begin:end].dot(Ar[k+1][begin:end])
        comm.Allreduce(local_alpha, alpha)
        comm.Allreduce(local_beta, beta)
        comm.Allreduce(local_delta, delta)

        # MrRでの1反復(解と残差の更新)
        d = alpha[2] * delta[0] - beta[1] ** 2
        zeta = alpha[1] * delta[0] / d
        eta = -alpha[1] * beta[1] / d
        Ay[0] = eta * Ay[0] + zeta * Ar[1]
        z = eta * z - zeta * Ar[0]
        Ar[0] -= Ay[0]
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
            comm.Allgather(A[begin:end].dot(Ar[0]), Ar[1])
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

    num_of_iter = i
    if rank == 0:
        elapsed_time = finish(start_time, isConverged,
                              num_of_iter, residual[index], k)
        return elapsed_time, num_of_solution_updates[:index+1], residual[:index+1], k_history[:index+1]
    else:
        exit(0)
