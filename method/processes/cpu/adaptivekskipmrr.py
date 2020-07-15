import numpy as np
from numpy import dot
from numpy.linalg import norm
from mpi4py import MPI

from ..common import start, end
from .common import init, init_matvec, init_vecvec, mpi_matvec, mpi_vecvec


def adaptive_k_skip_mrr(A, b, epsilon, k, T=np.float64):
    # 共通初期化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_of_process = comm.Get_size()
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process,T)
    local_A, Ax, local_Ax = init_matvec(N, local_N, T)
    local_a, local_b = init_vecvec(local_N, T)
    comm.Scatter(A, local_A, root=0)
    # root
    Ar = np.empty((k + 3, N), T)
    Ay = np.empty((k + 2, N), T)
    alpha = np.empty(2*k + 3, T)
    beta = np.empty(2*k + 2, T)
    beta[0] = 0
    delta = np.empty(2*k + 1, T)
    # local
    local_alpha = np.empty(2*k + 3, T)
    local_beta = np.empty(2*k + 2, T)
    local_beta[0] = 0
    local_delta = np.empty(2*k + 1, T)
    
    dif = 0
    k_history = np.zeros(max_iter+1, np.int)
    k_history[0] = k

    # 初期残差
    Ar[0] = b - mpi_matvec(local_A, x, Ax, local_Ax, comm)
    residual[0] = norm(Ar[0]) / b_norm
    pre_residual = residual[0]

    # 初期反復
    if rank == 0:
        start_time = start(method_name='k-skip MrR', k=k)
    Ar[1] = mpi_matvec(local_A, Ar[0], Ax, local_Ax, comm)
    zeta = mpi_vecvec(Ar[0], Ar[1], local_a, local_b, comm) / mpi_vecvec(Ar[1], Ar[1], local_a, local_b, comm)
    Ay[0] = zeta * Ar[1]
    z = -zeta * Ar[0]
    Ar[0] -= Ay[0]
    x -= z
    num_of_solution_updates[1] = 1
    k_history[1] = k

    # 反復計算
    for i in range(1, max_iter):
        cur_residual = norm(Ar[0]) / b_norm
        # 残差減少判定
        isIncreaese = np.array([cur_residual > pre_residual], dtype=bool)
        comm.Bcast(isIncreaese, root=0)
        if isIncreaese:
            # 解と残差を再計算
            x = pre_x.copy()
            Ar[0] = b - mpi_matvec(local_A, x, Ax, local_Ax, comm)
            Ar[1] = mpi_matvec(local_A, Ar[0], Ax, local_Ax, comm)
            zeta = mpi_vecvec(Ar[0], Ar[1], local_a, local_b, comm) / mpi_vecvec(Ar[1], Ar[1], local_a, local_b, comm)
            Ay[0] = zeta * Ar[1]
            z = -zeta * Ar[0]
            Ar[0] -= Ay[0]
            x -= z

            # kを下げて収束を安定化させる
            if k > 1:
                dif += 1
                k -= 1
        else:
            pre_residual = cur_residual
            residual[i - dif] = cur_residual
            pre_x = x.copy()
            
        # 収束判定
        isConverged = np.array([cur_residual < epsilon], dtype=bool)
        comm.Bcast(isConverged, root=0)
        if isConverged:
            break

        # 事前計算
        for j in range(1, k + 2):
            Ar[j] = mpi_matvec(local_A, Ar[j-1], Ax, local_Ax, comm)
        for j in range(1, k + 1):
            Ay[j] = mpi_matvec(local_A, Ay[j-1], Ax, local_Ax, comm)
        comm.Bcast(Ar)
        comm.Bcast(Ay)
        for j in range(2*k + 3):
            jj = j // 2
            local_alpha[j] = dot(
                Ar[jj][rank * local_N: (rank+1) * local_N],
                Ar[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_alpha, alpha, root=0)
        for j in range(1, 2 * k + 2):
            jj = j//2
            local_beta[j] = dot(
                Ay[jj][rank * local_N: (rank+1) * local_N],
                Ar[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_beta, beta, root=0)
        for j in range(2 * k + 1):
            jj = j // 2
            local_delta[j] = dot(
                Ay[jj][rank * local_N: (rank+1) * local_N],
                Ay[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_delta, delta, root=0)

        # MrRでの1反復(解と残差の更新)
        d = alpha[2] * delta[0] - beta[1] ** 2
        zeta = alpha[1] * delta[0] / d
        eta = -alpha[1] * beta[1] / d
        Ay[0] = eta * Ay[0] + zeta * Ar[1]
        z = eta * z - zeta * Ar[0]
        Ar[0] -= Ay[0]
        Ar[1] = mpi_matvec(local_A, Ar[0], Ax, local_Ax, comm)
        x -= z

        # MrRでのk反復
        for j in range(0, k):
            delta[0] = zeta ** 2 * alpha[2] + eta * zeta * beta[1]
            alpha[0] -= zeta * alpha[1]
            delta[1] = eta ** 2 * delta[1] + 2 * eta * zeta * beta[2] + zeta ** 2 * alpha[3]
            beta[1] = eta * beta[1] + zeta * alpha[2] - delta[1]
            alpha[1] = -beta[1]
            for l in range(2, 2 * (k - j) + 1):
                delta[l] = eta ** 2 * delta[l] + 2 * eta * zeta * beta[l + 1] + zeta ** 2 * alpha[l + 2]
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
            Ar[1] = mpi_matvec(local_A, Ar[0], Ax, local_Ax, comm)
            x -= z

        num_of_solution_updates[i + 1 - dif] = num_of_solution_updates[i - dif] + k + 1
        k_history[i + 1 - dif] = k

    else:
        isConverged = False

    num_of_iter = i
    if rank == 0:
        elapsed_time = end(start_time, isConverged, num_of_iter, residual[num_of_iter])
        return elapsed_time, num_of_solution_updates[:num_of_iter+1], residual[:num_of_iter+1], k_history[:num_of_iter+1]
    else:
        exit(0)
