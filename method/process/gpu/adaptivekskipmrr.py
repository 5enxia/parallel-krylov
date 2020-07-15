import numpy as np
import cupy as cp
from cupy import dot
from cupy.linalg import norm
from mpi4py import MPI

from ..common import start, end
from .common import init, init_matvec, init_vecvec, mpi_matvec, mpi_vecvec


def adaptive_k_skip_mrr(A, b, epsilon, k, T=np.float64):
    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_of_process = comm.Get_size()
    # GPU
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)
    num_of_gpu = cp.cuda.runtime.getDeviceCount()
    cp.cuda.Device(rank % num_of_gpu).use()
    # 共通初期化
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T)
    local_A, Ax, local_Ax = init_matvec(N, local_N, T)
    local_a, local_b = init_vecvec(local_N, T)
    local_A_cpu = local_A.get()
    comm.Scatter(A, local_A_cpu, root=0)
    local_A = cp.asarray(local_A_cpu)
    # root_gpu
    Ar = cp.empty((k + 3, N), T)
    Ay = cp.empty((k + 2, N), T)
    alpha = cp.empty(2*k + 3, T)
    beta = cp.empty(2*k + 2, T)
    beta[0] = 0
    delta = cp.empty(2*k + 1, T)
    # root_cpu
    Ar_cpu = np.empty((k + 3, N), T)
    Ay_cpu = np.empty((k + 2, N), T)
    alpha_cpu = np.empty(2*k + 3, T)
    beta_cpu = np.empty(2*k + 2, T)
    beta_cpu[0] = 0
    delta_cpu = np.empty(2*k + 1, T)
    # local
    local_alpha = cp.empty(2*k + 3, T)
    local_beta = cp.empty(2*k + 2, T)
    local_delta = cp.empty(2*k + 1, T)

    # 初期kと現在のkの値の差
    dif = 0
    k_history = cp.zeros(max_iter+1, cp.int)
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
        isIncreaese = np.array([cur_residual.get() > pre_residual.get()], dtype=bool)
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
        isConverged = np.array([cur_residual.get() < epsilon], dtype=bool)
        comm.Bcast(isConverged, root=0)
        if isConverged:
            break

        # 事前計算
        for j in range(1, k + 2):
            Ar[j] = mpi_matvec(local_A, Ar[j-1], Ax, local_Ax, comm)
        for j in range(1, k + 1):
            Ay[j] = mpi_matvec(local_A, Ay[j-1], Ax, local_Ax, comm)
        Ar_cpu = Ar.get()
        Ay_cpu = Ay.get()
        comm.Bcast(Ar_cpu)
        comm.Bcast(Ay_cpu)
        Ar = cp.asarray(Ar_cpu)
        Ay = cp.asarray(Ay_cpu)
        for j in range(2*k + 3):
            jj = j // 2
            local_alpha[j] = dot(
                Ar[jj][rank * local_N: (rank+1) * local_N],
                Ar[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_alpha.get(), alpha_cpu, root=0)
        alpha = cp.asarray(alpha_cpu)
        for j in range(1, 2 * k + 2):
            jj = j//2
            local_beta[j] = dot(
                Ay[jj][rank * local_N: (rank+1) * local_N],
                Ar[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_beta.get(), beta_cpu, root=0)
        beta = cp.asarray(beta_cpu)
        for j in range(2 * k + 1):
            jj = j // 2
            local_delta[j] = dot(
                Ay[jj][rank * local_N: (rank+1) * local_N],
                Ay[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_delta.get(), delta_cpu, root=0)
        delta = cp.asarray(delta_cpu)

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

    else:
        isConverged = False
        
    num_of_iter = i
    if rank == 0:
        elapsed_time = end(start_time, isConverged, num_of_iter, residual[num_of_iter], k)
        return elapsed_time, num_of_solution_updates[:num_of_iter+1].get(), residual[:num_of_iter+1].get(), k_history[:num_of_iter+1].get()
    else:
        exit(0)
