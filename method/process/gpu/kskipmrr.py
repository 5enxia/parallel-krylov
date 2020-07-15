import sys

import numpy as np
import cupy as cp
from cupy import dot
from cupy.linalg import norm
from mpi4py import MPI

if __name__ == "__main__":
    sys.path.append('../../../../')
    from krylov.method.common import getConditionParams
    from krylov.method.process.gpu.common import init, init_matvec, init_vecvec, start, end, mpi_matvec, mpi_vecvec


def k_skip_mrr(A, b, epsilon, k, T=np.float64):
    # 共通初期化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_of_process = comm.Get_size()
    x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T)
    local_N, local_A, Ax, local_Ax = init_matvec(N, num_of_process, T)
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
    
    # 初期残差
    Ar[0] = b - mpi_matvec(local_A, x, Ax, local_Ax, comm)
    residual[0] = norm(Ar[0]) / b_norm

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

    # 反復計算
    for i in range(1, max_iter):
        # 収束判定
        residual[i] = norm(Ar[0]) / b_norm
        isConverged = np.array([residual[i].get() < epsilon], dtype=bool)
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
        for j in range(k):
            delta[0] = zeta ** 2 * alpha[2] + eta * zeta * beta[1]
            alpha[0] -= zeta * alpha[1]
            delta[1] = eta ** 2 * delta[1] + 2 * eta * zeta * beta[2] + zeta ** 2 * alpha[3]
            beta[1] = eta * beta[1] + zeta * alpha[2] - delta[1]
            alpha[1] = -beta[1]
            for l in range(2, 2 * (k - j) + 1):
                delta[l] = eta ** 2 * delta[l] + 2 * eta * zeta * beta[l+1] + zeta ** 2 * alpha[l + 2]
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

        num_of_solution_updates[i + 1] = num_of_solution_updates[i] + k + 1

    else:
        isConverged = False

    num_of_iter = i + 1
    residual_index = i
    if rank == 0:
        end(start_time, isConverged, num_of_iter, residual, residual_index)


if __name__ == "__main__":
    # GPU Memory Settings
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    # Current GPU Settings
    rank = MPI.COMM_WORLD.Get_rank()
    num_of_gpu = cp.cuda.runtime.getDeviceCount()
    cp.cuda.Device(rank % num_of_gpu).use

    A, b, epsilon, k, T = getConditionParams('condition.json')
    b = cp.asarray(b)

    k_skip_mrr(A, b, epsilon, k, T)
