import numpy as np
import cupy as cp
from cupy.linalg import norm
from mpi4py import MPI

from ..common import start, end
from .common import init, init_matvec, init_vecvec, mpi_matvec, mpi_vecvec


def mrr(A, b, epsilon, T=np.float64):
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

    # 初期残差
    r = b - mpi_matvec(local_A, x, Ax, local_Ax, comm)
    residual[0] = norm(r) / b_norm

    # 初期反復
    if rank == 0:
        start_time = start(method_name='MrR')
    Ar = mpi_matvec(local_A, r, Ax, local_Ax, comm)
    zeta = mpi_vecvec(r, Ar, local_a, local_b, comm) / mpi_vecvec(Ar, Ar, local_a, local_b, comm)
    y = zeta * Ar
    z = -zeta * r
    r -= y
    x -= z
    num_of_solution_updates[1] = 1

    # 反復計算
    for i in range(1, max_iter):
        # 収束判定
        residual[i] = norm(r) / b_norm
        isConverged = np.array([residual[i].get() < epsilon], dtype=bool)
        comm.Bcast(isConverged, root=0)
        if isConverged:
            break

        # 解の更新
        Ar = mpi_matvec(local_A, r, Ax, local_Ax, comm)
        nu = mpi_vecvec(y, Ar, local_a, local_b, comm)
        gamma = nu / mpi_vecvec(y, y, local_a, local_b, comm)
        s = Ar - gamma * y
        zeta = mpi_vecvec(r, s, local_a, local_b, comm) / mpi_vecvec(s, s, local_a, local_b, comm)
        eta = -zeta * gamma
        y = eta * y + zeta * Ar
        z = eta * z - zeta * r
        r -= y
        x -= z
        num_of_solution_updates[i + 1] = i + 1
    else:
        isConverged = False

    num_of_iter = i
    if rank == 0:
        elapsed_time = end(start_time, isConverged, num_of_iter, residual[num_of_iter])
        return elapsed_time, num_of_solution_updates[:num_of_iter+1].get(), residual[:num_of_iter+1].get()
    else:
        exit(0)
