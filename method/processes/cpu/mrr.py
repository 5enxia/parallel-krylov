import numpy as np
from numpy.linalg import norm

from ..common import start, end
from .common import init, init_mpi, init_matvec, init_vecvec, mpi_matvec, mpi_vecvec


def mrr(A, b, epsilon, T=np.float64):
    # 共通初期化
    comm, rank, num_of_process = init_mpi()
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T)
    local_A, Ax, local_Ax = init_matvec(N, local_N, T)
    local_a, local_b = init_vecvec(local_N, T)
    comm.Scatter(A, local_A, root=0)

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
    i = 1

    # 反復計算
    while i < max_iter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        isConverged = np.array([residual[i] < epsilon], dtype=bool)
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
        i += 1
        num_of_solution_updates[i] = i
    else:
        isConverged = False

    if rank == 0:
        elapsed_time = end(start_time, isConverged, i, residual[i])
        return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
    else:
        exit(0)
