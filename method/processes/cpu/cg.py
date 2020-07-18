import numpy as np
from numpy.linalg import norm

from ..common import start, end
from .common import init, init_mpi, init_matvec, init_vecvec, mpi_matvec, mpi_vecvec


def cg(A, b, epsilon, T=np.float64):
    # 共通初期化
    comm, rank, num_of_process = init_mpi()
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T)
    local_A, Ax, local_Ax = init_matvec(N, local_N, T)
    local_a, local_b = init_vecvec(local_N, T)
    comm.Scatter(A, local_A, root=0)

    # 初期残差
    r = b - mpi_matvec(local_A, x, Ax, local_Ax, comm)
    p = r.copy()

    # 反復計算
    i = 0
    if rank == 0:
        start_time = start(method_name='CG')
    while i < max_iter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        isConverged = np.array([residual[i] < epsilon], dtype=bool)
        comm.Bcast(isConverged, root=0)
        if isConverged:
            break

        # 解の更新
        v = mpi_matvec(local_A, p, Ax, local_Ax, comm)
        alpha = mpi_vecvec(r, p, local_a, local_b, comm) / mpi_vecvec(p, v, local_a, local_b, comm)
        x += alpha * p
        old_r = r.copy()
        r -= alpha * mpi_matvec(local_A, p, Ax, local_Ax, comm)
        beta = mpi_vecvec(r, r, local_a, local_b, comm) / mpi_vecvec(old_r, old_r, local_a, local_b, comm)
        p = r + beta * p
        i += 1
        num_of_solution_updates[i] = i
    else:
        isConverged = False

    if rank == 0:
        elapsed_time = end(start_time, isConverged, i, residual[i])
        return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
    else:
        exit(0)
