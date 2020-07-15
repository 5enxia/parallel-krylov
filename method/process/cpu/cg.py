import numpy as np
from numpy.linalg import norm
from mpi4py import MPI

    from .common import getConditionParams
    from krylov.method.process.cpu.common import init, init_matvec, init_vecvec, start, end, mpi_matvec, mpi_vecvec


def cg(A, b, epsilon, T=np.float64):
    # 共通初期化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_of_process = comm.Get_size()
    x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T)
    local_N, local_A, Ax, local_Ax = init_matvec(N, num_of_process, T)
    local_a, local_b = init_vecvec(local_N, T)
    comm.Scatter(A, local_A, root=0)

    # 初期残差
    r = b - mpi_matvec(local_A, x, Ax, local_Ax, comm)
    p = r.copy()

    # 反復計算
    if rank == 0:
        start_time = start(method_name='CG')
    for i in range(max_iter):
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
        num_of_solution_updates[i] = i + 1

    else:
        isConverged = False

    num_of_iter = i + 1
    residual_index = i
    if rank == 0:
        end(start_time, isConverged, num_of_iter, residual, residual_index)


if __name__ == "__main__":
    A, b, epsilon, k, T = getConditionParams('condition.json')
    cg(A, b, epsilon, T)
