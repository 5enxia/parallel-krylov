import numpy as np
from numpy.linalg import norm

from .common import start, finish, init, init_mpi


def mrr(A, b, epsilon, T):
    # MPI初期化
    comm, rank, num_of_process = init_mpi()

    # 共通初期化
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T)
    begin, end = rank * local_N, (rank+1) * local_N

    # 初期化
    Ax = np.empty(N, T)
    Ar = np.empty(N, T)
    s = np.empty(N, T)
    rs = np.empty(1, T)
    ss = np.empty(1, T)
    nu = np.empty(1, T)
    mu = np.empty(1, T)

    # 初期残差
    comm.Allgather(A[begin:end].dot(x), Ax)
    r = b - Ax
    residual[0] = norm(r) / b_norm

    # 初期反復
    if rank == 0:
        start_time = start(method_name='MrR')
    comm.Allgather(A[begin:end].dot(r), Ar)
    comm.Allreduce(r[begin:end].dot(Ar[begin:end]), rs)
    comm.Allreduce(Ar[begin:end].dot(Ar[begin:end]), ss)
    zeta = rs / ss
    y = zeta * Ar
    z = -zeta * r
    r -= y
    x -= z

    i = 1
    num_of_solution_updates[1] = 1

    # 反復計算
    while i < max_iter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        isConverged = residual[i] < epsilon
        if isConverged:
            break

        # 解の更新
        comm.Allgather(A[begin:end].dot(r), Ar)
        comm.Allreduce(y[begin:end].dot(Ar[begin:end]), nu)
        comm.Allreduce(y[begin:end].dot(y[begin:end]), mu)
        gamma = nu / mu
        s = Ar - gamma * y
        comm.Allreduce(r[begin:end].dot(s[begin:end]), rs)
        comm.Allreduce(s[begin:end].dot(s[begin:end]), ss)
        zeta = rs / ss
        eta = -zeta * gamma
        y = eta * y + zeta * Ar
        z = eta * z - zeta * r
        r -= y
        x -= z
        i += 1
        num_of_solution_updates[i] = i
    else:
        isConverged = False
        residual[i] = norm(r) / b_norm

    if rank == 0:
        elapsed_time = finish(start_time, isConverged, i, residual[i])
        return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
    else:
        exit(0)
