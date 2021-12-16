import numpy as np

from .common import start, end as finish, init, init_mpi


def _mrr_gpu(A, b, epsilon, T, pu):
    import cupy as cp
    from cupy.linalg import norm
    from cupy import dot

    from .common import init_gpu

    # 共通初期化
    comm, rank, num_of_process = init_mpi()
    init_gpu(rank)
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    begin, end = rank * local_N, (rank+1) * local_N

    # 初期化
    Ax = cp.empty(N, T)
    Ar = cp.empty(N, T)
    s = cp.empty(N, T)
    rs = cp.empty(1, T)
    ss = cp.empty(1, T)
    nu = cp.empty(1, T)
    mu = cp.empty(1, T)

    # 初期残差
    comm.Allgather(A[begin:end].dot(x), Ax)
    r = b - Ax
    residual[0] = norm(r) / b_norm

    # 初期反復
    if rank == 0:
        start_time = start(method_name='MrR')
    comm.Allgather(A[begin:end].dot(r), Ar)
    rs = dot(r, Ar)
    ss = dot(Ar, Ar)
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
        
        comm.Allreduce(y[begin:end].dot(y[begin:end]).get(), mu_cpu)
        nu = dot(y, Ar)
        mu = dot(y, y)
        gamma = nu / mu
        s = Ar - gamma * y
        rs = dot(r, s)
        ss = dot(s, s)
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


def mrr(A, b, epsilon, T, pu):
    comm, rank, num_of_process = init_mpi()
    _mrr = _mrr_cpu if pu == 'cpu' else _mrr_gpu
    if rank == 0:
        return _mrr(A, b, epsilon, T, pu)
    else:
        _mrr(A, b, epsilon, T, pu)
