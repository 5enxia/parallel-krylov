import numpy as np

from .common import start, end as finish, init, init_mpi


def _cg_cpu(A, b, epsilon, T, pu):
    from numpy.linalg import norm
    from numpy import dot

    # 共通初期化
    comm, rank, num_of_process = init_mpi()
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    begin, end = rank * local_N, (rank+1) * local_N
    Ax = np.empty(N, T)

    # 初期残差
    comm.Allgather(A[begin:end].dot(x), Ax)
    r = b - Ax
    p = r.copy()
    gamma = dot(r, r)

    # 反復計算
    i = 0
    if rank == 0:
        start_time = start(method_name=f'CG + {pu} + mpi')
    while i < max_iter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        isConverged = residual[i] < epsilon
        if isConverged:
            break

        # 解の更新
        v = dot(A, p)
        sigma = dot(p, v)
        alpha = gamma / sigma
        x += alpha * p
        r -= alpha * v
        old_gamma = gamma.copy()
        gamma = dot(r, r)
        beta = gamma / old_gamma
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


def _cg_gpu(A, b, epsilon, T, pu):
    import cupy as cp
    from cupy.linalg import norm
    from cupy import dot

    # 共通初期化
    comm, rank, num_of_process = init_mpi()
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    begin, end = rank * local_N, (rank+1) * local_N

    # cpu
    Ax = np.empty(N, T)
    v_cpu = np.empty(N, T)

    # 初期残差
    comm.Allgather(A[begin:end].dot(x).get(), Ax)
    r = b - cp.asarray(Ax)
    p = r.copy()
    gamma = dot(r, r)

    # 反復計算
    i = 0
    if rank == 0:
        start_time = start(method_name=f'CG + {pu} + mpi')
    while i < max_iter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        isConverged = residual[i] < epsilon
        if isConverged:
            break

        # 解の更新
        comm.Allgather(A[begin:end].dot(p).get(), v_cpu)
        v = cp.asarray(v_cpu)
        sigma = dot(p, v)
        alpha = gamma / sigma
        x += alpha * p
        r -= alpha * v
        old_gamma = gamma.copy()
        gamma = dot(r, r)
        beta = gamma / old_gamma
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


def cg(A, b, epsilon, T, pu):
    comm, rank, num_of_process = init_mpi()
    _cg = _cg_cpu if pu == 'cpu' else _cg_gpu
    if rank == 0:
        return _cg(A, b, epsilon, T, pu)
    else:
        _cg(A, b, epsilon, T, pu)