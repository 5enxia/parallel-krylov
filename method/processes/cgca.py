import numpy as np

from .common import start, end, init, init_mpi


def cg(A, b, epsilon, T, pu):
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
    v = cp.empty(N, T)

    # 初期残差
    comm.Allgather(A[begin:end].dot(x), Ax)
    r = b - Ax
    p = r.copy()
    gamma = dot(r, r)

    # 反復計算
    i = 0
    if rank == 0:
        start_time = start(method_name='CG')
    while i < max_iter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        if residual[i] < epsilon:
            isConverged = True
            break

        # 解の更新
        comm.Allgather(A[begin:end].dot(p), v)
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
