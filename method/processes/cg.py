import numpy as np

from .common import start, end, init, init_mpi


def cg(A, b, epsilon, T, pu):
    comm, rank, num_of_process = init_mpi()
    if pu == 'cpu':
        from numpy.linalg import norm
        from .cpu import init_matvec, init_vecvec, mpi_matvec, mpi_vecvec1, mpi_vecvec2
    else:
        from cupy.linalg import norm
        from .common import init_gpu
        init_gpu(rank)

    # 共通初期化
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    local_A, Ax, local_Ax = init_matvec(N, local_N, T)
    local_a, local_b = init_vecvec(local_N, T)
    comm.Scatter(A, local_A, root=0)

    # 初期残差
    r = b - mpi_matvec(local_A, x, Ax, local_Ax, comm)
    p = r.copy()
    gamma = mpi_vecvec1(r, local_a, comm)

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
        sigma = mpi_vecvec2(p, v, local_a, local_b, comm)
        alpha = gamma / sigma
        x += alpha * p
        r -= alpha * v
        old_gamma = gamma.copy()
        gamma = mpi_vecvec1(r, local_a, comm)
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
