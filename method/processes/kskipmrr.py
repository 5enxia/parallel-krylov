import numpy as np

from .common import start, end, init, init_mpi


def kskipmrr(A, b, epsilon, k, T, pu):
    comm, rank, num_of_process = init_mpi()
    if pu == 'cpu':
        import numpy as xp
        from numpy import dot
        from numpy.linalg import norm
        from .common import init_matvec, init_vecvec, mpi_matvec, mpi_vecvec1, mpi_vecvec2
    else:
        import cupy as xp
        from cupy import dot
        from cupy.linalg import norm
        from .common import init_gpu
        init_gpu(rank)

    # 共通初期化
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    local_A, Ax, local_Ax = init_matvec(N, local_N, T)
    local_a, local_b = init_vecvec(local_N, T)
    comm.Scatter(A, local_A, root=0)
    # root
    Ar = xp.empty((k + 3, N), T)
    Ay = xp.empty((k + 2, N), T)
    alpha = xp.empty(2*k + 3, T)
    beta = xp.empty(2*k + 2, T)
    beta[0] = 0
    delta = xp.empty(2*k + 1, T)
    # local
    local_alpha = xp.empty(2*k + 3, T)
    local_beta = xp.empty(2*k + 2, T)
    local_beta[0] = 0
    local_delta = xp.empty(2*k + 1, T)
    
    # 初期残差
    Ar[0] = b - mpi_matvec(local_A, x, Ax, local_Ax, comm)
    residual[0] = norm(Ar[0]) / b_norm

    # 初期反復
    if rank == 0:
        start_time = start(method_name='k-skip MrR', k=k)
    Ar[1] = mpi_matvec(local_A, Ar[0], Ax, local_Ax, comm)
    zeta = mpi_vecvec2(Ar[0], Ar[1], local_a, local_b, comm) / mpi_vecvec1(Ar[1], local_a, comm)
    Ay[0] = zeta * Ar[1]
    z = -zeta * Ar[0]
    Ar[0] -= Ay[0]
    x -= z
    num_of_solution_updates[1] = 1
    i = 1
    index = 1

    # 反復計算
    while i < max_iter:
        # 収束判定
        residual[index] = norm(Ar[0]) / b_norm
        isConverged = np.array([residual[index] < epsilon], dtype=bool)
        comm.Bcast(isConverged, root=0)
        if isConverged:
            break

        # 事前計算
        for j in range(1, k + 2):
            Ar[j] = mpi_matvec(local_A, Ar[j-1], Ax, local_Ax, comm)
        for j in range(1, k + 1):
            Ay[j] = mpi_matvec(local_A, Ay[j-1], Ax, local_Ax, comm)
        comm.Bcast(Ar)
        comm.Bcast(Ay)
        for j in range(2*k + 3):
            jj = j // 2
            local_alpha[j] = dot(
                Ar[jj][rank * local_N: (rank+1) * local_N],
                Ar[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_alpha, alpha, root=0)
        for j in range(1, 2 * k + 2):
            jj = j//2
            local_beta[j] = dot(
                Ay[jj][rank * local_N: (rank+1) * local_N],
                Ar[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_beta, beta, root=0)
        for j in range(2 * k + 1):
            jj = j // 2
            local_delta[j] = dot(
                Ay[jj][rank * local_N: (rank+1) * local_N],
                Ay[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_delta, delta, root=0)

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

        i += (k + 1)
        index += 1
        num_of_solution_updates[index] = i
    else:
        isConverged = False
        residual[index] = norm(Ar[0]) / b_norm

    if rank == 0:
        elapsed_time = end(start_time, isConverged, i, residual[index])
        return elapsed_time, num_of_solution_updates[:index+1], residual[:index+1]
    else:
        exit(0)
