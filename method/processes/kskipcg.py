import numpy as np

from .common import start, end, init, init_mpi


def kskipcg(A, b, epsilon, k, T, pu):
    comm, rank, num_of_process = init_mpi()
    if pu == 'cpu':
        import numpy as xp
        from numpy import dot
        from numpy.linalg import norm
        from .cpu import init_matvec, mpi_matvec
    else:
        import cupy as xp
        from cupy import dot
        from cupy.linalg import norm
        from .common import init_gpu
        init_gpu(rank)
        from .gpu import init_matvec,  mpi_matvec

    # 共通初期化
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    local_A, Ax, local_Ax = init_matvec(N, local_N, T)
    if pu == 'cpu':
        comm.Scatter(A, local_A)
    else:
        local_A_cpu = np.empty((local_N, N), T)
        comm.Scatter(A, local_A_cpu)
        local_A = cp.asarray(local_A_cpu)
    # root
    Ar = xp.zeros((k + 2, N), T)
    Ap = xp.zeros((k + 3, N), T)
    a = xp.zeros(2*k + 2, T)
    f = xp.zeros(2*k + 4, T)
    c = xp.zeros(2*k + 2, T)
    # local
    local_a = xp.zeros(2*k + 2, T)
    local_f = xp.zeros(2*k + 4, T)
    local_c = xp.zeros(2*k + 2, T)

    # 初期残差
    Ar[0] = b - mpi_matvec(local_A, x, Ax, local_Ax, comm)
    Ap[0] = Ar[0].copy()

    # 反復計算
    i = 0
    index = 0
    if rank == 0:
        start_time = start(method_name='k-skip CG', k=k)
    while i < max_iter:
        # 収束判定
        residual[index] = norm(Ar[0]) / b_norm
        isConverged = np.array([residual[index] < epsilon], bool)
        comm.Bcast(isConverged)
        if isConverged:
            break

        # 事前計算
        for j in range(1, k + 1):
            Ar[j] = mpi_matvec(local_A, Ar[j-1], Ax, local_Ax, comm)
        for j in range(1, k + 2):
            Ap[j] = mpi_matvec(local_A, Ap[j-1], Ax, local_Ax, comm)
        comm.Bcast(Ar)
        comm.Bcast(Ap)
        for j in range(2 * k + 1):
            jj = j // 2
            local_a[j] = dot(
                Ar[jj][rank * local_N: (rank+1) * local_N],
                Ar[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_a, a, root=0)
        for j in range(2 * k + 4):
            jj = j // 2
            local_f[j] = dot(
                Ap[jj][rank * local_N: (rank+1) * local_N],
                Ap[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_f, f, root=0)
        for j in range(2 * k + 2):
            jj = j // 2
            local_c[j] = dot(
                Ar[jj][rank * local_N: (rank+1) * local_N],
                Ap[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_c, c, root=0)

        # CGでの1反復
        # 解の更新
        alpha = a[0] / f[1]
        beta = alpha ** 2 * f[2] / a[0] - 1
        x += alpha * Ap[0]
        Ar[0] -= alpha * Ap[1]
        Ap[0] = Ar[0] + beta * Ap[0]
        Ap[1] = mpi_matvec(local_A, Ap[0], Ax, local_Ax, comm)

        # CGでのk反復
        for j in range(k):
            for l in range(0, 2*(k-j)+1):
                a[l] += alpha*(alpha*f[l+2] - 2*c[l+1])
                d = c[l] - alpha*f[l+1]
                c[l] = a[l] + d*beta
                f[l] = c[l] + beta*(d + beta*f[l])
            
            # 解の更新
            alpha = a[0] / f[1]
            beta = alpha ** 2 * f[2] / a[0] - 1
            x += alpha * Ap[0]
            Ar[0] -= alpha * Ap[1]
            Ap[0] = Ar[0] + beta * Ap[0]
            Ap[1] = mpi_matvec(local_A, Ap[0], Ax, local_Ax, comm)

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
