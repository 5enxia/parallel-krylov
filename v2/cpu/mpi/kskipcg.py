import numpy as np
from numpy import dot
from numpy.linalg import norm

from .common import start, finish, init, init_mpi


def kskipcg(A, b, epsilon, k, T):
    comm, rank, num_of_process = init_mpi()

    # 共通初期化
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(
        A, b, T)
    begin, end = rank * local_N, (rank+1) * local_N
    # local_A, Ax, local_Ax = init_matvec(N, local_N, T)
    Ax = np.empty(N, T)

    # root
    Ar = np.zeros((k + 2, N), T)
    Ap = np.zeros((k + 3, N), T)
    a = np.zeros(2*k + 2, T)
    f = np.zeros(2*k + 4, T)
    c = np.zeros(2*k + 2, T)
    # local
    local_a = np.zeros(2*k + 2, T)
    local_f = np.zeros(2*k + 4, T)
    local_c = np.zeros(2*k + 2, T)

    # 初期残差
    comm.Allgather(A[begin:end].dot(x), Ax)
    Ar[0] = b - Ax
    Ap[0] = Ar[0].copy()

    # 反復計算
    i = 0
    index = 0
    if rank == 0:
        start_time = start(method_name='k-skip CG + MPI', k=k)
    while i < max_iter:
        # 収束判定
        residual[index] = norm(Ar[0]) / b_norm
        if residual[index] < epsilon:
            isConverged = True
            break

        # 基底計算
        for j in range(1, k + 1):
            comm.Allgather(A[begin:end].dot(Ar[j-1]), Ar[j])
            comm.Allgather(A[begin:end].dot(Ap[j-1]), Ap[j])
        comm.Allgather(A[begin:end].dot(Ap[k]), Ap[k+1])

        # 係数計算
        for j in range(2 * k + 1):
            jj = j // 2
            local_a[j] = dot(Ar[jj][begin:end], Ar[jj + j % 2][begin:end])
        comm.Allreduce(local_a, a)
        for j in range(2 * k + 4):
            jj = j // 2
            local_f[j] = dot(Ap[jj][begin:end], Ap[jj + j % 2][begin:end])
        comm.Allreduce(local_f, f)
        for j in range(2 * k + 2):
            jj = j // 2
            local_c[j] = dot(Ar[jj][begin:end], Ap[jj + j % 2][begin:end])
        comm.Allreduce(local_c, c)

        # CGでの1反復
        # 解の更新
        alpha = a[0] / f[1]
        beta = alpha ** 2 * f[2] / a[0] - 1
        x += alpha * Ap[0]
        Ar[0] -= alpha * Ap[1]
        Ap[0] = Ar[0] + beta * Ap[0]
        comm.Allgather(A[begin:end].dot(Ap[0]), Ap[1])

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
            comm.Allgather(A[begin:end].dot(Ap[0]), Ap[1])

        i += (k + 1)
        index += 1
        num_of_solution_updates[index] = i
    else:
        isConverged = False
        residual[index] = norm(Ar[0]) / b_norm

    if rank == 0:
        elapsed_time = finish(start_time, isConverged, i, residual[index])
        return elapsed_time, num_of_solution_updates[:index+1], residual[:index+1]
    else:
        exit(0)
