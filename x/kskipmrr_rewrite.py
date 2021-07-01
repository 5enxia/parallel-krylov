import numpy as np

from .common import start, end as finish, init, init_mpi


def _kskipmrr_gpu(A, b, epsilon, k, T, pu):
    import cupy as cp
    from cupy.linalg import norm

    from .common import init_gpu

    # 共通初期化
    comm, rank, num_of_process = init_mpi()
    init_gpu(rank)
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    begin, end = rank * local_N, (rank+1) * local_N

    # 初期化
    ArAy = cp.zeros((2, k+2, N))
    local_ArAy = cp.zeros((2, k+2, local_N))
    ArAy_cpu = np.zeros((2, k+2, local_N), T)
    abd = cp.zeros((3, 2*k + 3), T)
    abd_cpu = np.zeros((3, 2*k + 3))

    # 初期残差
    ArAy[0][0] = b - A.dot(x)
    residual[0] = norm(ArAy[0][0]) / b_norm

    # 初期反復
    if rank == 0:
        start_time = start(method_name='k-skip MrR', k=k)
    ArAy[0][1] = A.dot(ArAy[0][0])
    zeta = ArAy[0][0].dot(ArAy[0][1]) / ArAy[0][1].dot(ArAy[0][1])
    ArAy[1][0] = zeta * ArAy[0][1]
    z = -zeta * ArAy[0][0]
    ArAy[0][0] -= ArAy[1][0]
    x -= z
    i = 1
    index = 1
    num_of_solution_updates[1] = 1

    # 反復計算
    while i < max_iter:
        # 収束判定
        residual[index] = norm(ArAy[0][0]) / b_norm
        isConverged = residual[index] < epsilon
        if isConverged:
            break

        # 基底計算
        for j in range(1, k + 2):
            ArAy[0][j] = A.dot(ArAy[0][j-1])
        for j in range(1, k + 1):
            ArAy[1][j] = A.dot(ArAy[1][j-1])
        
        # 集団通信
        comm.Scatter(ArAy.get(), ArAy_cpu)
        local_ArAy = cp.asarray(ArAy_cpu)
        for j in range(2*k + 3):
            jj = j//2
            abd[0][j] = local_ArAy[0][jj].dot(local_ArAy[0][jj + j % 2])
        for j in range(1, 2*k + 2):
            jj = j//2
            abd[1][j] = local_ArAy[1][jj].dot(local_ArAy[0][jj + j % 2])
        for j in range(2*k + 1):
            jj = j//2
            abd[2][j] = local_ArAy[1][jj].dot(local_ArAy[1][jj + j % 2])
        comm.Reduce(abd.get(), abd_cpu)
        abd = cp.asarray(abd_cpu)

        # MrRでの1反復(解と残差の更新)
        d = abd[0][2] * abd[2][0] - abd[1][1] ** 2
        zeta = abd[0][1] * abd[2][0] / d
        eta = -abd[0][1] * abd[1][1] / d
        ArAy[1][0] = eta * ArAy[1][0] + zeta * ArAy[0][1]
        z = eta * z - zeta * ArAy[0][0]
        ArAy[0][0] -= ArAy[1][0]
        ArAy[0][1] = A.dot(ArAy[0][0])
        x -= z

        # MrRでのk反復
        for j in range(k):
            abd[2][0] = zeta ** 2 * abd[0][2] + eta * zeta * abd[1][1]
            abd[0][0] -= zeta * abd[0][1]
            abd[2][1] = eta ** 2 * abd[2][1] + 2 * eta * zeta * abd[1][2] + zeta ** 2 * abd[0][3]
            abd[1][1] = eta * abd[1][1] + zeta * abd[0][2] - abd[2][1]
            abd[0][1] = -abd[1][1]
            for l in range(2, 2 * (k - j) + 1):
                abd[2][l] = eta ** 2 * abd[2][l] + 2 * eta * zeta * abd[1][l+1] + zeta ** 2 * abd[0][l + 2]
                tau = eta * abd[1][l] + zeta * abd[0][l + 1]
                abd[1][l] = tau - abd[2][l]
                abd[0][l] -= tau + abd[1][l]
            # 解と残差の更新
            d = abd[0][2] * abd[2][0] - abd[1][1] ** 2
            zeta = abd[0][1] * abd[2][0] / d
            eta = -abd[0][1] * abd[1][1] / d
            ArAy[1][0] = eta * ArAy[1][0] + zeta * ArAy[0][1]
            z = eta * z - zeta * ArAy[0][0]
            ArAy[0][0] -= ArAy[1][0]
            ArAy[0][1] = A.dot(ArAy[0][0])
            x -= z

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


def kskipmrr(A, b, epsilon, k, T, pu):
    comm, rank, num_of_process = init_mpi()
    if pu == 'cpu':
        pass
    else:
        if rank == 0:
            return _kskipmrr_gpu(A, b, epsilon, k, T, pu)
        else:
            _kskipmrr_gpu(A, b, epsilon, k, T, pu)
