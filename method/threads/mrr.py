from .common import start, end


def _mrr(A, b, epsilon, x, b_norm, N, max_iter, residual, num_of_solution_updates, pu):
    if pu == 'cpu':
        from numpy import dot
        from numpy.linalg import norm
    else:
        from cupy import dot
        from cupy.linalg import norm

    # 初期残差
    r = b - dot(A, x)
    residual[0] = norm(r) / b_norm

    # 初期反復
    start_time = start(method_name='MrR')
    Ar = dot(A, r)
    zeta = dot(r, Ar) / dot(Ar, Ar)
    y = zeta * Ar
    z = -zeta * r
    r -= y
    x -= z
    num_of_solution_updates[1] = 1
    i = 1

    # 反復計算
    while i < max_iter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        if residual[i] < epsilon:
            isConverged = True
            break

        # 解の更新
        Ar = dot(A, r)
        gamma = dot(y, Ar) / dot(y, y)
        s = Ar - gamma * y
        zeta = dot(r, s) / dot(s, s)
        eta = -zeta * gamma
        y = eta * y + zeta * Ar
        z = eta * z - zeta * r
        r -= y
        x -= z
        i += 1
        num_of_solution_updates[i] = i
    else:
        isConverged = False

    elapsed_time = end(start_time, isConverged, i, residual[i])
    return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]


def mrr_cpu(A, b, epsilon, T):
    from .common import init_cpu
    x, b_norm, N, max_iter, residual, num_of_solution_updates = init_cpu(A, b, T)
    return _mrr(A, b, epsilon, x, b_norm, N, max_iter, residual, num_of_solution_updates, 'cpu')


def mrr_gpu(A, b, epsilon, T):
    from .common import init_gpu
    A, b, x, b_norm, N, max_iter, residual, num_of_solution_updates = init_gpu(A, b, T)
    return _mrr(A, b, epsilon, x, b_norm, N, max_iter, residual, num_of_solution_updates, 'gpu')
