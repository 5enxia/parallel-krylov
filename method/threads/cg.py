from .common import start, end


def _cg(A, b, epsilon, x, b_norm, N, max_iter, residual, num_of_solution_updates, pu):
    if pu == 'cpu':
        from numpy import dot
        from numpy.linalg import norm 
    else:
        from cupy import dot
        from cupy.linalg import norm 

    # 初期残差
    r = b - dot(A, x)
    p = r.copy()
    gamma = dot(r, r)

    # 反復計算
    i = 0
    start_time = start(method_name='CG')
    while i < max_iter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        if residual[i] < epsilon:
            isConverged = True
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

    elapsed_time = end(start_time, isConverged, i, residual[i])
    return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]


def cg_cpu(A, b, epsilon, T):
    from .common import init_cpu
    x, b_norm, N, max_iter, residual, num_of_solution_updates = init_cpu(A, b, T)
    return _cg(A, b, epsilon, x, b_norm, N, max_iter, residual, num_of_solution_updates, 'cpu')


def cg_gpu(A, b, epsilon, T):
    from .common import init_gpu
    A, b, x, b_norm, N, max_iter, residual, num_of_solution_updates = init_gpu(A, b, T)
    return _cg(A, b, epsilon, x, b_norm, N, max_iter, residual, num_of_solution_updates, 'gpu')
