import sys
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm, multi_dot

if __name__ == "__main__":
    homedir = os.path.expanduser("~")
    directory = os.path.join(homedir, 'krylov')
    sys.path.append(directory)

from krylov.method.threads.common import start, end as finish, init
import krylov.method.threads.pcg as mpcg
import krylov.method.threads.chronopoulos_gear as mcg
import krylov.method.threads.pipeline as mpl
import krylov.method.threads.gropp as mg


# def pcg(A, b, M, epsilon, callback=None, T=np.float64):
def pcg(A, b, ilu, epsilon, callback=None, T=np.float64):
    isConverged = False
    x, b_norm, N, max_iter, residual, solution_updates = init(
        A, b, T, pu='cpu')

    start_time = start(method_name='Preconditioned CG')

    r = b - dot(A, x)
    residual[0] = norm(r)/b_norm

    # u = dot(M, r)
    u = ilu.solve(r)
    p = u.copy()

    i = 0
    for i in range(1, max_iter):
        s = dot(A, p)
        # s = ilu.solve(p)

        alpha = dot(r, u)/dot(s, p)
        x += alpha*p
        r -= alpha*s
        residual[i] = norm(r)/b_norm
        if residual[i] < epsilon:
            isConverged = True
            break
        old_r = r.copy()
        old_u = u.copy()

        # u = dot(M, r)
        u = ilu.solve(r)

        beta = dot(r, u)/dot(old_r, old_u)
        p = u + beta*p

    end(start_time, isConverged, i, residual[i])


def chronopoulos_gear(A, b, ilu, epsilon, callback=None, T=np.float64):
    isConverged = False
    x, b_norm, N, max_iter, residual, solution_updates = init(
        A, b, T, pu='cpu')

    start_time = start(method_name='chronopoulos gear')

    r = b - dot(A, x)
    residual[0] = norm(r)/b_norm
    # u = dot(M, r)
    u = ilu.solve(r)
    w = dot(A, u)

    alpha = dot(r, u)/dot(w, u)
    beta = 0
    gamma = dot(r, u)
    old_gamma = gamma

    p = np.zeros(N, T)
    s = np.zeros(N, T)

    for i in range(1, max_iter):
        p = u + beta*p
        s = w + beta*s
        x += alpha*p
        r -= alpha*s

        residual[i] = norm(r)/b_norm
        print(residual[i])
        if residual[i] < epsilon:
            isConverged = True
            break

        # u = dot(M, r)
        u = ilu.solve(r)
        w = dot(A, u)
        gamma = dot(r, u)
        delta = dot(w, u)
        beta = gamma/old_gamma
        alpha = gamma/(delta - beta*gamma/alpha)

    end(start_time, isConverged, i, residual[i])


def pipeline(A, b, ilu, epsilon, callback=None, T=np.float64):
    isConverged = False
    x, b_norm, N, max_iter, residual, solution_updates = init(
        A, b, T, pu='cpu')

    start_time = start(method_name='pipeline')

    r = b - dot(A, x)
    residual[0] = norm(r) / b_norm

    # u = dot(M, r)
    u = ilu.solve(r)

    w = dot(A, u)

    z = np.zeros(N)
    q = np.zeros(N)
    s = np.zeros(N)
    p = np.zeros(N)

    i = 0 
    for i in range(1, max_iter):
        gamma = dot(r, u)
        old_gamma = gamma
        delta = dot(w, u)

        # m = dot(M, w)
        m = ilu.solve(r)

        n = dot(A, m)
        if i > 1:
            beta = gamma/old_gamma
            alpha = gamma/(delta - beta*gamma/alpha)
        else:
            beta = 0
            alpha = gamma/delta
        z = n + beta*z
        q = m + beta*q
        s = w + beta*s
        p = u + beta*p
        x += alpha*p
        r -= alpha*s
        residual[i] = norm(r) / b_norm
        if residual[i] < epsilon:
            isConverged = True
            break
        u -= alpha*q
        w -= alpha*z

    end(start_time, isConverged, i, residual[i])


# def gropp(A, b, M, epsilon, callback=None, T=np.float64):
def gropp(A, b, ilu, epsilon, callback=None, T=np.float64):
    isConverged = False
    x, b_norm, N, max_iter, residual, solution_updates = init(
        A, b, T, pu='cpu')

    start_time = start(method_name='gropp')

    r = b - dot(A, x)
    residual[0] = norm(r) / b_norm
    # u = dot(M, r)
    u = ilu.solve(r)
    p = u.copy()
    s = dot(A, p)
    gamma = dot(r, u)

    i = 0 
    for i in range(1, max_iter):
        delta = dot(p, s)
        # q = dot(M, s)
        q = ilu.solve(s)
        alpha = gamma/delta
        x += alpha*p
        r -= alpha*s
        residual[i] = norm(r)/b_norm
        if residual[i] < epsilon:
            isConverged = True
            break
        u -= alpha*q
        gamma = dot(r, u)
        old_gamma = gamma
        w = dot(A, u)
        beta = gamma/old_gamma
        p = u + beta*p
        s = w + beta*s

    end(start_time, isConverged, i, residual[i])
    return isConverged


if __name__ == "__main__":
    from krylov.util import toepliz_matrix_generator, precondition, load_condition
    params = load_condition.load_condition_params('../../condition.json')

    N = params['N']
    diag = params['diag']
    sub_diag = params['sub_diag']
    eps = params['epsilon']

    mn = params['method_name']
    mt = params['matrix_type']
    length = params['length']
    k = params['k']
    pt = params['processor_type']
    nop = params['number_of_process']

    T = np.float64
    A, b = toepliz_matrix_generator.generate(N, diag, sub_diag, T)
    # ilu = precondition.ilu(A)
    ilu = precondition.lu(A)

    # pcg(A, b, ilu, eps, T)
    # chronopoulos_gear(A, b, ilu, eps, T)
    # pipeline(A, b, ilu, eps, T)
    # gropp(A, b, ilu, eps, T)
    
    mpcg.pcg(A, b, ilu, eps, T)
    mcg.chronopoulos_gear(A, b, ilu, eps, T)
    mpl.pipeline(A, b, ilu, eps, T)
    mg.gropp(A, b, ilu, eps, T)
