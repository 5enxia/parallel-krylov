import sys
import numpy as np
from numpy import dot
from numpy.linalg import norm, multi_dot

from krylov.method.single.common import init, start, end 

def mrr(A, b, epsilon, callback = None, T = np.float64):
    x, b_norm, N, max_iter, residual, solution_updates = init(A, b, T)

    start_time = start(method_name = 'MrR')
    
    r = np.zeros(max_iter, T)
    r = b - dot(A,x) 
    residual[0] = norm(r) / b_norm
    z = np.zeros(N, T)

    # ======= first iter ====== #
    Ar = dot(A,r)
    zeta = dot(r,Ar) / dot(Ar,Ar)
    y = zeta * Ar
    z = -zeta * r
    r -= y
    x -= z
    solution_updates[1] = 1
    # ========================= #

    for i in range(1, max_iter):
        
        residual[i] = norm(r) / b_norm
        if residual[i] < epsilon:
            isConverged = True
            break

        Ar = dot(A,r)
        nu = dot(y,Ar)
        gamma = nu / dot(y,y)
        s = Ar - gamma * y
        zeta = dot(r,s) / dot(s,s)
        eta = -zeta * gamma
        y = eta * y + zeta * Ar
        z = eta * z - zeta * r
        r -= y
        x -= z
        
        solution_updates[i] = i + 1
    
    else:
        isConverged = False
    
    num_of_iter = i + 1
    residual_index = i 

    end(start_time, isConverged, num_of_iter, residual, residual_index)
    
    return isConverged