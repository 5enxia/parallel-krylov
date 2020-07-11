import json

import numpy as np

from krylov.util import toepliz_matrix_generator


def _start(method_name='', k=None):
    print('# ============== INFO ================= #')
    print(f'Method:\t{ method_name }')
    print(f'k:\t{ k }')


def _end(
    elapsed_time: float, isConverged: bool, num_of_iter: int, residual, residual_index,
    final_k=None
):
    print(f'time:\t{ elapsed_time } s')
    status = 'converged' if isConverged else 'diverged'
    print(f'status:\t{ status }')
    if isConverged:
        print(f'iteration:\t{ num_of_iter } times')
        print(f'residual:\t{residual[residual_index]}')
        if final_k:
            print(f'final k:\t{final_k}')
    print('# ===================================== #')


def getConditionParams(filename: str):
    with open(filename) as f:
        params = json.load(f)
    f.close()
    T = np.float64
    epsilon = params['epsilon']
    N = params['N']
    diag = params['diag']
    sub_diag = params['sub_diag']
    A, b = toepliz_matrix_generator.generate(N, diag, sub_diag, T)
    k = params['k']
    return A, b, epsilon, k, T
