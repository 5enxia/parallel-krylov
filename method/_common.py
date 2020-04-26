def _start(method_name = '', k = None):
    print('# ============== INFO ================= #')
    print(f'Method:\t{ method_name }')
    print(f'k:\t{ k }')

def _end(elapsed_time, isConverged, num_of_iter, residual, residual_index, final_k = None):
    print(f'time:\t{ elapsed_time } s')
    status = 'converged' if isConverged else 'diverged'
    print(f'status:\t{ status }')
    if isConverged:
        print(f'iteration:\t{ num_of_iter } times')
        print(f'residual:\t{residual[residual_index]}')
        if final_k:
            print(f'final k:\t{final_k}')
    print('# ===================================== #')