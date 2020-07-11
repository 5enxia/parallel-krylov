def matvec(A, local_A_cpu, local_A_gpu, x, local_x, local_x_gpu):
    comm.Bcast(x, root=0)
    comm.Scatter(A, local_A_cpu)
    with cp.cuda.Devise(0):
        local_A_gpu, b_gpu = cp.asarray(local_A_cpu), cp.asarray(b)
        local_y_gpu = cp.dot(local_A_gpu, x_gpu)
        local_y_cpu = local_y_gpu.get()
    comm.Gather(local_y_cpu, y_cpu)
    y_gpu = cp.asarray(y_cpu)
    return y_gpu

def vecvec(a, local_a_cpu, local_a_gpu, b, local_b_cpu, local_b_gpu):
    comm.Scatter(a, local_a_cpu, root=0)
    comm.Scatter(b, local_b_cpu, root=0)
    with cp.cuda.Device(0):
        local_y = cp.dot(cp.array(local_a), cp.array(local_b))
    comm.Reduce(local_y.get(), y,root=0)
    return cp.asarray(y)

def start():
    print('# ============== INFO ================= #')
    print(f'Method:\tadaptive k skip MrR')
    print(f'k:\t{ k }')
    return MPI.Wtime()

def end(start_time, isConverged, num_of_iter, residual, residual_index, final_k = None):
    elapsed_time = MPI.Wtime() - start_time
    print(f'time:\t{ elapsed_time } s')
    status = 'converged' if isConverged else 'diverged'
    print(f'status:\t{ status }')
    if isConverged:
        print(f'iteration:\t{ num_of_iter } times')
        print(f'residual:\t{residuals[residual_index]}')
        if final_k:
            print(f'final k:\t{final_k}')
    print('# ===================================== #')