from mpi4py import MPI
from krylov.method._common import _start, _end

def start(method_name = '', k = None):
    _start(method_name, k)
    return MPI.Wtime()

def end(start_time, isConverged, num_of_iter, residual, residual_index, final_k = None):
    elapsed_time = MPI.Wtime() - start_time
    _end(elapsed_time, isConverged, num_of_iter, residual, residual_index, final_k)
    return elapsed_time