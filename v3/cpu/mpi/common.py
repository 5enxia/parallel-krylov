import numpy as np
import scipy
from numpy.linalg import norm

from mpi4py import MPI

from ..common import _start, _finish, init


def start(method_name='', k=None):
    _start(method_name, k)
    return MPI.Wtime()


def finish(start_time, isConverged, num_of_iter, final_residual, final_k=None):
    elapsed_time = MPI.Wtime() - start_time
    _finish(elapsed_time, isConverged, num_of_iter, final_residual, final_k)
    return elapsed_time


class MultiProc(object):
    # mpi
    comm = None
    # dim
    local_N: int = 0
    # out
    out: np.ndarray = None

    @classmethod
    def joint_mpi(cls, comm):
        cls.comm = comm

    @classmethod
    def alloc(cls, local_A, T):
        cls.local_N = local_A.shape[0]
        cls.A = local_A
        cls.out = np.zeros(cls.local_N, T)

    @classmethod
    def dot(cls, _, x, out):
        cls.out = cls.A.dot(x)
        cls.comm.Allgather(cls.out, out)
        return out