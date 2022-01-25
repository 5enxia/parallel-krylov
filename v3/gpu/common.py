import time

import numpy as np
import scipy
import cupy as cp
from cupy.cuda import Device
from cupy.cuda.runtime import getDeviceCount()

from ..common import _start, _finish


# 計測開始
def start(method_name: str = '', k: int = None) -> float:
    _start(method_name, k)
    return time.perf_counter()


# 計測終了
def finish(start_time: float, isConverged: bool, num_of_iter: int, final_residual: float, final_k: int = None) -> float:
    elapsed_time = time.perf_counter() - start_time
    _finish(elapsed_time, isConverged, num_of_iter, final_residual, final_k)
    return elapsed_time


# パラメータの初期化
def init(b, x=None, maxiter=None) -> tuple:
    T = np.float64
    b_norm = np.linalg.norm(b)
    N = b.size
    if isinstance(x, np.ndarray):
        x = cp.array(x)
    else:
        x = cp.zeros(N, dtype=T)

    if maxiter == None:
        maxiter = N
    residual = cp.zeros(maxiter+1, T)
    num_of_solution_updates = cp.zeros(maxiter+1, np.int)

    return x, maxiter, b_norm, N, residual, num_of_solution_updates


class MultiGpu(object):
    # numbers
    begin: int = 0
    end: int = 0
    num_of_gpu: int = 0
    # dimentinal size
    N: int = 0
    local_N: int = 0
    # matrix
    A: list = []
    # vector
    x: list = []
    y: list = []
    out: np.ndarray = None
    # byte size
    nbytes: int = 0
    local_nbytes: int = 0
    # gpu stream
    streams = None

    # GPUの初期化
    @classmethod
    def init(cls):
        cls.num_of_gpu = getDeviceCount()
        cls.streams = [None] * cls.num_of_gpu

        # init memory allocator
        for i in range(cls.num_of_gpu):
            Device(i).use()
            pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
            cp.cuda.set_allocator(pool.malloc)
            cls.streams[i] = cp.cuda.Stream()

            # Enable P2P
            for j in range(cls.num_of_gpu):
                if i == j:
                    continue
                cp.cuda.runtime.deviceEnablePeerAccess(j)

    # メモリー領域を確保
    @classmethod
    def alloc(cls, A, b, T):
        # dimentional size
        cls.N = b.size
        cls.local_N = cls.N // cls.num_of_gpu
        # byte size
        cls.nbytes = b.nbytes
        cls.local_nbytes = b.nbytes // cls.num_of_gpu

        # init list
        cls.A = [None] * cls.num_of_gpu
        cls.x = [None] * cls.num_of_gpu
        cls.y = [None] * cls.num_of_gpu

        # allocate A, x, y
        for i in range(cls.num_of_gpu):
            Device(i).use()
            # divide A
            if isinstance(A, np.ndarray):
                cls.A[i] = cp.array(A[i*cls.local_N:(i+1)*cls.local_N], T)
            else:
                from cupyx.scipy.sparse import csr_matrix
                cls.A[i] = csr_matrix(A[i*cls.local_N:(i+1)*cls.local_N])
            cls.x[i] = cp.zeros(cls.N, T)
            cls.y[i] = cp.zeros(cls.local_N, T)

        # allocate output vector
        cls.out = cp.zeros(cls.N, T)

    # matvec with multi-gpu
    @classmethod
    def dot(cls, A, x):
        for i in range(cls.num_of_gpu):
            Device(i).use()
            # copy to workers
            cp.cuda.runtime.memcpyPeerAsync(cls.x[i].data.ptr, i, x.data.ptr, cls.end, cls.nbytes, cls.streams[i].ptr)
            # dot
            cls.y[i] = cls.A[i].dot(cls.x[i])
        for i in range(cls.num_of_gpu):
            # copy to master
            cp.cuda.runtime.memcpyPeerAsync(cls.out[i*cls.local_N].data.ptr, cls.end, cls.y[i].data.ptr, i, cls.y[i].nbytes, cls.streams[i].ptr)
        for i in range(cls.num_of_gpu):
            # sync
            cls.streams[i].synchronize()
        return cls.out
