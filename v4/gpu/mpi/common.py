import os
import numpy as np
import scipy
import cupy as cp
from cupy.cuda import Device
from mpi4py import MPI

# import socket

from ..common import _start, _finish


# 計測開始
def start(method_name='', k=None):
    _start(method_name, k)
    return MPI.Wtime()


# 計測終了
def finish(start_time, isConverged, num_of_iter, final_residual, final_k=None):
    elapsed_time = MPI.Wtime() - start_time
    _finish(elapsed_time, isConverged, num_of_iter, final_residual, final_k)
    return elapsed_time


# パラメータの初期化
def init(b, x=None, maxiter=None) -> tuple:
    T = np.float64
    b = cp.array(b)
    b_norm = cp.linalg.norm(b)
    N = b.size
    if isinstance(x, np.ndarray):
        x = cp.array(x)
    else:
        x = cp.zeros(N, dtype=T)

    if maxiter == None:
        maxiter = N
    residual = cp.zeros(maxiter+1, T)
    num_of_solution_updates = cp.zeros(maxiter+1, np.int)

    return b, x, maxiter, b_norm, N, residual, num_of_solution_updates


class MultiGpu(object):
    # numbers
    begin: int = 0
    end: int = 0
    num_of_gpu: int = 0
    num_of_process: int = 0
    # dimentinal size
    N: int = 0
    local_N: int = 0
    local_local_N: int = 0
    # matrix
    A: list = []
    # vector
    x: list = []
    y: list = []
    out: np.ndarray = None
    # byte size
    nbytes: int = 0
    local_nbytes: int = 0
    local_local_nbytes: int = 0
    # mpi
    comm = None
    # gpu stream
    streams = None

    # GPUの初期化
    @classmethod
    def init(cls):
        # ip = socket.gethostbyname(socket.gethostname())
        # rank = os.environ['MV2_COMM_WORLD_RANK']
        # local_rank = os.environ['MV2_COMM_WORLD_LOCAL_RANK']
        ids = os.environ['GPU_IDS'].split(',')
        cls.begin = int(ids[0])
        cls.end = int(ids[-1])
        cls.num_of_gpu = cls.end - cls.begin + 1
        cls.streams = [None] * cls.num_of_gpu

        # init memory allocator
        for i in range(cls.begin, cls.end+1):
            Device(i).use()
            pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
            cp.cuda.set_allocator(pool.malloc)
            cls.streams[i-cls.begin] = cp.cuda.Stream(non_blocking=False)
        
            # Enable P2P
            for j in range(4):
                if i == j:
                    continue
                cp.cuda.runtime.deviceEnablePeerAccess(j)
    
    # メモリー領域を確保
    @classmethod
    def alloc(cls, local_A, b, T):
        # dimentional size
        cls.local_N, cls.N = local_A.shape
        cls.local_local_N = cls.local_N // cls.num_of_gpu
        # byte size
        cls.nbytes = b.nbytes
        cls.local_nbytes = cls.nbytes // cls.num_of_process
        cls.local_local_nbytes = cls.local_nbytes // cls.num_of_gpu

        # init list
        cls.A = [None] * cls.num_of_gpu
        cls.x = [None] * cls.num_of_gpu
        cls.y = [None] * cls.num_of_gpu

        # divide single A -> multi local_A
        # allocate x, y
        for i in range(cls.begin, cls.end+1):
            Device(i).use()
            index = i-cls.begin
            # local_Aは1/8
            begin, end = index*cls.local_local_N, (index+1)*cls.local_local_N
            # npy
            if isinstance(local_A, np.ndarray):
                cls.A[index] = cp.array(local_A[begin:end], T)
            # npz
            elif isinstance(local_A, scipy.sparse.csr.csr_matrix):
                from cupyx.scipy.sparse import csr_matrix
                cls.A[index] = csr_matrix(local_A[begin:end])
            cls.x[index] = cp.zeros(cls.N, T)
            cls.y[index] = cp.zeros(cls.local_local_N, T)

        # init out vector
        cls.out = cp.zeros(cls.local_N, T)

    # マルチGPUを用いた行列ベクトル積
    @classmethod
    def dot(cls, local_A, x, out):
        # Copy vector data to All devices
        for i in range(cls.begin, cls.end+1):
            Device(i).use()
            index = i-cls.begin
            cp.cuda.runtime.memcpyPeerAsync(cls.x[index].data.ptr, i, x.data.ptr, cls.end, cls.nbytes, cls.streams[index].ptr)
            # dot
            cls.y[index] = cls.A[index].dot(cls.x[index])
        # Gather caculated element from All devices
        for i in range(cls.begin, cls.end+1):
            Device(i).synchronize()
            index = i-cls.begin
            cp.cuda.runtime.memcpyPeerAsync(cls.out[index*cls.local_local_N].data.ptr, cls.end, cls.y[index].data.ptr, i, cls.local_local_nbytes, cls.streams[index].ptr)

        # sync
        for i in range(cls.begin, cls.end+1):
            index = i-cls.begin
            cls.streams[index].synchronize()

        cls.comm.Allgather(cls.out, out)
        # return
        return out
    
    # joint comm
    @classmethod
    def joint_mpi(cls, comm):
        cls.comm = comm
        cls.num_of_process = comm.Get_size()
