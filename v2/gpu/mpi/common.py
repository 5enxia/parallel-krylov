import time

import numpy as np
import scipy
import cupy as cp
from cupy.cuda import Device
from mpi4py import MPI

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
def init(A, b, T, rank, num_of_process, num_of_all_of_gpu = 16) -> tuple:
    # 追加する要素数を算出
    old_N = b.size
    num_of_append = num_of_all_of_gpu - (old_N % num_of_all_of_gpu) # 足りない行を計算
    num_of_append = 0 if num_of_append == num_of_all_of_gpu else num_of_append
    N = old_N + num_of_append
    local_N = N // num_of_process
    begin, end = rank * local_N, (rank+1) * local_N

    ## A
    if num_of_append:
        # データをパディングする
        if isinstance(A, np.ndarray):
            if num_of_append:
                A = np.append(A, np.zeros((old_N, num_of_append)), axis=1)  # 右に0を追加
                A = np.append(A, np.zeros((num_of_append, N)), axis=0)  # 下に0を追加
        elif isinstance(A, scipy.sparse.csr.csr_matrix):
            from scipy.sparse import hstack, vstack, csr_matrix
            if num_of_append:
                A = hstack([A, csr_matrix((old_N, num_of_append))], 'csr') # 右にemptyを追加
                A = vstack([A, csr_matrix((num_of_append, N))], 'csr') # 下にemptyを追加
    local_A = A[begin:end]

    ## b
    b = cp.array(b, T)
    if num_of_append:
        b = cp.append(b, cp.zeros(num_of_append))  # 0を追加
    b_norm = cp.linalg.norm(b)

    # x
    x = cp.zeros(N, T)

    # その他パラメータ
    max_iter = old_N * 2
    residual = cp.zeros(max_iter+1, T)
    num_of_solution_updates = cp.zeros(max_iter+1, np.int)
    num_of_solution_updates[0] = 0

    return local_A, b, x, b_norm, N, max_iter, residual, num_of_solution_updates


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

    # GPUの初期化
    @classmethod
    def init_gpu(cls, begin: int, end: int, num_of_process: int):
        cls.begin = begin
        cls.end = end
        cls.num_of_gpu = end - begin + 1
        cls.num_of_process = num_of_process

        # init memory allocator
        for i in range(end, begin-1, -1):
            Device(i).use()
            pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
            cp.cuda.set_allocator(pool.malloc)

    
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
        for i in range(cls.end, cls.begin-1, -1):
            Device(i).use()
            index = i-cls.begin
            begin, end = i*cls.local_local_N, (i+1)*cls.local_local_N
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
    def dot(cls, A, x, out) -> None:
        # Copy vector data to All devices
        for i in range(cls.end, cls.begin-1, -1):
            Device(i).use()
            index = i-cls.begin
            cp.cuda.runtime.memcpyPeer(cls.x[index].data.ptr, i, x.data.ptr, cls.begin, cls.nbytes)
        # dot
        # for i in range(cls.end, cls.begin-1, -1):
        #     Device(i).use()
        #     index = i-cls.begin
            cls.y[index] = cls.A[index].dot(cls.x[index])
        # Gather caculated element from All devices
        for i in range(cls.end, cls.begin-1, -1):
            Device(i).use()
            Device(i).synchronize()
            index = i-cls.begin
            cp.cuda.runtime.memcpyPeer(cls.out[index*cls.local_local_N].data.ptr, cls.begin, cls.y[index].data.ptr, i, cls.y[index].nbytes)

        for i in range(cls.end, cls.begin-1, -1):
            Device(i).use()
            Device(i).synchronize()

        cls.comm.Allgather(cls.out, out)
        cls.comm.Barrier()
    
    # joint comm
    @classmethod
    def joint_mpi(cls, comm):
        cls.comm = comm

    @classmethod
    def sync(cls):
        Device(cls.begin).synchronize()


# mpi
def init_mpi():
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


# gpu
def calc_alloc_gpu(rank: int, num_of_process: int) -> tuple:
    # local
    if num_of_process == 2:
        return rank, rank
    # ito
    elif num_of_process == 4:
        return 0, 3
    elif num_of_process == 8:
        # odd
        if rank % 2 == 0:
            return 0, 1
        else:
            return 2, 3
    elif num_of_process == 16:
        _id = rank % 4
        return _id, _id 
    else:
        return 0, 0
