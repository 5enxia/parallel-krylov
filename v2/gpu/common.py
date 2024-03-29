import time

import numpy as np
import scipy
import cupy as cp
from cupy.cuda import Device

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
def init(A, b, T, num_of_thread):
    # 追加する要素数を算出
    old_N = b.size
    num_of_append: int = num_of_thread - (old_N % num_of_thread)  # 足りない行を計算
    num_of_append = 0 if num_of_append == num_of_thread else num_of_append
    N: int = old_N + num_of_append

    # A
    if num_of_append:
        # データをパディングする
        if isinstance(A, np.ndarray):
            if num_of_append:
                A = np.append(A, np.zeros((old_N, num_of_append)), axis=1)  # 右に0を追加
                A = np.append(A, np.zeros((num_of_append, N)), axis=0)  # 下に0を追加
        elif isinstance(A, scipy.sparse.csr.csr_matrix):
            from scipy.sparse import hstack, vstack, csr_matrix
            if num_of_append:
                A = hstack([A, csr_matrix((old_N, num_of_append))], 'csr')  # 右にemptyを追加
                A = vstack([A, csr_matrix((num_of_append, N))], 'csr')  # 下にemptyを追加

    # b
    b = cp.array(b, T)
    if num_of_append:
        b = cp.append(b, cp.zeros(num_of_append))  # 0を追加
    b_norm = cp.linalg.norm(b)

    # x
    x = cp.zeros(N, T)

    # その他パラメータ
    max_iter = old_N
    residual = cp.zeros(max_iter+16, T)
    num_of_solution_updates = cp.zeros(max_iter+16, np.int)
    num_of_solution_updates[0] = 0

    return A, b, x, b_norm, N, max_iter, residual, num_of_solution_updates


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
    def init_gpu(cls, begin: int, end: int):
        cls.begin = begin
        cls.end = end
        cls.num_of_gpu = end - begin + 1
        cls.streams = [None] * cls.num_of_gpu

        # init memory allocator
        for i in range(cls.begin, cls.end+1):
            Device(i).use()
            pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
            cp.cuda.set_allocator(pool.malloc)
            cls.streams[i-begin] = cp.cuda.Stream()

            # Enable P2P
            for j in range(4):
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

        # divide single A -> multi local_A
        # allocate x, y
        for i in range(cls.begin, cls.end+1):
            Device(i).use()
            index = i-cls.begin
            # npy
            if isinstance(A, np.ndarray):
                cls.A[index] = cp.array(A[i*cls.local_N:(i+1)*cls.local_N], T)
            # npz
            elif isinstance(A, scipy.sparse.csr.csr_matrix):
                from cupyx.scipy.sparse import csr_matrix
                cls.A[index] = csr_matrix(A[i*cls.local_N:(i+1)*cls.local_N])
            cls.x[index] = cp.zeros(cls.N, T)
            cls.y[index] = cp.zeros(cls.local_N, T)

        # init out vector
        cls.out = cp.zeros(cls.N, T)

    # マルチGPUを用いた行列ベクトル積
    @classmethod
    def dot(cls, A, x):
        # Copy vector data to All devices
        for i in range(cls.begin, cls.end+1):
            Device(i).use()
            index = i-cls.begin
            # cp.cuda.runtime.memcpyPeer(cls.x[index].data.ptr, i, x.data.ptr, cls.begin, cls.nbytes)
            cp.cuda.runtime.memcpyPeerAsync(cls.x[index].data.ptr, i, x.data.ptr, cls.end, cls.nbytes, cls.streams[index].ptr)
            # dot
            cls.y[index] = cls.A[index].dot(cls.x[index])
        # Gather caculated element from All devices
        for i in range(cls.begin, cls.end+1):
            index = i-cls.begin
            # cp.cuda.runtime.memcpyPeer(cls.out[index*cls.local_N].data.ptr, cls.begin, cls.y[index].data.ptr, i, cls.y[index].nbytes)
            cp.cuda.runtime.memcpyPeerAsync(cls.out[index*cls.local_N].data.ptr, cls.end, cls.y[index].data.ptr, i, cls.y[index].nbytes, cls.streams[index].ptr)
        # sync
        for i in range(cls.begin, cls.end+1):
            index = i-cls.begin
            cls.streams[index].synchronize()
            # Device(i).synchronize()
        # return
        return cls.out
