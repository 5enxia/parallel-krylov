import sys
import numpy as np
import six
import fastrlock
import cupy as cp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(f'Device ID: {rank}')
