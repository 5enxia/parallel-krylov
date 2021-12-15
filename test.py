import argparse
from typing import Optional

import numpy as np
from scipy.linalg import toeplitz

T = np.float64
N = 32
k = 0
elements = np.zeros(N, T)
elements[0] = 2
elements[1] = 1e-3

def normal(method):
	if method == 'cg':
		from v2.cpu.cg import cg
		A, b = toeplitz(elements), np.ones(N, T)
		cg(A, b, 1e-10, T)

	if method == 'mrr':
		from v2.cpu.mrr import mrr
		A, b = toeplitz(elements), np.ones(N, T)
		mrr(A, b, 1e-10, T)

	if method == 'kskipcg':
		from v2.cpu.kskipcg import kskipcg
		A, b = toeplitz(elements), np.ones(N, T)
		kskipcg(A, b, 1e-10, k, T)

	if method == 'kskipmrr':
		from v2.cpu.kskipmrr import kskipmrr
		A, b = toeplitz(elements), np.ones(N, T)
		kskipmrr(A, b, 1e-10, k, T)

	if method == 'adaptivekskipmrr':
		from v2.cpu.adaptivekskipmrr import adaptivekskipmrr
		A, b = toeplitz(elements), np.ones(N, T)
		adaptivekskipmrr(A, b, 1e-10, k, T)

def mpi(method):
	if method == 'cg':
		from v2.cpu.mpi.cg import cg
		A, b = toeplitz(elements), np.ones(N, T)
		cg(A, b, 1e-10, T)

	if method == 'mrr':
		from v2.cpu.mpi.mrr import mrr
		A, b = toeplitz(elements), np.ones(N, T)
		mrr(A, b, 1e-10, T)

	if method == 'kskipcg':
		from v2.cpu.mpi.kskipcg import kskipcg
		A, b = toeplitz(elements), np.ones(N, T)
		kskipcg(A, b, 1e-10, k, T)

	if method == 'kskipmrr':
		from v2.cpu.mpi.kskipmrr import kskipmrr
		A, b = toeplitz(elements), np.ones(N, T)
		kskipmrr(A, b, 1e-10, k, T)

	if method == 'adaptivekskipmrr':
		from v2.cpu.mpi.adaptivekskipmrr import adaptivekskipmrr
		A, b = toeplitz(elements), np.ones(N, T)
		adaptivekskipmrr(A, b, 1e-10, k, T)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='test exce cuter')
	parser.add_argument("-m", "--mpi", action="store_true", help="is mpi")
	parser.add_argument("method", help="")

	# parse argv
	args = parser.parse_args()
	if args.mpi:
		mpi(args.method)
	else:
		normal(args.method)