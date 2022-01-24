import argparse
from numpy import load as load_npy
import os

from scipy.io import mmwrite
from scipy.sparse import coo_matrix

parser = argparse.ArgumentParser(description='.npy to .npz converter')
parser.add_argument("path", help="data file path")

args = parser.parse_args()

npy = load_npy(args.path)
coo = coo_matrix(npy)

mmwrite(args.path. csr)
