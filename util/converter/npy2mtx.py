import argparse
import numpy as np
import os

from scipy.sparse import csr_matrix
from scipy.io import mmwrite, mmread 

# help
parser = argparse.ArgumentParser(description='.npy to .mtx(csr) converter')
parser.add_argument("path", help="data file path")

# parse argv
args = parser.parse_args()
path, basename = os.path.split(args.path)
filename, ext = os.path.splitext(basename)

# load npy
npy = np.load(args.path)

# compress
csr = csr_matrix(npy)

# save
mmwrite(f'{path}/{filename}.mtx', csr)

# test
# a = mmread(f'{path}/{filename}.mtx')
# print(a)