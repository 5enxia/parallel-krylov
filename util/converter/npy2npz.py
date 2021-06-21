import argparse
import numpy as np
import os

from scipy.sparse import csr_matrix, save_npz, load_npz

# help
parser = argparse.ArgumentParser(description='.npy to .npz(csr) converter')
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
save_npz(f'{path}/{filename}.npz', csr)

# test
# a = load_npz(f'{path}/{filename}.npz')
# print(a)