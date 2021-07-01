import os
import argparse
from scipy.io import mmread
from scipy.sparse import save_npz, csr_matrix

parser = argparse.ArgumentParser(description='.mtx to .npz converter')
parser.add_argument("path", help="data file path")

args = parser.parse_args()
path, basename = os.path.split(args.path)
filename, ext = os.path.splitext(basename)

mtx = mmread(args.path)
mtx = csr_matrix(mtx)
save_npz(f'{path}/{filename}.npz', mtx)
