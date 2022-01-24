import argparse
import numpy as np
from scipy.sparse import save_npz, bsr_matrix, csc_matrix, csr_matrix, coo_matrix, dia_matrix, dok_matrix, lil_matrix

methods = {
    "bsr": bsr_matrix,
    "csc": csc_matrix,
    "csr": csr_matrix,
    "coo": coo_matrix,
    "dia": dia_matrix,
    "dok": dok_matrix,
    "lil": lil_matrix,
}

parser = argparse.ArgumentParser(description='.npy to .npz(csr) converter')
parser.add_argument("path", help="data file path")
parser.add_argument("-f", "--format", default="coo",
                    help="Matrix Storage Format")

args = parser.parse_args()

npy = np.load(args.path)
method = methods[args.format]
npz = method(npy)

save_npz(args.path.replace('.npy', '.npz'), npz)