import argparse
from scipy.io import mmread
from scipy.sparse import save_npz, bsr_matrix, csc_matrix, csr_matrix, coo_matrix, dia_matrix, dok_matrix, lil_matrix

parser = argparse.ArgumentParser(description='.mtx to .npz converter')
parser.add_argument("path", help="data file path")
parser.add_argument("-f", "--format", default="csr",
                    help="Matrix Storage Format")

args = parser.parse_args()

methods = {
	"bsr": bsr_matrix,
    "csc": csc_matrix,
    "csr": csr_matrix,
    "coo": coo_matrix,
    "dia": dia_matrix,
    "dok": dok_matrix,
    "lil": lil_matrix,
}

mtx = mmread(args.path)
method = methods[args.format]
npz = method(mtx)
save_npz(args.path.replace('.mtx', '.npz'), npz)
