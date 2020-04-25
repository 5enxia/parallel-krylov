import sys
import unittest

sys.path.append('../../')
from krylov.util import loader,toepliz_matrix_generator
from krylov.single.k_skip_cg import k_skip_cg 
import numpy as np 

class TestCgMethod(unittest.TestCase):
    epsilon = 1e-8
    T = np.float64

    def test_single_k_skip_cg_method(self):
        k = 5
        A, b = toepliz_matrix_generator.generate()
        self.assertTrue(k_skip_cg(A, b, k, TestCgMethod.epsilon, TestCgMethod.T))

if __name__ == "__main__":
    unittest.main()