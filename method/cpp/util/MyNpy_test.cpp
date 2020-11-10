#include <iostream>
#include <cassert>

#include "MyNpy.h"

using namespace std;

using Matrix = MyNpy::Matrix;

int main() {
  cout << "matrix" << endl;
  Matrix mat = MyNpy::loadMatrix("./matrix_test.npy");
  cout << mat.size() << endl;
  assert(mat.size() == 10);

  cout << mat[0].size() << endl;
  assert(mat[0].size() == 10);
  for(int i=0; i < mat.size(); i++) {
    for(int j=0; j < mat[0].size(); j++) {
      cout << mat[i][j] << " ";
    }
    cout << endl;
  }

  cout << "vector" << endl;
  vector<double> vec = MyNpy::loadVector("./vector_test.npy");
  cout << vec.size() << endl;
  assert(vec.size() == 10);
  for(int i=0; i < vec.size(); i++) {
    cout << vec[i] << " ";
  }
  cout << endl;
}
