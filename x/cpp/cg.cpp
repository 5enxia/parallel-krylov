#include <string>

#include "util/MyTypes.h"
#include "util/MyBlas.h"
#include "util/MyNpy.h"
#include "util/MyTimer.h"

#include "cnpy.h" // npy

using namespace std;
using namespace MyTypes;
using namespace MyNpy;
using namespace MyBlas;
using namespace cnpy;

double cg(const Matrix &A, const vector<double> &b, const double epsilon);
// vector<double> cg(const Matrix &A, const vector<double> &b, const double epsilon);

int main() {
  string matrixFilePath = "/Users/takayasu/krylov/data/Meshless-Matrix-Reduced/matrix_EFG-2801.npy";
  string vectorFilePath = "/Users/takayasu/krylov/data/Meshless-Matrix-Reduced/vector_EFG-2801.npy";

  Matrix A = loadMatrix(matrixFilePath);
  vector<double> b = loadVector(vectorFilePath);

  const double epsilon = 1e-8;

  // timer
  std::chrono::system_clock::time_point  start, end;
  start = std::chrono::system_clock::now();
  auto r = cg(A, b, epsilon);
  end = std::chrono::system_clock::now();
  double elapsed = chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  cout << elapsed << "ms" << endl;
  cout << "final residual:" << r << endl;
  return 0;
}

double cg(const Matrix &A, const vector<double> &b, const double epsilon){
// vector<double> cg(const Matrix &A, const vector<double> &b, const double epsilon){
  const ui N = A.size();
  const ul max_iter = N * 2;
  const double b_norm = MyBlas::vecvec(b, b);
	vector<double> x(N, 0.0);
  double residual = 1;

  // 初期残差
	vector<double> r = b - A * x;
	vector<double> p = r;
  double gamma = r*r;

  // 反復計算
  ul i = 0;

  while (i < max_iter)
  {
    // 収束判定
    // double residual = vecvec(r, r) / b_norm;
    residual = vecvec(r, r) / b_norm;
    if (residual < epsilon) {
      break;
    }
    // 解の更新
    vector<double> v = matvec(A, p);
    double sigma = vecvec(p, v);
    double alpha = gamma / sigma;
    x = x + alpha * p;
    r = r - alpha * v;
    double old_gamma = gamma;
    gamma = vecvec(r, r);
    double beta = gamma / old_gamma;
    p = r + beta * p;
    i++;
  }

  return residual;
	// return x;
}
