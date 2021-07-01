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


vector<double> mrr(const Matrix &A, const vector<double> &b, const double epsilon);

int main() {
  string matrixFilePath = "../data/matrix.npy";
  string vectorFilePath = "../data/vector.npy";

  Matrix A = loadMatrix(matrixFilePath);
  vector<double> b = loadVector(vectorFilePath);

  const double epsilon = 1e-8;

  // timer
  std::chrono::system_clock::time_point  start, end;
  start = std::chrono::system_clock::now();
  auto x = mrr(A, b, epsilon);
  end = std::chrono::system_clock::now();
  double elapsed = chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  cout << elapsed << "ms" << endl;
  return 0;
}

vector<double> mrr(const Matrix &A, const vector<double> &b, const double epsilon){
  const ul N = A.size();
  const ul max_iter = N*2;
  const double b_norm = vecvec(b, b);
	vector<double> x(N, 0.0);

  // 初期残差
	vector<double> r = b - A*x;
  double residual = vecvec(r, r) / b_norm;

  // 初期反復
  vector<double> Ar = matvec(A, r);
  double zeta = vecvec(r, Ar) / vecvec(Ar, Ar);
  vector<double> y = zeta * Ar;
  vector<double> z = -zeta * r;
  r = r - y;
  x = x - z;
  ul i = 1;

  // 反復計算
  while (i < max_iter){
    // 収束判定
    residual = vecvec(r, r) / b_norm;
    if (residual < epsilon) {
      break;
    }

    // 解の更新
    Ar = matvec(A, r);
    double mu = vecvec(y, y);
    double nu = vecvec(y, Ar);
    double gamma = nu / mu;
    vector<double> s = Ar - gamma * y;
    double rs = vecvec(r, s);
    double ss = vecvec(s, s);
    zeta = rs / ss;
    double eta = -zeta * gamma;
    y = eta * y + zeta * Ar;
    z = eta * z - zeta * r;
    r = r - y;
    x = x - z;
    i++;
  }

  return x;
}
