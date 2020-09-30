#include "MyBlas.h"

using namespace std;

template<typename T>
vector<T> mrr(const Matrix &A, const vector<T> &b, const double epsilon);

int main() {
  Matrix A = {
		{5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0}
	};
	vector<double> b = {3.0, 1.0, 4.0, 0.0, 5.0, -1.0, 6.0, -2.0, 7.0, -15.0};
  const double epsilon = 1e-8;

  auto x = mrr(A, b, epsilon);
  for(auto e: x) cout << e << endl;
}

template<typename T>
vector<T> mrr(const Matrix &A, const vector<T> &b, const double epsilon){
  const ul N = A.size();
  const ul max_iter = N*2;
  const double b_norm = vecvec(b, b);
	vector<T> x(N, 0.0);

  // 初期残差
	vector<T> r = b - A*x;
  double residual = vecvec(r, r) / b_norm;

  // 初期反復
  vector<T> Ar = matvec(A, r);
  double zeta = vecvec(r, Ar) / vecvec(Ar, Ar);
  vector<T> y = zeta * Ar;
  vector<T> z = -zeta * r;
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
    vector<T> s = Ar - gamma * y;
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
