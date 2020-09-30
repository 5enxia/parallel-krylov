#include "MyBlas.h"

using namespace std;

template<typename T>
vector<T> kskipcg(const Matrix &A, const vector<T> &b, const double epsilon, const unsigned int k);

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
  const unsigned int k = 0; 

  auto x = kskipcg(A, b, epsilon, k);
  for(auto e: x) cout << e << endl;
}

template<typename T>
vector<T> kskipcg(const Matrix &A, const vector<T> &b, const double epsilon, const unsigned int k){
  const ul N = A.size();
  const ul max_iter = N*2;
  const double b_norm = vecvec(b, b);
	vector<T> x(N, 0.0);

  // 初期化
  vv Ar(k + 2, vector<double>(N, 0));
  vv Ap(k + 3, vector<double>(N, 0));
  vector<double> a(2 * k + 2, 0);
  vector<double> f(2 * k + 4, 0);
  vector<double> c(2 * k + 2, 0);

  // 初期残差
  Ar[0] = b - matvec(A, x);
  Ap[0] = Ar[0];

  // 反復計算
  ul i = 0;
  ul index = 0;
  while (i < max_iter) {
    // 収束判定
    double residual = vecvec(Ar[0], Ar[0]) / b_norm;
    if (residual < epsilon) break;

    // 事前計算
    ul jj;
    for (ui j=1;j<k+1;j++) Ar[j]=matvec(A, Ar[j-1]);
    for (ui j=1;j<k+2;j++) Ap[j]=matvec(A, Ap[j-1]);
    for (ui j=0;j<2*k+1;j++) { 
      jj = j;
      a[j] = vecvec(Ar[jj], Ar[jj + j % 2]);
    }
    for (ui j=0;j<2*k+4;j++) { 
      jj = j;
      f[j] = vecvec(Ap[jj], Ap[jj + j % 2]);
    }
    for (ui j=0;j<2*k+2;j++) { 
      jj = j;
      c[j] = vecvec(Ar[jj], Ap[jj + j % 2]);
    }

    // CGでの1反復
    double alpha = a[0] / f[1];
    double beta = pow(alpha,2) * f[2] / a[0] - 1;
    x = x + alpha * Ap[0];
    Ar[0] = Ar[0] - alpha * Ap[1];
    Ap[0] = Ar[0] + beta * Ap[0];
    Ap[1] = matvec(A, Ap[0]);

    // CGでのk反復
    for (ui j=0;j<k;j++) {
      for (ui l=0;l<2*(k-j)+1;l++) {
        a[l] += alpha*(alpha*f[l+2] - 2*c[l+1]);
        double d = c[l] - alpha*f[l+1];
        c[l] = a[l] + d*beta;
        f[l] = c[l] + beta*(d + beta*f[l]);
      }
      // 解の更新
      alpha = a[0] / f[1];
      beta = pow(alpha,2) * f[2] / a[0] - 1;
      x = x + alpha * Ap[0];
      Ar[0] = Ar[0] - alpha * Ap[1];
      Ap[0] = Ar[0] + beta * Ap[0];
      Ap[1] = matvec(A, Ap[0]);
    }

    i += (k + 1);
    index += 1;
  }

	return x;
}
