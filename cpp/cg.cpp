#include "MyBlas.h"

using namespace std;

template<typename T>
vector<T> cg(const Matrix &A, const vector<T> &b, const double epsilon);

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

  std::chrono::system_clock::time_point  start, end;
  start = std::chrono::system_clock::now(); // 開始
  auto x = cg(A, b, epsilon);
  end = std::chrono::system_clock::now();  // 終了
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  for(auto e: x) cout << e << endl;
  cout << elapsed << endl;
}

template<typename T>
vector<T> cg(const Matrix &A, const vector<T> &b, const double epsilon){
  const ul N = A.size();
  const ul max_iter = N * 2;
  const double b_norm = vecvec(b, b);
	vector<T> x(N, 0.0);

  // 初期残差
	vector<T> r = b - A*x;
	vector<T> p = r;
  double gamma = r*r;

  // 反復計算
  ul i = 0;

  

  while (i < max_iter)
  {
    // 収束判定
    double residual = vecvec(r, r) / b_norm;
    if (residual < epsilon) {
      break;
    }
    // 解の更新
    vector<T> v = matvec(A, p);
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

	return x;
}
