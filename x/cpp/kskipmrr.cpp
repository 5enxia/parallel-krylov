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

vector<double> kskipmrr(const Matrix &A, const vector<double> &b, const double epsilon, const int k);

int main() {
  string matrixFilePath = "../data/matrix.npy";
  string vectorFilePath = "../data/vector.npy";

  Matrix A = loadMatrix(matrixFilePath);
  vector<double> b = loadVector(vectorFilePath);
  const double epsilon = 1e-8;
  const unsigned int k = 0; 

  // timer
  std::chrono::system_clock::time_point  start, end;
  start = std::chrono::system_clock::now();
  auto x = kskipmrr(A, b, epsilon, k);
  end = std::chrono::system_clock::now();
  double elapsed = chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  cout << elapsed << "ms"<< endl;
  printVec(x);
  return 0;
}

vector<double> kskipmrr(const Matrix &A, const vector<double> &b, const double epsilon, const int k){
  const ul N = A.size();
  const ul max_iter = N*2;
  const double b_norm = vecvec(b, b);
	vector<double> x(N, 0.0);

  // 初期化
  vv Ar(k + 2, vector<double>(N));
  vv Ay(k + 1, vector<double>(N));
  vector<double> alpha(2*k+3);
  vector<double> beta(2*k+2);
  vector<double> delta(2*k+1);
  beta[0] = 0;

  // 初期残差
  Ar[0] = b - matvec(A, x);
  double residual = vecvec(Ar[0], Ar[0]) / b_norm;

  // 初期反復
  Ar[1] = matvec(A, Ar[0]);
  double zeta = vecvec(Ar[0], Ar[1]) / vecvec(Ar[1], Ar[1]);
  Ay[0] = zeta * Ar[1];
  vector<double> z = -zeta * Ar[0];
  Ar[0] = Ar[0] - Ay[0];
  x = x - z;
  ui i = 1;
  ui index = 1;

  // 反復計算
  while (i < max_iter) {
    // 収束判定
    residual = vecvec(Ar[0], Ar[0]) / b_norm;
    if (residual < epsilon) break;

    // 事前計算
    int jj;
    for (int j=1;j<k+2;j++) Ar[j]=matvec(A, Ar[j-1]);
    for (int j=1;j<k+1;j++) Ay[j]=matvec(A, Ay[j-1]);
    for (int j=0;j<2*k+3;j++) { 
      jj = j;
      alpha[j] = vecvec(Ar[jj], Ar[jj + j % 2]);
    }
    for (int j=1;j<2*k+2;j++) { 
      jj = j;
      beta[j] = vecvec(Ay[jj], Ar[jj + j % 2]);
    }
    for (int j=0;j<2*k+1;j++) { 
      jj = j;
      delta[j] = vecvec(Ay[jj], Ay[jj + j % 2]);
    }

    // MrRでの1反復(解の更新)
    double d = alpha[2] * delta[0] - pow(beta[1],2);
    zeta = alpha[1] * delta[0] / d;
    double eta = -alpha[1] * beta[1] / d;
    Ay[0] = eta * Ay[0] + zeta * Ar[1];
    z = eta * z - zeta * Ar[0];
    Ar[0] = Ar[0] - Ay[0];
    Ar[1] = matvec(A, Ar[0]);
    x = x - z;

    // MrRでのk反復
    for (int j=0;j<k;j++) {
      delta[0] = pow(zeta,2) * alpha[2] + eta * zeta * beta[1];
      alpha[0] = alpha[0] - zeta * alpha[1];
      delta[1] = pow(eta,2) * delta[1] + 2 * eta * zeta * beta[2] + pow(zeta,2) * alpha[3];
      beta[1] = eta * beta[1] + zeta * alpha[2] - delta[1];
      alpha[1] = -beta[1];
      for (int l=2;l<2*(k-j)+1;l++) {
        delta[l] = pow(eta,2) * delta[l] + 2 * eta * zeta * beta[l+1] + pow(zeta,2) * alpha[l + 2];
        double tau = eta * beta[l] + zeta * alpha[l + 1];
        beta[l] = tau - delta[l];
        alpha[l] = alpha[l] - tau + beta[l];
      }
      // 解の更新
      d = alpha[2] * delta[0] - pow(beta[1],2);
      zeta = alpha[1] * delta[0] / d;
      eta = -alpha[1] * beta[1] / d;
      Ay[0] = eta * Ay[0] + zeta * Ar[1];
      z = eta * z - zeta * Ar[0];
      Ar[0] = Ar[0] - Ay[0];
      Ar[1] = matvec(A, Ar[0]);
      x = x - z;
    }

    i += (k + 1);
    index++;
  }
  
  return x;
}