#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>
#include <cblas.h>

using namespace std;

using Matrix = vector<vector<double>>;
using vv = vector<vector<double>>;
using ui = unsigned int;
using ul = unsigned long;
using ll = long long;

double vecvec(const vector<double> &a, const vector<double> &b){
	if( a.size() != b.size() ) return {}; 
	int N = a.size();
	double c;

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			1, 1, N,	// Aの行数、Bの列数、Aの列数(Bの行数)
			1.0,			// alpha
			&a[0], N,	// A
			&b[0], 1,	// B
			0.0,			// beta
			&c, 1			// C
		);

	return c;
}

vector<double> matvec(const Matrix &A, const vector<double> &b){
	if( A.size() != b.size() ) return {};
	ul N = b.size();
	for(auto a: A) if( N != a.size() ) return {};
	vector<double> _A;
	for(auto a: A) std::copy(a.begin(), a.end(), std::back_inserter(_A));
	vector<double> C(N);

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			N, 1, N,		// Aの行数、Bの列数、Aの列数(Bの行数)
			1.0,				// alpha
			&_A[0], N,	// A
			&b[0], 1,		// B
			0.0,				// beta
			&C[0], 1		// C
		);

	return C;
}

// vec+vec
template<typename T>
vector<T> operator+(const vector<T>& t1, const vector<T>& t2){
	vector<T> ret( max(t1.size(),t2.size()) );
	for(ul i=0; i<t1.size(); i++) ret[i] += t1[i];
	for(ul i=0; i<t2.size(); i++) ret[i] += t2[i];

	return ret;
}


// vec-vec
template<typename T>
vector<T> operator-(const vector<T>& t1, const vector<T>& t2){
	vector<T> ret( max(t1.size(),t2.size()) );
	for(ul i=0; i<t1.size(); i++) ret[i] += t1[i];
	for(ul i=0; i<t2.size(); i++) ret[i] -= t2[i];

	return ret;
}

// double*vec
template<typename T>
vector<T> operator*(const double& b, const vector<T>& t2){
	vector<T> ret = t2;
	for(auto &&t: ret) t *= b;

	return ret;
}

// vec*vec
template<typename T>
T operator*(const vector<T>& t1, const vector<T>& t2){
	return vecvec(t1, t2);
}

// mat*vec
template<typename T>
vector<T> operator*(const Matrix& A, const vector<T>& b){
	return matvec(A, b);
}
