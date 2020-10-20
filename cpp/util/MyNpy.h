// standard library
#include<iostream>
#include<string>
#include<vector>

// package
#include "cnpy.h" // npy
#include "MyTypes.h" // define type

namespace MyNpy
{
	using namespace std;
	using namespace cnpy;
	using namespace MyTypes; 
	

	unsigned long i,j;

	vector<double> loadVector(string path) {
		NpyArray array = npy_load(path);
		double *loadedData = array.data<double>();

		vector<double> vec(array.shape[0], 0);
		for(i=0; i < array.shape[0]; i++) vec[i] = loadedData[i];
		return vec;
	}

	Matrix loadMatrix(string path) {
		NpyArray array = npy_load(path);
		double *loadedData = array.data<double>();

		Matrix mat(array.shape[0], vector<double>(array.shape[1], 0));
		for(i=0; i < array.shape[0]; i++) {
			for(j=0; j < array.shape[1]; j++) mat[i][j] = loadedData[i * array.shape[0] + j];
		}
		return mat;
	}
}
