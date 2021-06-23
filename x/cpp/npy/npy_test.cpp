#include<iostream>

#include "../util/MyNpy.h"

using namespace std;

int main()
{
  vector<double> *vec = MyNpy::loadVector("../../data/vector.npy");
  for(int i =0; vec->size(); i++;) {
    cout << *vec[i] << endl;
  }
}
