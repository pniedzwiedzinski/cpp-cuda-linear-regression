//**************************************
//
//   Author:   Patryk Niedźwiedziński
//
//**************************************

#include "load.h"
#include <iostream>
#include "matrix.h"

using namespace std;

Matrix* loadData(int n, int p) {
    Matrix* data = new Matrix(n, p);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            float a;
            cin >> a;
            data->update(i, j, a);
        }
    }
    return data;
}
