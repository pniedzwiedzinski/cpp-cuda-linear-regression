//**************************************
//
//   Author:   Patryk Niedźwiedziński
//
//**************************************

#include "load.h"
#include <iostream>

using namespace std;

Array2D* loadData(int n, int p) {
    Array2D* data = new Array2D(n, p);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            float a;
            cin >> a;
            data->update(i, j, a);
        }
    }
    return data;
}
