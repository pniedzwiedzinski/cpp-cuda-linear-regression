//**************************************
//
//   Author:   Patryk Niedźwiedziński
//
//**************************************

#include <iostream>
// #include <"matrix.h">
#include "load.h"

#define FILENAME "data.csv"

using namespace std;

int main() {
    int n, p;
    cin >> n;
    cin >> p;
    Array2D* data = loadData(n, p);
    cout << data->index(0, 0);

    return 0;
}
