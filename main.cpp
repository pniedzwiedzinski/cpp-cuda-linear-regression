//**************************************
//
//   Author:   Patryk Niedźwiedziński
//
//**************************************

#include <iostream>
#include "load.h"

using namespace std;

int main() {
    int n, p;
    cin >> n;
    cin >> p;
    Matrix* data = loadData(n, p);
    cout << data->index(0, 0);

    return 0;
}
