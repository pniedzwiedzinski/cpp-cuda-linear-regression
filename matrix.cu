//**************************************
//
//   Author:   Patryk Niedźwiedziński
//
//**************************************

#include "matrix.h"
#include "load.h"

using namespace std;

Matrix::Matrix(int r, int c) {
  array = new float[r * c];
  columns = c;
}

float& Matrix::index(int row, int column) {
  return array[columns * row + column];
}
void Matrix::update(int row, int column, float value) {
  array[columns * row + column] = value;
}

__global__ void transpose2D (Matrix* A, Matrix* B) {
  int x = threadIdx.x;
  int y = blockIdx.x;

  if (y!=x) {
    B->update(y, x, A->index(x, y));
  }
}

__global__ void mean2D (Matrix* matrix, int cols, float* acc) {
  int row = threadIdx.x;
  float sum = 0.0;
  for (int i = 0; i < cols; i++) {
    sum += matrix->index(i, row);
  }
  *acc = sum/cols;
}
