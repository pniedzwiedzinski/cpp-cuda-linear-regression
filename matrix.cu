//**************************************
//
//   Author:   Patryk Niedźwiedziński
//
//**************************************

#include "matrix.h"
#include "load.h"

using namespace std;

Matrix::Matrix(int r, int c) {
  array = new float*[r * c];
  columns = c;
  rows = r;
}

float** Matrix::operator[](int row) {
  return slice(array[row], columns, 0);
}

__global__ void transpose2D (Matrix* A, Matrix* B) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (y!=x) {
    B[y][x] = A[x][y];
  }
}

__global__ void mean2D (Matrix* matrix, int cols, float* acc) {
  int row = threadIdx.x;
  float sum = 0.0;
  for (int i = 0; i < cols; i++) {
    sum += matrix[i][row];
  }
  *acc = sum/cols;
}
