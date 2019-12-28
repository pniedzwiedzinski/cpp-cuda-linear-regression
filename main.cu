#include <iostream>

#define COLUMNS 2
#define ROWS 100

#define i_in_ROWS (int i = 0; i < ROWS; i++)
#define j_in_COLUMNS (int j = 0; j < COLUMNS; j++)

/*
* Matrix is represented as 1D array. To access value in n-th row and m-th column
* type `array[n*COLUMNS + m];`
*/


/*
* This functions returns transpose of the matrix.
* You should call this function like this:
*
*     transpose2D<<< columns, rows >>>(matrix, acc);
*/
__global__ void transpose2D (float* matrix, float* acc) {
  int row = threadIdx.x;
  int col = blockIdx.x;

  if (row!=col) {
    acc[col * COLUMNS + row] = matrix[row * COLUMNS + col];
  }
}


/*
* This function returns average of every value in matrix
*
*     mean2D<<< rows >>>(matrix, acc);
*/
__global__ void mean2D (float* matrix, float* acc) {
  int row = threadIdx.x;
  float sum = 0.0;
  for (int i = 0; i < COLUMNS; i++) {
    sum += matrix[row * COLUMNS + i];
  }
  *acc = sum/COLUMNS;
  printf("%v", *acc);
}


/*
* This function return dot product of two matrices.
*
*     matmul2D<<< columns, rows >>>(A, B, Acc);
*/
__global__ void matmul2D (float* A, float* B, float* Acc) {
  int row = threadIdx.x;
  int col = blockIdx.x;

  int sum = 0;
  for(int i = 0; i < COLUMNS; i++) {
    sum += A[row * COLUMNS + i] * B[i * COLUMNS + col];
  }

  Acc[row * COLUMNS + col] = sum;
}

float* createMatrix() {
  float* m = new float[ROWS * COLUMNS];
  for i_in_ROWS {
    for j_in_COLUMNS {
      m[i * COLUMNS + j] = 0;
    }
  }
  return m;
}

void printMatrix(float* matrix) {
  for i_in_ROWS {
    std::cout << "| ";
    for j_in_COLUMNS {
      cout << matrix[i * COLUMNS + j] << " ";
    }
    std::cout << "|\n";
  }
}

int main() {
  float* acc;
  float* zeroMatrix = createMatrix();
  printMatrix(zreoMatrix);
  int size = (ROWS*COLUMNS)*sizeof(float);

  float* cudaMatrix, cudaAcc;
  cudaMalloc((void**) &cudaMatrix, size);
  cudaMalloc((void**) &cudaAcc, sizeof(float));
  cudaMemcpy(cudaMatrix, zeroMatrix, size, cudaMemcpyHostToDevice);

  mean2D <<< ROWS >>> (cudaMatrix, cudaAcc);
  return 0;
}
