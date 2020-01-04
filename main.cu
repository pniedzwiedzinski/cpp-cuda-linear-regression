#include <iostream>


struct Matrix {
    float* arr;
    int rows;
    int columns;
}


/*
* Some basic functions to play out with CUDA
*/
__global__ void cuda_hello_world() {
    printf("Hello World from GPU!\n");
}

__global__ void assign (int* a, int* b) {
    *b = *a;
}

__global__ void add(int *a, int *b, int *c)
{
    *c = *a + *b;
}

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
  *acc = sum / COLUMNS;
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

void printMatrix(Matrix matrix) {
  for (i = 0; i < matrix.cols; i++) {
    std::cout << "| ";
    for (j = 0; j < matrix.rows; j++) {
      std::cout << matrix[i * COLUMNS + j] << " ";
    }
    std::cout << "|\n";
  }
}

int main() {
  //
  //
  // SOME MATRIX
  //float* acc;
  //int size = (ROWS*COLUMNS)*sizeof(float);
  //float* zeroMatrix = (float*)malloc(size);
  //printMatrix(zeroMatrix);
//
  //float* cudaMatrix;
  //float* cudaAcc;
  //cudaMalloc((void**) &cudaMatrix, size);
  //cudaMalloc((void**) &cudaAcc, sizeof(float));
  //cudaMemcpy(cudaMatrix, zeroMatrix, size, cudaMemcpyHostToDevice);
  //std::cout << "test\n";
  //mean2D <<< 1, ROWS >>> (cudaMatrix, cudaAcc);
  //cudaMemcpy(acc, cudaAcc, sizeof(float), cudaMemcpyDeviceToHost);
  //std::cout << *acc;
  //
  //
  // ASSIGN
  //int a = 5, b;
  //int *cudaA, *cudaB;
  //int size = sizeof(int);
  //cudaMalloc((void**) &cudaA, size);
  //cudaMalloc((void**) &cudaB, size);
  //cudaMemcpy(cudaA, &a, size, cudaMemcpyHostToDevice);
  //assign <<< 1, 1 >>> (cudaA, cudaB);
  //cudaMemcpy(&b, cudaB, size, cudaMemcpyDeviceToHost);
  //std::cout << b;
  //
  //
  // HELLO WORLD
  //cuda_hello_world<<<1,1>>>();
  //
  //
  // ADD
  //int a = 2, b = 3, c;
  //int *cudaA, *cudaB, *cudaC;
  //int size = sizeof(int);
  //cudaMalloc((void **)&cudaA, size);
  //cudaMalloc((void **)&cudaB, size);
  //cudaMalloc((void **)&cudaC, size);
//
  //cudaMemcpy(cudaA, &a, size, cudaMemcpyHostToDevice);
  //cudaMemcpy(cudaB, &b, size, cudaMemcpyHostToDevice);
  //add<<<1,1>>>(cudaA, cudaB, cudaC);
//
  //cudaMemcpy(&c, cudaC, size, cudaMemcpyDeviceToHost);
//
  //std::cout << c;
  return 0;
}
