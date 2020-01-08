#include <iostream>

/*
* Matrix is represented as 1D array. To access value in n-th row and m-th column
* type `array[n*COLUMNS + m];`
*/

struct Matrix {
    float* arr;
    int rows;
    int columns;

    Matrix(int _rows, int _columns):
        rows(_rows), columns(_columns), arr(new float[_rows * _columns]) {}

    int size() {
        return sizeof(float) * rows * columns;
    }

    void set(int row, int col, float value) {
        arr[row * columns + col] = value;
    }
};


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
* This functions returns transpose of the matrix.
* You should call this function like this:
*
*     transpose2D<<< columns, rows >>>(matrix, acc);
*/
__global__ void transpose2D (float* matrix, float* acc, int COLUMNS) {
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
//__global__ void mean2D (float* matrix, float* acc) {
  //int row = threadIdx.x;
  //float sum = 0.0;
  //for (int i = 0; i < COLUMNS; i++) {
    //sum += matrix[row * COLUMNS + i];
  //}
  //*acc = sum / COLUMNS;
//}


/*
* This function return dot product of two matrices.
*
*     matmul2D<<< columns, rows >>>(A, B, Acc);
*/
__global__ void matmul2D (float* A, float* B, float* Acc, int COLUMNS) {
  int row = threadIdx.x;
  int col = blockIdx.x;

  int sum = 0;
  for(int i = 0; i < COLUMNS; i++) {
    sum += A[row * COLUMNS + i] * B[i * COLUMNS + col];
  }

  Acc[row * COLUMNS + col] = sum;
}

void printMatrix(Matrix matrix) {
  for (int i = 0; i < matrix.columns; i++) {
    std::cout << "| ";
    for (int j = 0; j < matrix.rows; j++) {
      std::cout << matrix.arr[i * matrix.columns + j] << " ";
    }
    std::cout << "|\n";
  }
}

Matrix* transpose(Matrix* matrix) {
   Matrix* result = new Matrix(matrix->columns, matrix->rows);
   float* cudaOriginal;
   float* cudaResult;
   cudaMalloc((void**) &cudaOriginal, matrix->size());
   cudaMalloc((void**) &cudaResult, result->size());

   cudaMemcpy(cudaOriginal, matrix->arr, matrix->size(), cudaMemcpyHostToDevice);

   transpose2d <<< matrix->columns, matrix->rows >>> (cudaOriginal, cudaResult);
   cudaMemcpy(result->arr, cudaResult, result->size, cudaMemcpyDeviceToHost);
   return result;
}

/*
 * Return product of two matrices
*/
Matrix* matmul(Matrix* A, Matrix* B) {
    if (A->columns != B->rows) {
        std::cout << "Matrix dim mismatch, got ("
                  << A->rows
                  << ","
                  << A->columns
                  << ") and ("
                  << B->rows
                  << ","
                  << B->columns
                  << ")\n";
        throw std::invalid_argument( "Invalid input" );
    }
    Matrix* result = new Matrix(A->rows, B->columns);
    float* cudaA;
    float* cudaB;
    float* cudaAcc;
    cudaMalloc((void**) &cudaA, A->size());
    cudaMalloc((void**) &cudaB, B->size());
    cudaMalloc((void**) &cudaAcc, result->size());

    cudaMemcpy(cudaA, A->arr, A->size(), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, B->arr, B->size(), cudaMemcpyHostToDevice);


    matmul2D <<< A->columns, A->rows >>> (cudaA, cudaB, cudaAcc, A->columns);

    cudaMemcpy(result->arr, cudaAcc, result->size(), cudaMemcpyDeviceToHost);

    return result;
}

Matrix* loadData() {
    int n, p;
    float temp;
    cin >> n;
    cin >> p;

    Matrix* data = new Matrix(n, p);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p+1; j++) {
            cin >> temp;
            data->set(i, j, temp);
        }
    }
    return data;
}

int main() {

  Matrix* A = new Matrix(2, 2);
  A->arr = new float[2*2]{1, 1, 2, 2};
  Matrix* B = new Matrix(2, 2);
  B->arr = new float[2*2]{3, 3, 4, 4};
  //Matrix* A = loadData();
  //Matrix* B = loadData();


  Matrix* mul = matmul(A, B);

  printMatrix(*mul);

  Matrix* t = transpose(mul);
  printMatrix(*t);
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
