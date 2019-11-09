//**************************************
//
//   Author:   Patryk Niedźwiedziński
//
//**************************************

using namespace std;

__global__ void transpose2D (float** A, float** B) {
  int x = threadIdx.x;
  int y = blockIdx.x;

  if (y!=x) {
    B[y][x] = A[x][y];
  }
}
