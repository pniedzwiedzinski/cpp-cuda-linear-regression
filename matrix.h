class Matrix {
   public:
    Matrix(int r, int c);

    float** array;
    int columns;
    int rows;

    float** operator[](int row);
};

void transpose2D(Matrix* A, Matrix* B);
void mean2D(Matrix* matrix, int cols, float* acc);
