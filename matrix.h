class Matrix {
   public:
    Matrix(int r, int c);

    float* array;
    int columns;

    float& index(int row, int column);
    void update(int row, int column, float value);
};

void transpose2D(Matrix* A, Matrix* B);
void mean2D(Matrix* matrix, int cols, float* acc);
