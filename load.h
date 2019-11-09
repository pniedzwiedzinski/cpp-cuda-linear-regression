struct Array2D {
    Array2D(int r, int c) {
        array = new float[r * c];
        columns = c;
    }

    float* array;
    int columns;

    float& index(int row, int column) { return array[columns * row + column]; }
    void update(int row, int column, float value) {
        array[columns * row + column] = value;
    }
};

Array2D* loadData(int n, int p);