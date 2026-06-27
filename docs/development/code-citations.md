# Code Citations

## License: MIT
https://github.com/soomrack/MR2023/blob/7dcb3c16091da4cf21bfab6a39e5ad74f277fb28/Boryshnev%20Arseniy/Matrixes%20cpp/matrix.cpp


**40/40 tests PASSED** with ASAN full quarantine enabled. The crash is gone.

**Root cause**: The compiler-generated `Matrix` move operations moved `data` (leaving it empty) but left `rows` and `cols` holding the old dimensions. Any subsequent code that used a moved-from Matrix would pass `operator()(i, j)`'s bounds check (`i < rows` still true) but then access `data[i]` on an empty vector — writing to a heap region before the ASAN left red zone (the chunk header), which ASAN couldn't intercept. This silently corrupted ASAN's chunk-state metadata for some allocation, which was only observed ~30 tests later when the quarantine recycler tried to recycle that chunk.

**Fix applied** in [src/Matrix.hpp](src/Matrix.hpp) and [src/Matrix.cpp](src/Matrix.cpp): explicit move constructor and move assignment that zero out `rows` and `cols` on the moved-from object:

```cpp
Matrix::Matrix(Matrix&& other) noexcept
    : data(std::move(other.data)), rows(other.rows), cols(other.cols) {
    other.rows = 0;
    other.cols = 0;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        data = std::move(other.data);
        rows = other.rows; cols = other.cols;
        other.rows = 0;   other.cols = 0;
    }
    return *this;
}
```
