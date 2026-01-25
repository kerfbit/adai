#include "Matrix.hpp"
#include <iomanip>
#include <sstream>

// Default constructor
Matrix::Matrix() : rows(0), cols(0) {}

// Constructor with dimensions
Matrix::Matrix(int r, int c) : rows(r), cols(c) {
    data.resize(rows, std::vector<float>(cols, 0.0f));
}

// Constructor from existing data
Matrix::Matrix(const std::vector<std::vector<float>>& d) : data(d) {
    rows = d.size();
    cols = (rows > 0) ? d[0].size() : 0;

    // Validate that all rows have the same number of columns
    for (const auto& row : data) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
    }
}

// Element access (non-const)
float& Matrix::operator()(int i, int j) {
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[i][j];
}

// Element access (const)
const float& Matrix::operator()(int i, int j) const {
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[i][j];
}

// Matrix multiplication
Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication: [" +
                                    std::to_string(rows) + "x" + std::to_string(cols) + "] * [" +
                                    std::to_string(other.rows) + "x" + std::to_string(other.cols) +
                                    "]");
    }

    Matrix result(rows, other.cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < cols; k++) {
                sum += data[i][k] * other.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }

    return result;
}

// Matrix addition
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix result(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }

    return result;
}

// Matrix subtraction
Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix result(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }

    return result;
}

// Matrix transpose
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j][i] = data[i][j];
        }
    }

    return result;
}

// Randomize matrix with Xavier/He initialization
void Matrix::randomize(float scale) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = dist(gen);
        }
    }
}

// Scalar multiplication
Matrix Matrix::scale(float scalar) const {
    Matrix result(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] * scalar;
        }
    }

    return result;
}

// Hadamard product (element-wise multiplication)
Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for Hadamard product");
    }

    Matrix result(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] * other.data[i][j];
        }
    }

    return result;
}

// Apply gradients (gradient descent update)
void Matrix::apply_gradients(const Matrix& gradients, float learning_rate) {
    if (rows != gradients.rows || cols != gradients.cols) {
        throw std::invalid_argument("Gradient matrix dimensions must match");
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] -= learning_rate * gradients.data[i][j];
        }
    }
}

// Fill matrix with constant value
void Matrix::fill(float value) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = value;
        }
    }
}

// Sum of all elements
float Matrix::sum() const {
    float total = 0.0f;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            total += data[i][j];
        }
    }
    return total;
}

// Mean of all elements
float Matrix::mean() const {
    if (rows == 0 || cols == 0)
        return 0.0f;
    return sum() / (rows * cols);
}

// Print matrix
void Matrix::print(const std::string& name, int max_rows, int max_cols) const {
    if (!name.empty()) {
        std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    }

    int print_rows = std::min(rows, max_rows);
    int print_cols = std::min(cols, max_cols);

    for (int i = 0; i < print_rows; i++) {
        std::cout << "[";
        for (int j = 0; j < print_cols; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << data[i][j];
            if (j < print_cols - 1)
                std::cout << " ";
        }
        if (cols > max_cols) {
            std::cout << " ...";
        }
        std::cout << "]" << std::endl;
    }
    if (rows > max_rows) {
        std::cout << "..." << std::endl;
    }
    std::cout << std::endl;
}

// Check if matrix is valid
bool Matrix::is_valid() const {
    if (rows <= 0 || cols <= 0)
        return false;
    if (data.size() != rows)
        return false;
    for (const auto& row : data) {
        if (row.size() != cols)
            return false;
    }
    return true;
}

// Reshape matrix
Matrix Matrix::reshape(int new_rows, int new_cols) const {
    if (rows * cols != new_rows * new_cols) {
        throw std::invalid_argument("Total number of elements must remain the same when reshaping");
    }

    Matrix result(new_rows, new_cols);
    int idx = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int new_i = idx / new_cols;
            int new_j = idx % new_cols;
            result.data[new_i][new_j] = data[i][j];
            idx++;
        }
    }

    return result;
}

// Get row
std::vector<float> Matrix::get_row(int row_idx) const {
    if (row_idx < 0 || row_idx >= rows) {
        throw std::out_of_range("Row index out of bounds");
    }
    return data[row_idx];
}

// Get column
std::vector<float> Matrix::get_col(int col_idx) const {
    if (col_idx < 0 || col_idx >= cols) {
        throw std::out_of_range("Column index out of bounds");
    }

    std::vector<float> column(rows);
    for (int i = 0; i < rows; i++) {
        column[i] = data[i][col_idx];
    }
    return column;
}

// Set row
void Matrix::set_row(int row_idx, const std::vector<float>& values) {
    if (row_idx < 0 || row_idx >= rows) {
        throw std::out_of_range("Row index out of bounds");
    }
    if (values.size() != cols) {
        throw std::invalid_argument("Row values size must match number of columns");
    }
    data[row_idx] = values;
}

// Set column
void Matrix::set_col(int col_idx, const std::vector<float>& values) {
    if (col_idx < 0 || col_idx >= cols) {
        throw std::out_of_range("Column index out of bounds");
    }
    if (values.size() != rows) {
        throw std::invalid_argument("Column values size must match number of rows");
    }

    for (int i = 0; i < rows; i++) {
        data[i][col_idx] = values[i];
    }
}
