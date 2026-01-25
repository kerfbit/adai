# Contributing to ADAI

Thank you for your interest in contributing to the Advanced Deep Learning AI (ADAI) project! This guide will help you get started with development and ensure a smooth contribution process.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Contributions](#submitting-contributions)
- [Code Review Process](#code-review-process)

## Code of Conduct

By participating in this project, you agree to:

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other contributors

## Getting Started

### Prerequisites

Before you begin, ensure you have:

- **C++17 compatible compiler**
  - GCC 7+ or Clang 5+ (Linux/macOS)
  - MSVC 2017+ (Windows)
- **CMake 3.10+**
- **Git**
- **Google Test** (automatically fetched by CMake)

### Optional Tools

- **clang-format** - For code formatting
- **clang-tidy** - For static analysis
- **lcov** - For coverage reports
- **cppcheck** - For additional static analysis

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/adai.git
cd adai

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/adai.git
```

### 2. Build the Project

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build (use -j for parallel builds)
make -j$(nproc)

# Run tests to verify setup
ctest --output-on-failure
```

### 3. Set Up Development Branch

```bash
# Ensure you're on develop branch
git checkout develop
git pull upstream develop

# Create your feature branch
git checkout -b feature/your-feature-name
```

## Coding Standards

### C++ Style Guidelines

We follow the **Google C++ Style Guide** with some modifications:

#### Naming Conventions

```cpp
// Classes: PascalCase
class MultiHeadAttention { };
class BPETokenizer { };

// Functions/Methods: camelCase
void calculateGradient();
Matrix forward(const Matrix& input);

// Variables: snake_case
int num_layers = 6;
float learning_rate = 0.001f;

// Constants: UPPER_SNAKE_CASE
const int MAX_SEQUENCE_LENGTH = 512;
const float DEFAULT_EPSILON = 1e-8f;

// Member variables: snake_case (no prefix)
class Encoder {
    int d_model;
    int num_heads;
    std::vector<EncoderBlock*> blocks;
};

// Template parameters: PascalCase
template<typename DataType>
class Matrix { };
```

#### File Organization

```cpp
// Header file (.hpp)
#pragma once

#include <vector>      // System includes first
#include <string>

#include "Matrix.hpp"  // Project includes second

/**
 * @brief Class description
 * 
 * Detailed description with usage examples
 */
class MyClass {
public:
    // Public interface first
    MyClass();
    void publicMethod();
    
private:
    // Private implementation last
    void privateMethod();
    int member_variable;
};
```

#### Code Formatting

- **Indentation**: 4 spaces (no tabs)
- **Line length**: 100 characters maximum
- **Braces**: K&R style (opening brace on same line)
- **Pointer/Reference**: Left-aligned (`Matrix* ptr`, not `Matrix *ptr`)

```cpp
// Good
if (condition) {
    doSomething();
} else {
    doSomethingElse();
}

// Function declarations
void myFunction(int param1, 
                const std::string& param2,
                Matrix* output);

// Constructor initialization
MyClass::MyClass(int size) 
    : member1(size),
      member2(0),
      member3(nullptr) {
    // Constructor body
}
```

### Automated Formatting

If you have `clang-format` installed:

```bash
# Format a single file
clang-format -i src/MyFile.cpp

# Format all C++ files in src/
find src/ -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
```

### Comments and Documentation

```cpp
/**
 * @brief Brief one-line description
 * 
 * Detailed description that explains what the function does,
 * including any important algorithms or implementation notes.
 * 
 * @param input Input matrix of shape [seq_len, d_model]
 * @param mask Optional attention mask
 * @return Output matrix of shape [seq_len, d_model]
 * 
 * @throws std::invalid_argument if dimensions don't match
 * 
 * Example:
 * @code
 * Matrix input(10, 64);
 * Matrix output = attention.forward(input);
 * @endcode
 */
Matrix forward(const Matrix& input, const Matrix* mask = nullptr);

// Use inline comments for complex logic
for (int i = 0; i < num_heads; ++i) {
    // Split input into attention heads
    // Each head processes d_model / num_heads dimensions
    Matrix head = splitHead(input, i);
}
```

### Error Handling

```cpp
// Use exceptions for error conditions
if (d_model % num_heads != 0) {
    throw std::invalid_argument(
        "d_model must be divisible by num_heads");
}

// Validate inputs early
if (matrix.rows == 0 || matrix.cols == 0) {
    throw std::runtime_error("Matrix dimensions cannot be zero");
}

// Provide meaningful error messages
throw std::runtime_error(
    "Expected input shape [" + std::to_string(expected_rows) + 
    ", " + std::to_string(expected_cols) + "], got [" + 
    std::to_string(input.rows) + ", " + std::to_string(input.cols) + "]");
```

### Modern C++ Practices

```cpp
// Use smart pointers
std::unique_ptr<EncoderBlock> block = 
    std::make_unique<EncoderBlock>(d_model, num_heads, d_ff);

// Use range-based for loops
for (const auto& token : tokens) {
    processToken(token);
}

// Use auto where type is obvious
auto result = matrix1.multiply(matrix2);  // Type is clearly Matrix

// Prefer const correctness
void process(const Matrix& input);  // Won't modify input
const Matrix& getWeights() const;   // Const member function

// Use nullptr instead of NULL
Matrix* ptr = nullptr;

// Use std::vector instead of raw arrays
std::vector<float> data(size);
```

## Testing Guidelines

### Test Structure

```cpp
#include <gtest/gtest.h>
#include "test_base.hpp"  // Use test base classes
#include "../src/MyClass.hpp"

// Use descriptive test suite names
TEST(MyClassTest, ConstructorInitializesCorrectly) {
    MyClass obj(param);
    
    EXPECT_EQ(obj.getParam(), param);
    EXPECT_TRUE(obj.isValid());
}

// Use test fixtures for complex setups
class MyClassTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        obj = std::make_unique<MyClass>(10);
    }
    
    std::unique_ptr<MyClass> obj;
};

TEST_F(MyClassTestFixture, MethodProducesExpectedOutput) {
    Matrix result = obj->process(input);
    EXPECT_EQ(result.rows, expected_rows);
}
```

### Test Coverage

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **Edge Cases**: Test boundary conditions, empty inputs, large values
- **Error Cases**: Test that errors are thrown appropriately

```cpp
// Good test coverage example
TEST(MatrixTest, MultiplyCorrectDimensions) {
    Matrix a(2, 3);
    Matrix b(3, 4);
    Matrix c = a.multiply(b);
    EXPECT_EQ(c.rows, 2);
    EXPECT_EQ(c.cols, 4);
}

TEST(MatrixTest, MultiplyThrowsOnIncompatibleDimensions) {
    Matrix a(2, 3);
    Matrix b(4, 5);  // Incompatible
    EXPECT_THROW(a.multiply(b), std::invalid_argument);
}

TEST(MatrixTest, MultiplyWithZeroMatrix) {
    Matrix a(2, 2);
    Matrix b(2, 2);  // All zeros
    Matrix c = a.multiply(b);
    // Verify all elements are zero
}
```

### Running Tests

```bash
# Run all tests
cd build
ctest --output-on-failure

# Run specific test suite
./tests/matrixTests

# Run with verbose output
ctest -V

# Run tests matching pattern
ctest -R "Matrix.*"
```

### Adding New Tests

1. Create test file in `tests/` directory
2. Add test executable to `tests/CMakeLists.txt`
3. Link necessary libraries
4. Add test to CTest

```cmake
# In tests/CMakeLists.txt
add_executable(myNewTests my_new_test.cpp)
target_link_libraries(myNewTests 
    adai_core 
    ${GTEST_LIBRARIES} 
    pthread
)
add_test(NAME MyNewTests COMMAND myNewTests)
```

## Documentation

### Code Documentation

- Document all public APIs with Doxygen-style comments
- Explain complex algorithms and design decisions
- Provide usage examples for non-trivial functions
- Keep comments up to date with code changes

### Markdown Documentation

When adding features, update relevant documentation:

- **API Reference** (`docs/api/`) - Component interface documentation
- **Guides** (`docs/guides/`) - Usage tutorials and how-tos
- **Architecture** (`docs/architecture/`) - Design decisions

## Submitting Contributions

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts with target branch

### Submission Process

1. **Push your changes**
   ```bash
   git push origin feature/your-feature
   ```

2. **Create Pull Request on GitHub**
   - Base branch: `develop`
   - Compare branch: `feature/your-feature`
   - Fill in PR template completely

3. **PR Checklist**
   - Clear title following commit convention
   - Description explains what and why
   - Links to related issues
   - Screenshots for UI changes (if applicable)
   - Test plan included

### Pull Request Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes Made
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Related Issues
Fixes #123
Related to #456
```

## Code Review Process

### What Reviewers Look For

- **Correctness**: Does the code work as intended?
- **Testing**: Are there adequate tests?
- **Style**: Does it follow coding standards?
- **Performance**: Are there obvious inefficiencies?
- **Documentation**: Is the code well-documented?
- **Security**: Are there potential security issues?

### Responding to Feedback

- Be receptive to constructive criticism
- Ask for clarification if feedback is unclear
- Make requested changes in new commits
- Mark conversations as resolved when addressed
- Thank reviewers for their time

### After Approval

- Squash commits if requested
- Ensure CI passes
- Wait for maintainer to merge
- Delete your feature branch after merge

## Development Tips

### Debugging

```bash
# Build with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

# Use GDB for debugging
gdb ./tests/myTests
(gdb) break MyClass::myMethod
(gdb) run
```

### Memory Checking

```bash
# Build with AddressSanitizer
cmake -DENABLE_ASAN=ON ..
make
./tests/myTests
```

### Code Coverage

```bash
# Build with coverage enabled
cmake -DENABLE_COVERAGE=ON ..
make
ctest
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory coverage_html
```

### Performance Profiling

```bash
# Use perf (Linux)
perf record ./build/myProgram
perf report

# Use valgrind
valgrind --tool=callgrind ./build/myProgram
```

## Getting Help

- **Documentation**: Check `docs/` directory first
- **Issues**: Search existing issues on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Process Plan**: See `PROCESS_IMPROVEMENT_PLAN.md`

## Recognition

Contributors will be:
- Listed in project contributors
- Credited in release notes
- Acknowledged in documentation

Thank you for contributing to ADAI! 🚀
