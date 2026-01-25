#!/bin/bash
#
# Run all tests with optional sanitizers
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              ADAI Test Runner                                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Parse arguments
SANITIZER=""
VERBOSE=false
COVERAGE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --asan)
            SANITIZER="asan"
            shift
            ;;
        --ubsan)
            SANITIZER="ubsan"
            shift
            ;;
        --tsan)
            SANITIZER="tsan"
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--asan|--ubsan|--tsan] [--coverage] [--verbose]"
            exit 1
            ;;
    esac
done

# Reconfigure if sanitizer or coverage requested
if [ -n "$SANITIZER" ] || [ "$COVERAGE" = true ]; then
    echo "🔧 Reconfiguring build..."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    CMAKE_OPTS=""
    if [ "$SANITIZER" = "asan" ]; then
        CMAKE_OPTS="-DENABLE_ASAN=ON"
        echo "   AddressSanitizer enabled"
    elif [ "$SANITIZER" = "ubsan" ]; then
        CMAKE_OPTS="-DENABLE_UBSAN=ON"
        echo "   UndefinedBehaviorSanitizer enabled"
    elif [ "$SANITIZER" = "tsan" ]; then
        CMAKE_OPTS="-DENABLE_TSAN=ON"
        echo "   ThreadSanitizer enabled"
    fi
    
    if [ "$COVERAGE" = true ]; then
        CMAKE_OPTS="$CMAKE_OPTS -DENABLE_COVERAGE=ON"
        echo "   Code coverage enabled"
    fi
    
    cmake .. $CMAKE_OPTS
    echo ""
    
    echo "🔨 Building tests..."
    make -j$(nproc)
    echo ""
fi

# Run tests
cd "$BUILD_DIR"
echo "🧪 Running tests..."
echo ""

if [ "$VERBOSE" = true ]; then
    ctest --output-on-failure --verbose
else
    ctest --output-on-failure
fi

TEST_RESULT=$?
echo ""

# Generate coverage report if requested
if [ "$COVERAGE" = true ]; then
    echo "📊 Generating coverage report..."
    lcov --capture --directory . --output-file coverage.info 2>/dev/null || true
    lcov --remove coverage.info '/usr/*' '*/gtest/*' --output-file coverage.info 2>/dev/null || true
    
    if [ -f coverage.info ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📈 Coverage Summary"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        lcov --list coverage.info
        echo ""
        echo "💡 Generate HTML report: genhtml coverage.info -o coverage_html"
    fi
fi

exit $TEST_RESULT
