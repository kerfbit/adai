#!/bin/bash
# ============================================================================
# ADAI Build and Vocabulary Creation Tool
# ============================================================================
# This script helps you:
# 1. Build the ADAI project with optimal settings
# 2. Create vocabulary from training data
# 3. Train a chatbot model
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="build"
RELEASE_BUILD=true
NUM_CORES=$(nproc 2>/dev/null || echo 4)

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# ============================================================================
# Build Functions
# ============================================================================

build_project() {
    print_header "Building ADAI Project"
    
    # Create build directory
    if [ ! -d "$BUILD_DIR" ]; then
        print_info "Creating build directory: $BUILD_DIR"
        mkdir -p "$BUILD_DIR"
    fi
    
    cd "$BUILD_DIR"
    
    # Configure with CMake
    print_info "Configuring with CMake..."
    if [ "$RELEASE_BUILD" = true ]; then
        cmake -DCMAKE_BUILD_TYPE=Release \
              -DBUILD_TESTING=ON \
              -DBUILD_EXAMPLES=ON \
              ..
        print_success "Configured in Release mode (optimized)"
    else
        cmake -DCMAKE_BUILD_TYPE=Debug \
              -DBUILD_TESTING=ON \
              -DBUILD_EXAMPLES=ON \
              ..
        print_success "Configured in Debug mode"
    fi
    
    # Build
    print_info "Building with $NUM_CORES cores..."
    make -j"$NUM_CORES"
    
    cd ..
    print_success "Build completed successfully!"
    
    # Show built binaries
    print_info "Built executables:"
    ls -lh "$BUILD_DIR/bin/" 2>/dev/null || echo "  (binaries will be in $BUILD_DIR/bin/)"
}

# ============================================================================
# Vocabulary Creation Functions
# ============================================================================

create_vocabulary() {
    print_header "Creating Vocabulary"
    
    # Check if training data exists
    if [ ! -f "sample_training_data.txt" ]; then
        print_error "Training data not found: sample_training_data.txt"
        print_info "Please create a training data file in INPUT/RESPONSE format"
        return 1
    fi
    
    # Extract text for vocabulary building
    print_info "Extracting text from training data..."
    grep "^INPUT:\|^RESPONSE:" sample_training_data.txt | sed 's/^INPUT: //;s/^RESPONSE: //' > /tmp/vocab_texts.txt
    
    local num_lines=$(wc -l < /tmp/vocab_texts.txt)
    print_info "Found $num_lines text samples"
    
    # Parameters
    local vocab_size=${1:-5000}
    local output_file=${2:-vocab.txt}
    
    print_info "Vocabulary size: $vocab_size"
    print_info "Output file: $output_file"
    
    # Check if chatbot_trainer exists
    if [ ! -f "$BUILD_DIR/bin/chatbot_trainer" ]; then
        print_error "chatbot_trainer not found. Please build the project first."
        return 1
    fi
    
    # Use ChatbotTrainer to build vocabulary
    print_info "Building vocabulary using ChatbotTrainer..."
    "$BUILD_DIR/bin/chatbot_trainer" \
        --data sample_training_data.txt \
        --build-vocab \
        --vocab-size "$vocab_size" \
        --output-vocab "$output_file"
    
    if [ -f "$output_file" ]; then
        print_success "Vocabulary created: $output_file"
        print_info "Vocabulary statistics:"
        head -n 10 "$output_file"
        echo "  ..."
        local vocab_count=$(grep -c "^[^#]" "$output_file" || echo "unknown")
        print_info "Total tokens: $vocab_count"
    else
        print_error "Failed to create vocabulary"
        return 1
    fi
}

# ============================================================================
# Training Functions
# ============================================================================

train_model() {
    print_header "Training Chatbot Model"
    
    local data_file=${1:-sample_training_data.txt}
    local vocab_file=${2:-vocab.txt}
    local model_output=${3:-chatbot_model.bin}
    local epochs=${4:-10}
    
    # Check files exist
    if [ ! -f "$data_file" ]; then
        print_error "Training data not found: $data_file"
        return 1
    fi
    
    if [ ! -f "$vocab_file" ]; then
        print_error "Vocabulary not found: $vocab_file"
        print_info "Run: $0 vocab"
        return 1
    fi
    
    if [ ! -f "$BUILD_DIR/bin/chatbot_trainer" ]; then
        print_error "chatbot_trainer not found. Please build the project first."
        return 1
    fi
    
    print_info "Training data: $data_file"
    print_info "Vocabulary: $vocab_file"
    print_info "Output model: $model_output"
    print_info "Epochs: $epochs"
    
    # Train model
    "$BUILD_DIR/bin/chatbot_trainer" \
        --data "$data_file" \
        --vocab "$vocab_file" \
        --output "$model_output" \
        --epochs "$epochs" \
        --learning-rate 0.001 \
        --batch-size 4
    
    print_success "Model training completed!"
    print_info "Model files:"
    ls -lh "$model_output"* | head -n 10
}

# ============================================================================
# Interactive Mode
# ============================================================================

interactive_menu() {
    while true; do
        print_header "ADAI Build & Vocabulary Tool"
        echo "1. Build project (Release mode)"
        echo "2. Build project (Debug mode)"
        echo "3. Create vocabulary (5000 tokens)"
        echo "4. Create vocabulary (custom size)"
        echo "5. Train model (10 epochs)"
        echo "6. Train model (custom epochs)"
        echo "7. Full workflow (build + vocab + train)"
        echo "8. Clean build directory"
        echo "9. Exit"
        echo ""
        read -p "Select option [1-9]: " choice
        
        case $choice in
            1)
                RELEASE_BUILD=true
                build_project
                ;;
            2)
                RELEASE_BUILD=false
                build_project
                ;;
            3)
                create_vocabulary 5000 vocab.txt
                ;;
            4)
                read -p "Enter vocabulary size: " size
                read -p "Enter output filename [vocab.txt]: " filename
                filename=${filename:-vocab.txt}
                create_vocabulary "$size" "$filename"
                ;;
            5)
                train_model sample_training_data.txt vocab.txt chatbot_model.bin 10
                ;;
            6)
                read -p "Enter number of epochs: " epochs
                train_model sample_training_data.txt vocab.txt chatbot_model.bin "$epochs"
                ;;
            7)
                print_header "Full Workflow"
                RELEASE_BUILD=true
                build_project
                create_vocabulary 5000 vocab.txt
                train_model sample_training_data.txt vocab.txt chatbot_model.bin 10
                print_success "Full workflow completed!"
                ;;
            8)
                print_info "Cleaning build directory..."
                rm -rf "$BUILD_DIR"
                print_success "Build directory cleaned"
                ;;
            9)
                print_info "Exiting..."
                exit 0
                ;;
            *)
                print_error "Invalid option"
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# ============================================================================
# Main Script
# ============================================================================

show_usage() {
    cat << EOF
ADAI Build and Vocabulary Creation Tool

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    build               Build the project (Release mode)
    build-debug         Build the project (Debug mode)
    vocab [SIZE] [OUT]  Create vocabulary (default: 5000 tokens, vocab.txt)
    train [EPOCHS]      Train model (default: 10 epochs)
    full                Run full workflow (build + vocab + train)
    clean               Clean build directory
    interactive         Launch interactive menu (default)
    help                Show this help message

Examples:
    $0 build                      # Build in release mode
    $0 vocab 10000 my_vocab.txt   # Create 10k vocabulary
    $0 train 20                   # Train for 20 epochs
    $0 full                       # Complete workflow

Environment Variables:
    BUILD_DIR           Build directory (default: build)
    NUM_CORES           Number of cores for compilation (default: auto-detect)

EOF
}

# Parse command line arguments
case "${1:-interactive}" in
    build)
        RELEASE_BUILD=true
        build_project
        ;;
    build-debug)
        RELEASE_BUILD=false
        build_project
        ;;
    vocab)
        create_vocabulary "${2:-5000}" "${3:-vocab.txt}"
        ;;
    train)
        train_model sample_training_data.txt vocab.txt chatbot_model.bin "${2:-10}"
        ;;
    full)
        RELEASE_BUILD=true
        build_project
        create_vocabulary 5000 vocab.txt
        train_model sample_training_data.txt vocab.txt chatbot_model.bin 10
        print_success "🎉 Full workflow completed!"
        ;;
    clean)
        rm -rf "$BUILD_DIR"
        print_success "Build directory cleaned"
        ;;
    interactive)
        interactive_menu
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
