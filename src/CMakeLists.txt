cmake_minimum_required(VERSION 3.10)

# ============================================================================
# OpenMP Support for Parallel Processing
# ============================================================================

# Find OpenMP package
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling parallel matrix operations")
    set(ENABLE_OPENMP ON)
else()
    message(WARNING "OpenMP not found - matrix operations will run sequentially. Install with: sudo apt-get install libomp-dev")
    set(ENABLE_OPENMP OFF)
endif()

# ============================================================================
# Core Libraries - Build Once, Link Many
# ============================================================================

# Core mathematical operations
add_library(adai_core STATIC
    Matrix.cpp
    Activation.cpp
    Optimizer.cpp
)

# OpenMP support (optional but recommended)
if(OpenMP_CXX_FOUND)
    target_link_libraries(adai_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(adai_core PUBLIC ADAI_ENABLE_OPENMP)
    message(STATUS "OpenMP support added to adai_core - parallel CPU operations enabled")
endif()

# GPU support (optional)
if(ENABLE_GPU)
    # Add GPU library
    add_library(adai_gpu STATIC
        gpu/MatrixGPU.cu
    )
    
    # Set CUDA properties
    set_target_properties(adai_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    # Link with CUDA libraries
    target_link_libraries(adai_gpu
        CUDA::cudart
        CUDA::cublas
    )
    
    # Include GPU headers
    target_include_directories(adai_gpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    # Link core library with GPU library
    target_link_libraries(adai_core adai_gpu)
    
    message(STATUS "GPU support added to adai_core")
endif()

# Precompiled headers for commonly used STL headers
target_precompile_headers(adai_core
    PUBLIC
        <vector>
        <string>
        <memory>
        <algorithm>
        <cmath>
        <iostream>
        <stdexcept>
        <unordered_map>
        <utility>
)

# Layer normalization and utilities
add_library(adai_layers STATIC
    LayerNorm.cpp
    PositionalEncoding.cpp
    TokenEmbedding.cpp
)
target_link_libraries(adai_layers adai_core)

# Attention mechanisms
add_library(adai_attention STATIC
    MultiHeadAttention.cpp
    CrossAttention.cpp
)
target_link_libraries(adai_attention adai_core)

# Feed-forward networks
add_library(adai_feedforward STATIC
    FeedForward.cpp
)
target_link_libraries(adai_feedforward adai_core)

# Transformer blocks
add_library(adai_transformer STATIC
    EncoderBlock.cpp
    DecoderBlock.cpp
)
target_link_libraries(adai_transformer 
    adai_attention 
    adai_feedforward 
    adai_layers
)

# Complete models
add_library(adai_models STATIC
    LLMEncoder.cpp
    Decoder.cpp
    LanguageModelHead.cpp
    EncoderDecoderModel.cpp
)
target_link_libraries(adai_models adai_transformer)

# Tokenization and generation
add_library(adai_nlp STATIC
    BPETokenizer.cpp
    TextGenerator.cpp
    ConversationContext.cpp
)
target_link_libraries(adai_nlp adai_core)

# Additional precompiled headers for NLP components
target_precompile_headers(adai_nlp
    PRIVATE
        <fstream>
        <sstream>
        <regex>
        <queue>
        <unordered_set>
        <cctype>
)

# API layer (optional, requires httplib)
option(BUILD_API_SERVER "Build REST API server" ON)
if(BUILD_API_SERVER)
    # Try to find cpp-httplib
    find_path(HTTPLIB_INCLUDE_DIR httplib.h
        PATHS
            /usr/include
            /usr/local/include
            ${CMAKE_SOURCE_DIR}/external/cpp-httplib
    )
    
    if(HTTPLIB_INCLUDE_DIR)
        add_library(adai_api STATIC
            ChatbotAPI.cpp
        )
        target_link_libraries(adai_api 
            adai_models 
            adai_nlp
            pthread
        )
        target_include_directories(adai_api PRIVATE ${HTTPLIB_INCLUDE_DIR})
        message(STATUS "Building API server (cpp-httplib found at ${HTTPLIB_INCLUDE_DIR})")
    else()
        message(WARNING "cpp-httplib not found. API server will not be built. Install with: sudo apt-get install libhttplib-dev or download from https://github.com/yhirose/cpp-httplib")
        set(BUILD_API_SERVER OFF)
    endif()
endif()

# ============================================================================
# Production Executables
# ============================================================================

# Chatbot CLI - Interactive chatbot application with parallel optimizations (P1-P5)
add_executable(chatbot ChatbotCLI_main.cpp ChatbotCLI.cpp)
target_link_libraries(chatbot 
    adai_models 
    adai_nlp
    adai_attention    # Priority 4: Parallel attention heads
    adai_core         # Priority 1: OpenMP matrix operations
)
if(OpenMP_CXX_FOUND)
    message(STATUS "Building chatbot with full parallel optimizations")
endif()

# Chatbot Trainer - Model training tool with parallel optimizations (P1-P5)
add_executable(chatbot_trainer ChatbotTrainer.cpp)
target_link_libraries(chatbot_trainer 
    adai_models 
    adai_nlp
    adai_attention    # Priority 4: Parallel attention heads
    adai_core         # Priority 1: OpenMP matrix operations
)
if(OpenMP_CXX_FOUND)
    message(STATUS "Building chatbot_trainer with full parallel optimizations (OpenMP + parallel attention)")
endif()

# Vocabulary Builder - Standalone BPE vocabulary creation tool
add_executable(vocab_builder VocabBuilder.cpp)
target_link_libraries(vocab_builder adai_nlp)
set_target_properties(vocab_builder PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
message(STATUS "Building vocab_builder - standalone vocabulary creation tool")

# OpenMP Benchmark - Test parallel performance (Priority 1 implementation)
add_executable(openmp_benchmark OpenMPBenchmark.cpp)
target_link_libraries(openmp_benchmark adai_core)
if(OpenMP_CXX_FOUND)
    message(STATUS "Building openmp_benchmark with OpenMP support")
endif()

# Augmentation Benchmark - Test parallel data augmentation (Priority 2 implementation)
add_executable(augmentation_benchmark AugmentationBenchmark.cpp)
target_link_libraries(augmentation_benchmark adai_core)
if(OpenMP_CXX_FOUND)
    message(STATUS "Building augmentation_benchmark with OpenMP support")
endif()

# Batched Inference Benchmark - Test batched inference throughput (Priority 3 implementation)
add_executable(batched_inference_benchmark BatchedInferenceBenchmark.cpp)
target_link_libraries(batched_inference_benchmark adai_nlp adai_core)
message(STATUS "Building batched_inference_benchmark")

# Attention Head Benchmark - Test parallel attention head computation (Priority 4 implementation)
add_executable(attention_head_benchmark AttentionHeadBenchmark.cpp)
target_link_libraries(attention_head_benchmark adai_attention adai_core)
if(OpenMP_CXX_FOUND)
    message(STATUS "Building attention_head_benchmark with OpenMP support")
endif()

# Pipeline Benchmark - Test encoder-decoder pipeline parallelism (Priority 5 implementation)
add_executable(pipeline_benchmark PipelineBenchmark.cpp)
target_link_libraries(pipeline_benchmark adai_models adai_nlp adai_core pthread)
message(STATUS "Building pipeline_benchmark")

# Integrated Benchmark - Test all optimizations together (Priority 1-5 combined)
add_executable(integrated_benchmark IntegratedBenchmark.cpp)
target_link_libraries(integrated_benchmark adai_models adai_nlp adai_attention adai_core pthread)
if(OpenMP_CXX_FOUND)
    message(STATUS "Building integrated_benchmark with full parallelization (OpenMP + pthread)")
endif()

# Chatbot GUI - Qt-based graphical interface (optional)
option(BUILD_GUI "Build Qt GUI for chatbot" ON)
if(BUILD_GUI)
    # Find Qt5 or Qt6
    find_package(Qt6 COMPONENTS Widgets QUIET)
    if(NOT Qt6_FOUND)
        find_package(Qt5 COMPONENTS Widgets QUIET)
    endif()
    
    if(Qt6_FOUND OR Qt5_FOUND)
        # Enable Qt's automatic MOC (Meta-Object Compiler)
        set(CMAKE_AUTOMOC ON)
        set(CMAKE_AUTOUIC ON)
        set(CMAKE_AUTORCC ON)
        
        # Build the actual GUI binary (will be renamed)
        add_executable(chatbot_gui_binary 
            ChatbotGUI_main.cpp 
            ChatbotGUI.cpp
        )
        
        # Link with appropriate Qt version and full parallel processing libraries
        if(Qt6_FOUND)
            target_link_libraries(chatbot_gui_binary 
                adai_models 
                adai_nlp
                adai_attention    # Priority 4: Parallel attention heads
                adai_core         # Priority 1: OpenMP matrix operations
                Qt6::Widgets
            )
            if(OpenMP_CXX_FOUND)
                message(STATUS "Building chatbot_gui with Qt6 and full parallel optimizations (OpenMP + parallel attention)")
            else()
                message(STATUS "Building chatbot_gui with Qt6")
            endif()
        else()
            target_link_libraries(chatbot_gui_binary 
                adai_models 
                adai_nlp
                adai_attention    # Priority 4: Parallel attention heads
                adai_core         # Priority 1: OpenMP matrix operations
                Qt5::Widgets
            )
            if(OpenMP_CXX_FOUND)
                message(STATUS "Building chatbot_gui with Qt5 and full parallel optimizations (OpenMP + parallel attention)")
            else()
                message(STATUS "Building chatbot_gui with Qt5")
            endif()
        endif()
        
        # Build the wrapper that fixes environment issues
        add_executable(chatbot_gui ChatbotGUI_wrapper.cpp)
        
        # Make sure the binary is built before the wrapper
        add_dependencies(chatbot_gui chatbot_gui_binary)
        
        message(STATUS "Building chatbot_gui wrapper to fix library path issues")
    else()
        message(WARNING "Qt5/Qt6 not found. GUI will not be built. Install with: sudo apt-get install qt5-default or qtbase5-dev")
        set(BUILD_GUI OFF)
    endif()
endif()

# Chatbot API Server - REST API service
if(BUILD_API_SERVER)
    add_executable(chatbot_api_server ChatbotAPIServer.cpp)
    target_link_libraries(chatbot_api_server 
        adai_api
        adai_models 
        adai_nlp
    )
    target_include_directories(chatbot_api_server PRIVATE ${HTTPLIB_INCLUDE_DIR})
    message(STATUS "Building chatbot_api_server executable")
endif()

# ============================================================================
# Example Programs (optional build)
# ============================================================================

if(BUILD_EXAMPLES)
    # Tokenizer example
    add_executable(tokenizer_example BPETokenizerexample.cpp)
    target_link_libraries(tokenizer_example adai_nlp)
    
    # LLM Encoder example
    add_executable(encoder_example encoder.cpp)
    target_link_libraries(encoder_example adai_models adai_nlp)
    
    # EncoderBlock example
    add_executable(encoderblock_example EncoderBlockExample.cpp)
    target_link_libraries(encoderblock_example adai_transformer)
    
    # MultiHeadAttention example
    add_executable(mha_example MultiHeadAttentionExample.cpp)
    target_link_libraries(mha_example adai_attention)
    
    # FeedForward example
    add_executable(feedforward_example FeedForwardExample.cpp)
    target_link_libraries(feedforward_example adai_feedforward)
    
    # DecoderBlock example
    add_executable(decoderblock_example DecoderBlockExample.cpp)
    target_link_libraries(decoderblock_example adai_transformer)
    
    # TextGenerator example
    add_executable(textgenerator_example TextGeneratorExample.cpp)
    target_link_libraries(textgenerator_example adai_nlp)
    
    # EncoderDecoderModel example
    add_executable(encoder_decoder_example EncoderDecoderModelExample.cpp)
    target_link_libraries(encoder_decoder_example adai_models adai_nlp)
    
    # Optimizer example
    add_executable(optimizer_example OptimizerExample.cpp)
    target_link_libraries(optimizer_example adai_core)
    
    # Inference Optimization Benchmark
    add_executable(inference_optimization_benchmark InferenceOptimizationBenchmark.cpp)
    target_link_libraries(inference_optimization_benchmark adai_models adai_nlp)
    
    # Enhanced Training Pipeline Example
    add_executable(enhanced_training_example EnhancedTrainingExample.cpp)
    target_link_libraries(enhanced_training_example adai_models adai_nlp)
    
    # Dataset Enhanced Features Example
    add_executable(dataset_enhanced_example DatasetEnhancedExample.cpp)
    target_link_libraries(dataset_enhanced_example adai_core)
    
    # Phase 5 Advanced Features Examples
    add_executable(phase5_examples Phase5Examples.cpp)
    target_link_libraries(phase5_examples adai_core)
    
    # Data Pipeline Enhancement Example
    add_executable(data_pipeline_example DataPipelineExample.cpp)
    target_link_libraries(data_pipeline_example adai_nlp)
    
    # Dataset Batch Processing Example
    add_executable(dataset_batch_processing_example DatasetBatchProcessingExample.cpp)
    target_link_libraries(dataset_batch_processing_example adai_nlp)
    
    # GPU Acceleration Example (always build, shows status even without GPU)
    add_executable(gpu_example GPUExample.cpp)
    target_link_libraries(gpu_example adai_core)
    
    message(STATUS "Building example programs (disable with -DBUILD_EXAMPLES=OFF)")
endif()

# ============================================================================
# Build Summary
# ============================================================================

message(STATUS "ADAI Build Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build Tests: ${BUILD_TESTING}")
