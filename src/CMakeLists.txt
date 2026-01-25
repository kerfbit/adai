cmake_minimum_required(VERSION 3.10)

# ============================================================================
# Core Libraries - Build Once, Link Many
# ============================================================================

# Core mathematical operations
add_library(adai_core STATIC
    Matrix.cpp
    Activation.cpp
    Optimizer.cpp
)

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

# Chatbot CLI - Interactive chatbot application
add_executable(chatbot ChatbotCLI.cpp)
target_link_libraries(chatbot 
    adai_models 
    adai_nlp
)

# Chatbot Trainer - Model training tool
add_executable(chatbot_trainer ChatbotTrainer.cpp)
target_link_libraries(chatbot_trainer 
    adai_models 
    adai_nlp
)

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
