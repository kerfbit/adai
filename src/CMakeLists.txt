cmake_minimum_required(VERSION 3.10)

# ============================================================================
# OpenMP Support for Parallel Processing
# ============================================================================

# Apple Clang does not bundle libomp; provide hints when Homebrew's libomp is
# installed (brew install libomp) so that find_package(OpenMP) succeeds.
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE BREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(BREW_LIBOMP_PREFIX)
        set(OpenMP_C_FLAGS   "-Xpreprocessor -fopenmp -I${BREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${BREW_LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES   "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY   "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

# Find OpenMP package
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling parallel matrix operations")
    set(ENABLE_OPENMP ON)
else()
    if(APPLE)
        message(WARNING "OpenMP not found - matrix operations will run sequentially. Install with: brew install libomp")
    else()
        message(WARNING "OpenMP not found - matrix operations will run sequentially. Install with: sudo apt-get install libomp-dev")
    endif()
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
    Logger.cpp
    TrainingMetricsService.cpp
    MetricsPushClient.cpp
    GenerationQualityMetrics.cpp
    RegistryTransport.cpp
    DatasetRegistry.cpp
    DataFetcher.cpp
)
target_link_libraries(adai_core PUBLIC spdlog::spdlog)

# Add httplib support for metrics push (if available)
if(HTTPLIB_INCLUDE_DIR)
    target_include_directories(adai_core PRIVATE ${HTTPLIB_INCLUDE_DIR})
    target_compile_definitions(adai_core PRIVATE BUILD_METRICS_API_SERVER)
    message(STATUS "Metrics push support added to adai_core (httplib found)")
endif()

# OpenMP support (optional but recommended)
if(OpenMP_CXX_FOUND)
    target_link_libraries(adai_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(adai_core PUBLIC ADAI_ENABLE_OPENMP)
    message(STATUS "OpenMP support added to adai_core - parallel CPU operations enabled")
endif()

# ============================================================================
# BLAS support (TD-007) — OpenBLAS / Intel MKL / Apple Accelerate
# Enable with: -DENABLE_BLAS=ON (default). Requires libblas-dev or libopenblas-dev.
# ============================================================================
if(ENABLE_BLAS)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        # locate the CBLAS C interface header (required for cblas_sgemm)
        find_path(CBLAS_INCLUDE_DIR cblas.h
            PATHS
                /usr/include
                /usr/local/include
                /usr/include/x86_64-linux-gnu
                /usr/include/aarch64-linux-gnu
                /usr/include/arm-linux-gnueabihf
                /usr/local/opt/openblas/include    # macOS Homebrew OpenBLAS
                /opt/homebrew/opt/openblas/include  # macOS Apple Silicon Homebrew
        )
        if(CBLAS_INCLUDE_DIR)
            target_include_directories(adai_core PRIVATE ${CBLAS_INCLUDE_DIR})
            target_link_libraries(adai_core PUBLIC ${BLAS_LIBRARIES})
            target_compile_definitions(adai_core PUBLIC ADAI_ENABLE_BLAS)
            message(STATUS "BLAS enabled: ${BLAS_LIBRARIES} (cblas.h: ${CBLAS_INCLUDE_DIR})")
        else()
            message(STATUS "BLAS found (${BLAS_LIBRARIES}) but cblas.h not located — "
                           "install libblas-dev or libopenblas-dev to enable BLAS acceleration")
        endif()
    else()
        message(STATUS "BLAS not found — install libblas-dev or libopenblas-dev to enable BLAS acceleration")
    endif()
endif()

# ============================================================================
# SIMD support (TD-007) — AVX2/FMA (x86) and NEON (ARM)
# Enable with: -DENABLE_SIMD=ON (default).
# Release builds using -march=native already enable AVX2+FMA where supported.
# This option additionally enables intrinsics in Debug/RelWithDebInfo builds.
# ============================================================================
if(ENABLE_SIMD)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx2" ADAI_HAVE_AVX2)
        if(ADAI_HAVE_AVX2)
            target_compile_options(adai_core PRIVATE -mavx2)
            check_cxx_compiler_flag("-mfma" ADAI_HAVE_FMA)
            if(ADAI_HAVE_FMA)
                target_compile_options(adai_core PRIVATE -mfma)
                message(STATUS "SIMD: AVX2 + FMA intrinsics enabled for adai_core")
            else()
                message(STATUS "SIMD: AVX2 intrinsics enabled for adai_core (no FMA)")
            endif()
        else()
            message(STATUS "SIMD: compiler does not support -mavx2; intrinsics disabled")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(adai_core PRIVATE /arch:AVX2)
        message(STATUS "SIMD: AVX2 intrinsics enabled via /arch:AVX2")
    endif()
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
    
    # Link with CUDA libraries — use static cudart so the binary needs no
    # libcudart.so on the target machine (driver-only install is enough).
    # cublas has no simple static form; set RPATH=$ORIGIN so the bundled
    # libcublas.so found next to the executable at runtime.
    target_link_libraries(adai_gpu
        CUDA::cudart_static
        CUDA::cublas
    )
    
    # Include GPU headers
    target_include_directories(adai_gpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    # Link core library with GPU library
    target_link_libraries(adai_core PUBLIC adai_gpu)

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
)  # spdlog transitive via adai_core

# Complete models
add_library(adai_models STATIC
    LLMEncoder.cpp
    Decoder.cpp
    LanguageModelHead.cpp
    EncoderDecoderModel.cpp
    ModelSerializer.cpp
)
target_link_libraries(adai_models adai_transformer)
target_include_directories(adai_models PRIVATE ${CMAKE_SOURCE_DIR}/external)

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

# Try to find cpp-httplib
find_path(HTTPLIB_INCLUDE_DIR httplib.h
    PATHS
        /usr/include
        /usr/local/include
        ${CMAKE_SOURCE_DIR}/external/cpp-httplib
)

# ============================================================================
# libcurl — required for FTP dataset transport (Phase 10)
# Enable with: -DENABLE_FTP_TRANSPORT=ON (default ON when curl is found)
# ============================================================================
option(ENABLE_FTP_TRANSPORT "Enable FTP dataset transport via libcurl" ON)
if(ENABLE_FTP_TRANSPORT)
    find_package(CURL QUIET)
    if(CURL_FOUND)
        message(STATUS "libcurl found: ${CURL_LIBRARIES} — FTP dataset transport enabled")
    else()
        message(STATUS "libcurl not found — FTP dataset transport disabled. "
                       "Install with: sudo apt-get install libcurl4-openssl-dev")
        set(ENABLE_FTP_TRANSPORT OFF)
    endif()
endif()

# ============================================================================
# OpenSSL — required for FTPS (TLS) and HMAC-SHA256 token signing (Phase 11/3)
# Enable with: -DENABLE_FTPS=ON (default ON when OpenSSL is found)
# ============================================================================
option(ENABLE_FTPS "Enable FTPS (TLS) for FTP dataset transport" ON)
if(ENABLE_FTPS)
    find_package(OpenSSL QUIET)
    if(OPENSSL_FOUND)
        message(STATUS "OpenSSL found: ${OPENSSL_LIBRARIES} — FTPS enabled")
    else()
        message(STATUS "OpenSSL not found — FTPS disabled. "
                       "Install with: sudo apt-get install libssl-dev")
        set(ENABLE_FTPS OFF)
    endif()
endif()

# API layer (optional, requires httplib)
option(BUILD_API_SERVER "Build REST API server" ON)
if(BUILD_API_SERVER)
    
    if(HTTPLIB_INCLUDE_DIR)
        add_library(adai_api STATIC
            ChatbotAPI.cpp
            DocumentStore.cpp
            RAGInference.cpp
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

# Training Metrics API layer (optional, requires httplib)
option(BUILD_METRICS_API_SERVER "Build Training Metrics REST API server" ON)
if(BUILD_METRICS_API_SERVER)
    
    if(HTTPLIB_INCLUDE_DIR)
        add_library(adai_metrics_api STATIC
            TrainingMetricsAPI.cpp
        )
        target_link_libraries(adai_metrics_api
            adai_core
            pthread
        )
        target_include_directories(adai_metrics_api PRIVATE ${HTTPLIB_INCLUDE_DIR})
        message(STATUS "Building Metrics API server (cpp-httplib found at ${HTTPLIB_INCLUDE_DIR})")
    else()
        message(WARNING "cpp-httplib not found. Metrics API server will not be built. Install with: sudo apt-get install libhttplib-dev or download from https://github.com/yhirose/cpp-httplib")
        set(BUILD_METRICS_API_SERVER OFF)
    endif()
endif()

# ============================================================================
# Production Executables
# ============================================================================

# Chatbot CLI - Interactive chatbot API client
add_executable(chatbot ChatbotCLI_main.cpp ChatbotCLI.cpp)
target_include_directories(chatbot PRIVATE ${HTTPLIB_INCLUDE_DIR})
find_package(Threads)
target_link_libraries(chatbot Threads::Threads)
set_target_properties(chatbot PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
message(STATUS "Building chatbot as lightweight API client")


# Incremental Training Tool - Incremental model training with data management
add_executable(incremental_trainer IncrementalTrainingTool.cpp IncrementalTrainer.cpp
    ChatbotTrainer.cpp Config.cpp TrainingMetricsService.cpp)  # Logger.cpp compiled into adai_core
target_link_libraries(incremental_trainer 
    adai_models 
    adai_nlp
    adai_attention
    adai_core
    spdlog::spdlog
    $<$<CXX_COMPILER_ID:GNU>:stdc++fs>  # GCC/libstdc++ only; not available on macOS (libc++)
)
if(BUILD_METRICS_API_SERVER AND HTTPLIB_INCLUDE_DIR)
    target_compile_definitions(incremental_trainer PRIVATE BUILD_METRICS_API_SERVER)
    target_link_libraries(incremental_trainer adai_metrics_api)
    target_include_directories(incremental_trainer PRIVATE ${HTTPLIB_INCLUDE_DIR})
endif()
if(ENABLE_FTP_TRANSPORT AND CURL_FOUND)
    target_compile_definitions(incremental_trainer PRIVATE BUILD_FTP_TRANSPORT)
    target_link_libraries(incremental_trainer CURL::libcurl)
    message(STATUS "FTP dataset transport enabled for incremental_trainer")
endif()
set_target_properties(incremental_trainer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
if(ENABLE_GPU)
    # Allow bundled cublas .so files placed next to the binary to be found at runtime
    set_target_properties(incremental_trainer PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_RPATH "$ORIGIN"
    )
endif()
if(OpenMP_CXX_FOUND)
    message(STATUS "Building incremental_trainer with full parallel optimizations")
endif()

# Find SQLite3 (pkg-config not always available; fall back to find_path/find_library)
find_path(SQLITE3_INCLUDE_DIR sqlite3.h
    PATHS /usr/include /usr/local/include /usr/include/x86_64-linux-gnu)
find_library(SQLITE3_LIBRARY NAMES sqlite3
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
if(SQLITE3_INCLUDE_DIR AND SQLITE3_LIBRARY)
    message(STATUS "SQLite3 found: ${SQLITE3_LIBRARY} (headers: ${SQLITE3_INCLUDE_DIR})")
    set(SQLITE3_FOUND TRUE)
else()
    message(WARNING "SQLite3 not found — adai_mns will not be built. Install with: sudo apt-get install libsqlite3-dev")
    set(SQLITE3_FOUND FALSE)
endif()

# Model Name Service library (requires cpp-httplib + SQLite3)
if(HTTPLIB_INCLUDE_DIR AND SQLITE3_FOUND)
    add_library(adai_mns STATIC
        ModelNameService.cpp
        ModelNameClient.cpp
    )
    target_link_libraries(adai_mns
        adai_core
        pthread
        ${SQLITE3_LIBRARY}
    )
    target_include_directories(adai_mns PRIVATE ${HTTPLIB_INCLUDE_DIR} ${SQLITE3_INCLUDE_DIR})
    target_compile_definitions(adai_mns PUBLIC BUILD_MNS_SERVER)
    message(STATUS "Building adai_mns - Model Name Service library (SQLite backend)")
else()
    message(STATUS "adai_mns not built (cpp-httplib or SQLite3 not found)")
endif()

# Model Name Service daemon (mns_server)
if(HTTPLIB_INCLUDE_DIR)
    add_executable(mns_server ModelNameServiceServer.cpp)
    target_link_libraries(mns_server
        adai_mns
        adai_core
        spdlog::spdlog
        pthread
        $<$<CXX_COMPILER_ID:GNU>:stdc++fs>
    )
    target_include_directories(mns_server PRIVATE ${HTTPLIB_INCLUDE_DIR})
    set_target_properties(mns_server PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    message(STATUS "Building mns_server - Model Name Service daemon (port 8083)")
else()
    message(STATUS "mns_server not built (cpp-httplib not found)")
endif()

# Model Name Service CLI (mns_cli)
if(HTTPLIB_INCLUDE_DIR AND TARGET adai_mns)
    add_executable(mns_cli MnsCliTool.cpp Config.cpp)
    target_link_libraries(mns_cli
        adai_mns
        adai_core
        spdlog::spdlog
        pthread
        $<$<CXX_COMPILER_ID:GNU>:stdc++fs>
    )
    target_include_directories(mns_cli PRIVATE ${HTTPLIB_INCLUDE_DIR} ${SQLITE3_INCLUDE_DIR})
    set_target_properties(mns_cli PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    message(STATUS "Building mns_cli - Model Name Service CLI tool")
endif()

# Link adai_mns into dependent targets now that the target exists
if(TARGET adai_mns)
    target_link_libraries(incremental_trainer adai_mns)
    target_include_directories(incremental_trainer PRIVATE ${SQLITE3_INCLUDE_DIR})
    if(TARGET chatbot_api_server)
        target_link_libraries(chatbot_api_server adai_mns)
        target_include_directories(chatbot_api_server PRIVATE ${SQLITE3_INCLUDE_DIR})
    endif()
    if(TARGET adai_metrics_api)
        target_link_libraries(adai_metrics_api adai_mns)
        target_include_directories(adai_metrics_api PRIVATE ${SQLITE3_INCLUDE_DIR})
    endif()
endif()

# Registry Server - Distributed dataset queue coordination daemon (TD-028 Phase 9)
# Requires cpp-httplib; skipped when not available.
if(HTTPLIB_INCLUDE_DIR)
    add_executable(registry_server RegistryServer.cpp Config.cpp)
    target_link_libraries(registry_server
        adai_core
        spdlog::spdlog
        pthread
        $<$<CXX_COMPILER_ID:GNU>:stdc++fs>
    )
    target_include_directories(registry_server PRIVATE ${HTTPLIB_INCLUDE_DIR})
    target_compile_definitions(registry_server PRIVATE BUILD_METRICS_API_SERVER)
    set_target_properties(registry_server PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    if(ENABLE_FTPS AND OPENSSL_FOUND)
        target_compile_definitions(registry_server PRIVATE BUILD_FTPS)
        target_link_libraries(registry_server OpenSSL::SSL OpenSSL::Crypto)
        message(STATUS "FTPS enabled for registry_server (OpenSSL found)")
    endif()
    message(STATUS "Building registry_server - distributed dataset queue coordination")
else()
    message(STATUS "registry_server not built (cpp-httplib not found)")
endif()

# Dataset Manager - Standalone data queue management tool (TD-028 Phase 6)
# Links adai_core (DatasetRegistry, DataFetcher, Logger), Config.cpp, and optionally adai_mns.
add_executable(dataset_manager DatasetManagerTool.cpp Config.cpp)
target_link_libraries(dataset_manager
    adai_core
    spdlog::spdlog
    $<$<CXX_COMPILER_ID:GNU>:stdc++fs>
)
if(TARGET adai_mns)
    target_link_libraries(dataset_manager adai_mns)
    target_include_directories(dataset_manager PRIVATE ${HTTPLIB_INCLUDE_DIR} ${SQLITE3_INCLUDE_DIR})
endif()
set_target_properties(dataset_manager PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
message(STATUS "Building dataset_manager - standalone dataset queue management tool")

# Vocabulary Builder - Standalone BPE vocabulary creation tool
add_executable(vocab_builder VocabBuilder.cpp)
target_link_libraries(vocab_builder adai_nlp)
set_target_properties(vocab_builder PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
message(STATUS "Building vocab_builder - standalone vocabulary creation tool")

# OpenMP Benchmark - Test parallel performance (Priority 1 implementation)
add_executable(openmp_benchmark ../benchmarks/OpenMPBenchmark.cpp)
target_link_libraries(openmp_benchmark adai_core)
if(OpenMP_CXX_FOUND)
    message(STATUS "Building openmp_benchmark with OpenMP support")
endif()

# Augmentation Benchmark - Test parallel data augmentation (Priority 2 implementation)
add_executable(augmentation_benchmark ../benchmarks/AugmentationBenchmark.cpp)
target_link_libraries(augmentation_benchmark adai_core)
if(OpenMP_CXX_FOUND)
    message(STATUS "Building augmentation_benchmark with OpenMP support")
endif()

# Batched Inference Benchmark - Test batched inference throughput (Priority 3 implementation)
add_executable(batched_inference_benchmark ../benchmarks/BatchedInferenceBenchmark.cpp)
target_link_libraries(batched_inference_benchmark adai_nlp adai_core)
message(STATUS "Building batched_inference_benchmark")

# Attention Head Benchmark - Test parallel attention head computation (Priority 4 implementation)
add_executable(attention_head_benchmark ../benchmarks/AttentionHeadBenchmark.cpp)
target_link_libraries(attention_head_benchmark adai_attention adai_core)
if(OpenMP_CXX_FOUND)
    message(STATUS "Building attention_head_benchmark with OpenMP support")
endif()

# Pipeline Benchmark - Test encoder-decoder pipeline parallelism (Priority 5 implementation)
add_executable(pipeline_benchmark ../benchmarks/PipelineBenchmark.cpp)
target_link_libraries(pipeline_benchmark adai_models adai_nlp adai_core pthread)
message(STATUS "Building pipeline_benchmark")

# Integrated Benchmark - Test all optimizations together (Priority 1-5 combined)
add_executable(integrated_benchmark ../benchmarks/IntegratedBenchmark.cpp)
target_link_libraries(integrated_benchmark adai_models adai_nlp adai_attention adai_core pthread)
if(OpenMP_CXX_FOUND)
    message(STATUS "Building integrated_benchmark with full parallelization (OpenMP + pthread)")
endif()

# Chatbot GUI - Qt-based graphical interface (optional)
# BUILD_GUI option and Qt find_package are declared in the top-level CMakeLists.txt.
if(BUILD_GUI)
    
    if(Qt6_FOUND OR Qt5_FOUND)
        # Enable Qt's automatic MOC (Meta-Object Compiler)
        set(CMAKE_AUTOMOC ON)
        set(CMAKE_AUTOUIC ON)
        set(CMAKE_AUTORCC ON)
        
        # Build the actual GUI binary (will be renamed)
        add_executable(chatbot_gui_binary
            ChatbotGUI_main.cpp
            ChatbotGUI.cpp
            Config.cpp
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
        message(WARNING "Qt5/Qt6 not found. GUI will not be built. Install with: sudo apt-get install qtbase5-dev")
        set(BUILD_GUI OFF)
    endif()
endif()

# Chatbot API Server - REST API service
if(BUILD_API_SERVER)
    add_executable(chatbot_api_server
        ChatbotAPIServer.cpp
        Config.cpp
    )  # Logger.cpp compiled into adai_core
    target_link_libraries(chatbot_api_server
        adai_api
        adai_models
        adai_nlp
        adai_attention    # Priority 4: Parallel attention heads
        adai_core         # Priority 1: OpenMP matrix operations
        spdlog::spdlog
    )
    target_include_directories(chatbot_api_server PRIVATE ${HTTPLIB_INCLUDE_DIR})
    set_target_properties(chatbot_api_server PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    if(OpenMP_CXX_FOUND)
        message(STATUS "Building chatbot_api_server with full parallel optimizations (OpenMP + parallel attention)")
    else()
        message(STATUS "Building chatbot_api_server with structured logging")
    endif()
endif()

# Training Metrics API Server - REST API for training metrics polling
if(BUILD_METRICS_API_SERVER)
    add_executable(metrics_api_server
        TrainingMetricsAPIServer.cpp
    )
    target_link_libraries(metrics_api_server
        adai_metrics_api
        adai_core
        spdlog::spdlog
    )
    if(TARGET adai_mns)
        target_link_libraries(metrics_api_server adai_mns)
        target_include_directories(metrics_api_server PRIVATE ${SQLITE3_INCLUDE_DIR})
    endif()
    target_include_directories(metrics_api_server PRIVATE ${HTTPLIB_INCLUDE_DIR})
    set_target_properties(metrics_api_server PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    message(STATUS "Building metrics_api_server for training metrics polling")
endif()

# ============================================================================
# Example Programs (optional build)
# ============================================================================

if(BUILD_EXAMPLES)
    # Tokenizer example
    add_executable(tokenizer_example ../examples/BPETokenizerexample.cpp)
    target_link_libraries(tokenizer_example adai_nlp)
    
    # LLM Encoder example - Disabled: encoder.cpp moved to legacy (LLMEncoder.cpp is used in library)
    # add_executable(encoder_example encoder.cpp)
    # target_link_libraries(encoder_example adai_models adai_nlp)
    
    # EncoderBlock example
    add_executable(encoderblock_example ../examples/EncoderBlockExample.cpp)
    target_link_libraries(encoderblock_example adai_transformer)
    
    # MultiHeadAttention example
    add_executable(mha_example ../examples/MultiHeadAttentionExample.cpp)
    target_link_libraries(mha_example adai_attention)
    
    # FeedForward example
    add_executable(feedforward_example ../examples/FeedForwardExample.cpp)
    target_link_libraries(feedforward_example adai_feedforward)
    
    # DecoderBlock example
    add_executable(decoderblock_example ../examples/DecoderBlockExample.cpp)
    target_link_libraries(decoderblock_example adai_transformer)
    
    # TextGenerator example
    add_executable(textgenerator_example ../examples/TextGeneratorExample.cpp)
    target_link_libraries(textgenerator_example adai_nlp)
    
    # EncoderDecoderModel example
    add_executable(encoder_decoder_example ../examples/EncoderDecoderModelExample.cpp)
    target_link_libraries(encoder_decoder_example adai_models adai_nlp)
    
    # Optimizer example
    add_executable(optimizer_example ../examples/OptimizerExample.cpp)
    target_link_libraries(optimizer_example adai_core)
    
    # Inference Optimization Benchmark
    add_executable(inference_optimization_benchmark ../benchmarks/InferenceOptimizationBenchmark.cpp)
    target_link_libraries(inference_optimization_benchmark adai_models adai_nlp)
    
    # Enhanced Training Pipeline Example
    add_executable(enhanced_training_example ../examples/EnhancedTrainingExample.cpp)
    target_link_libraries(enhanced_training_example adai_models adai_nlp)
    
    # Dataset Enhanced Features Example
    add_executable(dataset_enhanced_example ../examples/DatasetEnhancedExample.cpp)
    target_link_libraries(dataset_enhanced_example adai_core)
    
    # Phase 5 Advanced Features Examples
    add_executable(phase5_examples ../examples/Phase5Examples.cpp)
    target_link_libraries(phase5_examples adai_core)
    
    # Data Pipeline Enhancement Example
    add_executable(data_pipeline_example ../examples/DataPipelineExample.cpp)
    target_link_libraries(data_pipeline_example adai_nlp)
    
    # Dataset Batch Processing Example
    add_executable(dataset_batch_processing_example ../examples/DatasetBatchProcessingExample.cpp)
    target_link_libraries(dataset_batch_processing_example adai_nlp)
    
    # GPU Acceleration Example (always build, shows status even without GPU)
    add_executable(gpu_example ../examples/GPUExample.cpp)
    target_link_libraries(gpu_example adai_core)
    
    # RAG (Retrieval-Augmented Generation) Example
    add_executable(rag_example ../examples/RAGExample.cpp DocumentStore.cpp RAGInference.cpp)
    target_link_libraries(rag_example adai_models adai_nlp)
    message(STATUS "Building rag_example - Retrieval-Augmented Generation demo")
    
    # Training Metrics Service Example
    add_executable(training_metrics_example ../examples/TrainingMetricsServiceExample.cpp
        TrainingMetricsService.cpp IncrementalTrainer.cpp ChatbotTrainer.cpp Config.cpp
        ModelNameClient.cpp)
    target_link_libraries(training_metrics_example
        adai_models
        adai_nlp
        adai_core
        pthread
        $<$<CXX_COMPILER_ID:GNU>:stdc++fs>  # GCC/libstdc++ only; not available on macOS (libc++)
    )
    if(HTTPLIB_INCLUDE_DIR)
        target_include_directories(training_metrics_example PRIVATE ${HTTPLIB_INCLUDE_DIR})
    endif()
    message(STATUS "Building training_metrics_example - Real-time metrics monitoring demo")
    
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
