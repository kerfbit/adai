cmake_minimum_required(VERSION 3.10)

enable_testing()

# Google Test is now fetched via FetchContent in root CMakeLists.txt
# Use GTest::gtest_main and GTest::gtest targets
include_directories(${CMAKE_SOURCE_DIR}/src)

# ============================================================================
# Core Component Tests - Now using libraries
# ============================================================================

# Tokenizer tests
add_executable(runTests tokenizer_test.cpp)
target_link_libraries(runTests adai_nlp GTest::gtest_main)
add_test(NAME TokenizerTests COMMAND runTests)

# Tokenizer error handling tests (TD-002)
add_executable(tokenizerErrorHandlingTests tokenizer_error_handling_test.cpp)
target_link_libraries(tokenizerErrorHandlingTests adai_nlp GTest::gtest_main)
add_test(NAME TokenizerErrorHandlingTests COMMAND tokenizerErrorHandlingTests)

# Matrix tests
add_executable(matrixTests matrix_test.cpp)
target_link_libraries(matrixTests adai_core GTest::gtest_main)
add_test(NAME MatrixTests COMMAND matrixTests)

# Matrix SIMD tests (TD-007)
add_executable(matrixSIMDTests matrix_simd_test.cpp)
target_link_libraries(matrixSIMDTests adai_core GTest::gtest_main)
add_test(NAME MatrixSIMDTests COMMAND matrixSIMDTests)

# Quantization tests (TD-075 — ASYMMETRIC_INT8 int8_t storage/sign-extension fix)
add_executable(quantizationTests quantization_test.cpp)
target_link_libraries(quantizationTests adai_core GTest::gtest_main)
add_test(NAME QuantizationTests COMMAND quantizationTests)

# Activation tests
add_executable(activationTests activation_test.cpp)
target_link_libraries(activationTests adai_core GTest::gtest_main)
add_test(NAME ActivationTests COMMAND activationTests)

# Optimizer tests
add_executable(optimizerTests optimizer_test.cpp)
target_link_libraries(optimizerTests adai_core GTest::gtest_main)
add_test(NAME OptimizerTests COMMAND optimizerTests)

# ============================================================================
# Layer Tests
# ============================================================================

# LayerNorm tests
add_executable(layernormTests layernorm_test.cpp)
target_link_libraries(layernormTests adai_layers GTest::gtest_main)
add_test(NAME LayerNormTests COMMAND layernormTests)

# PositionalEncoding tests
add_executable(positionalencodingTests positionalencoding_test.cpp)
target_link_libraries(positionalencodingTests adai_layers GTest::gtest_main)
add_test(NAME PositionalEncodingTests COMMAND positionalencodingTests)

# SpecialTokens tests (header-only; no adai lib dependency)
add_executable(specialtokensTests specialtokens_test.cpp)
target_link_libraries(specialtokensTests GTest::gtest_main)
add_test(NAME SpecialTokensTests COMMAND specialtokensTests)

# TokenEmbedding tests
add_executable(tokenembeddingTests tokenembedding_test.cpp)
target_link_libraries(tokenembeddingTests adai_layers GTest::gtest_main)
add_test(NAME TokenEmbeddingTests COMMAND tokenembeddingTests)

# ============================================================================
# Attention Tests
# ============================================================================

# MultiHeadAttention tests
add_executable(multiheadattentionTests multiheadattention_test.cpp)
target_link_libraries(multiheadattentionTests adai_attention GTest::gtest_main)
add_test(NAME MultiHeadAttentionTests COMMAND multiheadattentionTests)

# CrossAttention tests
add_executable(crossattentionTests crossattention_test.cpp)
target_link_libraries(crossattentionTests adai_attention GTest::gtest_main)
add_test(NAME CrossAttentionTests COMMAND crossattentionTests)

# ============================================================================
# Feed-Forward Tests
# ============================================================================

# FeedForward tests
add_executable(feedforwardTests feedforward_test.cpp)
target_link_libraries(feedforwardTests adai_feedforward GTest::gtest_main)
add_test(NAME FeedForwardTests COMMAND feedforwardTests)

# ============================================================================
# Transformer Block Tests
# ============================================================================

# EncoderBlock tests
add_executable(encoderblockTests encoderblock_test.cpp)
target_link_libraries(encoderblockTests adai_transformer GTest::gtest_main)
add_test(NAME EncoderBlockTests COMMAND encoderblockTests)

# DecoderBlock tests
add_executable(decoderblockTests decoderblock_test.cpp)
target_link_libraries(decoderblockTests adai_transformer GTest::gtest_main)
add_test(NAME DecoderBlockTests COMMAND decoderblockTests)

# ============================================================================
# Model Tests
# ============================================================================

# LanguageModelHead tests
add_executable(languagemodelheadTests languagemodelhead_test.cpp)
target_link_libraries(languagemodelheadTests adai_models GTest::gtest_main)
add_test(NAME LanguageModelHeadTests COMMAND languagemodelheadTests)

# Decoder tests
add_executable(decoderTests decoder_test.cpp)
target_link_libraries(decoderTests adai_models GTest::gtest_main)
add_test(NAME DecoderTests COMMAND decoderTests)

# EncoderDecoderModel tests
add_executable(encoderdecoderTests encoderdecoder_test.cpp)
target_link_libraries(encoderdecoderTests adai_models adai_nlp GTest::gtest_main)
add_test(NAME EncoderDecoderTests COMMAND encoderdecoderTests)

# LLMEncoder tests
add_executable(llmencoderTests 
    llmencoder_test.cpp
    ../src/LLMEncoder.cpp  # Include implementation
)
target_link_libraries(llmencoderTests 
    adai_models
    adai_nlp
    adai_transformer
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    GTest::gtest_main
)
add_test(NAME LLMEncoderTests COMMAND llmencoderTests)

# ============================================================================
# NLP Component Tests
# ============================================================================

# TextGenerator tests
add_executable(textgeneratorTests textgenerator_test.cpp)
target_link_libraries(textgeneratorTests adai_nlp GTest::gtest_main)
add_test(NAME TextGeneratorTests COMMAND textgeneratorTests)

# ConversationContext tests
add_executable(conversationcontextTests conversationcontext_test.cpp)
target_link_libraries(conversationcontextTests adai_nlp GTest::gtest_main)
add_test(NAME ConversationContextTests COMMAND conversationcontextTests)

# Inference Optimization tests (KV Cache, Batch Processing, Profiling)
add_executable(inferenceOptimizationTests inference_optimization_test.cpp)
target_link_libraries(inferenceOptimizationTests adai_models adai_nlp GTest::gtest_main)
add_test(NAME InferenceOptimizationTests COMMAND inferenceOptimizationTests)

# ============================================================================
# Application Tests
# ============================================================================

# ChatbotTrainer tests (comprehensive unit tests with header support)
add_executable(chatbottrainerTests 
    chatbottrainer_test.cpp
    ../src/ChatbotTrainer.cpp  # Include implementation
)
target_link_libraries(chatbottrainerTests
    adai_models
    adai_nlp
    adai_transformer
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    GTest::gtest_main
)
add_test(NAME ChatbotTrainerTests COMMAND chatbottrainerTests)

# ChatbotCLI tests (unit tests for command parsing and helper functions)
add_executable(chatbotcliTests chatbotcli_test.cpp)
target_link_libraries(chatbotcliTests GTest::gtest_main)
add_test(NAME ChatbotCLITests COMMAND chatbotcliTests)

# ChatbotCLI improved tests (full unit tests with header support)
add_executable(chatbotcliImprovedTests 
    chatbotcli_improved_test.cpp
    ../src/ChatbotCLI.cpp  # Include implementation
    ../src/Config.cpp
)
target_link_libraries(chatbotcliImprovedTests 
    adai_models 
    adai_nlp 
    GTest::gtest_main
)
if(HTTPLIB_INCLUDE_DIR)
    target_include_directories(chatbotcliImprovedTests PRIVATE ${HTTPLIB_INCLUDE_DIR})
endif()
add_test(NAME ChatbotCLIImprovedTests COMMAND chatbotcliImprovedTests)

# Dataset tests (data loading and splitting)
add_executable(datasetTests dataset_test.cpp)
target_link_libraries(datasetTests GTest::gtest_main)
add_test(NAME DatasetTests COMMAND datasetTests)

# MetricsTracker tests (training metrics tracking)
add_executable(metricstrackerTests metricstracker_test.cpp)
target_link_libraries(metricstrackerTests GTest::gtest_main)
add_test(NAME MetricsTrackerTests COMMAND metricstrackerTests)

# CheckpointManager tests (model checkpoint management)
add_executable(checkpointmanagerTests checkpointmanager_test.cpp)
target_link_libraries(checkpointmanagerTests GTest::gtest_main)
add_test(NAME CheckpointManagerTests COMMAND checkpointmanagerTests)

# ============================================================================
# Service Infrastructure Tests (Daemon Service - TD Future Enhancements)
# ============================================================================

# Config tests (Configuration loading, validation, hot-reload)
add_executable(configTests 
    config_test.cpp
    ../src/Config.cpp
    ../src/Logger.cpp
)
target_link_libraries(configTests 
    spdlog::spdlog
    GTest::gtest_main
)
add_test(NAME ConfigTests COMMAND configTests)

# Logger tests (Structured logging, file rotation, level filtering)
add_executable(loggerTests 
    logger_test.cpp
    ../src/Logger.cpp
)
target_link_libraries(loggerTests 
    spdlog::spdlog
    GTest::gtest_main
)
add_test(NAME LoggerTests COMMAND loggerTests)

# IncrementalTrainer tests (Session management, checkpoints, data registry)
add_executable(incrementaltrainerTests 
    incrementaltrainer_test.cpp
    ../src/IncrementalTrainer.cpp
    ../src/ChatbotTrainer.cpp
    ../src/Logger.cpp
    ../src/Config.cpp
)
target_link_libraries(incrementaltrainerTests 
    adai_models
    adai_nlp
    adai_transformer
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    spdlog::spdlog
    GTest::gtest_main
)
if(HTTPLIB_INCLUDE_DIR)
    target_link_libraries(incrementaltrainerTests adai_metrics_api)
    target_include_directories(incrementaltrainerTests PRIVATE ${HTTPLIB_INCLUDE_DIR})
endif()
add_test(NAME IncrementalTrainerTests COMMAND incrementaltrainerTests)

# ============================================================================
# Phase 5 Advanced Features Tests
# ============================================================================

# Phase 5 tests (RLHF, LoRA, Quantization, Speculative Decoding)
add_executable(phase5Tests phase5_test.cpp)
target_link_libraries(phase5Tests adai_core GTest::gtest_main)
add_test(NAME Phase5Tests COMMAND phase5Tests)

# Data Pipeline tests (Efficient Batching, Parallel Data Loading)
add_executable(datapipelineTests datapipeline_test.cpp)
target_link_libraries(datapipelineTests adai_nlp GTest::gtest_main)
add_test(NAME DataPipelineTests COMMAND datapipelineTests)

# ChatbotAPI tests (REST API with batch processing)
add_executable(chatbotapiTests chatbotapi_test.cpp)
target_link_libraries(chatbotapiTests 
    adai_api
    adai_models 
    adai_nlp 
    adai_transformer 
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    GTest::gtest_main
)
add_test(NAME ChatbotAPITests COMMAND chatbotapiTests)

# ParallelDataLoader tests (multi-threaded data loading with prefetching)
add_executable(paralleldataloaderTests paralleldataloader_test.cpp)
target_link_libraries(paralleldataloaderTests adai_nlp GTest::gtest_main)
add_test(NAME ParallelDataLoaderTests COMMAND paralleldataloaderTests)

# IntegratedInferenceEngine tests (combined batching + pipeline + OpenMP + attention parallelism)
add_executable(integratedinferenceengineTests integratedinferenceengine_test.cpp)
target_link_libraries(integratedinferenceengineTests
    adai_models
    adai_nlp
    adai_transformer
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    GTest::gtest_main
)
add_test(NAME IntegratedInferenceEngineTests COMMAND integratedinferenceengineTests)

# PipelineInferenceEngine tests (pipeline parallelism, ThreadSafeQueue, mock encoder/decoder)
add_executable(pipelineinferenceengineTests pipelineinferenceengine_test.cpp)
target_link_libraries(pipelineinferenceengineTests
    adai_models
    adai_nlp
    adai_transformer
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    GTest::gtest_main
)
add_test(NAME PipelineInferenceEngineTests COMMAND pipelineinferenceengineTests)

# BatchedInferenceEngine tests (continuous batching, request queuing, stats)
add_executable(batchedinferenceengineTests batchedinferenceengine_test.cpp)
target_link_libraries(batchedinferenceengineTests
    adai_models
    adai_nlp
    adai_transformer
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    GTest::gtest_main
)
add_test(NAME BatchedInferenceEngineTests COMMAND batchedinferenceengineTests)

# SpeculativeDecoding tests (draft/target pair, acceptance rate, stats)
add_executable(speculativedecodingTests speculativedecoding_test.cpp)
target_link_libraries(speculativedecodingTests
    adai_models
    adai_nlp
    adai_transformer
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    GTest::gtest_main
)
add_test(NAME SpeculativeDecodingTests COMMAND speculativedecodingTests)

# VocabBuilder tests (file-loading helpers + BPETokenizer build workflow)
add_executable(vocabbuilderTests vocabbuilder_test.cpp)
target_link_libraries(vocabbuilderTests
    adai_nlp
    adai_core
    GTest::gtest_main
)
add_test(NAME VocabBuilderTests COMMAND vocabbuilderTests)

# ============================================================================
# Integration Tests
# ============================================================================

# Integration tests - End-to-end workflows
add_executable(integrationTests integration_test.cpp)
target_link_libraries(integrationTests 
    adai_models 
    adai_nlp 
    adai_transformer 
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    GTest::gtest_main
)
add_test(NAME IntegrationTests COMMAND integrationTests)

# RAG vs BERT Comparison Test
add_executable(ragBertComparisonTests
    rag_bert_comparison_test.cpp
    ../src/RAGInference.cpp   # Include RAG inference implementation
    ../src/DocumentStore.cpp  # Include document store implementation
)
target_link_libraries(ragBertComparisonTests 
    adai_models 
    adai_nlp 
    adai_transformer
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    GTest::gtest_main
)
add_test(NAME RAGBERTComparisonTests COMMAND ragBertComparisonTests)

# DocumentStore tests (RAG component)
add_executable(documentstoreTests 
    documentstore_test.cpp
    ../src/DocumentStore.cpp  # Include implementation
)
target_link_libraries(documentstoreTests 
    adai_models
    adai_nlp
    adai_transformer
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    GTest::gtest_main
)
add_test(NAME DocumentStoreTests COMMAND documentstoreTests)

# RAGInference tests (RAG pipeline)
add_executable(raginferenceTests 
    raginference_test.cpp
    ../src/RAGInference.cpp  # Include implementation
    ../src/DocumentStore.cpp  # DocumentStore dependency
)
target_link_libraries(raginferenceTests 
    adai_models
    adai_nlp
    adai_transformer
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    GTest::gtest_main
)
add_test(NAME RAGInferenceTests COMMAND raginferenceTests)

# BatchProcessor tests (batch processing utilities)
add_executable(batchprocessorTests 
    batchprocessor_test.cpp
)
target_link_libraries(batchprocessorTests 
    adai_core
    GTest::gtest_main
)
add_test(NAME BatchProcessorTests COMMAND batchprocessorTests)

# ============================================================================
# Test Summary
# ============================================================================

# Generation quality metrics tests (BLEU/ROUGE)
add_executable(generationQualityTests generation_quality_test.cpp)
target_include_directories(generationQualityTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(generationQualityTests
    adai_core
    GTest::gtest_main
)
add_test(NAME GenerationQualityTests COMMAND generationQualityTests)

# TD-023: Async generation quality scoring tests
add_executable(generationQualityAsyncTests
    generation_quality_async_test.cpp
    ../src/ChatbotTrainer.cpp
)
target_link_libraries(generationQualityAsyncTests
    adai_models
    adai_nlp
    adai_transformer
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    GTest::gtest_main
)
add_test(NAME GenerationQualityAsyncTests COMMAND generationQualityAsyncTests)

# Batch padding efficiency tests (TrainingMetricsService + pure arithmetic helpers)
add_executable(paddingEfficiencyTests padding_efficiency_test.cpp)
target_include_directories(paddingEfficiencyTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(paddingEfficiencyTests
    adai_core
    GTest::gtest_main
)
add_test(NAME PaddingEfficiencyTests COMMAND paddingEfficiencyTests)

# TrainingMetricsService resume-state regression tests
add_executable(trainingMetricsServiceResumeTests training_metrics_service_resume_test.cpp)
target_include_directories(trainingMetricsServiceResumeTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(trainingMetricsServiceResumeTests
    adai_core
    GTest::gtest_main
)
add_test(NAME TrainingMetricsServiceResumeTests COMMAND trainingMetricsServiceResumeTests)

# TD-018 phase 2: MetricsSessionRegistry tests
add_executable(metricsSessionRegistryTests metrics_session_registry_test.cpp)
target_include_directories(metricsSessionRegistryTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(metricsSessionRegistryTests
    adai_core
    GTest::gtest_main
)
add_test(NAME MetricsSessionRegistryTests COMMAND metricsSessionRegistryTests)

# TD-020: Metrics Database Persistence tests
add_executable(metricsDatabaseTests MetricsDatabaseTest.cpp)
target_include_directories(metricsDatabaseTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(metricsDatabaseTests
    adai_core
    GTest::gtest_main
)
if(SQLITE3_FOUND)
    target_include_directories(metricsDatabaseTests PRIVATE ${SQLITE3_INCLUDE_DIR})
endif()
add_test(NAME MetricsDatabaseTests COMMAND metricsDatabaseTests)

# TD-042: PostgresMetricsDatabase test coverage, parameterizing the same IMetricsDatabase-level
# behavior MetricsDatabaseTest.cpp covers for SQLite against a real local Postgres instance
# (initdb/pg_ctl'd by the test itself — see postgres_metrics_database_test.cpp). Only buildable
# when ENABLE_POSTGRES_METRICS links PostgresMetricsDatabase.cpp/libpq into adai_core at all
# (PostgreSQL::PostgreSQL and ADAI_ENABLE_POSTGRES flow in transitively via that PUBLIC link, per
# src/CMakeLists.txt — no separate include/link needed here).
if(ENABLE_POSTGRES_METRICS)
    add_executable(postgresMetricsDatabaseTests postgres_metrics_database_test.cpp)
    target_include_directories(postgresMetricsDatabaseTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(postgresMetricsDatabaseTests
        adai_core
        GTest::gtest_main
    )
    add_test(NAME PostgresMetricsDatabaseTests COMMAND postgresMetricsDatabaseTests)
    # Real initdb + 17 CREATE DATABASE/DROP DATABASE round trips is slower than the in-process
    # SQLite suite; keep it well under the blanket 1200s but give it real headroom.
    set_tests_properties(PostgresMetricsDatabaseTests PROPERTIES TIMEOUT 300)
endif()

# Daemon admin config persistence (metrics_api_server / mns_server / registry_server)
add_executable(daemonConfigStoreTests daemonconfigstore_test.cpp)
target_include_directories(daemonConfigStoreTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(daemonConfigStoreTests
    adai_core
    GTest::gtest_main
)
if(SQLITE3_FOUND)
    target_include_directories(daemonConfigStoreTests PRIVATE ${SQLITE3_INCLUDE_DIR})
endif()
add_test(NAME DaemonConfigStoreTests COMMAND daemonConfigStoreTests)

# Native Parquet -> JSONL reader (replaces the Python pandas/pyarrow subprocess)
add_executable(parquetReaderTests parquet_reader_test.cpp)
target_include_directories(parquetReaderTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(parquetReaderTests
    adai_core
    GTest::gtest_main
)
add_test(NAME ParquetReaderTests COMMAND parquetReaderTests)

# TD-021: IncrementalTrainer × Metrics Service Decoupling tests
add_executable(incrementalTrainerDecouplingTests
    incremental_trainer_registry_test.cpp
    ../src/IncrementalTrainer.cpp
    ../src/ChatbotTrainer.cpp
    ../src/Logger.cpp
    ../src/Config.cpp
)
target_link_libraries(incrementalTrainerDecouplingTests
    adai_models
    adai_nlp
    adai_transformer
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    spdlog::spdlog
    GTest::gtest_main
)
if(HTTPLIB_INCLUDE_DIR)
    target_compile_definitions(incrementalTrainerDecouplingTests PRIVATE BUILD_METRICS_API_SERVER)
    target_link_libraries(incrementalTrainerDecouplingTests adai_metrics_api)
    target_include_directories(incrementalTrainerDecouplingTests PRIVATE ${HTTPLIB_INCLUDE_DIR})
endif()
add_test(NAME IncrementalTrainerDecouplingTests COMMAND incrementalTrainerDecouplingTests)

# TD-021: MetricsPushClient unit tests
add_executable(metricsPushClientTests
    metrics_push_client_test.cpp
)
target_include_directories(metricsPushClientTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(metricsPushClientTests
    adai_core
    GTest::gtest_main
)
if(HTTPLIB_INCLUDE_DIR)
    target_compile_definitions(metricsPushClientTests PRIVATE BUILD_METRICS_API_SERVER)
    target_include_directories(metricsPushClientTests PRIVATE ${HTTPLIB_INCLUDE_DIR})
endif()
add_test(NAME MetricsPushClientTests COMMAND metricsPushClientTests)

# TD-018 phase 3: TrainingMetricsAPI session routing + compatibility aliases
if(TARGET adai_metrics_api)
    add_executable(trainingMetricsApiRoutesTests training_metrics_api_routes_test.cpp)
    target_include_directories(trainingMetricsApiRoutesTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(trainingMetricsApiRoutesTests
        adai_metrics_api
        adai_core
        GTest::gtest_main
    )
    if(HTTPLIB_INCLUDE_DIR)
        target_include_directories(trainingMetricsApiRoutesTests PRIVATE ${HTTPLIB_INCLUDE_DIR})
    endif()
    add_test(NAME TrainingMetricsAPIRoutesTests COMMAND trainingMetricsApiRoutesTests)

    # Live-server integration tests — connect to a running metrics_api_server.
    # Override target with: METRICS_SERVER_HOST=<host> METRICS_SERVER_PORT=<port>
    # Tests are skipped automatically when the server is unreachable.
    add_executable(trainingMetricsApiLiveTests training_metrics_api_live_test.cpp)
    target_include_directories(trainingMetricsApiLiveTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(trainingMetricsApiLiveTests
        adai_core
        GTest::gtest_main
    )
    if(HTTPLIB_INCLUDE_DIR)
        target_include_directories(trainingMetricsApiLiveTests PRIVATE ${HTTPLIB_INCLUDE_DIR})
    endif()
    add_test(NAME TrainingMetricsAPILiveTests COMMAND trainingMetricsApiLiveTests)

    # Live-server integration tests for registry_server (TD-028 Phase 9).
    # Override target with: REGISTRY_SERVER_HOST=<host> REGISTRY_SERVER_PORT=<port>
    # Tests are skipped automatically when the server is unreachable.
    add_executable(datasetRegistryLiveTests dataset_registry_live_test.cpp)
    target_include_directories(datasetRegistryLiveTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(datasetRegistryLiveTests
        adai_core
        GTest::gtest_main
    )
    if(HTTPLIB_INCLUDE_DIR)
        target_include_directories(datasetRegistryLiveTests PRIVATE ${HTTPLIB_INCLUDE_DIR})
    endif()
    add_test(NAME DatasetRegistryLiveTests COMMAND datasetRegistryLiveTests)
endif()

# TD-040: permanent regression test for the FTP-token path-confinement fix.
# Unlike datasetRegistryLiveTests above (connects to an externally-managed
# instance, skips if unreachable), this spawns its own registry_server
# subprocess with a --data-dir it fully controls, so it runs unattended in
# any environment and can deliberately construct an out-of-tree pending
# entry. Needs the actual registry_server executable (not just adai_core) —
# gated on that target existing, independent of the adai_metrics_api block
# above.
if(TARGET registry_server AND HTTPLIB_INCLUDE_DIR)
    add_executable(registryFtpConfinementTests registry_ftp_confinement_test.cpp)
    target_include_directories(registryFtpConfinementTests PRIVATE
        ${CMAKE_SOURCE_DIR}/src ${HTTPLIB_INCLUDE_DIR})
    target_compile_definitions(registryFtpConfinementTests PRIVATE
        REGISTRY_SERVER_BINARY_PATH="$<TARGET_FILE:registry_server>")
    target_link_libraries(registryFtpConfinementTests
        adai_core
        pthread
        GTest::gtest_main
    )
    add_dependencies(registryFtpConfinementTests registry_server)
    add_test(NAME RegistryFtpConfinementTests COMMAND registryFtpConfinementTests)
endif()

# Model Name Service live-server integration tests.
# Override target with: MNS_SERVER_HOST=<host> MNS_SERVER_PORT=<port>
# Tests auto-skip when mns_server is unreachable.
if(TARGET adai_mns)
    add_executable(mnsLiveTests model_name_service_live_test.cpp)
    target_include_directories(mnsLiveTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(mnsLiveTests
        adai_core
        GTest::gtest_main
    )
    if(HTTPLIB_INCLUDE_DIR)
        target_include_directories(mnsLiveTests PRIVATE ${HTTPLIB_INCLUDE_DIR})
    endif()
    add_test(NAME MNSLiveTests COMMAND mnsLiveTests)

    # MNS Manager GUI unit + integration tests.
    # Part 1 (json helpers, URL parser) always runs; Part 2 auto-skips without a server.
    if(HTTPLIB_INCLUDE_DIR)
        add_executable(mnsManagerGuiTests mns_manager_gui_test.cpp)
        target_include_directories(mnsManagerGuiTests PRIVATE
            ${CMAKE_SOURCE_DIR}/src ${HTTPLIB_INCLUDE_DIR})
        target_link_libraries(mnsManagerGuiTests
            adai_core
            GTest::gtest_main
        )
        add_test(NAME MnsManagerGuiTests COMMAND mnsManagerGuiTests)
    endif()

    # ModelNameClient unit tests — spins up a real httplib::Server standing in
    # for mns_server and verifies both request construction and response parsing.
    add_executable(modelnameclientTests modelnameclient_test.cpp)
    target_include_directories(modelnameclientTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(modelnameclientTests
        adai_mns
        adai_core
        GTest::gtest_main
    )
    if(HTTPLIB_INCLUDE_DIR)
        target_include_directories(modelnameclientTests PRIVATE ${HTTPLIB_INCLUDE_DIR})
    endif()
    add_test(NAME ModelNameClientTests COMMAND modelnameclientTests)
endif()

# TD-017: Adaptive gradient clipping tests (TrainingMetricsService)
add_executable(adaptiveClippingTests adaptive_clipping_test.cpp)
target_include_directories(adaptiveClippingTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(adaptiveClippingTests
    adai_core
    GTest::gtest_main
)
add_test(NAME AdaptiveClippingTests COMMAND adaptiveClippingTests)

# HuggingFace SafeTensors import/export tests
add_executable(modelSerializerTests model_serializer_test.cpp)
target_include_directories(modelSerializerTests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external
)
target_link_libraries(modelSerializerTests
    adai_models
    adai_nlp
    adai_transformer
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    GTest::gtest_main
)
add_test(NAME ModelSerializerTests COMMAND modelSerializerTests)

# TD-019: Stale metrics detection tests (TrainingMetricsService staleness logic)
add_executable(staleDetectionTests stale_detection_test.cpp)
target_include_directories(staleDetectionTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(staleDetectionTests
    adai_core
    GTest::gtest_main
)
add_test(NAME StaleDetectionTests COMMAND staleDetectionTests)

# TD-025: IncrementalTrainer Background Launch with PID Message
add_executable(incrementalTrainerBackgroundTests
    incremental_trainer_background_test.cpp
)
target_include_directories(incrementalTrainerBackgroundTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(incrementalTrainerBackgroundTests
    GTest::gtest_main
)
add_test(NAME IncrementalTrainerBackgroundTests COMMAND incrementalTrainerBackgroundTests)

# TD-028 Phase 8: LocalTransport unit tests
add_executable(registryTransportTests registry_transport_test.cpp)
target_link_libraries(registryTransportTests
    adai_core
    GTest::gtest_main
)
add_test(NAME RegistryTransportTests COMMAND registryTransportTests)

# TD-028 Phase 9: LocalTransport acquire/release/commit_trained tests
add_executable(registryTransportPhase9Tests RegistryTransportTests.cpp)
target_link_libraries(registryTransportPhase9Tests
    adai_core
    GTest::gtest_main
)
add_test(NAME RegistryTransportPhase9Tests COMMAND registryTransportPhase9Tests)

# TD-028 Phase 10: DatasetRegistry unit tests
add_executable(datasetRegistryTests DatasetRegistryTests.cpp)
target_link_libraries(datasetRegistryTests
    adai_core
    GTest::gtest_main
)
add_test(NAME DatasetRegistryTests COMMAND datasetRegistryTests)

# TD-028 Phase 10: DataFetcher unit tests (offline only)
add_executable(dataFetcherTests DataFetcherTests.cpp)
target_link_libraries(dataFetcherTests
    adai_core
    GTest::gtest_main
)
add_test(NAME DataFetcherTests COMMAND dataFetcherTests)

# TrainingSampleMeta unit tests (SampleMeta, sample_to_jsonl, parse_jsonl_sample)
add_executable(trainingSampleMetaTests training_sample_meta_test.cpp)
target_include_directories(trainingSampleMetaTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(trainingSampleMetaTests GTest::gtest_main)
add_test(NAME TrainingSampleMetaTests COMMAND trainingSampleMetaTests)

# TD-028 Phase 10 (FTP transport): FtpDataServer unit tests
# Tests TokenStore, ftp_detail helpers, issue_token, AcquireResponse.
# Does NOT start the FTP server (no network/port binding).
# Phase 3: BUILD_FTPS enabled when OpenSSL is found so HMAC and rate-limit
# tests exercise the real crypto path.
find_package(OpenSSL QUIET)
add_executable(ftpDataServerTests FtpDataServerTests.cpp)
target_link_libraries(ftpDataServerTests
    adai_core
    GTest::gtest_main
)
if(OPENSSL_FOUND)
    target_compile_definitions(ftpDataServerTests PRIVATE BUILD_FTPS)
    target_link_libraries(ftpDataServerTests OpenSSL::SSL OpenSSL::Crypto)
    message(STATUS "FtpDataServerTests: BUILD_FTPS enabled (OpenSSL found)")
endif()
add_test(NAME FtpDataServerTests COMMAND ftpDataServerTests)

# TD-028 Phase 10 (FTP transport): DataTransport unit tests
# Compiled WITHOUT BUILD_FTP_TRANSPORT to exercise the stub implementations.
# Also validates FileToken and AcquireResponse struct defaults.
add_executable(dataTransportTests DataTransportTests.cpp)
target_link_libraries(dataTransportTests
    adai_core
    GTest::gtest_main
)
add_test(NAME DataTransportTests COMMAND dataTransportTests)

# TD-028 Phase 2 (FTP transport): DataTransport unit tests WITH libcurl.
# Tests Condition B (skip re-download for complete files), parallel fetch_all
# result ordering, and error propagation — all without a running FTP server.
# find_package result is cached so this is a no-op if src/CMakeLists already ran it.
find_package(CURL QUIET)
if(ENABLE_FTP_TRANSPORT AND CURL_FOUND)
    add_executable(dataTransportFtpTests DataTransportFtpTests.cpp)
    target_compile_definitions(dataTransportFtpTests PRIVATE BUILD_FTP_TRANSPORT)
    target_link_libraries(dataTransportFtpTests
        adai_core
        CURL::libcurl
        GTest::gtest_main
    )
    add_test(NAME DataTransportFtpTests COMMAND dataTransportFtpTests)
endif()

# TD-028 Phase 2 (startup sweep): startup_sweep() unit tests.
# Tests all four cleanup conditions (A, D, G, B/C) using a real
# DatasetRegistry backed by a temp directory — no FTP server needed.
add_executable(startupSweepTests StartupSweepTests.cpp)
target_link_libraries(startupSweepTests
    adai_core
    GTest::gtest_main
)
add_test(NAME StartupSweepTests COMMAND startupSweepTests)

# TD-003 GPU-native kernel tests (activation_saturation_ratio/attention_entropy
# GPU-path fix + CUDA TD-003 port). Only meaningful — and only compilable —
# when a GPU backend is configured; adai_gpu doesn't exist otherwise.
if(ENABLE_GPU OR ENABLE_SYCL)
    add_executable(matrixgpuTd003Tests matrixgpu_td003_test.cpp)
    target_link_libraries(matrixgpuTd003Tests adai_gpu GTest::gtest_main)
    add_test(NAME MatrixGPUTD003Tests COMMAND matrixgpuTd003Tests)
endif()

# ============================================================================
# Trainer admin control daemon (`incremental_trainer serve`) tests
# ============================================================================

# TrainerControlState — header-only, but its log ring buffer mirrors every entry
# through adai::Logger (see TrainerControlState::log()), so it now needs adai_core
# (Logger.cpp) + spdlog to link, unlike before that feature existed.
add_executable(trainerControlStateTests trainer_control_state_test.cpp)
target_include_directories(trainerControlStateTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(trainerControlStateTests
    adai_core
    spdlog::spdlog
    GTest::gtest_main
    pthread
)
add_test(NAME TrainerControlStateTests COMMAND trainerControlStateTests)

# TrainerAdminAPI — real httplib::Server exercised with a real httplib::Client,
# same style as RegistryServer/ModelNameService's own routes tests. Gated on
# HTTPLIB_INCLUDE_DIR like the daemon itself (see src/CMakeLists.txt's
# BUILD_TRAINER_ADMIN block) — nothing to test when it isn't built.
if(HTTPLIB_INCLUDE_DIR)
    add_executable(trainerAdminApiTests
        trainer_admin_api_test.cpp
        ../src/TrainerAdminAPI.cpp
    )
    target_include_directories(trainerAdminApiTests PRIVATE ${CMAKE_SOURCE_DIR}/src
        ${HTTPLIB_INCLUDE_DIR})
    target_link_libraries(trainerAdminApiTests
        adai_core
        spdlog::spdlog
        pthread
        GTest::gtest_main
    )
    add_test(NAME TrainerAdminAPITests COMMAND trainerAdminApiTests)
else()
    message(STATUS "TrainerAdminAPITests not built (cpp-httplib not found)")
endif()

# IncrementalTrainer × TrainerControlState integration — proves
# should_auto_save()/cleanup_old_sessions() actually prefer an attached
# control state's live-tunable values over the config-file ones, via the
# same friend-class pattern ChatbotTrainerCacheTest uses.
add_executable(incrementalTrainerControlTests
    incremental_trainer_control_test.cpp
    ../src/IncrementalTrainer.cpp
    ../src/ChatbotTrainer.cpp
    ../src/Logger.cpp
    ../src/Config.cpp
)
target_link_libraries(incrementalTrainerControlTests
    adai_models
    adai_nlp
    adai_transformer
    adai_attention
    adai_feedforward
    adai_layers
    adai_core
    spdlog::spdlog
    GTest::gtest_main
)
if(HTTPLIB_INCLUDE_DIR)
    target_link_libraries(incrementalTrainerControlTests adai_metrics_api)
    target_include_directories(incrementalTrainerControlTests PRIVATE ${HTTPLIB_INCLUDE_DIR})
endif()
add_test(NAME IncrementalTrainerControlTests COMMAND incrementalTrainerControlTests)

# Note: ChatbotTrainer's cooperative-abort-flag tests (was_aborted(), early
# epoch-loop exit on /admin/pause) live in chatbottrainer_test.cpp
# (ChatbotTrainerAbortTest fixture) — no separate target needed, it's part
# of the existing chatbottrainerTests binary above.
#
# There is deliberately no separate trainer_serve_loop_test.cpp: `serve`'s
# supervisory loop lives inline in IncrementalTrainingTool.cpp's main() with
# no extracted seam to call into directly, so its two real testable pieces —
# the idle-poll wake()/interruptible_sleep() primitive, and the
# IncrementalTrainer×TrainerControlState wiring it drives each iteration —
# are covered above (TrainerControlStateTests, IncrementalTrainerControlTests)
# instead of duplicating the loop's control flow against fabricated state.

# ============================================================================
# TD-043: scripts/ test suite
#
# A lightweight, dependency-free bash assertion harness (tests/scripts/
# harness.sh) plus one *_test.sh file per deployment-critical shell script,
# and two Python unittest suites for the two Python tools (check_file_status.py,
# gen_status_report.py) — neither bats-core nor pytest is installed or
# otherwise part of this project's toolchain, so this avoids introducing
# either as a new CI/dependency-file change for no benefit over plain bash/
# unittest here. Each test file is self-contained (locates its own target
# script via $CMAKE_SOURCE_DIR-relative path, never assumes a particular
# CWD) and safe to run as a normal, non-root ctest invocation: every
# install_*.sh/package_*.sh test either exercises argument-parsing/error
# paths that run before anything privileged or destructive, or — where a
# script supports it (install_oneapi_libs.sh's --dry-run, package_server_
# bundle.sh needing no root at all) — exercises its real happy path inside
# an isolated scratch directory.
set(ADAI_SCRIPTS_TEST_DIR "${CMAKE_SOURCE_DIR}/tests/scripts")
foreach(_script_test IN ITEMS
    build_windows_test.sh
    chatbot_gui_fixed_test.sh
    check_ports_test.sh
    check_tech_debt_test.sh
    docker_build_test.sh
    install_chatbot_api_test.sh
    install_cloudflared_test.sh
    install_incremental_trainer_test.sh
    install_metrics_service_test.sh
    install_mns_server_test.sh
    install_oneapi_libs_test.sh
    install_server_bundle_test.sh
    manual_test_reload_test.sh
    model_service_test.sh
    package_server_bundle_test.sh
    package_sycl_test.sh
    package_windows_test.sh
    run_chatbot_gui_test.sh
    run_chatbot_test.sh
    run_tests_test.sh
    test_chatbot_gui_comprehensive_test.sh
    test_chatbot_gui_test.sh
    test_config_reload_test.sh
    test_log_rotation_test.sh
    test_signal_handling_test.sh
    verify_cli_parallel_test.sh
    verify_gui_parallel_test.sh
)
    string(REPLACE ".sh" "" _test_name "${_script_test}")
    add_test(NAME ScriptsTests_${_test_name}
             COMMAND bash "${ADAI_SCRIPTS_TEST_DIR}/${_script_test}")
endforeach()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    # TD-045: same rationale as the TD-043 Python suites above — standard
    # library unittest only, run each test file directly rather than via
    # `-m unittest discover` so a single file's own name identifies it in
    # `ctest -R`.
    foreach(_py_test IN ITEMS
        test_check_file_status.py
        test_gen_status_report.py
        test_batch_api_client.py
        test_check_intel_driver_updates.py
        test_fix_markdown_lint.py
        test_monitor_training.py
        test_serve_dashboard.py
    )
        string(REPLACE "test_" "" _py_test_name "${_py_test}")
        string(REPLACE ".py" "" _py_test_name "${_py_test_name}")
        add_test(NAME ScriptsTests_${_py_test_name}
                 COMMAND ${Python3_EXECUTABLE} "${ADAI_SCRIPTS_TEST_DIR}/${_py_test}")
    endforeach()
endif()

# ============================================================================
# TD-036: thin main() wrapper smoke tests
#
# ChatbotCLI_main.cpp, ChatbotGUI_main.cpp, ChatbotGUI_wrapper.cpp, and
# MnsManagerGUI_main.cpp are argv-parsing shims around already-tested library
# classes — a GTest unit test isn't a good fit for a main(), so each gets a
# scripted smoke test (same harness.sh as the TD-043 scripts/ suite above)
# invoking the real built binary with --help/-h and checking exit code +
# output, the same category as the manual QA scripts in scripts/, just
# automated. The two Qt-GUI ones are gated on the binary's own CMake target
# existing rather than re-deriving BUILD_GUI/Qt5_FOUND/Qt6_FOUND here, so
# they automatically no-op (rather than fail) on a non-GUI configure.
set(ADAI_SMOKE_TEST_DIR "${CMAKE_SOURCE_DIR}/tests/smoke")
add_test(NAME SmokeTests_chatbot_main
         COMMAND bash "${ADAI_SMOKE_TEST_DIR}/chatbot_main_test.sh")
if(TARGET chatbot_gui_binary)
    add_test(NAME SmokeTests_chatbot_gui_binary_main
             COMMAND bash "${ADAI_SMOKE_TEST_DIR}/chatbot_gui_binary_main_test.sh")
endif()
if(TARGET chatbot_gui)
    add_test(NAME SmokeTests_chatbot_gui_wrapper
             COMMAND bash "${ADAI_SMOKE_TEST_DIR}/chatbot_gui_wrapper_test.sh")
endif()
if(TARGET mns_manager_gui)
    add_test(NAME SmokeTests_mns_manager_gui_main
             COMMAND bash "${ADAI_SMOKE_TEST_DIR}/mns_manager_gui_main_test.sh")
endif()

# TD-064: no test in this file had a TIMEOUT property, so ctest had no way to
# bound a hung test — confirmed the hard way when paralleldataloaderTests hung
# for ~9 hours under a full `ctest -j8` run (both its threads genuinely
# blocked in futex_wait_queue per /proc, 0% CPU — a real, if not yet
# root-caused, intermittent hang, not a busy-spin). 1200s (20 min) is chosen
# with headroom above the slowest legitimately-long suites observed in this
# same run (EncoderDecoderTests ~623s, IncrementalTrainerTests ~531s,
# RAGBERTComparisonTests ~508s), so a genuine hang now fails loudly within a
# bounded time instead of blocking a CI run indefinitely. Set last so it
# applies to every test already registered above.
get_property(_adai_all_tests DIRECTORY PROPERTY TESTS)
if(_adai_all_tests)
    set_tests_properties(${_adai_all_tests} PROPERTIES TIMEOUT 1200)
endif()

message(STATUS "Configured 57+ test suites using library targets (including Config, Logger, IncrementalTrainer, LLMEncoder, DocumentStore, RAGInference, BatchProcessor, ParallelDataLoader, IntegratedInferenceEngine, PipelineInferenceEngine, BatchedInferenceEngine, SpeculativeDecoding, VocabBuilder, ChatbotAPI, ChatbotTrainer, GenerationQuality, GenerationQualityAsync, PaddingEfficiency, AdaptiveClipping, ModelSerializer, StaleDetection, BackgroundLaunch, integration, RAG/BERT comparison, Phase 5, DatasetRegistry, DatasetRegistryLive, DataFetcher, FtpDataServer, DataTransport, DataTransportFtp, StartupSweep, TrainingSampleMeta, and TrainerAdminAPI tests)")