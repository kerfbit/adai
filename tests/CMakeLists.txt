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

message(STATUS "Configured 52 test suites using library targets (including Config, Logger, IncrementalTrainer, LLMEncoder, DocumentStore, RAGInference, BatchProcessor, ParallelDataLoader, IntegratedInferenceEngine, PipelineInferenceEngine, BatchedInferenceEngine, SpeculativeDecoding, VocabBuilder, ChatbotAPI, ChatbotTrainer, GenerationQuality, GenerationQualityAsync, PaddingEfficiency, AdaptiveClipping, ModelSerializer, StaleDetection, BackgroundLaunch, integration, RAG/BERT comparison, Phase 5, DatasetRegistry and DataFetcher tests)")