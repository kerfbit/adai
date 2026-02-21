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
target_compile_definitions(chatbottrainerTests PRIVATE CHATBOT_TRAINER_TEST_BUILD)
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
)
target_link_libraries(chatbotcliImprovedTests 
    adai_models 
    adai_nlp 
    GTest::gtest_main
)
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
add_executable(ragBertComparisonTests rag_bert_comparison_test.cpp)
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

# ============================================================================
# Test Summary
# ============================================================================

message(STATUS "Configured 28 test suites using library targets (including ChatbotAPI, ChatbotTrainer, integration, RAG/BERT comparison and Phase 5 tests)")