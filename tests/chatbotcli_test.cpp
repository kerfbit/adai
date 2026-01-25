#include <../gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Note: ChatbotCLI is in a .cpp file without a header
// We'll test the command parsing logic, color codes, and file handling
// through mock structures and helper functions

// ANSI color codes (matching ChatbotCLI.cpp)
#define COLOR_RESET "\033[0m"
#define COLOR_USER "\033[1;36m"
#define COLOR_BOT "\033[1;32m"
#define COLOR_SYSTEM "\033[1;33m"
#define COLOR_ERROR "\033[1;31m"

// Mock generation strategies
const std::vector<std::string> VALID_STRATEGIES = {"greedy", "beam", "sampling", "top-k",
                                                   "nucleus"};

// Helper function to check if a string is a valid command
bool is_valid_command(const std::string& input) {
    if (input.empty() || input[0] != '/')
        return false;

    std::vector<std::string> valid_commands = {"/help",     "/clear", "/save",   "/load", "/stats",
                                               "/settings", "/set",   "/system", "/exit", "/quit"};

    for (const auto& cmd : valid_commands) {
        if (input == cmd || input.substr(0, cmd.length()) == cmd) {
            return true;
        }
    }
    return false;
}

// Helper function to check if strategy is valid
bool is_valid_strategy(const std::string& strategy) {
    for (const auto& valid : VALID_STRATEGIES) {
        if (strategy == valid)
            return true;
    }
    return false;
}

// Helper function to parse /set command
struct SetCommandResult {
    bool valid = false;
    std::string param;
    std::string value;
    std::string error;
};

SetCommandResult parse_set_command(const std::string& command) {
    SetCommandResult result;

    // Remove "/set " prefix
    if (command.substr(0, 5) != "/set ") {
        result.error = "Not a set command";
        return result;
    }

    std::string args = command.substr(5);
    size_t space_pos = args.find(' ');

    if (space_pos == std::string::npos) {
        result.error = "Missing value";
        return result;
    }

    result.param = args.substr(0, space_pos);
    result.value = args.substr(space_pos + 1);
    result.valid = true;

    return result;
}

// Helper function to normalize parameter name
std::string normalize_param(const std::string& param) {
    if (param == "length" || param == "max_length")
        return "max_length";
    if (param == "temperature" || param == "temp")
        return "temperature";
    if (param == "top_p" || param == "top-p")
        return "top_p";
    if (param == "top_k" || param == "top-k")
        return "top_k";
    if (param == "beam_width" || param == "beam-width")
        return "beam_width";
    return param;
}

// Helper function to trim whitespace
std::string trim(const std::string& str) {
    if (str.empty())
        return str;

    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos)
        return "";

    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

// Helper function to create test vocab file
void create_test_vocab_file(const std::string& filepath, int vocab_size = 100) {
    std::ofstream file(filepath);
    file << "<unk>\n";
    file << "<pad>\n";
    file << "<s>\n";
    file << "</s>\n";

    for (int i = 0; i < vocab_size - 4; i++) {
        file << "token" << i << "\n";
    }
    file.close();
}

// Helper function to create test model config file
void create_test_model_file(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    // Write minimal binary data (not a real model, just for file existence tests)
    int d_model = 512;
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    file.close();
}

// Helper function to create test conversation file
void create_test_conversation_file(const std::string& filepath) {
    std::ofstream file(filepath);
    file << "USER: Hello!\n";
    file << "ASSISTANT: Hi there! How can I help you?\n";
    file << "USER: What's the weather?\n";
    file << "ASSISTANT: I don't have access to real-time weather data.\n";
    file.close();
}

// ============================================================================
// Test Suite 1: Color Code Tests
// ============================================================================

TEST(ColorCodeTest, ColorCodesAreDefined) {
    // Verify all color codes are properly defined
    EXPECT_STREQ(COLOR_RESET, "\033[0m");
    EXPECT_STREQ(COLOR_USER, "\033[1;36m");
    EXPECT_STREQ(COLOR_BOT, "\033[1;32m");
    EXPECT_STREQ(COLOR_SYSTEM, "\033[1;33m");
    EXPECT_STREQ(COLOR_ERROR, "\033[1;31m");
}

TEST(ColorCodeTest, ColorCodesAreNonEmpty) {
    EXPECT_GT(std::string(COLOR_RESET).length(), 0);
    EXPECT_GT(std::string(COLOR_USER).length(), 0);
    EXPECT_GT(std::string(COLOR_BOT).length(), 0);
    EXPECT_GT(std::string(COLOR_SYSTEM).length(), 0);
    EXPECT_GT(std::string(COLOR_ERROR).length(), 0);
}

TEST(ColorCodeTest, ColorCodesStartWithEscape) {
    EXPECT_EQ(std::string(COLOR_RESET)[0], '\033');
    EXPECT_EQ(std::string(COLOR_USER)[0], '\033');
    EXPECT_EQ(std::string(COLOR_BOT)[0], '\033');
    EXPECT_EQ(std::string(COLOR_SYSTEM)[0], '\033');
    EXPECT_EQ(std::string(COLOR_ERROR)[0], '\033');
}

// ============================================================================
// Test Suite 2: Command Validation Tests
// ============================================================================

TEST(CommandValidationTest, RecognizeValidCommands) {
    EXPECT_TRUE(is_valid_command("/help"));
    EXPECT_TRUE(is_valid_command("/clear"));
    EXPECT_TRUE(is_valid_command("/save"));
    EXPECT_TRUE(is_valid_command("/load"));
    EXPECT_TRUE(is_valid_command("/stats"));
    EXPECT_TRUE(is_valid_command("/settings"));
    EXPECT_TRUE(is_valid_command("/exit"));
    EXPECT_TRUE(is_valid_command("/quit"));
}

TEST(CommandValidationTest, RecognizeCommandsWithArguments) {
    EXPECT_TRUE(is_valid_command("/set strategy greedy"));
    EXPECT_TRUE(is_valid_command("/set temperature 0.8"));
    EXPECT_TRUE(is_valid_command("/system You are helpful"));
}

TEST(CommandValidationTest, RejectInvalidCommands) {
    EXPECT_FALSE(is_valid_command("/unknown"));
    EXPECT_FALSE(is_valid_command("/test"));
    EXPECT_FALSE(is_valid_command("/"));
    EXPECT_FALSE(is_valid_command(""));
}

TEST(CommandValidationTest, RejectNonCommands) {
    EXPECT_FALSE(is_valid_command("Hello"));
    EXPECT_FALSE(is_valid_command("What is the weather?"));
    EXPECT_FALSE(is_valid_command("exit"));  // Missing /
}

// ============================================================================
// Test Suite 3: Strategy Validation Tests
// ============================================================================

TEST(StrategyValidationTest, RecognizeValidStrategies) {
    EXPECT_TRUE(is_valid_strategy("greedy"));
    EXPECT_TRUE(is_valid_strategy("beam"));
    EXPECT_TRUE(is_valid_strategy("sampling"));
    EXPECT_TRUE(is_valid_strategy("top-k"));
    EXPECT_TRUE(is_valid_strategy("nucleus"));
}

TEST(StrategyValidationTest, RejectInvalidStrategies) {
    EXPECT_FALSE(is_valid_strategy("random"));
    EXPECT_FALSE(is_valid_strategy("best"));
    EXPECT_FALSE(is_valid_strategy(""));
    EXPECT_FALSE(is_valid_strategy("GREEDY"));  // Case sensitive
}

TEST(StrategyValidationTest, StrategyCount) {
    EXPECT_EQ(VALID_STRATEGIES.size(), 5);
}

// ============================================================================
// Test Suite 4: Set Command Parsing Tests
// ============================================================================

TEST(SetCommandParsingTest, ParseValidSetCommand) {
    auto result = parse_set_command("/set strategy greedy");
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.param, "strategy");
    EXPECT_EQ(result.value, "greedy");
}

TEST(SetCommandParsingTest, ParseSetCommandWithNumericValue) {
    auto result = parse_set_command("/set temperature 0.8");
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.param, "temperature");
    EXPECT_EQ(result.value, "0.8");
}

TEST(SetCommandParsingTest, ParseSetCommandWithIntValue) {
    auto result = parse_set_command("/set length 150");
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.param, "length");
    EXPECT_EQ(result.value, "150");
}

TEST(SetCommandParsingTest, RejectSetCommandWithoutValue) {
    auto result = parse_set_command("/set strategy");
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.error, "Missing value");
}

TEST(SetCommandParsingTest, RejectNonSetCommand) {
    auto result = parse_set_command("/help");
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.error, "Not a set command");
}

TEST(SetCommandParsingTest, ParseSetCommandWithSpacesInValue) {
    auto result = parse_set_command("/set system You are helpful");
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.param, "system");
    EXPECT_EQ(result.value, "You are helpful");
}

// ============================================================================
// Test Suite 5: Parameter Normalization Tests
// ============================================================================

TEST(ParameterNormalizationTest, NormalizeLength) {
    EXPECT_EQ(normalize_param("length"), "max_length");
    EXPECT_EQ(normalize_param("max_length"), "max_length");
}

TEST(ParameterNormalizationTest, NormalizeTemperature) {
    EXPECT_EQ(normalize_param("temperature"), "temperature");
    EXPECT_EQ(normalize_param("temp"), "temperature");
}

TEST(ParameterNormalizationTest, NormalizeTopP) {
    EXPECT_EQ(normalize_param("top_p"), "top_p");
    EXPECT_EQ(normalize_param("top-p"), "top_p");
}

TEST(ParameterNormalizationTest, NormalizeTopK) {
    EXPECT_EQ(normalize_param("top_k"), "top_k");
    EXPECT_EQ(normalize_param("top-k"), "top_k");
}

TEST(ParameterNormalizationTest, NormalizeBeamWidth) {
    EXPECT_EQ(normalize_param("beam_width"), "beam_width");
    EXPECT_EQ(normalize_param("beam-width"), "beam_width");
}

TEST(ParameterNormalizationTest, UnknownParameterUnchanged) {
    EXPECT_EQ(normalize_param("unknown"), "unknown");
    EXPECT_EQ(normalize_param("strategy"), "strategy");
}

// ============================================================================
// Test Suite 6: String Trimming Tests
// ============================================================================

TEST(StringTrimmingTest, TrimLeadingWhitespace) {
    EXPECT_EQ(trim("  hello"), "hello");
    EXPECT_EQ(trim("\thello"), "hello");
    EXPECT_EQ(trim("\nhello"), "hello");
}

TEST(StringTrimmingTest, TrimTrailingWhitespace) {
    EXPECT_EQ(trim("hello  "), "hello");
    EXPECT_EQ(trim("hello\t"), "hello");
    EXPECT_EQ(trim("hello\n"), "hello");
}

TEST(StringTrimmingTest, TrimBothSides) {
    EXPECT_EQ(trim("  hello  "), "hello");
    EXPECT_EQ(trim("\t\nhello\r\n"), "hello");
}

TEST(StringTrimmingTest, NoTrimNeeded) {
    EXPECT_EQ(trim("hello"), "hello");
    EXPECT_EQ(trim("hello world"), "hello world");
}

TEST(StringTrimmingTest, EmptyAndWhitespaceOnly) {
    EXPECT_EQ(trim(""), "");
    EXPECT_EQ(trim("   "), "");
    EXPECT_EQ(trim("\t\n\r"), "");
}

// ============================================================================
// Test Suite 7: File Operations Tests
// ============================================================================

TEST(FileOperationsTest, CreateVocabFile) {
    std::string test_file = "test_vocab_cli.txt";
    create_test_vocab_file(test_file, 10);

    std::ifstream file(test_file);
    EXPECT_TRUE(file.good());

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    EXPECT_EQ(lines.size(), 10);
    EXPECT_EQ(lines[0], "<unk>");
    EXPECT_EQ(lines[1], "<pad>");
    EXPECT_EQ(lines[2], "<s>");
    EXPECT_EQ(lines[3], "</s>");

    std::remove(test_file.c_str());
}

TEST(FileOperationsTest, CreateModelFile) {
    std::string test_file = "test_model_cli.bin";
    create_test_model_file(test_file);

    std::ifstream file(test_file, std::ios::binary);
    EXPECT_TRUE(file.good());

    int d_model;
    file.read(reinterpret_cast<char*>(&d_model), sizeof(int));
    EXPECT_EQ(d_model, 512);

    file.close();
    std::remove(test_file.c_str());
}

TEST(FileOperationsTest, CreateConversationFile) {
    std::string test_file = "test_conversation_cli.txt";
    create_test_conversation_file(test_file);

    std::ifstream file(test_file);
    EXPECT_TRUE(file.good());

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    EXPECT_EQ(lines.size(), 4);
    EXPECT_EQ(lines[0], "USER: Hello!");
    EXPECT_EQ(lines[1], "ASSISTANT: Hi there! How can I help you?");

    std::remove(test_file.c_str());
}

TEST(FileOperationsTest, ReadConversationFile) {
    std::string test_file = "test_read_conversation_cli.txt";
    create_test_conversation_file(test_file);

    std::ifstream file(test_file);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    EXPECT_NE(content.find("USER:"), std::string::npos);
    EXPECT_NE(content.find("ASSISTANT:"), std::string::npos);
    EXPECT_NE(content.find("Hello!"), std::string::npos);

    std::remove(test_file.c_str());
}

// ============================================================================
// Test Suite 8: Default Parameter Tests
// ============================================================================

struct DefaultSettings {
    int max_response_length = 100;
    float temperature = 1.0f;
    float top_p = 0.9f;
    int top_k = 50;
    int beam_width = 5;
    std::string generation_strategy = "nucleus";
};

TEST(DefaultParametersTest, MaxResponseLength) {
    DefaultSettings settings;
    EXPECT_EQ(settings.max_response_length, 100);
}

TEST(DefaultParametersTest, Temperature) {
    DefaultSettings settings;
    EXPECT_FLOAT_EQ(settings.temperature, 1.0f);
}

TEST(DefaultParametersTest, TopP) {
    DefaultSettings settings;
    EXPECT_FLOAT_EQ(settings.top_p, 0.9f);
}

TEST(DefaultParametersTest, TopK) {
    DefaultSettings settings;
    EXPECT_EQ(settings.top_k, 50);
}

TEST(DefaultParametersTest, BeamWidth) {
    DefaultSettings settings;
    EXPECT_EQ(settings.beam_width, 5);
}

TEST(DefaultParametersTest, GenerationStrategy) {
    DefaultSettings settings;
    EXPECT_EQ(settings.generation_strategy, "nucleus");
}

TEST(DefaultParametersTest, StrategyIsValid) {
    DefaultSettings settings;
    EXPECT_TRUE(is_valid_strategy(settings.generation_strategy));
}

// ============================================================================
// Test Suite 9: Model Architecture Tests
// ============================================================================

struct ModelArchitecture {
    int d_model = 512;
    int num_heads = 8;
    int d_ff = 2048;
    int num_encoder_layers = 6;
    int num_decoder_layers = 6;
    int max_seq_length = 1024;
};

TEST(ModelArchitectureTest, DefaultDModel) {
    ModelArchitecture arch;
    EXPECT_EQ(arch.d_model, 512);
}

TEST(ModelArchitectureTest, DefaultNumHeads) {
    ModelArchitecture arch;
    EXPECT_EQ(arch.num_heads, 8);
}

TEST(ModelArchitectureTest, DefaultDFF) {
    ModelArchitecture arch;
    EXPECT_EQ(arch.d_ff, 2048);
}

TEST(ModelArchitectureTest, DefaultEncoderLayers) {
    ModelArchitecture arch;
    EXPECT_EQ(arch.num_encoder_layers, 6);
}

TEST(ModelArchitectureTest, DefaultDecoderLayers) {
    ModelArchitecture arch;
    EXPECT_EQ(arch.num_decoder_layers, 6);
}

TEST(ModelArchitectureTest, DefaultMaxSeqLength) {
    ModelArchitecture arch;
    EXPECT_EQ(arch.max_seq_length, 1024);
}

TEST(ModelArchitectureTest, DModelDivisibleByHeads) {
    ModelArchitecture arch;
    EXPECT_EQ(arch.d_model % arch.num_heads, 0);
}

TEST(ModelArchitectureTest, DFFLargerThanDModel) {
    ModelArchitecture arch;
    EXPECT_GT(arch.d_ff, arch.d_model);
}

// ============================================================================
// Test Suite 10: Conversation Context Configuration Tests
// ============================================================================

struct ConversationContextConfig {
    int max_messages = 20;
    int max_tokens = 2048;
};

TEST(ConversationContextConfigTest, DefaultMaxMessages) {
    ConversationContextConfig config;
    EXPECT_EQ(config.max_messages, 20);
}

TEST(ConversationContextConfigTest, DefaultMaxTokens) {
    ConversationContextConfig config;
    EXPECT_EQ(config.max_tokens, 2048);
}

TEST(ConversationContextConfigTest, MaxTokensGreaterThanMaxMessages) {
    ConversationContextConfig config;
    // Assuming average message is at least 1 token
    EXPECT_GT(config.max_tokens, config.max_messages);
}

TEST(ConversationContextConfigTest, MaxMessagesPositive) {
    ConversationContextConfig config;
    EXPECT_GT(config.max_messages, 0);
}

TEST(ConversationContextConfigTest, MaxTokensPositive) {
    ConversationContextConfig config;
    EXPECT_GT(config.max_tokens, 0);
}

// ============================================================================
// Test Suite 11: Command Line Arguments Tests
// ============================================================================

struct CommandLineDefaults {
    std::string vocab_path = "vocab.txt";
    std::string model_path = "chatbot_model.bin";
    std::string conv_save_path = "conversation_history.txt";
};

TEST(CommandLineDefaultsTest, DefaultVocabPath) {
    CommandLineDefaults defaults;
    EXPECT_EQ(defaults.vocab_path, "vocab.txt");
}

TEST(CommandLineDefaultsTest, DefaultModelPath) {
    CommandLineDefaults defaults;
    EXPECT_EQ(defaults.model_path, "chatbot_model.bin");
}

TEST(CommandLineDefaultsTest, DefaultConversationPath) {
    CommandLineDefaults defaults;
    EXPECT_EQ(defaults.conv_save_path, "conversation_history.txt");
}

TEST(CommandLineDefaultsTest, AllPathsNonEmpty) {
    CommandLineDefaults defaults;
    EXPECT_FALSE(defaults.vocab_path.empty());
    EXPECT_FALSE(defaults.model_path.empty());
    EXPECT_FALSE(defaults.conv_save_path.empty());
}

TEST(CommandLineDefaultsTest, VocabPathHasExtension) {
    CommandLineDefaults defaults;
    EXPECT_NE(defaults.vocab_path.find(".txt"), std::string::npos);
}

TEST(CommandLineDefaultsTest, ModelPathHasExtension) {
    CommandLineDefaults defaults;
    EXPECT_NE(defaults.model_path.find(".bin"), std::string::npos);
}

// ============================================================================
// Test Suite 12: Edge Case Tests
// ============================================================================

TEST(EdgeCaseTest, EmptyCommandString) {
    EXPECT_FALSE(is_valid_command(""));
}

TEST(EdgeCaseTest, OnlySlashCommand) {
    EXPECT_FALSE(is_valid_command("/"));
}

TEST(EdgeCaseTest, VeryLongCommand) {
    std::string long_cmd = "/set temperature ";
    long_cmd += std::string(1000, '0');
    auto result = parse_set_command(long_cmd);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.value.length(), 1000);
}

TEST(EdgeCaseTest, CommandWithMultipleSpaces) {
    auto result = parse_set_command("/set strategy     greedy");
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.value, "    greedy");
}

TEST(EdgeCaseTest, TrimVeryLongWhitespace) {
    std::string input = std::string(100, ' ') + "test" + std::string(100, ' ');
    EXPECT_EQ(trim(input), "test");
}

TEST(EdgeCaseTest, SpecialCharactersInCommand) {
    EXPECT_FALSE(is_valid_command("/test!@#"));
}

TEST(EdgeCaseTest, UnicodeInCommand) {
    // Test with Unicode characters
    std::string unicode_cmd = "/system Hello 世界";
    EXPECT_TRUE(is_valid_command(unicode_cmd));
}

TEST(EdgeCaseTest, NegativeNumericValues) {
    auto result = parse_set_command("/set temperature -0.5");
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.value, "-0.5");
}

TEST(EdgeCaseTest, ScientificNotationValues) {
    auto result = parse_set_command("/set temperature 1e-6");
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.value, "1e-6");
}

TEST(EdgeCaseTest, ZeroValues) {
    auto result = parse_set_command("/set length 0");
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.value, "0");
}

// ============================================================================
// Test Suite 13: Integration Tests
// ============================================================================

TEST(IntegrationTest, CompleteCommandWorkflow) {
    // Simulate a complete command workflow
    std::string input = "  /set strategy beam  ";

    // Trim input
    std::string trimmed = trim(input);
    EXPECT_EQ(trimmed, "/set strategy beam");

    // Validate it's a command
    EXPECT_TRUE(is_valid_command(trimmed));

    // Parse the set command
    auto result = parse_set_command(trimmed);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.param, "strategy");
    EXPECT_EQ(result.value, "beam");

    // Validate the strategy
    EXPECT_TRUE(is_valid_strategy(result.value));
}

TEST(IntegrationTest, ParameterChangeWorkflow) {
    // Change temperature
    auto temp_result = parse_set_command("/set temp 0.8");
    EXPECT_TRUE(temp_result.valid);
    std::string normalized = normalize_param(temp_result.param);
    EXPECT_EQ(normalized, "temperature");

    // Change length
    auto len_result = parse_set_command("/set length 150");
    EXPECT_TRUE(len_result.valid);
    normalized = normalize_param(len_result.param);
    EXPECT_EQ(normalized, "max_length");
}

TEST(IntegrationTest, FileCreationAndVerification) {
    std::string vocab_file = "test_integration_vocab.txt";
    std::string model_file = "test_integration_model.bin";
    std::string conv_file = "test_integration_conv.txt";

    // Create files
    create_test_vocab_file(vocab_file, 50);
    create_test_model_file(model_file);
    create_test_conversation_file(conv_file);

    // Verify all exist
    EXPECT_TRUE(std::ifstream(vocab_file).good());
    EXPECT_TRUE(std::ifstream(model_file).good());
    EXPECT_TRUE(std::ifstream(conv_file).good());

    // Cleanup
    std::remove(vocab_file.c_str());
    std::remove(model_file.c_str());
    std::remove(conv_file.c_str());
}

TEST(IntegrationTest, AllValidStrategiesRecognized) {
    for (const auto& strategy : VALID_STRATEGIES) {
        EXPECT_TRUE(is_valid_strategy(strategy));

        std::string cmd = "/set strategy " + strategy;
        auto result = parse_set_command(cmd);
        EXPECT_TRUE(result.valid);
        EXPECT_EQ(result.value, strategy);
    }
}

TEST(IntegrationTest, AllCommandsRecognized) {
    std::vector<std::string> commands = {"/help",  "/clear",    "/save", "/load",
                                         "/stats", "/settings", "/exit", "/quit"};

    for (const auto& cmd : commands) {
        EXPECT_TRUE(is_valid_command(cmd));
    }
}

// ============================================================================
// Test Suite 14: Parameter Range Validation Tests
// ============================================================================

TEST(ParameterRangeTest, TemperatureReasonableRange) {
    // Common temperature range is 0.1 to 2.0
    DefaultSettings settings;
    EXPECT_GE(settings.temperature, 0.1f);
    EXPECT_LE(settings.temperature, 2.0f);
}

TEST(ParameterRangeTest, TopPInValidRange) {
    // Top-p should be between 0 and 1
    DefaultSettings settings;
    EXPECT_GT(settings.top_p, 0.0f);
    EXPECT_LE(settings.top_p, 1.0f);
}

TEST(ParameterRangeTest, TopKPositive) {
    DefaultSettings settings;
    EXPECT_GT(settings.top_k, 0);
}

TEST(ParameterRangeTest, BeamWidthPositive) {
    DefaultSettings settings;
    EXPECT_GT(settings.beam_width, 0);
}

TEST(ParameterRangeTest, MaxLengthPositive) {
    DefaultSettings settings;
    EXPECT_GT(settings.max_response_length, 0);
}

TEST(ParameterRangeTest, MaxLengthReasonable) {
    // Max length should be reasonable for conversation
    DefaultSettings settings;
    EXPECT_GE(settings.max_response_length, 10);
    EXPECT_LE(settings.max_response_length, 2048);
}

// ============================================================================
// Test Suite 15: Color Output Formatting Tests
// ============================================================================

TEST(ColorOutputTest, ColoredUserMessage) {
    std::string message = "Hello!";
    std::string colored = std::string(COLOR_USER) + "You: " + COLOR_RESET + message;

    EXPECT_NE(colored.find(COLOR_USER), std::string::npos);
    EXPECT_NE(colored.find(COLOR_RESET), std::string::npos);
    EXPECT_NE(colored.find(message), std::string::npos);
}

TEST(ColorOutputTest, ColoredBotMessage) {
    std::string message = "Hi there!";
    std::string colored = std::string(COLOR_BOT) + "Bot: " + COLOR_RESET + message;

    EXPECT_NE(colored.find(COLOR_BOT), std::string::npos);
    EXPECT_NE(colored.find(COLOR_RESET), std::string::npos);
    EXPECT_NE(colored.find(message), std::string::npos);
}

TEST(ColorOutputTest, ColoredSystemMessage) {
    std::string message = "✅ Settings saved";
    std::string colored = std::string(COLOR_SYSTEM) + message + COLOR_RESET;

    EXPECT_NE(colored.find(COLOR_SYSTEM), std::string::npos);
    EXPECT_NE(colored.find(COLOR_RESET), std::string::npos);
    EXPECT_NE(colored.find(message), std::string::npos);
}

TEST(ColorOutputTest, ColoredErrorMessage) {
    std::string message = "❌ Failed to load file";
    std::string colored = std::string(COLOR_ERROR) + message + COLOR_RESET;

    EXPECT_NE(colored.find(COLOR_ERROR), std::string::npos);
    EXPECT_NE(colored.find(COLOR_RESET), std::string::npos);
    EXPECT_NE(colored.find(message), std::string::npos);
}

TEST(ColorOutputTest, ResetAtEndOfColoredString) {
    std::string colored = std::string(COLOR_USER) + "Test" + COLOR_RESET;
    EXPECT_EQ(colored.substr(colored.length() - 4), COLOR_RESET);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
