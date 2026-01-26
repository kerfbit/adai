# ChatbotCLI Testing Context Documentation

## Overview

This document provides comprehensive documentation for the ChatbotCLI test suites. The tests cover command parsing, parameter validation, color formatting, file operations, integration workflows, and modern C++ implementation for the command-line chatbot interface.

## Test Suites

### Legacy Test Suite

**Test File:** `tests/chatbotcli_test.cpp`  
**Test Executable:** `chatbotcliTests`  
**Total Tests:** 83 tests across 15 test suites  
**Test Framework:** Google Test (GTest) 1.14.0  
**Approach:** Helper functions and mocks (ChatbotCLI was in .cpp without header)

### Improved Test Suite (2026)

**Test File:** `tests/chatbotcli_improved_test.cpp`  
**Test Executable:** `chatbotcliImprovedTests`  
**Total Tests:** 23 tests across 4 test suites  
**Test Framework:** Google Test (GTest) 1.14.0  
**Approach:** Direct class instantiation and testing via header file

**Key Improvements:**
- ✅ Tests actual ChatbotCLI class (not just helper functions)
- ✅ Full constructor, accessor, and mutator coverage
- ✅ Direct command handling tests via `handle_setting()`
- ✅ Test fixture with automatic setup/teardown
- ✅ All tests passing with modern C++ implementation

## Improved Test Suite Coverage

| Test Suite | Tests | Purpose |
|------------|-------|---------|
| ChatbotCLITest | 17 | Class functionality (constructor, getters, setters, commands) |
| CommandValidationTest | 3 | Command recognition and validation |
| StrategyValidationTest | 2 | Generation strategy validation |
| ColorCodeTest | 1 | ANSI color code validation |
| **Total** | **23** | **Core CLI functionality with header support** |

### ChatbotCLITest Suite (17 tests)

**Purpose:** Test ChatbotCLI class members and methods directly

**Test Fixture:**
```cpp
class ChatbotCLITest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Create test vocab file
        vocab_file = "test_vocab.txt";
        model_file = "test_model.bin";
        conv_file = "test_conversation.txt";
        
        std::ofstream vocab(vocab_file);
        vocab << "hello 100\\n";
        vocab << "world 50\\n";
        vocab << "test 25\\n";
        vocab.close();
    }
    
    void TearDown() override {
        std::remove(vocab_file.c_str());
        std::remove(model_file.c_str());
        std::remove(conv_file.c_str());
    }
    
    std::string vocab_file;
    std::string model_file;
    std::string conv_file;
};
```

#### Constructor Tests (3 tests)

**ConstructorSetsDefaultParameters**
```cpp
TEST_F(ChatbotCLITest, ConstructorSetsDefaultParameters)
```
**Validates:**
- `max_response_length` = 100
- `temperature` = 1.0f
- `top_p` = 0.9f
- `top_k` = 50
- `beam_width` = 5
- `generation_strategy` = "nucleus"

**ConstructorStoresFilePaths**
```cpp
TEST_F(ChatbotCLITest, ConstructorStoresFilePaths)
```
**Validates:**
- Correct storage of `vocab_path`
- Correct storage of `model_path`
- Correct storage of `conversation_save_path`

**DefaultConversationSavePath**
```cpp
TEST_F(ChatbotCLITest, DefaultConversationSavePath)
```
**Validates:**
- Default path is "conversation_history.txt"

#### Accessor/Mutator Tests (6 tests)

**SetAndGetGenerationStrategy**
- Tests setting and retrieving: greedy, beam, sampling

**SetAndGetMaxResponseLength**
- Tests setting values: 50, 200

**SetAndGetTemperature**
- Tests setting values: 0.5f, 1.5f

**SetAndGetTopP**
- Tests setting values: 0.8f, 0.95f

**SetAndGetTopK**
- Tests setting values: 10, 100

**SetAndGetBeamWidth**
- Tests setting values: 3, 10

#### Command Handling Tests (8 tests)

All tests redirect stdout to suppress output during testing.

**HandleSettingStrategy**
```cpp
TEST_F(ChatbotCLITest, HandleSettingStrategy)
```
**Tests:**
```cpp
cli.handle_setting("strategy greedy");
cli.handle_setting("strategy beam");
cli.handle_setting("strategy nucleus");
```

**HandleSettingMaxLength**
- Tests: `"length 150"`, `"max_length 75"`

**HandleSettingTemperature**
- Tests: `"temperature 0.7"`, `"temp 1.2"`

**HandleSettingTopP**
- Tests: `"top_p 0.85"`, `"top-p 0.92"`

**HandleSettingTopK**
- Tests: `"top_k 20"`, `"top-k 40"`

**HandleSettingBeamWidth**
- Tests: `"beam_width 7"`, `"beam-width 12"`

**HandleSettingInvalidStrategy**
- Verifies invalid strategy doesn't change current value

**HandleSettingMissingValue**
- Verifies missing value doesn't change current value

## Legacy Test Suite Coverage

| Test Suite | Tests | Purpose |
|------------|-------|---------|
| ColorCodeTest | 3 | ANSI color code validation |
| CommandValidationTest | 4 | Command recognition and validation |
| StrategyValidationTest | 3 | Generation strategy validation |
| SetCommandParsingTest | 6 | `/set` command parsing logic |
| ParameterNormalizationTest | 6 | Parameter alias normalization |
| StringTrimmingTest | 5 | Whitespace trimming utility |
| FileOperationsTest | 4 | File creation and I/O |
| DefaultParametersTest | 7 | Default generation settings |
| ModelArchitectureTest | 8 | Model architecture configuration |
| ConversationContextConfigTest | 5 | Context manager configuration |
| CommandLineDefaultsTest | 6 | Default file paths |
| EdgeCaseTest | 10 | Edge cases and boundary conditions |
| IntegrationTest | 5 | End-to-end workflows |
| ParameterRangeTest | 6 | Parameter range validation |
| ColorOutputTest | 5 | Colored output formatting |
| **Total** | **83** | **Complete CLI functionality** |

## Legacy Test Suites Detailed

### 1. ColorCodeTest (3 tests)

**Purpose:** Validate ANSI escape codes for terminal color output

**Tests:**

#### ColorCodesAreDefined
```cpp
TEST(ColorCodeTest, ColorCodesAreDefined)
```
**Purpose:** Verify all color codes have correct ANSI values

**Validates:**
- `COLOR_RESET` = `"\033[0m"`
- `COLOR_USER` = `"\033[1;36m"` (Cyan)
- `COLOR_BOT` = `"\033[1;32m"` (Green)
- `COLOR_SYSTEM` = `"\033[1;33m"` (Yellow)
- `COLOR_ERROR` = `"\033[1;31m"` (Red)

**Expected:** All color codes match exact ANSI values

#### ColorCodesAreNonEmpty
```cpp
TEST(ColorCodeTest, ColorCodesAreNonEmpty)
```
**Purpose:** Ensure all color codes contain content

**Expected:** Every color code has length > 0

#### ColorCodesStartWithEscape
```cpp
TEST(ColorCodeTest, ColorCodesStartWithEscape)
```
**Purpose:** Verify all codes start with escape character

**Expected:** First character is `'\033'` for all colors

---

### 2. CommandValidationTest (4 tests)

**Purpose:** Validate command recognition logic

**Tests:**

#### RecognizeValidCommands
```cpp
TEST(CommandValidationTest, RecognizeValidCommands)
```
**Purpose:** Verify all valid commands are recognized

**Valid Commands Tested:**
- `/help` - Show help message
- `/clear` - Clear conversation
- `/save` - Save conversation
- `/load` - Load conversation
- `/stats` - Show statistics
- `/settings` - Show settings
- `/exit` - Exit chatbot
- `/quit` - Exit chatbot

**Expected:** `is_valid_command()` returns `true` for all

#### RecognizeCommandsWithArguments
```cpp
TEST(CommandValidationTest, RecognizeCommandsWithArguments)
```
**Purpose:** Recognize commands that take arguments

**Examples:**
- `/set strategy greedy`
- `/set temperature 0.8`
- `/system You are helpful`

**Expected:** Commands with arguments recognized correctly

#### RejectInvalidCommands
```cpp
TEST(CommandValidationTest, RejectInvalidCommands)
```
**Purpose:** Reject unknown commands

**Invalid Commands:**
- `/unknown`
- `/test`
- `/` (only slash)
- `` (empty string)

**Expected:** All return `false`

#### RejectNonCommands
```cpp
TEST(CommandValidationTest, RejectNonCommands)
```
**Purpose:** Reject user messages as commands

**Examples:**
- `"Hello"` - Normal message
- `"What is the weather?"` - Question
- `"exit"` - Missing slash

**Expected:** All return `false`

---

### 3. StrategyValidationTest (3 tests)

**Purpose:** Validate generation strategy options

**Tests:**

#### RecognizeValidStrategies
```cpp
TEST(StrategyValidationTest, RecognizeValidStrategies)
```
**Purpose:** Verify all 5 generation strategies recognized

**Valid Strategies:**
1. `greedy` - Greedy decoding
2. `beam` - Beam search
3. `sampling` - Temperature sampling
4. `top-k` - Top-k sampling
5. `nucleus` - Nucleus (top-p) sampling

**Expected:** `is_valid_strategy()` returns `true` for all

#### RejectInvalidStrategies
```cpp
TEST(StrategyValidationTest, RejectInvalidStrategies)
```
**Purpose:** Reject unknown strategies

**Invalid Examples:**
- `"random"`
- `"best"`
- `""` (empty)
- `"GREEDY"` (case sensitive)

**Expected:** All return `false`

#### StrategyCount
```cpp
TEST(StrategyValidationTest, StrategyCount)
```
**Purpose:** Verify exactly 5 strategies available

**Expected:** `VALID_STRATEGIES.size() == 5`

---

### 4. SetCommandParsingTest (6 tests)

**Purpose:** Test `/set` command parsing logic

**Tests:**

#### ParseValidSetCommand
```cpp
TEST(SetCommandParsingTest, ParseValidSetCommand)
```
**Purpose:** Parse basic `/set` command

**Input:** `/set strategy greedy`

**Expected Output:**
- `valid = true`
- `param = "strategy"`
- `value = "greedy"`

#### ParseSetCommandWithNumericValue
```cpp
TEST(SetCommandParsingTest, ParseSetCommandWithNumericValue)
```
**Purpose:** Parse float parameter values

**Input:** `/set temperature 0.8`

**Expected Output:**
- `valid = true`
- `param = "temperature"`
- `value = "0.8"`

#### ParseSetCommandWithIntValue
```cpp
TEST(SetCommandParsingTest, ParseSetCommandWithIntValue)
```
**Purpose:** Parse integer parameter values

**Input:** `/set length 150`

**Expected Output:**
- `valid = true`
- `param = "length"`
- `value = "150"`

#### RejectSetCommandWithoutValue
```cpp
TEST(SetCommandParsingTest, RejectSetCommandWithoutValue)
```
**Purpose:** Reject `/set` without value

**Input:** `/set strategy`

**Expected Output:**
- `valid = false`
- `error = "Missing value"`

#### RejectNonSetCommand
```cpp
TEST(SetCommandParsingTest, RejectNonSetCommand)
```
**Purpose:** Reject non-set commands

**Input:** `/help`

**Expected Output:**
- `valid = false`
- `error = "Not a set command"`

#### ParseSetCommandWithSpacesInValue
```cpp
TEST(SetCommandParsingTest, ParseSetCommandWithSpacesInValue)
```
**Purpose:** Handle multi-word values

**Input:** `/set system You are helpful`

**Expected Output:**
- `valid = true`
- `param = "system"`
- `value = "You are helpful"`

---

### 5. ParameterNormalizationTest (6 tests)

**Purpose:** Test parameter alias normalization

**Tests:**

#### NormalizeLength
```cpp
TEST(ParameterNormalizationTest, NormalizeLength)
```
**Aliases:** `length`, `max_length`

**Expected:** Both normalize to `"max_length"`

#### NormalizeTemperature
```cpp
TEST(ParameterNormalizationTest, NormalizeTemperature)
```
**Aliases:** `temperature`, `temp`

**Expected:** Both normalize to `"temperature"`

#### NormalizeTopP
```cpp
TEST(ParameterNormalizationTest, NormalizeTopP)
```
**Aliases:** `top_p`, `top-p`

**Expected:** Both normalize to `"top_p"`

#### NormalizeTopK
```cpp
TEST(ParameterNormalizationTest, NormalizeTopK)
```
**Aliases:** `top_k`, `top-k`

**Expected:** Both normalize to `"top_k"`

#### NormalizeBeamWidth
```cpp
TEST(ParameterNormalizationTest, NormalizeBeamWidth)
```
**Aliases:** `beam_width`, `beam-width`

**Expected:** Both normalize to `"beam_width"`

#### UnknownParameterUnchanged
```cpp
TEST(ParameterNormalizationTest, UnknownParameterUnchanged)
```
**Purpose:** Unknown parameters remain unchanged

**Examples:**
- `"unknown"` → `"unknown"`
- `"strategy"` → `"strategy"` (no alias)

**Expected:** Input equals output

---

### 6. StringTrimmingTest (5 tests)

**Purpose:** Test whitespace trimming utility

**Tests:**

#### TrimLeadingWhitespace
```cpp
TEST(StringTrimmingTest, TrimLeadingWhitespace)
```
**Examples:**
- `"  hello"` → `"hello"`
- `"\thello"` → `"hello"`
- `"\nhello"` → `"hello"`

**Expected:** Leading whitespace removed

#### TrimTrailingWhitespace
```cpp
TEST(StringTrimmingTest, TrimTrailingWhitespace)
```
**Examples:**
- `"hello  "` → `"hello"`
- `"hello\t"` → `"hello"`
- `"hello\n"` → `"hello"`

**Expected:** Trailing whitespace removed

#### TrimBothSides
```cpp
TEST(StringTrimmingTest, TrimBothSides)
```
**Examples:**
- `"  hello  "` → `"hello"`
- `"\t\nhello\r\n"` → `"hello"`

**Expected:** Both sides trimmed

#### NoTrimNeeded
```cpp
TEST(StringTrimmingTest, NoTrimNeeded)
```
**Examples:**
- `"hello"` → `"hello"`
- `"hello world"` → `"hello world"`

**Expected:** String unchanged

#### EmptyAndWhitespaceOnly
```cpp
TEST(StringTrimmingTest, EmptyAndWhitespaceOnly)
```
**Examples:**
- `""` → `""`
- `"   "` → `""`
- `"\t\n\r"` → `""`

**Expected:** All return empty string

---

### 7. FileOperationsTest (4 tests)

**Purpose:** Test file creation and I/O operations

**Tests:**

#### CreateVocabFile
```cpp
TEST(FileOperationsTest, CreateVocabFile)
```
**Purpose:** Create and validate vocabulary file

**Process:**
1. Create vocab file with 10 tokens
2. Verify file exists
3. Read and validate contents
4. Check special tokens: `<unk>`, `<pad>`, `<s>`, `</s>`
5. Cleanup

**Expected:** 10 lines with correct special tokens

#### CreateModelFile
```cpp
TEST(FileOperationsTest, CreateModelFile)
```
**Purpose:** Create binary model file

**Process:**
1. Create binary file with `d_model = 512`
2. Read back binary data
3. Verify value matches
4. Cleanup

**Expected:** Binary read returns 512

#### CreateConversationFile
```cpp
TEST(FileOperationsTest, CreateConversationFile)
```
**Purpose:** Create conversation history file

**Process:**
1. Create file with 4 lines (2 exchanges)
2. Validate line count
3. Check USER/ASSISTANT markers
4. Cleanup

**Example Content:**
```
USER: Hello!
ASSISTANT: Hi there! How can I help you?
USER: What's the weather?
ASSISTANT: I don't have access to real-time weather data.
```

**Expected:** 4 lines with correct format

#### ReadConversationFile
```cpp
TEST(FileOperationsTest, ReadConversationFile)
```
**Purpose:** Read and parse conversation file

**Process:**
1. Create conversation file
2. Read entire content as string
3. Verify presence of markers and text
4. Cleanup

**Expected:** Content contains USER, ASSISTANT, "Hello!"

---

### 8. DefaultParametersTest (7 tests)

**Purpose:** Validate default generation parameter values

**Tests:**

#### MaxResponseLength
```cpp
TEST(DefaultParametersTest, MaxResponseLength)
```
**Expected:** `max_response_length = 100`

#### Temperature
```cpp
TEST(DefaultParametersTest, Temperature)
```
**Expected:** `temperature = 1.0f`

#### TopP
```cpp
TEST(DefaultParametersTest, TopP)
```
**Expected:** `top_p = 0.9f`

#### TopK
```cpp
TEST(DefaultParametersTest, TopK)
```
**Expected:** `top_k = 50`

#### BeamWidth
```cpp
TEST(DefaultParametersTest, BeamWidth)
```
**Expected:** `beam_width = 5`

#### GenerationStrategy
```cpp
TEST(DefaultParametersTest, GenerationStrategy)
```
**Expected:** `generation_strategy = "nucleus"`

#### StrategyIsValid
```cpp
TEST(DefaultParametersTest, StrategyIsValid)
```
**Purpose:** Default strategy is valid

**Expected:** `is_valid_strategy("nucleus") == true`

---

### 9. ModelArchitectureTest (8 tests)

**Purpose:** Validate default model architecture settings

**Tests:**

#### DefaultDModel
```cpp
TEST(ModelArchitectureTest, DefaultDModel)
```
**Expected:** `d_model = 512`

#### DefaultNumHeads
```cpp
TEST(ModelArchitectureTest, DefaultNumHeads)
```
**Expected:** `num_heads = 8`

#### DefaultDFF
```cpp
TEST(ModelArchitectureTest, DefaultDFF)
```
**Expected:** `d_ff = 2048`

#### DefaultEncoderLayers
```cpp
TEST(ModelArchitectureTest, DefaultEncoderLayers)
```
**Expected:** `num_encoder_layers = 6`

#### DefaultDecoderLayers
```cpp
TEST(ModelArchitectureTest, DefaultDecoderLayers)
```
**Expected:** `num_decoder_layers = 6`

#### DefaultMaxSeqLength
```cpp
TEST(ModelArchitectureTest, DefaultMaxSeqLength)
```
**Expected:** `max_seq_length = 1024`

#### DModelDivisibleByHeads
```cpp
TEST(ModelArchitectureTest, DModelDivisibleByHeads)
```
**Purpose:** Validate architecture constraint

**Expected:** `d_model % num_heads == 0` (512 % 8 = 0)

#### DFFLargerThanDModel
```cpp
TEST(ModelArchitectureTest, DFFLargerThanDModel)
```
**Purpose:** Validate feed-forward expansion

**Expected:** `d_ff > d_model` (2048 > 512)

---

### 10. ConversationContextConfigTest (5 tests)

**Purpose:** Validate conversation context configuration

**Tests:**

#### DefaultMaxMessages
```cpp
TEST(ConversationContextConfigTest, DefaultMaxMessages)
```
**Expected:** `max_messages = 20`

#### DefaultMaxTokens
```cpp
TEST(ConversationContextConfigTest, DefaultMaxTokens)
```
**Expected:** `max_tokens = 2048`

#### MaxTokensGreaterThanMaxMessages
```cpp
TEST(ConversationContextConfigTest, MaxTokensGreaterThanMaxMessages)
```
**Purpose:** Validate sensible defaults

**Expected:** `max_tokens > max_messages` (2048 > 20)

#### MaxMessagesPositive
```cpp
TEST(ConversationContextConfigTest, MaxMessagesPositive)
```
**Expected:** `max_messages > 0`

#### MaxTokensPositive
```cpp
TEST(ConversationContextConfigTest, MaxTokensPositive)
```
**Expected:** `max_tokens > 0`

---

### 11. CommandLineDefaultsTest (6 tests)

**Purpose:** Validate default file paths

**Tests:**

#### DefaultVocabPath
```cpp
TEST(CommandLineDefaultsTest, DefaultVocabPath)
```
**Expected:** `vocab_path = "vocab.txt"`

#### DefaultModelPath
```cpp
TEST(CommandLineDefaultsTest, DefaultModelPath)
```
**Expected:** `model_path = "chatbot_model.bin"`

#### DefaultConversationPath
```cpp
TEST(CommandLineDefaultsTest, DefaultConversationPath)
```
**Expected:** `conv_save_path = "conversation_history.txt"`

#### AllPathsNonEmpty
```cpp
TEST(CommandLineDefaultsTest, AllPathsNonEmpty)
```
**Expected:** All paths have length > 0

#### VocabPathHasExtension
```cpp
TEST(CommandLineDefaultsTest, VocabPathHasExtension)
```
**Expected:** Path contains `".txt"`

#### ModelPathHasExtension
```cpp
TEST(CommandLineDefaultsTest, ModelPathHasExtension)
```
**Expected:** Path contains `".bin"`

---

### 12. EdgeCaseTest (10 tests)

**Purpose:** Test boundary conditions and edge cases

**Tests:**

#### EmptyCommandString
```cpp
TEST(EdgeCaseTest, EmptyCommandString)
```
**Input:** `""`

**Expected:** `is_valid_command() == false`

#### OnlySlashCommand
```cpp
TEST(EdgeCaseTest, OnlySlashCommand)
```
**Input:** `"/"`

**Expected:** `is_valid_command() == false`

#### VeryLongCommand
```cpp
TEST(EdgeCaseTest, VeryLongCommand)
```
**Input:** `/set temperature ` + 1000 zeros

**Expected:** Parses successfully, value length = 1000

#### CommandWithMultipleSpaces
```cpp
TEST(EdgeCaseTest, CommandWithMultipleSpaces)
```
**Input:** `/set strategy     greedy` (5 spaces)

**Expected:** Parses with spaces preserved in value

#### TrimVeryLongWhitespace
```cpp
TEST(EdgeCaseTest, TrimVeryLongWhitespace)
```
**Input:** 100 spaces + `"test"` + 100 spaces

**Expected:** Trims to `"test"`

#### SpecialCharactersInCommand
```cpp
TEST(EdgeCaseTest, SpecialCharactersInCommand)
```
**Input:** `/test!@#`

**Expected:** `is_valid_command() == false`

#### UnicodeInCommand
```cpp
TEST(EdgeCaseTest, UnicodeInCommand)
```
**Input:** `/system Hello 世界`

**Expected:** Recognized as valid command

#### NegativeNumericValues
```cpp
TEST(EdgeCaseTest, NegativeNumericValues)
```
**Input:** `/set temperature -0.5`

**Expected:** Parses successfully, value = `"-0.5"`

#### ScientificNotationValues
```cpp
TEST(EdgeCaseTest, ScientificNotationValues)
```
**Input:** `/set temperature 1e-6`

**Expected:** Parses successfully, value = `"1e-6"`

#### ZeroValues
```cpp
TEST(EdgeCaseTest, ZeroValues)
```
**Input:** `/set length 0`

**Expected:** Parses successfully, value = `"0"`

---

### 13. IntegrationTest (5 tests)

**Purpose:** Test end-to-end workflows

**Tests:**

#### CompleteCommandWorkflow
```cpp
TEST(IntegrationTest, CompleteCommandWorkflow)
```
**Purpose:** Complete command processing workflow

**Steps:**
1. Input: `"  /set strategy beam  "`
2. Trim to `"/set strategy beam"`
3. Validate as command
4. Parse set command
5. Validate strategy

**Expected:** All steps succeed

#### ParameterChangeWorkflow
```cpp
TEST(IntegrationTest, ParameterChangeWorkflow)
```
**Purpose:** Change multiple parameters

**Steps:**
1. Parse `/set temp 0.8`
2. Normalize `temp` → `temperature`
3. Parse `/set length 150`
4. Normalize `length` → `max_length`

**Expected:** All normalization correct

#### FileCreationAndVerification
```cpp
TEST(IntegrationTest, FileCreationAndVerification)
```
**Purpose:** Create and verify all file types

**Steps:**
1. Create vocab file
2. Create model file
3. Create conversation file
4. Verify all exist
5. Cleanup all

**Expected:** All files created successfully

#### AllValidStrategiesRecognized
```cpp
TEST(IntegrationTest, AllValidStrategiesRecognized)
```
**Purpose:** Test all 5 strategies end-to-end

**Process:**
For each strategy in `[greedy, beam, sampling, top-k, nucleus]`:
1. Verify `is_valid_strategy()` returns true
2. Create `/set strategy <name>` command
3. Parse command
4. Validate parsed value matches strategy

**Expected:** All 5 strategies pass all checks

#### AllCommandsRecognized
```cpp
TEST(IntegrationTest, AllCommandsRecognized)
```
**Purpose:** Verify all commands recognized

**Commands Tested:**
`/help`, `/clear`, `/save`, `/load`, `/stats`, `/settings`, `/exit`, `/quit`

**Expected:** All return `true` from `is_valid_command()`

---

### 14. ParameterRangeTest (6 tests)

**Purpose:** Validate parameter value ranges

**Tests:**

#### TemperatureReasonableRange
```cpp
TEST(ParameterRangeTest, TemperatureReasonableRange)
```
**Expected Range:** 0.1 ≤ temperature ≤ 2.0

**Default:** 1.0 (within range)

#### TopPInValidRange
```cpp
TEST(ParameterRangeTest, TopPInValidRange)
```
**Expected Range:** 0.0 < top_p ≤ 1.0

**Default:** 0.9 (within range)

#### TopKPositive
```cpp
TEST(ParameterRangeTest, TopKPositive)
```
**Expected:** `top_k > 0`

**Default:** 50

#### BeamWidthPositive
```cpp
TEST(ParameterRangeTest, BeamWidthPositive)
```
**Expected:** `beam_width > 0`

**Default:** 5

#### MaxLengthPositive
```cpp
TEST(ParameterRangeTest, MaxLengthPositive)
```
**Expected:** `max_response_length > 0`

**Default:** 100

#### MaxLengthReasonable
```cpp
TEST(ParameterRangeTest, MaxLengthReasonable)
```
**Expected Range:** 10 ≤ max_response_length ≤ 2048

**Default:** 100 (within range)

---

### 15. ColorOutputTest (5 tests)

**Purpose:** Test colored output formatting

**Tests:**

#### ColoredUserMessage
```cpp
TEST(ColorOutputTest, ColoredUserMessage)
```
**Format:** `COLOR_USER + "You: " + COLOR_RESET + message`

**Expected:** Contains USER color, RESET, and message

#### ColoredBotMessage
```cpp
TEST(ColorOutputTest, ColoredBotMessage)
```
**Format:** `COLOR_BOT + "Bot: " + COLOR_RESET + message`

**Expected:** Contains BOT color, RESET, and message

#### ColoredSystemMessage
```cpp
TEST(ColorOutputTest, ColoredSystemMessage)
```
**Format:** `COLOR_SYSTEM + message + COLOR_RESET`

**Expected:** Contains SYSTEM color, RESET, and message

#### ColoredErrorMessage
```cpp
TEST(ColorOutputTest, ColoredErrorMessage)
```
**Format:** `COLOR_ERROR + message + COLOR_RESET`

**Expected:** Contains ERROR color, RESET, and message

#### ResetAtEndOfColoredString
```cpp
TEST(ColorOutputTest, ResetAtEndOfColoredString)
```
**Purpose:** Verify RESET at string end

**Expected:** Last 4 characters are `COLOR_RESET`

---

## Helper Functions

### is_valid_command()
```cpp
bool is_valid_command(const std::string& input)
```
**Purpose:** Check if input is a valid command

**Returns:** `true` if starts with `/` and matches known command

### is_valid_strategy()
```cpp
bool is_valid_strategy(const std::string& strategy)
```
**Purpose:** Validate generation strategy

**Returns:** `true` if strategy in `[greedy, beam, sampling, top-k, nucleus]`

### parse_set_command()
```cpp
SetCommandResult parse_set_command(const std::string& command)
```
**Purpose:** Parse `/set param value` command

**Returns:** Struct with `valid`, `param`, `value`, `error` fields

### normalize_param()
```cpp
std::string normalize_param(const std::string& param)
```
**Purpose:** Normalize parameter aliases to canonical names

**Mappings:**
- `length`, `max_length` → `max_length`
- `temperature`, `temp` → `temperature`
- `top_p`, `top-p` → `top_p`
- `top_k`, `top-k` → `top_k`
- `beam_width`, `beam-width` → `beam_width`

### trim()
```cpp
std::string trim(const std::string& str)
```
**Purpose:** Remove leading/trailing whitespace

**Handles:** Spaces, tabs, newlines, carriage returns

### create_test_vocab_file()
```cpp
void create_test_vocab_file(const std::string& filepath, int vocab_size = 100)
```
**Purpose:** Create test vocabulary file

**Format:**
```
<unk>
<pad>
<s>
</s>
token0
token1
...
```

### create_test_model_file()
```cpp
void create_test_model_file(const std::string& filepath)
```
**Purpose:** Create minimal binary model file for testing

### create_test_conversation_file()
```cpp
void create_test_conversation_file(const std::string& filepath)
```
**Purpose:** Create test conversation history file

---

## Building and Running Tests

### Build Test Executable

```bash
cd /home/rodney/Repos/adai/build
cmake ..
make chatbotcliTests
```

### Run All Tests

```bash
./tests/chatbotcliTests
```

**Expected Output:**
```
[==========] Running 83 tests from 15 test suites.
[----------] Global test environment set-up.
...
[  PASSED  ] 83 tests.
```

### Run Specific Test Suite

```bash
./tests/chatbotcliTests --gtest_filter=CommandValidationTest.*
```

### Run Specific Test

```bash
./tests/chatbotcliTests --gtest_filter=CommandValidationTest.RecognizeValidCommands
```

### Verbose Output

```bash
./tests/chatbotcliTests --gtest_print_time=1
```

---

## Test Results

### Summary Statistics

**Total Tests:** 83  
**Test Suites:** 15  
**Passed:** 83 (100%)  
**Failed:** 0  
**Execution Time:** 1 ms

### Coverage by Category

| Category | Tests | Coverage |
|----------|-------|----------|
| Command Parsing | 10 | 100% of command types |
| Parameter Validation | 19 | All 6 parameters + aliases |
| String Utilities | 5 | All whitespace cases |
| File Operations | 4 | All file types |
| Configuration | 26 | All defaults validated |
| Edge Cases | 10 | Comprehensive boundaries |
| Integration | 5 | End-to-end workflows |
| Color Formatting | 5 | All color codes |

### Components Tested

✅ **Command Recognition:** All 8 commands  
✅ **Generation Strategies:** All 5 strategies  
✅ **Parameter Aliases:** All 5 parameter types  
✅ **ANSI Colors:** All 5 color codes  
✅ **File Operations:** Vocab, model, conversation files  
✅ **Default Values:** All 13 default settings  
✅ **Edge Cases:** Empty, long, special characters, unicode  
✅ **Integration:** Complete workflows  

---

## Coverage Analysis

### What's Tested

**✅ Command Line Interface:**
- Command validation (8 commands)
- Command parsing logic
- Parameter extraction
- Argument handling

**✅ Generation Parameters:**
- All 5 strategies
- 6 configurable parameters
- Parameter aliases (10 total)
- Default values
- Range validation

**✅ Color System:**
- All 5 ANSI color codes
- Color formatting functions
- Reset code handling

**✅ File Operations:**
- Vocabulary file I/O
- Model file creation
- Conversation history
- File validation

**✅ Configuration:**
- Model architecture (6 params)
- Context settings (2 params)
- Generation defaults (6 params)
- File paths (3 paths)

**✅ Utilities:**
- Whitespace trimming
- String parsing
- Parameter normalization

### What's Not Tested (Requires Full Class)

**❌ Full ChatbotCLI Instantiation:**
- Constructor with real components
- Initialization sequence
- Component integration
- Model loading
- Tokenizer integration

**❌ Interactive Loop:**
- Main run() loop
- User input handling
- Response generation
- Conversation flow

**❌ Component Integration:**
- EncoderDecoderModel interaction
- ConversationContext usage
- BPETokenizer integration
- TextGenerator usage

**❌ Generation Execution:**
- Actual text generation
- Strategy implementation
- Response formatting
- Error handling in generation

**Reason:** ChatbotCLI is defined in `.cpp` file without header, making full class instantiation difficult in unit tests. These require integration tests or end-to-end testing.

---

## Testing Best Practices Demonstrated

### 1. Comprehensive Coverage

**Example:** Strategy validation tests all 5 strategies individually and in integration tests

```cpp
TEST(StrategyValidationTest, RecognizeValidStrategies) {
    EXPECT_TRUE(is_valid_strategy("greedy"));
    EXPECT_TRUE(is_valid_strategy("beam"));
    EXPECT_TRUE(is_valid_strategy("sampling"));
    EXPECT_TRUE(is_valid_strategy("top-k"));
    EXPECT_TRUE(is_valid_strategy("nucleus"));
}
```

### 2. Edge Case Testing

**Example:** Test extreme inputs (empty, very long, special characters)

```cpp
TEST(EdgeCaseTest, VeryLongCommand) {
    std::string long_cmd = "/set temperature " + std::string(1000, '0');
    auto result = parse_set_command(long_cmd);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.value.length(), 1000);
}
```

### 3. Integration Testing

**Example:** Complete workflow testing

```cpp
TEST(IntegrationTest, CompleteCommandWorkflow) {
    std::string input = "  /set strategy beam  ";
    std::string trimmed = trim(input);
    EXPECT_TRUE(is_valid_command(trimmed));
    auto result = parse_set_command(trimmed);
    EXPECT_TRUE(result.valid);
    EXPECT_TRUE(is_valid_strategy(result.value));
}
```

### 4. Boundary Validation

**Example:** Test parameter ranges

```cpp
TEST(ParameterRangeTest, TopPInValidRange) {
    DefaultSettings settings;
    EXPECT_GT(settings.top_p, 0.0f);
    EXPECT_LE(settings.top_p, 1.0f);
}
```

### 5. File Cleanup

**Example:** Always cleanup test files

```cpp
TEST(FileOperationsTest, CreateVocabFile) {
    create_test_vocab_file(test_file, 10);
    // ... test logic ...
    std::remove(test_file.c_str());  // Cleanup
}
```

### 6. Positive and Negative Testing

**Example:** Test both valid and invalid inputs

```cpp
// Positive
TEST(CommandValidationTest, RecognizeValidCommands)

// Negative
TEST(CommandValidationTest, RejectInvalidCommands)
```

### 7. Descriptive Test Names

All test names clearly describe what they test:
- `ColorCodesAreDefined` - Clear expectation
- `ParseValidSetCommand` - Clear input type
- `RejectInvalidCommands` - Clear negative test

### 8. Isolated Tests

Each test is independent and doesn't rely on others:
- No shared state between tests
- Each test creates its own test data
- File operations include cleanup

---

## Known Limitations

### 1. Legacy Tests - No Full Class Testing

**Limitation:** Original ChatbotCLI defined in `.cpp` without header

**Impact:** Cannot instantiate full class in legacy tests

**Workaround:** Test helper functions and logic separately

**Resolution (2026):** ✅ **Solved** - Header file created, improved test suite added

### 2. No Interactive Testing

**Limitation:** Cannot test interactive user input loop

**Impact:** Main `run()` loop not tested

**Future:** Consider extracting to testable components

### 3. No Component Integration

**Limitation:** Cannot test model/tokenizer integration

**Impact:** Generation pipeline not validated

**Future:** Mock components or integration tests

### 4. No Terminal Output Validation

**Limitation:** Cannot verify actual colored terminal output

**Impact:** Only test color code presence

**Future:** Capture stdout/stderr for validation

### 5. Limited File I/O Testing

**Limitation:** Basic file operations only

**Impact:** Error conditions not fully tested

**Future:** Test file permissions, disk space, etc.

---

## Future Enhancements

### 1. Mock Component Testing

Create mock versions of:
- `EncoderDecoderModel`
- `ConversationContext`
- `BPETokenizer`

**Benefit:** Test full class initialization and methods

### 2. Interactive Simulation

Create test harness to simulate user input:
```cpp
TEST(InteractiveTest, CommandSequence) {
    // Simulate: /set strategy beam → hello → /stats → /exit
}
```

### 3. Error Condition Testing

Test error scenarios:
- File not found
- Permission denied
- Out of memory
- Invalid model format
- Tokenizer failures

### 4. Performance Testing

Add performance benchmarks:
- Command parsing speed
- File I/O performance
- Memory usage
- Large conversation handling

### 5. Terminal Output Capture

Capture and validate console output:
```cpp
TEST(OutputTest, WelcomeMessageFormat) {
    // Redirect stdout
    // Run print_welcome()
    // Validate output format
}
```

### 6. Extended Parameter Validation

Test parameter boundaries:
- Negative values
- Extreme values
- Type mismatches
- Overflow conditions

### 7. Unicode and Internationalization

Extended unicode testing:
- Various languages
- Emoji in commands
- Multi-byte characters
- Right-to-left text

### 8. Stress Testing

Test system limits:
- Very long conversations
- Rapid command input
- Large file operations
- Memory constraints

---

## Related Documentation

- **ChatbotCLI Header:** `src/ChatbotCLI.hpp`
- **ChatbotCLI Implementation:** `src/ChatbotCLI.cpp`
- **ChatbotCLI Context:** `docs/guides/chatbot-cli-internals.md`
- **Improved Test Suite:** `tests/chatbotcli_improved_test.cpp`
- **Legacy Test Suite:** `tests/chatbotcli_test.cpp`
- **EncoderDecoderModel Tests:** `tests/encoderdecoder_test.cpp`
- **ChatbotTrainer Tests:** `tests/chatbottrainer_test.cpp`
- **Optimizer Tests:** `tests/optimizer_test.cpp`
- **CMake Configuration:** `tests/CMakeLists.txt`

---

## Summary

The ChatbotCLI testing infrastructure provides comprehensive validation through two complementary test suites:

### Improved Test Suite (2026)
✅ **23 tests** with direct class instantiation  
✅ **4 test suites** covering core functionality  
✅ **100% pass rate** with modern C++ implementation  
✅ **Constructor testing** for all initialization paths  
✅ **Accessor/mutator testing** for all parameters  
✅ **Command handling** via actual class methods  
✅ **Test fixtures** with automatic setup/teardown  
✅ **Smart pointer support** testing modern implementation  

### Legacy Test Suite
✅ **83 tests** covering all testable CLI functionality  
✅ **15 test suites** organized by feature category  
✅ **100% pass rate** with fast execution (1ms)  
✅ **Command parsing** for all 8 commands  
✅ **Parameter validation** for all 6 parameters + aliases  
✅ **Generation strategies** for all 5 strategies  
✅ **File operations** for vocab, model, conversation files  
✅ **Edge cases** including unicode, long inputs, special chars  
✅ **Integration workflows** for complete command processing  
✅ **Color formatting** for all 5 ANSI color codes  
✅ **Default configuration** for all 17 default values  

**Combined Coverage:** 106 tests ensuring ChatbotCLI reliability

**Key Strengths:**
- Comprehensive coverage from helper functions to full class
- Well-organized test suites
- Clear, descriptive test names
- Proper file cleanup and fixtures
- Edge case coverage
- Integration testing
- Fast execution
- Modern C++ best practices

**Testing Evolution:**
The addition of the header file and improved test suite in 2026 resolved the original architectural limitation, enabling full class-level testing while maintaining backward compatibility with the extensive legacy test suite. Together, they provide defense-in-depth validation of all ChatbotCLI functionality.
