// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-11

#include "ChatbotApiServerArgs.hpp"
#include <cstdlib>

namespace adai {

std::optional<std::string> extract_config_path_arg(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            return std::string(argv[i + 1]);
        }
    }
    return std::nullopt;
}

ChatbotApiServerArgsResult apply_chatbot_api_server_args(int argc, char* argv[],
                                                         ServiceConfig& config) {
    ChatbotApiServerArgsResult result;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            result.help = true;
            return result;
        }
        if (arg == "--config") {
            ++i;  // Already applied in extract_config_path_arg(); skip its value here.
        } else if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (arg == "--vocab" && i + 1 < argc) {
            config.vocab_path = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = std::atoi(argv[++i]);
        } else if (arg == "--timeout" && i + 1 < argc) {
            config.session_timeout = std::atoi(argv[++i]);
        } else if (arg == "--log-level" && i + 1 < argc) {
            config.log_level = argv[++i];
        } else if (arg == "--d-model" && i + 1 < argc) {
            config.d_model = std::atoi(argv[++i]);
        } else if (arg == "--num-heads" && i + 1 < argc) {
            config.num_heads = std::atoi(argv[++i]);
        } else if (arg == "--d-ff" && i + 1 < argc) {
            config.d_ff = std::atoi(argv[++i]);
        } else if (arg == "--enc-layers" && i + 1 < argc) {
            config.num_encoder_layers = std::atoi(argv[++i]);
        } else if (arg == "--dec-layers" && i + 1 < argc) {
            config.num_decoder_layers = std::atoi(argv[++i]);
        } else if (arg == "--max-seq-len" && i + 1 < argc) {
            config.max_seq_length = std::atoi(argv[++i]);
        } else if (arg == "--max-gen-len" && i + 1 < argc) {
            config.max_gen_length = std::atoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            config.temperature = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--top-p" && i + 1 < argc) {
            config.top_p = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--strategy" && i + 1 < argc) {
            config.strategy = argv[++i];
        } else {
            result.error = true;
            result.error_message = "Unknown argument: " + arg;
            return result;
        }
    }

    return result;
}

std::optional<std::string> validate_chatbot_api_server_config(const ServiceConfig& config) {
    if (config.vocab_path.empty()) {
        return "Error: Vocabulary path is required (use --vocab, VOCAB_PATH env var, or config "
              "file)";
    }
    return std::nullopt;
}

}  // namespace adai
