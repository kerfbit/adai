// @adai-status: experimental
// @adai-version: 0.9.0
// @adai-reviewed: 2026-10-31

#include "ChatbotApiServerArgs.hpp"

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

// TODO: See TD-244 in TECHNICAL_DEBT.md - no --top-k/--beam-width flags.
ChatbotApiServerArgsResult apply_chatbot_api_server_args(int argc, char* argv[],
                                                         ServiceConfig& config) {
    ChatbotApiServerArgsResult result;

    // TD-240: numeric flags used atoi/atof, so "--port abc" became 0 and "--port 8080x" 8080. Each
    // value must now be the whole number (a size non-negative, a float finite), else an error.
    auto bad_number = [&](const std::string& flag, const char* value, const char* expected) {
        result.error = true;
        result.error_message = "Invalid value for " + flag + ": '" + value + "' (expected " + expected + ")";
        return result;
    };

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
        } else if (arg == "--draft-model" && i + 1 < argc) {
            config.draft_model_path = argv[++i];
        } else if (arg == "--speculative-candidates" && i + 1 < argc) {
            const auto v = parse_strict_int(argv[++i]);
            if (!v) return bad_number(arg, argv[i], "a whole number");
            config.speculative_num_candidates = *v;
        } else if (arg == "--port" && i + 1 < argc) {
            const auto v = parse_strict_int(argv[++i]);
            if (!v) return bad_number(arg, argv[i], "a whole number");
            config.port = *v;
        } else if (arg == "--timeout" && i + 1 < argc) {
            const auto v = parse_strict_int(argv[++i]);
            if (!v) return bad_number(arg, argv[i], "a whole number");
            config.session_timeout = *v;
        } else if (arg == "--log-level" && i + 1 < argc) {
            config.log_level = argv[++i];
        } else if (arg == "--d-model" && i + 1 < argc) {
            const auto v = parse_strict_size(argv[++i]);
            if (!v) return bad_number(arg, argv[i], "a non-negative whole number");
            config.d_model = *v;
        } else if (arg == "--num-heads" && i + 1 < argc) {
            const auto v = parse_strict_size(argv[++i]);
            if (!v) return bad_number(arg, argv[i], "a non-negative whole number");
            config.num_heads = *v;
        } else if (arg == "--d-ff" && i + 1 < argc) {
            const auto v = parse_strict_size(argv[++i]);
            if (!v) return bad_number(arg, argv[i], "a non-negative whole number");
            config.d_ff = *v;
        } else if (arg == "--enc-layers" && i + 1 < argc) {
            const auto v = parse_strict_size(argv[++i]);
            if (!v) return bad_number(arg, argv[i], "a non-negative whole number");
            config.num_encoder_layers = *v;
        } else if (arg == "--dec-layers" && i + 1 < argc) {
            const auto v = parse_strict_size(argv[++i]);
            if (!v) return bad_number(arg, argv[i], "a non-negative whole number");
            config.num_decoder_layers = *v;
        } else if (arg == "--max-seq-len" && i + 1 < argc) {
            const auto v = parse_strict_size(argv[++i]);
            if (!v) return bad_number(arg, argv[i], "a non-negative whole number");
            config.max_seq_length = *v;
        } else if (arg == "--max-gen-len" && i + 1 < argc) {
            const auto v = parse_strict_size(argv[++i]);
            if (!v) return bad_number(arg, argv[i], "a non-negative whole number");
            config.max_gen_length = *v;
        } else if (arg == "--temperature" && i + 1 < argc) {
            const auto v = parse_strict_float(argv[++i]);
            if (!v) return bad_number(arg, argv[i], "a number");
            config.temperature = *v;
        } else if (arg == "--top-p" && i + 1 < argc) {
            const auto v = parse_strict_float(argv[++i]);
            if (!v) return bad_number(arg, argv[i], "a number");
            config.top_p = *v;
        } else if (arg == "--strategy" && i + 1 < argc) {
            config.strategy = argv[++i];
        } else if (arg == "--profile") {
            result.profile = true;
        } else if (arg == "--batched-inference") {
            result.batched_inference = true;
        } else if (arg == "--batch-timeout-ms" && i + 1 < argc) {
            const auto v = parse_strict_int(argv[++i]);
            if (!v) return bad_number(arg, argv[i], "a whole number");
            result.batch_timeout_ms = *v;
        } else if (arg == "--pipeline-inference" || arg == "--integrated-inference") {
            // TD-474/TD-475, TD-417–TD-420: retired with their engines, which batched nothing through
            // the model and answered differently from /chat. Say so rather than "unknown argument".
            result.error = true;
            result.error_message =
                arg + " was removed (October 2026): the engine never batched requests and answered "
                "differently from /chat; see docs/proposals/real_batching_v2_plan.md";
            return result;
        } else {
            // TODO: See TD-245 in TECHNICAL_DEBT.md - a known flag missing its value (e.g. trailing --port)
            // lands here and is reported as "Unknown argument".
            // TODO: See TD-243 in TECHNICAL_DEBT.md - --batch-timeout-ms without --batched-inference is
            // accepted silently.
            result.error = true;
            result.error_message = "Unknown argument: " + arg;
            return result;
        }
    }

    return result;
}

// TODO: See TD-241 in TECHNICAL_DEBT.md - only vocab_path is checked; ConfigLoader::validate() (port,
// timeout, ...) runs only on SIGHUP reload, and nothing validates strategy/temperature/top_p.
std::optional<std::string> validate_chatbot_api_server_config(const ServiceConfig& config) {
    if (config.vocab_path.empty()) {
        return "Error: Vocabulary path is required (use --vocab, VOCAB_PATH env var, or config "
              "file)";
    }
    return std::nullopt;
}

}  // namespace adai
