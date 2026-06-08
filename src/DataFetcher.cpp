#include "DataFetcher.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <utility>
#include <vector>
#include "Logger.hpp"

using adai::Logger;
namespace fs = std::filesystem;

// ============================================================================
// Constructor
// ============================================================================

DataFetcher::DataFetcher(FetcherConfig cfg)
    : config_(std::move(cfg)) {}

// ============================================================================
// Public — Project Gutenberg
// ============================================================================

std::string DataFetcher::fetch_gutenberg(int book_id, int num_pairs) {
    if (!fs::exists(config_.gutenberg_output_dir)) {
        fs::create_directories(config_.gutenberg_output_dir);
    }

    std::string url = get_gutenberg_url(book_id);
    std::ostringstream text_path_oss;
    text_path_oss << config_.gutenberg_output_dir << "/gutenberg_" << book_id << ".txt";
    const std::string text_file = text_path_oss.str();

    if (!download_file(url, text_file)) {
        return "";
    }

    std::ostringstream training_path_oss;
    training_path_oss << config_.gutenberg_output_dir << "/gutenberg_" << book_id
                      << "_training.txt";
    const std::string training_file = training_path_oss.str();

    if (!convert_gutenberg_to_training_data(text_file, training_file, num_pairs)) {
        return "";
    }

    return training_file;
}

std::vector<std::string> DataFetcher::fetch_gutenberg_batch(const std::vector<int>& ids,
                                                              int num_pairs_each) {
    std::vector<std::string> paths;
    paths.reserve(ids.size());
    int success_count = 0;

    for (int book_id : ids) {
        std::string path = fetch_gutenberg(book_id, num_pairs_each);
        if (!path.empty()) {
            ++success_count;
        }
        paths.push_back(std::move(path));
    }

    Logger::info("Downloaded {}/{} books", success_count, ids.size());
    return paths;
}

// ============================================================================
// Public — HuggingFace
// ============================================================================

std::string DataFetcher::fetch_huggingface(const std::string& dataset_id, int num_pairs,
                                            const std::string& split,
                                            const std::string& input_field,
                                            const std::string& output_field) {
    // Build a filesystem-safe base name (replace '/' with '_')
    std::string safe_id = dataset_id;
    std::replace(safe_id.begin(), safe_id.end(), '/', '_');

    const std::string rows_dir =
        config_.huggingface_output_dir + "/" + safe_id + "_" + split;
    const std::string train_file =
        config_.huggingface_output_dir + "/" + safe_id + "_" + split + "_training.txt";

    if (!fs::exists(config_.huggingface_output_dir)) {
        fs::create_directories(config_.huggingface_output_dir);
    }
    if (!fs::exists(rows_dir)) {
        fs::create_directories(rows_dir);
    }

    std::cout << "🤖 Fetching HuggingFace dataset '" << dataset_id << "' (split=" << split
              << ", target=" << num_pairs << " pairs)\n";

    // datasets-server allows at most 100 rows per request.
    // Over-sample by 50 % to compensate for rows that yield no pairs.
    const int chunk_size   = 100;
    const int target_rows  = std::min(num_pairs * 3 / 2 + chunk_size, 500000);
    int downloaded         = 0;
    int chunk_idx          = 0;

    while (downloaded < target_rows) {
        int this_len = std::min(chunk_size, target_rows - downloaded);
        std::ostringstream chunk_path;
        chunk_path << rows_dir << "/chunk_" << chunk_idx << ".json";

        if (!download_hf_rows(dataset_id, split, downloaded, this_len, chunk_path.str())) {
            if (chunk_idx == 0) {
                std::cerr
                    << "❌ Failed to fetch dataset '" << dataset_id << "'.\n"
                    << "   • Check the dataset ID at https://huggingface.co/datasets\n"
                    << "   • Gated datasets require: export HF_TOKEN=hf_...\n"
                    << "   • Verify at: https://datasets-server.huggingface.co/is-valid?dataset="
                    << dataset_id << "\n";
                return "";
            }
            Logger::info("Download halted at chunk {} — converting partial data", chunk_idx);
            break;
        }

        downloaded += this_len;
        ++chunk_idx;
        std::cout << "  ✓ rows " << (downloaded - this_len + 1) << "-" << downloaded << "\n";
    }

    Logger::info("Downloaded {} rows in {} chunks for '{}'", downloaded, chunk_idx, dataset_id);

    if (!convert_hf_to_training_data(rows_dir, train_file, input_field, output_field, num_pairs)) {
        std::cerr << "❌ Could not extract training pairs from '" << dataset_id << "'.\n"
                  << "   Provide explicit field names: huggingface " << dataset_id << " "
                  << num_pairs << " " << split << " <input_field> <output_field>\n";
        return "";
    }

    return train_file;
}

// ============================================================================
// Private — Gutenberg helpers (moved verbatim from IncrementalTrainer.cpp)
// ============================================================================

/*static*/
std::string DataFetcher::get_gutenberg_url(int book_id) {
    std::ostringstream oss;
    oss << "https://www.gutenberg.org/files/" << book_id << "/" << book_id << "-0.txt";
    return oss.str();
}

/*static*/
bool DataFetcher::download_file(const std::string& url, const std::string& output_path) {
    Logger::info("Downloading: {}", url);

    std::ostringstream cmd;
    cmd << "curl -L -f -s -o \"" << output_path << "\" \"" << url << "\"";

    int result = std::system(cmd.str().c_str());

    if (result == 0 && fs::exists(output_path) && fs::file_size(output_path) > 0) {
        Logger::info("Downloaded to: {}", output_path);
        return true;
    }

    if (url.find("-0.txt") != std::string::npos) {
        std::string fallback_url = url;
        size_t pos = fallback_url.find("-0.txt");
        fallback_url.replace(pos, 6, ".txt");

        Logger::info("Trying fallback URL: {}", fallback_url);
        std::ostringstream fallback_cmd;
        fallback_cmd << "curl -L -f -s -o \"" << output_path << "\" \"" << fallback_url << "\"";

        result = std::system(fallback_cmd.str().c_str());

        if (result == 0 && fs::exists(output_path) && fs::file_size(output_path) > 0) {
            Logger::info("Downloaded to: {}", output_path);
            return true;
        }
    }

    Logger::error("Failed to download: {}", url);
    return false;
}

/*static*/
bool DataFetcher::download_gutenberg_book(int book_id, const std::string& output_dir) {
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }

    std::string url = get_gutenberg_url(book_id);
    std::ostringstream output_path;
    output_path << output_dir << "/gutenberg_" << book_id << ".txt";

    return download_file(url, output_path.str());
}

/*static*/
std::string DataFetcher::clean_gutenberg_text(const std::string& raw_text) {
    std::string cleaned = raw_text;

    // Remove Project Gutenberg header (before "*** START OF")
    size_t start_pos = cleaned.find("*** START OF");
    if (start_pos != std::string::npos) {
        start_pos = cleaned.find('\n', start_pos);
        if (start_pos != std::string::npos) {
            cleaned = cleaned.substr(start_pos + 1);
        }
    }

    // Remove Project Gutenberg footer (after "*** END OF")
    size_t end_pos = cleaned.find("*** END OF");
    if (end_pos != std::string::npos) {
        cleaned = cleaned.substr(0, end_pos);
    }

    // Remove excessive whitespace
    cleaned = std::regex_replace(cleaned, std::regex("[ \\t]+"), " ");
    cleaned = std::regex_replace(cleaned, std::regex("\\n{3,}"), "\n\n");

    return cleaned;
}

/*static*/
std::vector<std::string> DataFetcher::extract_sentences(const std::string& text) {
    std::vector<std::string> sentences;

    std::regex sentence_regex("[^.!?]+[.!?]+");

    auto sentences_begin = std::sregex_iterator(text.begin(), text.end(), sentence_regex);
    auto sentences_end   = std::sregex_iterator();

    for (std::sregex_iterator i = sentences_begin; i != sentences_end; ++i) {
        std::string sentence = i->str();

        sentence.erase(0, sentence.find_first_not_of(" \t\n\r"));
        sentence.erase(sentence.find_last_not_of(" \t\n\r") + 1);

        if (sentence.length() > 20 && sentence.length() < 500) {
            sentences.push_back(sentence);
        }
    }

    return sentences;
}

/*static*/
std::string DataFetcher::generate_question_from_sentence(const std::string& sentence) {
    std::vector<std::string> question_templates = {
        "What does this mean: ", "Can you explain: ", "Tell me about: ",
        "What is this about: ",  "Explain this: ",    "What does this say: "};

    int idx = rand() % static_cast<int>(question_templates.size());
    return question_templates[idx] + sentence;
}

/*static*/
std::vector<std::pair<std::string, std::string>> DataFetcher::create_qa_pairs_from_text(
    const std::vector<std::string>& sentences, int max_pairs) {
    std::vector<std::pair<std::string, std::string>> pairs;

    for (size_t i = 0; i < sentences.size() - 1 && static_cast<int>(pairs.size()) < max_pairs;
         i += 2) {
        std::string question = generate_question_from_sentence(sentences[i]);
        std::string answer   = sentences[i + 1];

        if (static_cast<int>(pairs.size()) < max_pairs) {
            pairs.emplace_back(question, answer);
        }

        if (i + 2 < sentences.size() && static_cast<int>(pairs.size()) < max_pairs) {
            std::string context = sentences[i] + " " + sentences[i + 1];
            std::string summary = sentences[i + 2];
            pairs.emplace_back("Summarize: " + context, summary);
        }
    }

    return pairs;
}

/*static*/
bool DataFetcher::convert_gutenberg_to_training_data(const std::string& text_file,
                                                      const std::string& output_file,
                                                      int max_pairs) {
    Logger::info("Converting Gutenberg text to training pairs: {}", text_file);

    std::ifstream file(text_file);
    if (!file.is_open()) {
        Logger::error("Cannot open: {}", text_file);
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string raw_text = buffer.str();
    file.close();

    std::string cleaned_text  = clean_gutenberg_text(raw_text);
    std::vector<std::string> sentences = extract_sentences(cleaned_text);
    Logger::info("Extracted {} sentences", sentences.size());

    if (sentences.empty()) {
        Logger::error("No valid sentences found in {}", text_file);
        return false;
    }

    auto pairs = create_qa_pairs_from_text(sentences, max_pairs);
    Logger::info("Created {} conversation pairs", pairs.size());

    std::ofstream out(output_file);
    if (!out.is_open()) {
        Logger::error("Cannot create: {}", output_file);
        return false;
    }

    for (const auto& pair : pairs) {
        out << "INPUT: " << pair.first << "\n";
        out << "RESPONSE: " << pair.second << "\n";
        out << "\n";
    }

    out.close();

    Logger::info("Training data saved to: {}", output_file);
    return true;
}

// ============================================================================
// Private — HuggingFace helpers (moved verbatim from IncrementalTrainer.cpp)
// ============================================================================

/*static*/
bool DataFetcher::download_hf_rows(const std::string& dataset_id, const std::string& split,
                                    int offset, int length, const std::string& output_path) {
    std::ostringstream url;
    url << "https://datasets-server.huggingface.co/rows"
        << "?dataset=" << dataset_id << "&config=default"
        << "&split=" << split << "&offset=" << offset << "&length=" << length;

    Logger::info("HF datasets-server: {} split={} offset={} length={}", dataset_id, split, offset,
                 length);

    std::ostringstream cmd;
    const char* hf_token = std::getenv("HF_TOKEN");
    if (hf_token && hf_token[0] != '\0') {
        cmd << "curl -L -f -s"
            << " -H \"Authorization: Bearer " << hf_token << "\""
            << " -o \"" << output_path << "\""
            << " \"" << url.str() << "\"";
    } else {
        cmd << "curl -L -f -s"
            << " -o \"" << output_path << "\""
            << " \"" << url.str() << "\"";
    }

    int rc = std::system(cmd.str().c_str());
    if (rc != 0 || !fs::exists(output_path) || fs::file_size(output_path) == 0) {
        Logger::error("Failed to fetch HuggingFace rows (offset={} length={})", offset, length);
        return false;
    }
    return true;
}

// ============================================================================
// JSON helpers — file-local statics (moved verbatim from IncrementalTrainer.cpp)
// Not declared in the header; only used by convert_hf_to_training_data().
// ============================================================================

/// Unescape a JSON string value (\n, \t, \r, \", \\, \/).
static std::string hf_unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            ++i;
            switch (s[i]) {
                case 'n':  out += '\n'; break;
                case 't':  out += '\t'; break;
                case 'r':  out += '\r'; break;
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                default:   out += '\\'; out += s[i]; break;
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

/// Extract the first quoted string value for the given JSON key.
static std::string hf_extract_string(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string::npos) return "";
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos) return "";

    size_t vpos = colon + 1;
    while (vpos < json.size() && std::isspace(static_cast<unsigned char>(json[vpos]))) ++vpos;
    if (vpos >= json.size() || json[vpos] != '"') return "";

    std::string raw;
    ++vpos;
    while (vpos < json.size() && json[vpos] != '"') {
        if (json[vpos] == '\\' && vpos + 1 < json.size()) {
            raw += '\\';
            raw += json[vpos + 1];
            vpos += 2;
        } else {
            raw += json[vpos++];
        }
    }
    return hf_unescape(raw);
}

/// Extract an array of quoted strings for a given JSON key.
static std::vector<std::string> hf_extract_string_array(const std::string& json,
                                                          const std::string& key) {
    std::vector<std::string> result;
    std::string needle = "\"" + key + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string::npos) return result;
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos) return result;
    size_t bracket = json.find('[', colon + 1);
    if (bracket == std::string::npos) return result;

    int    depth   = 1;
    size_t pos     = bracket + 1;
    size_t arr_end = std::string::npos;
    while (pos < json.size() && depth > 0) {
        char c = json[pos];
        if (c == '[') {
            ++depth; ++pos;
        } else if (c == ']') {
            --depth;
            if (depth == 0) { arr_end = pos; break; }
            ++pos;
        } else if (c == '"') {
            ++pos;
            while (pos < json.size() && json[pos] != '"') {
                if (json[pos] == '\\') ++pos;
                ++pos;
            }
            if (pos < json.size()) ++pos;
        } else {
            ++pos;
        }
    }
    if (arr_end == std::string::npos) return result;

    std::string arr = json.substr(bracket + 1, arr_end - bracket - 1);
    size_t p = 0;
    while (p < arr.size()) {
        if (arr[p] == '"') {
            ++p;
            std::string raw;
            while (p < arr.size() && arr[p] != '"') {
                if (arr[p] == '\\' && p + 1 < arr.size()) {
                    raw += '\\';
                    raw += arr[p + 1];
                    p += 2;
                } else {
                    raw += arr[p++];
                }
            }
            if (p < arr.size()) ++p;
            result.push_back(hf_unescape(raw));
        } else {
            ++p;
        }
    }
    return result;
}

/// Extract a JSON object ({ ... }) for the given key.
static std::string hf_extract_object(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string::npos) return "";
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos) return "";
    size_t brace = json.find('{', colon + 1);
    if (brace == std::string::npos) return "";

    int    depth = 1;
    size_t pos   = brace + 1;
    while (pos < json.size() && depth > 0) {
        char c = json[pos];
        if (c == '{') {
            ++depth; ++pos;
        } else if (c == '}') {
            --depth;
            if (depth == 0) break;
            ++pos;
        } else if (c == '"') {
            ++pos;
            while (pos < json.size() && json[pos] != '"') {
                if (json[pos] == '\\') ++pos;
                ++pos;
            }
            if (pos < json.size()) ++pos;
        } else {
            ++pos;
        }
    }
    return json.substr(brace, pos - brace + 1);
}

/// Parse the array of "row" sub-objects from a datasets-server JSON response.
static std::vector<std::string> hf_extract_rows(const std::string& response_json) {
    std::vector<std::string> rows;
    size_t rows_key = response_json.find("\"rows\"");
    if (rows_key == std::string::npos) return rows;
    size_t arr_start = response_json.find('[', rows_key + 6);
    if (arr_start == std::string::npos) return rows;

    size_t pos = arr_start + 1;
    while (pos < response_json.size()) {
        while (pos < response_json.size() &&
               (std::isspace(static_cast<unsigned char>(response_json[pos])) ||
                response_json[pos] == ',')) {
            ++pos;
        }

        if (pos >= response_json.size() || response_json[pos] == ']') break;
        if (response_json[pos] != '{') { ++pos; continue; }

        int    depth     = 1;
        size_t obj_start = pos;
        ++pos;
        while (pos < response_json.size() && depth > 0) {
            char c = response_json[pos];
            if (c == '{') {
                ++depth; ++pos;
            } else if (c == '}') {
                --depth;
                if (depth == 0) break;
                ++pos;
            } else if (c == '"') {
                ++pos;
                while (pos < response_json.size() && response_json[pos] != '"') {
                    if (response_json[pos] == '\\') ++pos;
                    ++pos;
                }
                if (pos < response_json.size()) ++pos;
            } else {
                ++pos;
            }
        }
        if (pos < response_json.size()) ++pos;

        std::string outer_obj = response_json.substr(obj_start, pos - obj_start);
        std::string row_obj   = hf_extract_object(outer_obj, "row");
        if (!row_obj.empty()) rows.push_back(row_obj);
    }
    return rows;
}

/// Try to infer input/output field names from the first row's JSON.
static std::pair<std::string, std::string> hf_detect_fields(const std::string& row_json) {
    static const std::array<std::pair<const char*, const char*>, 10> candidates = {{
        {"instruction", "output"},
        {"instruction", "response"},
        {"question",    "answer"},
        {"question",    "response"},
        {"input",       "output"},
        {"prompt",      "completion"},
        {"prompt",      "response"},
        {"context",     "response"},
        {"source",      "target"},
        {"text",        "label"},
    }};
    for (const auto& [in_f, out_f] : candidates) {
        bool has_in  = row_json.find(std::string("\"") + in_f  + "\"") != std::string::npos;
        bool has_out = row_json.find(std::string("\"") + out_f + "\"") != std::string::npos;
        if (has_in && has_out) return {in_f, out_f};
    }
    return {"", ""};
}

// ============================================================================
// Private — convert_hf_to_training_data (moved verbatim from IncrementalTrainer.cpp)
// ============================================================================

/*static*/
bool DataFetcher::convert_hf_to_training_data(const std::string& rows_dir,
                                               const std::string& output_file,
                                               const std::string& input_field,
                                               const std::string& output_field,
                                               int max_pairs) {
    std::ofstream out(output_file);
    if (!out.is_open()) {
        Logger::error("Cannot create output file: {}", output_file);
        return false;
    }

    int pair_count  = 0;
    std::string det_in  = input_field;
    std::string det_out = output_field;

    for (int chunk = 0; pair_count < max_pairs; ++chunk) {
        std::ostringstream chunk_path;
        chunk_path << rows_dir << "/chunk_" << chunk << ".json";
        if (!fs::exists(chunk_path.str())) break;

        std::ifstream f(chunk_path.str());
        if (!f.is_open()) {
            Logger::error("Cannot open chunk: {}", chunk_path.str());
            continue;
        }
        std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        f.close();

        auto rows = hf_extract_rows(json);
        if (rows.empty()) {
            Logger::info("No rows parsed from chunk {} — stopping", chunk);
            break;
        }

        for (const auto& row_json : rows) {
            if (pair_count >= max_pairs) break;

            if (det_in.empty()) {
                auto [df_in, df_out] = hf_detect_fields(row_json);
                det_in  = df_in;
                det_out = df_out;
                if (!det_in.empty()) {
                    Logger::info("Auto-detected HF fields: input='{}' output='{}'", det_in,
                                 det_out);
                }
            }

            if (!det_in.empty() && !det_out.empty() && det_in == det_out) {
                // Single-text-field datasets — split at a sentence boundary near the middle
                std::string full_text = hf_extract_string(row_json, det_in);
                if (full_text.size() >= 40) {
                    size_t mid       = full_text.size() / 2;
                    size_t split_pos = std::string::npos;
                    for (size_t radius = 0; radius < mid; ++radius) {
                        for (size_t pos : {mid + radius, mid - radius}) {
                            if (pos >= full_text.size() - 1) continue;
                            char c = full_text[pos];
                            if ((c == '.' || c == '!' || c == '?') &&
                                pos + 1 < full_text.size() &&
                                (full_text[pos + 1] == ' ' || full_text[pos + 1] == '\n')) {
                                split_pos = pos + 1;
                                break;
                            }
                        }
                        if (split_pos != std::string::npos) break;
                    }
                    if (split_pos != std::string::npos && split_pos < full_text.size() - 5) {
                        std::string input_text  = full_text.substr(0, split_pos);
                        std::string output_text = full_text.substr(split_pos);
                        size_t start = output_text.find_first_not_of(" \n\t");
                        if (start != std::string::npos) output_text = output_text.substr(start);
                        if (!input_text.empty() && !output_text.empty()) {
                            out << "INPUT: " << input_text << "\n"
                                << "RESPONSE: " << output_text << "\n\n";
                            ++pair_count;
                        }
                    }
                }
            } else if (!det_in.empty() && !det_out.empty()) {
                // Key-value datasets (alpaca, dolly, OpenHermes, …)
                std::string input_text = hf_extract_string(row_json, det_in);
                if (det_in == "instruction") {
                    std::string sub = hf_extract_string(row_json, "input");
                    if (!sub.empty()) input_text += "\n" + sub;
                }
                std::string output_text = hf_extract_string(row_json, det_out);
                if (!input_text.empty() && !output_text.empty()) {
                    out << "INPUT: " << input_text << "\n"
                        << "RESPONSE: " << output_text << "\n\n";
                    ++pair_count;
                }
            } else {
                // Dialog-array datasets (daily_dialog, BlendedSkillTalk, …)
                static const std::array<const char*, 5> dialog_keys = {
                    "dialog", "turns", "utterances", "conversations", nullptr};
                for (const char* dk : dialog_keys) {
                    if (dk == nullptr) break;
                    if (pair_count >= max_pairs) break;
                    auto turns = hf_extract_string_array(row_json, dk);
                    if (turns.empty()) continue;
                    for (size_t i = 0; i + 1 < turns.size() && pair_count < max_pairs; i += 2) {
                        if (!turns[i].empty() && !turns[i + 1].empty()) {
                            out << "INPUT: " << turns[i] << "\n"
                                << "RESPONSE: " << turns[i + 1] << "\n\n";
                            ++pair_count;
                        }
                    }
                    break;
                }

                // Last resort: single-text-field with sentence splitting
                if (pair_count == 0) {
                    static const std::array<const char*, 4> text_keys = {
                        "text", "content", "document", nullptr};
                    for (const char* tk : text_keys) {
                        if (tk == nullptr) break;
                        std::string full_text = hf_extract_string(row_json, tk);
                        if (full_text.size() >= 40) {
                            det_in  = tk;
                            det_out = tk;
                            Logger::info(
                                "Auto-detected single-text field: '{}' — will split at sentence "
                                "boundaries",
                                tk);
                            size_t mid       = full_text.size() / 2;
                            size_t split_pos = std::string::npos;
                            for (size_t radius = 0; radius < mid; ++radius) {
                                for (size_t pos : {mid + radius, mid - radius}) {
                                    if (pos >= full_text.size() - 1) continue;
                                    char c = full_text[pos];
                                    if ((c == '.' || c == '!' || c == '?') &&
                                        pos + 1 < full_text.size() &&
                                        (full_text[pos + 1] == ' ' ||
                                         full_text[pos + 1] == '\n')) {
                                        split_pos = pos + 1;
                                        break;
                                    }
                                }
                                if (split_pos != std::string::npos) break;
                            }
                            if (split_pos != std::string::npos &&
                                split_pos < full_text.size() - 5) {
                                std::string input_text  = full_text.substr(0, split_pos);
                                std::string output_text = full_text.substr(split_pos);
                                size_t start = output_text.find_first_not_of(" \n\t");
                                if (start != std::string::npos)
                                    output_text = output_text.substr(start);
                                if (!input_text.empty() && !output_text.empty()) {
                                    out << "INPUT: " << input_text << "\n"
                                        << "RESPONSE: " << output_text << "\n\n";
                                    ++pair_count;
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    out.close();
    Logger::info("Wrote {} training pairs from HuggingFace dataset to: {}", pair_count,
                 output_file);
    return pair_count > 0;
}
