// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07

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
#include "ParquetReader.hpp"
#include "TrainingSampleMeta.hpp"

using adai::Logger;
namespace fs = std::filesystem;

// ============================================================================
// Constructor
// ============================================================================

DataFetcher::DataFetcher(FetcherConfig cfg) : config_(std::move(cfg)) {}

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
                      << "_training.jsonl";
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
// Public — Gutenberg cache + rotating-slice serving (registry_server use)
// ============================================================================

std::string DataFetcher::ensure_gutenberg_cached(int book_id) {
    const std::string dataset_dir =
        config_.gutenberg_output_dir + "/gutenberg_" + std::to_string(book_id);
    const std::string sentences_path = dataset_dir + "/sentences.txt";

    if (fs::exists(sentences_path) && fs::file_size(sentences_path) > 0) {
        Logger::info("Gutenberg book #{} already cached at '{}' — skipping download", book_id,
                     sentences_path);
        return sentences_path;
    }

    if (!fs::exists(dataset_dir)) {
        fs::create_directories(dataset_dir);
    }

    Logger::info("Downloading Project Gutenberg book #{} to cache...", book_id);
    const std::string text_file = dataset_dir + "/book.txt";
    if (!download_file(get_gutenberg_url(book_id), text_file)) {
        return "";
    }

    std::ifstream file(text_file);
    if (!file.is_open()) {
        Logger::error("Cannot open downloaded Gutenberg text: {}", text_file);
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    const std::string cleaned_text = clean_gutenberg_text(buffer.str());
    const std::vector<std::string> sentences = extract_sentences(cleaned_text);
    if (sentences.empty()) {
        Logger::error("No valid sentences found in Gutenberg book #{}", book_id);
        return "";
    }

    std::ofstream out(sentences_path, std::ios::trunc);
    if (!out.is_open()) {
        Logger::error("Cannot create sentence cache: {}", sentences_path);
        return "";
    }
    for (const auto& s : sentences) {
        out << s << "\n";
    }
    out.close();

    Logger::info("Cached {} sentences for Gutenberg book #{} at '{}'", sentences.size(), book_id,
                 sentences_path);
    return sentences_path;
}

/*static*/
int DataFetcher::convert_gutenberg_slice(const std::string& cached_sentences,
                                         const std::string& output_file, int offset_index,
                                         int num_pairs, int& next_offset_index) {
    std::ifstream f(cached_sentences);
    if (!f.is_open()) {
        Logger::error("Cannot open cached Gutenberg sentences: {}", cached_sentences);
        next_offset_index = offset_index;
        return 0;
    }
    std::vector<std::string> sentences;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty())
            sentences.push_back(line);
    }
    f.close();

    if (sentences.empty()) {
        next_offset_index = offset_index;
        return 0;
    }

    // Clamp an out-of-range (e.g. stale) offset to size() so the loop below
    // makes zero progress and the wraparound path takes over, rather than
    // reading out of bounds.
    const size_t start = (offset_index >= 0 && static_cast<size_t>(offset_index) < sentences.size())
                             ? static_cast<size_t>(offset_index)
                             : sentences.size();

    SampleMeta meta;
    meta.task_type = "qa";
    meta.domain = "literature";
    meta.language = "en";

    std::ofstream out(output_file, std::ios::trunc);
    if (!out.is_open()) {
        Logger::error("Cannot create output file: {}", output_file);
        next_offset_index = static_cast<int>(start);
        return 0;
    }

    size_t next_index = start;
    auto pairs = create_qa_pairs_from_text(sentences, num_pairs, start, &next_index);
    for (const auto& p : pairs) {
        out << sample_to_jsonl(p.first, p.second, meta) << "\n";
    }
    int pairs_written = static_cast<int>(pairs.size());
    next_offset_index = static_cast<int>(next_index);

    // Book exhausted before num_pairs was satisfied — wrap around to
    // sentence 0 once and top up the remainder, appending to the same file.
    if (pairs_written < num_pairs) {
        size_t next_index2 = 0;
        auto pairs2 =
            create_qa_pairs_from_text(sentences, num_pairs - pairs_written, 0, &next_index2);
        for (const auto& p : pairs2) {
            out << sample_to_jsonl(p.first, p.second, meta) << "\n";
        }
        pairs_written += static_cast<int>(pairs2.size());
        next_offset_index = static_cast<int>(next_index2);
        if (!pairs2.empty()) {
            Logger::info(
                "Gutenberg slice wrapped around to sentence 0 to satisfy remaining pair(s)");
        }
    }
    out.close();

    return pairs_written;
}

// ============================================================================
// Public — HuggingFace
// ============================================================================

std::string DataFetcher::fetch_huggingface(const std::string& dataset_id, int num_pairs,
                                           const std::string& split, const std::string& input_field,
                                           const std::string& output_field) {
    std::string safe_id = dataset_id;
    std::replace(safe_id.begin(), safe_id.end(), '/', '_');

    const std::string dataset_dir = config_.huggingface_output_dir + "/" + safe_id + "_" + split;
    const std::string train_file =
        config_.huggingface_output_dir + "/" + safe_id + "_" + split + "_training.jsonl";

    if (!fs::exists(config_.huggingface_output_dir)) {
        fs::create_directories(config_.huggingface_output_dir);
    }
    if (!fs::exists(dataset_dir)) {
        fs::create_directories(dataset_dir);
    }

    std::cout << "🤖 Fetching HuggingFace dataset '" << dataset_id << "' (split=" << split
              << ", target=" << num_pairs << " pairs)\n";

    const std::string jsonl_path = download_hf_full_dataset(dataset_id, split, dataset_dir);
    if (jsonl_path.empty()) {
        std::cerr << "❌ Failed to download dataset '" << dataset_id << "'.\n"
                  << "   • Check the dataset ID at https://huggingface.co/datasets\n"
                  << "   • Gated datasets require: export HF_TOKEN=hf_...\n"
                  << "   • Verify at: https://datasets-server.huggingface.co/is-valid?dataset="
                  << dataset_id << "\n";
        return "";
    }

    if (!convert_hf_to_training_data(jsonl_path, train_file, input_field, output_field,
                                     num_pairs)) {
        std::cerr << "❌ Could not extract training pairs from '" << dataset_id << "'.\n"
                  << "   Provide explicit field names: huggingface " << dataset_id << " "
                  << num_pairs << " " << split << " <input_field> <output_field>\n";
        return "";
    }

    return train_file;
}

// ============================================================================
// Public — HuggingFace cache + rotating-slice serving (registry_server use)
// ============================================================================

std::string DataFetcher::ensure_huggingface_cached(const std::string& dataset_id,
                                                   const std::string& split) {
    std::string safe_id = dataset_id;
    std::replace(safe_id.begin(), safe_id.end(), '/', '_');

    const std::string dataset_dir = config_.huggingface_output_dir + "/" + safe_id + "_" + split;
    const std::string jsonl_path = dataset_dir + "/full_dataset.jsonl";

    if (fs::exists(jsonl_path) && fs::file_size(jsonl_path) > 0) {
        Logger::info("HuggingFace dataset '{}' (split={}) already cached at '{}' — skipping download",
                     dataset_id, split, jsonl_path);
        return jsonl_path;
    }

    if (!fs::exists(config_.huggingface_output_dir)) {
        fs::create_directories(config_.huggingface_output_dir);
    }
    if (!fs::exists(dataset_dir)) {
        fs::create_directories(dataset_dir);
    }

    Logger::info("Downloading full HuggingFace dataset '{}' (split={}) to cache...", dataset_id,
                 split);
    return download_hf_full_dataset(dataset_id, split, dataset_dir);
}

/*static*/
int DataFetcher::convert_huggingface_slice(const std::string& cached_jsonl,
                                           const std::string& output_file,
                                           const std::string& input_field,
                                           const std::string& output_field, int offset_rows,
                                           int num_pairs, int& next_offset_rows) {
    int rows_consumed = 0;
    int pairs_written = 0;
    convert_hf_to_training_data(cached_jsonl, output_file, input_field, output_field, num_pairs,
                                offset_rows, /*append=*/false, &rows_consumed, &pairs_written);
    next_offset_rows = offset_rows + rows_consumed;

    // Dataset exhausted before num_pairs was satisfied — wrap around to row 0
    // once and top up the remainder, appending to the same output file.
    if (pairs_written < num_pairs) {
        int rows_consumed2 = 0;
        int pairs_written2 = 0;
        convert_hf_to_training_data(cached_jsonl, output_file, input_field, output_field,
                                    num_pairs - pairs_written, /*offset_rows=*/0, /*append=*/true,
                                    &rows_consumed2, &pairs_written2);
        pairs_written += pairs_written2;
        next_offset_rows = rows_consumed2;
        if (pairs_written2 > 0) {
            Logger::info(
                "HuggingFace slice wrapped around to row 0 to satisfy remaining {} pair(s)",
                num_pairs - (pairs_written - pairs_written2));
        }
    }

    return pairs_written;
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

    // Phase 13: strip non-essential structural content before sentence
    // extraction. Order matters — strip_toc() needs the real in-body chapter
    // markers still present to find where the table of contents ends, so it
    // must run before strip_chapter_markers() removes them.
    cleaned = strip_toc(cleaned);
    cleaned = strip_illustration_markers(cleaned);
    cleaned = strip_chapter_markers(cleaned);

    // Remove excessive whitespace
    cleaned = std::regex_replace(cleaned, std::regex("[ \\t]+"), " ");
    cleaned = std::regex_replace(cleaned, std::regex("\\n{3,}"), "\n\n");

    return cleaned;
}

namespace {
// Trim leading/trailing whitespace (space, tab, CR) — used by the Gutenberg
// cleaning helpers below to compare whole-line content robustly regardless
// of trailing \r from CRLF-encoded source texts.
std::string trim_line(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r");
    if (first == std::string::npos)
        return "";
    const auto last = s.find_last_not_of(" \t\r");
    return s.substr(first, last - first + 1);
}
}  // namespace

/*static*/
bool DataFetcher::is_standalone_chapter_marker_line(const std::string& trimmed_line) {
    // Matches ONLY when the entire line is the marker — nothing else. This is
    // what distinguishes a real in-body chapter heading ("CHAPTER I." alone
    // on its own line) from a table-of-contents entry ("CHAPTER I.     Down
    // the Rabbit-Hole", title trailing on the same line).
    static const std::regex marker_regex(R"(^(CHAPTER|BOOK|PART)\s+([IVXLCDM]+|[0-9]+)\.?$)",
                                         std::regex::icase);
    return std::regex_match(trimmed_line, marker_regex);
}

/*static*/
std::string DataFetcher::strip_illustration_markers(const std::string& text) {
    // Well-known, unambiguous Project Gutenberg transcription conventions —
    // low false-positive risk, safe to strip unconditionally wherever found.
    static const std::regex marker_regex(
        R"(\[(Illustration|Figure|Frontispiece|Footnote)[^\]]*\])", std::regex::icase);
    return std::regex_replace(text, marker_regex, "");
}

/*static*/
std::string DataFetcher::strip_chapter_markers(const std::string& text) {
    std::vector<std::string> lines;
    {
        std::istringstream iss(text);
        std::string line;
        while (std::getline(iss, line))
            lines.push_back(line);
    }

    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < lines.size(); ++i) {
        if (!is_standalone_chapter_marker_line(trim_line(lines[i]))) {
            out += lines[i];
            out += '\n';
            continue;
        }

        // Drop the marker line. Also drop the immediately-following non-blank
        // line if it looks like a chapter title (short, no terminal sentence
        // punctuation) rather than the start of real prose — left in place it
        // would otherwise bleed into and corrupt the chapter's first sentence
        // during extraction (extract_sentences() spans newlines). This is a
        // heuristic: a chapter with no separate title line whose first line
        // of prose happens to be very short could be mistakenly dropped too.
        size_t j = i + 1;
        while (j < lines.size() && trim_line(lines[j]).empty())
            ++j;
        if (j < lines.size()) {
            const std::string next_trimmed = trim_line(lines[j]);
            const char back = next_trimmed.empty() ? '\0' : next_trimmed.back();
            if (next_trimmed.size() <= 60 && back != '.' && back != '!' && back != '?') {
                i = j;  // also consume the title line
            }
        }
    }
    return out;
}

/*static*/
std::string DataFetcher::strip_toc(const std::string& text) {
    std::vector<std::string> lines;
    {
        std::istringstream iss(text);
        std::string line;
        while (std::getline(iss, line))
            lines.push_back(line);
    }

    static const std::regex toc_header_regex(R"(^(table of contents|contents)$)", std::regex::icase);
    constexpr size_t kMaxScanLines = 500;

    for (size_t i = 0; i < lines.size(); ++i) {
        if (!std::regex_match(trim_line(lines[i]), toc_header_regex))
            continue;

        // Found a TOC header — scan forward (bounded) for the first real,
        // standalone chapter marker; everything strictly in between is the
        // TOC block. If no marker is found within the scan window, bail
        // without stripping anything for this occurrence — better to leave a
        // stray "Contents" line in than risk deleting real content.
        const size_t scan_limit = std::min(lines.size(), i + 1 + kMaxScanLines);
        size_t end = i + 1;
        bool found_marker = false;
        for (; end < scan_limit; ++end) {
            if (is_standalone_chapter_marker_line(trim_line(lines[end]))) {
                found_marker = true;
                break;
            }
        }
        if (!found_marker)
            continue;

        std::string out;
        out.reserve(text.size());
        for (size_t k = 0; k < lines.size(); ++k) {
            if (k >= i && k < end)
                continue;
            out += lines[k];
            out += '\n';
        }
        return out;  // only the first TOC block is stripped
    }
    return text;
}

/*static*/
std::vector<std::string> DataFetcher::extract_sentences(const std::string& text) {
    std::vector<std::string> sentences;

    // Protect decimal/version numbers ("3.0", "1.5") from being mistaken for
    // sentence boundaries by the naive splitter below — swap the internal
    // period for a placeholder byte that isn't '.'/'!'/'?', then restore it
    // per extracted sentence. Without this, e.g. "EDITION 3.0" splits into
    // "...EDITION 3." + a stray leading "0 " on the next sentence.
    static const std::regex decimal_regex(R"((\d)\.(\d))");
    const std::string protected_text = std::regex_replace(text, decimal_regex, "$1\x01$2");

    std::regex sentence_regex("[^.!?]+[.!?]+");

    auto sentences_begin =
        std::sregex_iterator(protected_text.begin(), protected_text.end(), sentence_regex);
    auto sentences_end = std::sregex_iterator();

    for (std::sregex_iterator i = sentences_begin; i != sentences_end; ++i) {
        std::string sentence = i->str();

        sentence.erase(0, sentence.find_first_not_of(" \t\n\r"));
        sentence.erase(sentence.find_last_not_of(" \t\n\r") + 1);

        std::replace(sentence.begin(), sentence.end(), '\x01', '.');

        // The regex spans newlines, so a "sentence" can carry embedded
        // whitespace/newlines from the original line-wrapped text (or from a
        // paragraph break within it). Collapse to single spaces — both a
        // training-data quality improvement and required so cached sentences
        // can be safely stored one-per-line (Phase 13: ensure_gutenberg_cached).
        sentence = std::regex_replace(sentence, std::regex("\\s+"), " ");

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
    const std::vector<std::string>& sentences, int max_pairs, size_t start_index,
    size_t* out_next_index) {
    std::vector<std::pair<std::string, std::string>> pairs;

    size_t i = start_index;
    for (; !sentences.empty() && i < sentences.size() - 1 &&
           static_cast<int>(pairs.size()) < max_pairs;
         i += 2) {
        std::string question = generate_question_from_sentence(sentences[i]);
        std::string answer = sentences[i + 1];

        if (static_cast<int>(pairs.size()) < max_pairs) {
            pairs.emplace_back(question, answer);
        }

        if (i + 2 < sentences.size() && static_cast<int>(pairs.size()) < max_pairs) {
            std::string context = sentences[i] + " " + sentences[i + 1];
            std::string summary = sentences[i + 2];
            pairs.emplace_back("Summarize: " + context, summary);
        }
    }

    if (out_next_index)
        *out_next_index = i;
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

    std::string cleaned_text = clean_gutenberg_text(raw_text);
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

    SampleMeta meta;
    meta.task_type = "qa";
    meta.domain = "literature";
    meta.language = "en";

    for (const auto& pair : pairs) {
        out << sample_to_jsonl(pair.first, pair.second, meta) << "\n";
    }

    out.close();

    Logger::info("Training data saved to: {}", output_file);
    return true;
}

// ============================================================================
// Private — HuggingFace helpers (moved verbatim from IncrementalTrainer.cpp)
// ============================================================================

// Forward declaration — defined after the JSON helper block below.
static std::vector<std::string> hf_extract_parquet_urls(const std::string& json);

/*static*/
std::string DataFetcher::download_hf_full_dataset(const std::string& dataset_id,
                                                  const std::string& split,
                                                  const std::string& output_dir) {
    const char* hf_token = std::getenv("HF_TOKEN");
    const std::string jsonl_path = output_dir + "/full_dataset.jsonl";

    // 1. Fetch the parquet file list from the datasets-server /parquet endpoint
    const std::string info_path = output_dir + "/parquet_info.json";
    {
        std::ostringstream url;
        url << "https://datasets-server.huggingface.co/parquet" << "?dataset=" << dataset_id
            << "&config=default&split=" << split;

        std::ostringstream cmd;
        if (hf_token && hf_token[0] != '\0') {
            cmd << "curl -L -f -s -H \"Authorization: Bearer " << hf_token << "\"" << " -o \""
                << info_path << "\" \"" << url.str() << "\"";
        } else {
            cmd << "curl -L -f -s -o \"" << info_path << "\" \"" << url.str() << "\"";
        }
        if (std::system(cmd.str().c_str()) != 0 || !fs::exists(info_path) ||
            fs::file_size(info_path) == 0) {
            Logger::error("Failed to fetch parquet info for '{}'", dataset_id);
            return "";
        }
    }

    std::vector<std::string> parquet_urls;
    {
        std::ifstream f(info_path);
        std::string info_json((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
        parquet_urls = hf_extract_parquet_urls(info_json);
    }

    if (parquet_urls.empty()) {
        Logger::error("No parquet files found for dataset '{}' split '{}'", dataset_id, split);
        return "";
    }
    Logger::info("Found {} parquet file(s) for '{}'", parquet_urls.size(), dataset_id);

    // 2. Download each parquet file and convert it natively to JSONL (ParquetReader),
    // appending directly to jsonl_path — no python3/pandas/pyarrow subprocess, no
    // per-part temp .jsonl + `cat` concatenation.
    const std::string parquet_dir = output_dir + "/parquet";
    fs::create_directories(parquet_dir);

    int converted = 0;
    for (size_t i = 0; i < parquet_urls.size(); ++i) {
        const std::string pq_file = parquet_dir + "/part_" + std::to_string(i) + ".parquet";

        // Download
        std::ostringstream dl_cmd;
        if (hf_token && hf_token[0] != '\0') {
            dl_cmd << "curl -L -f -s -H \"Authorization: Bearer " << hf_token << "\"" << " -o \""
                   << pq_file << "\" \"" << parquet_urls[i] << "\"";
        } else {
            dl_cmd << "curl -L -f -s -o \"" << pq_file << "\" \"" << parquet_urls[i] << "\"";
        }
        if (std::system(dl_cmd.str().c_str()) != 0 || !fs::exists(pq_file) ||
            fs::file_size(pq_file) == 0) {
            Logger::error("Failed to download parquet part {}: {}", i, parquet_urls[i]);
            continue;
        }

        // Convert to JSONL — first successful part truncates jsonl_path, subsequent
        // parts append, so a failure on part 0 never leaves a stale file from a
        // previous run behind.
        const long long rows = ParquetReader::convert_to_jsonl(pq_file, jsonl_path,
                                                                /*append=*/converted > 0);
        if (rows < 0) {
            Logger::error("Parquet to JSONL conversion failed for part {} ({})", i, pq_file);
            continue;
        }

        ++converted;
        std::cout << "  ✓ parquet part " << (i + 1) << "/" << parquet_urls.size() << " (" << rows
                  << " rows)\n";
    }

    if (converted == 0 || !fs::exists(jsonl_path) || fs::file_size(jsonl_path) == 0) {
        Logger::error("No parquet files successfully converted for '{}'", dataset_id);
        return "";
    }

    Logger::info("Dataset '{}': {}/{} parquet files converted to JSONL", dataset_id, converted,
                 parquet_urls.size());
    return jsonl_path;
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
                case 'n':
                    out += '\n';
                    break;
                case 't':
                    out += '\t';
                    break;
                case 'r':
                    out += '\r';
                    break;
                case '"':
                    out += '"';
                    break;
                case '\\':
                    out += '\\';
                    break;
                case '/':
                    out += '/';
                    break;
                case 'b':
                    out += '\b';
                    break;
                case 'f':
                    out += '\f';
                    break;
                default:
                    out += '\\';
                    out += s[i];
                    break;
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
    if (kpos == std::string::npos)
        return "";
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos)
        return "";

    size_t vpos = colon + 1;
    while (vpos < json.size() && std::isspace(static_cast<unsigned char>(json[vpos])))
        ++vpos;
    if (vpos >= json.size() || json[vpos] != '"')
        return "";

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
    if (kpos == std::string::npos)
        return result;
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos)
        return result;
    size_t bracket = json.find('[', colon + 1);
    if (bracket == std::string::npos)
        return result;

    int depth = 1;
    size_t pos = bracket + 1;
    size_t arr_end = std::string::npos;
    while (pos < json.size() && depth > 0) {
        char c = json[pos];
        if (c == '[') {
            ++depth;
            ++pos;
        } else if (c == ']') {
            --depth;
            if (depth == 0) {
                arr_end = pos;
                break;
            }
            ++pos;
        } else if (c == '"') {
            ++pos;
            while (pos < json.size() && json[pos] != '"') {
                if (json[pos] == '\\')
                    ++pos;
                ++pos;
            }
            if (pos < json.size())
                ++pos;
        } else {
            ++pos;
        }
    }
    if (arr_end == std::string::npos)
        return result;

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
            if (p < arr.size())
                ++p;
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
    if (kpos == std::string::npos)
        return "";
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos)
        return "";
    size_t brace = json.find('{', colon + 1);
    if (brace == std::string::npos)
        return "";

    int depth = 1;
    size_t pos = brace + 1;
    while (pos < json.size() && depth > 0) {
        char c = json[pos];
        if (c == '{') {
            ++depth;
            ++pos;
        } else if (c == '}') {
            --depth;
            if (depth == 0)
                break;
            ++pos;
        } else if (c == '"') {
            ++pos;
            while (pos < json.size() && json[pos] != '"') {
                if (json[pos] == '\\')
                    ++pos;
                ++pos;
            }
            if (pos < json.size())
                ++pos;
        } else {
            ++pos;
        }
    }
    return json.substr(brace, pos - brace + 1);
}

/// Try to infer input/output field names from the first row's JSON.
///
/// A candidate is only accepted if BOTH fields hold actual non-empty JSON
/// *string* values in this row — not merely if the key names are present.
/// A field that matches by name but holds a number/bool/null/array/object
/// (e.g. a classification dataset's numeric "label" column, which happens to
/// match the {"text","label"} candidate by name) is unusable as free-text
/// input/output and would silently produce zero pairs for the *entire* file,
/// since the detected field pair is locked in after this first check and
/// never revisited. Skipping such a candidate here lets the caller fall
/// through to the dialog-array / single-text-field fallbacks instead, which
/// correctly extract from just the "text" field in exactly this situation.
static std::pair<std::string, std::string> hf_detect_fields(const std::string& row_json) {
    static const std::array<std::pair<const char*, const char*>, 10> candidates = {{
        {"instruction", "output"},
        {"instruction", "response"},
        {"question", "answer"},
        {"question", "response"},
        {"input", "output"},
        {"prompt", "completion"},
        {"prompt", "response"},
        {"context", "response"},
        {"source", "target"},
        {"text", "label"},
    }};
    for (const auto& [in_f, out_f] : candidates) {
        if (!hf_extract_string(row_json, in_f).empty() && !hf_extract_string(row_json, out_f).empty())
            return {in_f, out_f};
    }
    return {"", ""};
}

/// Extract parquet file URLs from a datasets-server /parquet API response.
static std::vector<std::string> hf_extract_parquet_urls(const std::string& json) {
    std::vector<std::string> urls;
    size_t arr_key = json.find("\"parquet_files\"");
    if (arr_key == std::string::npos)
        return urls;
    size_t arr_start = json.find('[', arr_key);
    if (arr_start == std::string::npos)
        return urls;

    size_t pos = arr_start + 1;
    while (pos < json.size()) {
        while (pos < json.size() &&
               (std::isspace(static_cast<unsigned char>(json[pos])) || json[pos] == ','))
            ++pos;
        if (pos >= json.size() || json[pos] == ']')
            break;
        if (json[pos] != '{') {
            ++pos;
            continue;
        }

        int depth = 1;
        size_t obj_start = pos++;
        while (pos < json.size() && depth > 0) {
            char c = json[pos];
            if (c == '{') {
                ++depth;
                ++pos;
            } else if (c == '}') {
                if (--depth == 0)
                    break;
                ++pos;
            } else if (c == '"') {
                ++pos;
                while (pos < json.size() && json[pos] != '"') {
                    if (json[pos] == '\\')
                        ++pos;
                    ++pos;
                }
                if (pos < json.size())
                    ++pos;
            } else {
                ++pos;
            }
        }
        if (pos < json.size())
            ++pos;

        std::string obj = json.substr(obj_start, pos - obj_start);
        std::string url = hf_extract_string(obj, "url");
        if (!url.empty())
            urls.push_back(url);
    }
    return urls;
}

// ============================================================================
// Private — convert_hf_to_training_data
// Reads a JSONL file (one JSON object per line) produced by download_hf_full_dataset
// and writes up to max_pairs training JSONL lines with inferred metadata.
// ============================================================================

// Infer task_type from the auto-detected HuggingFace field names.
static std::string hf_task_type(const std::string& in_field) {
    if (in_field == "instruction")
        return "instruction";
    if (in_field == "question")
        return "qa";
    if (in_field == "prompt")
        return "completion";
    return "instruction";
}

/*static*/
bool DataFetcher::convert_hf_to_training_data(const std::string& jsonl_file,
                                              const std::string& output_file,
                                              const std::string& input_field,
                                              const std::string& output_field, int max_pairs,
                                              int offset_rows, bool append,
                                              int* out_rows_consumed, int* out_pairs_written) {
    std::ifstream f(jsonl_file);
    if (!f.is_open()) {
        Logger::error("Cannot open JSONL file: {}", jsonl_file);
        return false;
    }
    std::ofstream out(output_file, append ? std::ios::app : std::ios::trunc);
    if (!out.is_open()) {
        Logger::error("Cannot create output file: {}", output_file);
        return false;
    }

    int pair_count = 0;
    int row_index = 0;        // valid (non-empty, '{'-prefixed) rows seen so far
    int rows_consumed = 0;    // valid rows advanced past offset_rows this call
    std::string det_in = input_field;
    std::string det_out = output_field;
    std::string det_task_type;
    std::string line;

    // Helper: split a long text at a sentence boundary near its midpoint.
    auto mid_split = [](const std::string& text, std::string& left, std::string& right) -> bool {
        if (text.size() < 40)
            return false;
        size_t mid = text.size() / 2;
        size_t split_pos = std::string::npos;
        for (size_t radius = 0; radius < mid && split_pos == std::string::npos; ++radius) {
            for (size_t pos : {mid + radius, mid - radius}) {
                if (pos >= text.size() - 1)
                    continue;
                char c = text[pos];
                if ((c == '.' || c == '!' || c == '?') && pos + 1 < text.size() &&
                    (text[pos + 1] == ' ' || text[pos + 1] == '\n')) {
                    split_pos = pos + 1;
                    break;
                }
            }
        }
        if (split_pos == std::string::npos || split_pos >= text.size() - 5)
            return false;
        left = text.substr(0, split_pos);
        right = text.substr(split_pos);
        size_t start = right.find_first_not_of(" \n\t");
        if (start != std::string::npos)
            right = right.substr(start);
        return !left.empty() && !right.empty();
    };

    while (std::getline(f, line) && pair_count < max_pairs) {
        if (line.empty() || line.front() != '{')
            continue;

        // Skip rows already served on a prior call (Phase 12 rotating slices) —
        // still counts toward row_index so the cursor advances monotonically,
        // but doesn't run field detection or emit pairs.
        if (row_index < offset_rows) {
            ++row_index;
            continue;
        }
        ++row_index;
        ++rows_consumed;
        const std::string& row_json = line;

        if (det_in.empty()) {
            auto [df_in, df_out] = hf_detect_fields(row_json);
            det_in = df_in;
            det_out = df_out;
            det_task_type = det_in.empty() ? "" : hf_task_type(det_in);
            if (!det_in.empty()) {
                Logger::info("Auto-detected HF fields: input='{}' output='{}' task_type='{}'",
                             det_in, det_out, det_task_type);
            }
        }

        SampleMeta meta;
        meta.task_type = det_task_type;

        if (!det_in.empty() && !det_out.empty() && det_in == det_out) {
            // Single-text-field datasets — split at sentence boundary near middle
            std::string full_text = hf_extract_string(row_json, det_in);
            std::string left, right;
            if (mid_split(full_text, left, right)) {
                meta.task_type = "completion";
                out << sample_to_jsonl(left, right, meta) << "\n";
                ++pair_count;
            }
        } else if (!det_in.empty() && !det_out.empty()) {
            // Key-value datasets (alpaca, dolly, OpenHermes, …)
            std::string input_text = hf_extract_string(row_json, det_in);
            if (det_in == "instruction") {
                std::string sub = hf_extract_string(row_json, "input");
                if (!sub.empty())
                    input_text += "\n" + sub;
            }
            std::string output_text = hf_extract_string(row_json, det_out);
            if (!input_text.empty() && !output_text.empty()) {
                out << sample_to_jsonl(input_text, output_text, meta) << "\n";
                ++pair_count;
            }
        } else {
            // Dialog-array datasets (daily_dialog, BlendedSkillTalk, …)
            static const std::array<const char*, 5> dialog_keys = {"dialog", "turns", "utterances",
                                                                   "conversations", nullptr};
            bool handled = false;
            for (const char* dk : dialog_keys) {
                if (dk == nullptr || pair_count >= max_pairs)
                    break;
                auto turns = hf_extract_string_array(row_json, dk);
                if (turns.empty())
                    continue;
                SampleMeta chat_meta;
                chat_meta.task_type = "chat";
                for (size_t i = 0; i + 1 < turns.size() && pair_count < max_pairs; i += 2) {
                    if (!turns[i].empty() && !turns[i + 1].empty()) {
                        out << sample_to_jsonl(turns[i], turns[i + 1], chat_meta) << "\n";
                        ++pair_count;
                    }
                }
                handled = true;
                break;
            }

            // Last resort: single-text-field with sentence splitting
            if (!handled && pair_count == 0) {
                static const std::array<const char*, 4> text_keys = {"text", "content", "document",
                                                                     nullptr};
                for (const char* tk : text_keys) {
                    if (tk == nullptr)
                        break;
                    std::string full_text = hf_extract_string(row_json, tk);
                    if (full_text.size() < 40)
                        continue;
                    det_in = tk;
                    det_out = tk;
                    det_task_type = "completion";
                    Logger::info(
                        "Auto-detected single-text field: '{}' — will split at sentence boundaries",
                        tk);
                    std::string left, right;
                    if (mid_split(full_text, left, right)) {
                        SampleMeta comp_meta;
                        comp_meta.task_type = "completion";
                        out << sample_to_jsonl(left, right, comp_meta) << "\n";
                        ++pair_count;
                    }
                    break;
                }
            }
        }
    }

    out.close();
    Logger::info("Wrote {} training pairs from HuggingFace dataset to: {} (rows {}..{})",
                 pair_count, output_file, offset_rows, offset_rows + rows_consumed);

    if (out_rows_consumed)
        *out_rows_consumed = rows_consumed;
    if (out_pairs_written)
        *out_pairs_written = pair_count;
    return pair_count > 0;
}
