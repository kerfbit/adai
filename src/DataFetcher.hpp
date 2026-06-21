#pragma once

#include <string>
#include <utility>
#include <vector>

/**
 * @brief Configuration for DataFetcher.
 *
 * Controls where downloaded and converted data files are written.
 * All paths are relative to the working directory unless absolute.
 */
struct FetcherConfig {
    std::string gutenberg_output_dir   = "gutenberg_data";
    std::string huggingface_output_dir = "huggingface_data";
};

/**
 * @brief Stateless network fetcher for external training-data sources.
 *
 * Each public method downloads and converts a dataset from an external source,
 * writes the resulting JSONL training file to disk, and returns the
 * path.  The caller is responsible for enqueueing the result via
 * DatasetRegistry::add_file().
 *
 * DataFetcher has **no dependency on DatasetRegistry** or on any model or
 * tokenizer code.  It can be constructed and used without loading a model.
 *
 * Supported sources:
 *   - Project Gutenberg (plain-text books converted to QA pairs)
 *   - HuggingFace Datasets server (rows API, no Python required)
 *
 * Usage:
 * @code
 *   DataFetcher fetcher;
 *   std::string path = fetcher.fetch_gutenberg(1342, 500);
 *   if (!path.empty()) registry.add_file(path);
 * @endcode
 *
 * See docs/proposals/dataset_manager_separation.md (TD-028) for the full
 * design context and the planned Phase 6 dataset_manager CLI binary.
 */
class DataFetcher {
public:
    explicit DataFetcher(FetcherConfig cfg = {});

    // ── Project Gutenberg ──────────────────────────────────────────────────

    /**
     * @brief Download a Project Gutenberg book and convert it to training pairs.
     *
     * Downloads the book text, cleans the Gutenberg header/footer, extracts
     * sentences, and creates consecutive-sentence QA pairs.  The resulting
     * INPUT:/RESPONSE: file is written to
     * `config.gutenberg_output_dir/gutenberg_<id>_training.jsonl`.
     *
     * @param book_id   Gutenberg numeric book ID (e.g. 1342 for Pride and Prejudice).
     * @param num_pairs Maximum number of training pairs to generate (default 500).
     * @return Path to the produced `.jsonl` training file, or "" on any failure.
     */
    std::string fetch_gutenberg(int book_id, int num_pairs = 500);

    /**
     * @brief Download multiple Gutenberg books.
     *
     * Calls fetch_gutenberg() for each ID and collects the results.  Failed
     * books produce an empty string in the returned vector (same index as the
     * failed ID in @p ids).
     *
     * @param ids           List of Gutenberg book IDs to download.
     * @param num_pairs_each Maximum pairs per book (default 500).
     * @return Vector of training-file paths (empty string for each failure).
     */
    std::vector<std::string> fetch_gutenberg_batch(const std::vector<int>& ids,
                                                    int num_pairs_each = 500);

    // ── HuggingFace Datasets ───────────────────────────────────────────────

    /**
     * @brief Download a HuggingFace dataset and convert it to training pairs.
     *
     * Uses the HuggingFace datasets-server REST API — no Python or
     * huggingface_hub library required.  Rows are fetched as JSON in chunks of
     * 100 and written to the `config.huggingface_output_dir` directory before
     * conversion.
     *
     * Field names are auto-detected from the first row when @p input_field and
     * @p output_field are empty.  Supported auto-detection patterns include
     * instruction/output, question/answer, prompt/completion, and dialog arrays.
     * Set the HF_TOKEN environment variable to access gated datasets.
     *
     * @param dataset_id   HuggingFace dataset identifier, e.g. "daily_dialog" or
     *                     "tatsu-lab/alpaca".  Slashes are preserved.
     * @param num_pairs    Maximum training pairs to extract (default 500).
     * @param split        Dataset split (default "train").
     * @param input_field  JSON field for input text; empty = auto-detect.
     * @param output_field JSON field for output text; empty = auto-detect.
     * @return Path to the produced training file, or "" on any failure.
     */
    std::string fetch_huggingface(const std::string& dataset_id,
                                   int num_pairs = 500,
                                   const std::string& split        = "train",
                                   const std::string& input_field  = "",
                                   const std::string& output_field = "");

private:
    FetcherConfig config_;

    // ── Gutenberg private helpers ──────────────────────────────────────────

    static std::string get_gutenberg_url(int book_id);
    static bool        download_file(const std::string& url, const std::string& dest);
    static bool        download_gutenberg_book(int book_id, const std::string& output_dir);
    static std::string clean_gutenberg_text(const std::string& raw_text);
    static std::vector<std::string> extract_sentences(const std::string& text);
    static std::string generate_question_from_sentence(const std::string& sentence);
    static std::vector<std::pair<std::string, std::string>> create_qa_pairs_from_text(
        const std::vector<std::string>& sentences, int max_pairs);
    static bool convert_gutenberg_to_training_data(const std::string& text_file,
                                                    const std::string& output_file,
                                                    int max_pairs);

    // ── HuggingFace private helpers ────────────────────────────────────────

    static std::string download_hf_full_dataset(const std::string& dataset_id,
                                                 const std::string& split,
                                                 const std::string& output_dir);
    static bool convert_hf_to_training_data(const std::string& jsonl_file,
                                             const std::string& output_file,
                                             const std::string& input_field,
                                             const std::string& output_field,
                                             int max_pairs);
    // JSON helpers (hf_unescape, hf_extract_string, etc.) are file-local
    // static functions in DataFetcher.cpp and not exposed here.
};
