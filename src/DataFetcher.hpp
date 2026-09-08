#pragma once

// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07


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
    std::string gutenberg_output_dir = "gutenberg_data";
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
 *   - HuggingFace Datasets server (parquet exports, converted natively via
 *     ParquetReader — no Python required)
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

    // ── Gutenberg: cache + rotating-slice serving (registry_server use) ────

    /**
     * @brief Ensure the book is downloaded, cleaned, and split into
     *        sentences, cached to a stable path — downloading only on a
     *        cache miss.
     *
     * Mirrors ensure_huggingface_cached(): a prior fetch_gutenberg() or
     * ensure_gutenberg_cached() call for the same book_id counts as a cache
     * hit and no network request is made.
     *
     * @return Path to the cached sentences file (one cleaned sentence per
     *         line), or "" on failure.
     */
    std::string ensure_gutenberg_cached(int book_id);

    /**
     * @brief Convert up to @p num_pairs QA pairs starting at sentence
     *        @p offset_index from a cached sentence file (as produced by
     *        ensure_gutenberg_cached()) into @p output_file.
     *
     * Wraps around to sentence 0 exactly once if the book is exhausted
     * before @p num_pairs is satisfied — mirrors convert_huggingface_slice().
     *
     * @param[out] next_offset_index Sentence index to resume from next call.
     * @return Number of pairs actually written (0 = failure / empty book).
     */
    static int convert_gutenberg_slice(const std::string& cached_sentences,
                                       const std::string& output_file, int offset_index,
                                       int num_pairs, int& next_offset_index);

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
    std::string fetch_huggingface(const std::string& dataset_id, int num_pairs = 500,
                                  const std::string& split = "train",
                                  const std::string& input_field = "",
                                  const std::string& output_field = "");

    // ── HuggingFace: cache + rotating-slice serving (registry_server use) ──

    /**
     * @brief Ensure the full HuggingFace dataset is downloaded to a stable
     *        cache path, downloading only if not already present.
     *
     * Reuses the same `config.huggingface_output_dir/<safe_id>_<split>/
     * full_dataset.jsonl` layout as fetch_huggingface(), so a prior
     * fetch_huggingface() call (or a prior call to this method) counts as a
     * cache hit and no network request is made.
     *
     * @return Path to the cached full-dataset JSONL file, or "" on failure.
     */
    std::string ensure_huggingface_cached(const std::string& dataset_id, const std::string& split);

    /**
     * @brief Convert up to @p num_pairs pairs starting at row @p offset_rows
     *        from a cached full-dataset JSONL (as produced by
     *        ensure_huggingface_cached()) into @p output_file.
     *
     * Wraps around to row 0 exactly once if the dataset is exhausted before
     * @p num_pairs is satisfied, so repeated calls with an advancing
     * @p offset_rows eventually cycle back to the start rather than starving.
     *
     * @param[out] next_offset_rows Row index to resume from on the next call.
     * @return Number of pairs actually written (0 = failure / empty dataset).
     */
    static int convert_huggingface_slice(const std::string& cached_jsonl,
                                         const std::string& output_file,
                                         const std::string& input_field,
                                         const std::string& output_field, int offset_rows,
                                         int num_pairs, int& next_offset_rows);

   private:
    FetcherConfig config_;

    // ── Gutenberg private helpers ──────────────────────────────────────────

    static std::string get_gutenberg_url(int book_id);
    static bool download_file(const std::string& url, const std::string& dest);
    static bool download_gutenberg_book(int book_id, const std::string& output_dir);
    static std::string clean_gutenberg_text(const std::string& raw_text);
    // Cleaning sub-steps (Phase 13), applied by clean_gutenberg_text() before
    // whitespace normalization. Heuristic, best-effort on Gutenberg's loosely
    // standardized plain-text format — not a guarantee on every book.
    static std::string strip_illustration_markers(const std::string& text);
    static std::string strip_chapter_markers(const std::string& text);
    static std::string strip_toc(const std::string& text);
    // True if, after trimming, the line is *exactly* a chapter/book/part
    // marker ("CHAPTER I.", "BOOK 2", ...) with nothing else on it — this is
    // what distinguishes a real in-body marker from a table-of-contents entry
    // (which has the chapter title trailing on the same line). Shared by
    // strip_chapter_markers() and strip_toc().
    static bool is_standalone_chapter_marker_line(const std::string& trimmed_line);
    static std::vector<std::string> extract_sentences(const std::string& text);
    static std::string generate_question_from_sentence(const std::string& sentence);
    // start_index/out_next_index are Phase 13 additions (see
    // convert_gutenberg_slice) — both default so the existing call site in
    // convert_gutenberg_to_training_data() is unaffected.
    static std::vector<std::pair<std::string, std::string>> create_qa_pairs_from_text(
        const std::vector<std::string>& sentences, int max_pairs, size_t start_index = 0,
        size_t* out_next_index = nullptr);
    static bool convert_gutenberg_to_training_data(const std::string& text_file,
                                                   const std::string& output_file, int max_pairs);

    // ── HuggingFace private helpers ────────────────────────────────────────

    static std::string download_hf_full_dataset(const std::string& dataset_id,
                                                const std::string& split,
                                                const std::string& output_dir);
    // offset_rows/append/out_rows_consumed/out_pairs_written are Phase 12 additions
    // (see convert_huggingface_slice) — all default so the existing call site in
    // fetch_huggingface() is unaffected.
    static bool convert_hf_to_training_data(const std::string& jsonl_file,
                                            const std::string& output_file,
                                            const std::string& input_field,
                                            const std::string& output_field, int max_pairs,
                                            int offset_rows = 0, bool append = false,
                                            int* out_rows_consumed = nullptr,
                                            int* out_pairs_written = nullptr);
    // JSON helpers (hf_unescape, hf_extract_string, etc.) are file-local
    // static functions in DataFetcher.cpp and not exposed here.

    // Grants tests/DataFetcherTests.cpp direct access to the private Phase 13
    // Gutenberg-cleaning helpers (strip_toc, strip_chapter_markers, ...) so
    // they can be unit-tested with hand-written fixture strings without a
    // network call, without widening the public API surface.
    friend class DataFetcherGutenbergCleaningTest;
};
