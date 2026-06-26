#pragma once

#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// Hot-reloading implemented - see DAEMON_IMPLEMENTATION_COMPLETE.md
// TODO: See TECHNICAL_DEBT.md Future Enhancement #4 - JSON Configuration Format Support
//       Add support for JSON format in addition to key=value format
// TODO: See TECHNICAL_DEBT.md Future Enhancement #5 - Configuration Profiles
//       Support named profiles (dev, staging, prod) with inheritance

namespace adai {

/// GPU scheduling strategy — controls CUDA stream priority at initialisation.
enum class GPUStrategy : std::uint8_t {
    BACKGROUND, ///< Low-priority stream; yields to other GPU work (default)
    FULL,       ///< High-priority stream; maximises training throughput
};

/// Parse a GPU_STRATEGY string; warns and returns BACKGROUND on unknown input.
inline GPUStrategy gpu_strategy_from_string(const std::string& s) {
    if (s == "full")       return GPUStrategy::FULL;
    if (s == "background") return GPUStrategy::BACKGROUND;
    std::cerr << "Warning: unknown GPU_STRATEGY '" << s << "', using 'background'\n";
    return GPUStrategy::BACKGROUND;
}

/**
 * @brief Configuration for the ADAI chatbot service.
 *
 * Configuration values are loaded with the following priority (highest to lowest):
 * 1. Environment variables
 * 2. Configuration file
 * 3. Default values
 */
struct ServiceConfig {
    // ============================================================
    // Server Configuration
    // ============================================================

    /// Path to the model file
    std::string model_path;

    /// Path to the vocabulary file
    std::string vocab_path;

    /// Server port (default: 8080)
    int port = 8080;

    /// Session timeout in minutes (default: 30)
    int session_timeout = 30;

    /// Logging level: DEBUG, INFO, WARN, ERROR (default: INFO)
    std::string log_level = "INFO";

    /// Log file path (empty = console only, set to enable file logging)
    std::string log_file_path;

    /// Directory for training session artefacts (checkpoints, logs, metrics)
    std::string session_dir = "training_sessions";

    /// Maximum log file size in MB before rotation (default: 10)
    size_t log_max_size_mb = 10;

    /// Maximum number of rotated log files to keep (default: 5)
    size_t log_max_files = 5;

    /// Enable log file compression (default: false)
    bool log_compress = false;

    // ============================================================
    // Model Architecture Parameters
    // ============================================================

    /// Model dimension (default: 512)
    size_t d_model = 512;

    /// Number of attention heads (default: 8)
    size_t num_heads = 8;

    /// Feed-forward dimension (default: 2048)
    size_t d_ff = 2048;

    /// Number of encoder layers (default: 6)
    size_t num_encoder_layers = 6;

    /// Number of decoder layers (default: 6)
    size_t num_decoder_layers = 6;

    /// Maximum sequence length (default: 1024)
    size_t max_seq_length = 1024;

    // ============================================================
    // Training Hyperparameters
    // ============================================================

    /// Initial learning rate (default: 0.0001)
    float learning_rate = 0.0001f;

    /// Number of training epochs (default: 10)
    int num_epochs = 10;

    /// L2 weight decay regularization (default: 0.01)
    float weight_decay = 0.01f;

    /// Gradient clipping norm; 0 = disabled (default: 1.0)
    float gradient_clip = 1.0f;

    // Adaptive gradient clipping (TD-017)
    bool adaptive_gradient_clip = false;    ///< Master switch; false = legacy fixed-clip
    float gradient_clip_min = 0.1f;         ///< Hard floor — threshold never drops below this
    float gradient_clip_max = 5.0f;         ///< Hard ceiling — threshold never rises above this
    float gradient_clip_ema_decay = 0.05f;  ///< EMA smoothing factor α (0 < α ≤ 1)
    float gradient_clip_headroom = 2.0f;    ///< Threshold = ema_norm × headroom
    int gradient_clip_warmup_steps = 100;   ///< Steps before adaptive logic activates
    float gradient_clip_spike_k = 5.0f;     ///< Outlier: norms > k×ema are not fed into EMA

    /// Batch size / gradient accumulation steps (default: 1)
    int batch_size = 1;

    // ============================================================
    // Generation Parameters
    // ============================================================

    /// Maximum generation length (default: 100)
    size_t max_gen_length = 100;

    /// Generation temperature (default: 1.0)
    float temperature = 1.0f;

    /// Nucleus sampling threshold (default: 0.9)
    float top_p = 0.9f;

    /// Top-k sampling parameter (default: 50)
    int top_k = 50;

    /// Beam search width (default: 4)
    int beam_width = 4;

    /// Generation strategy: greedy, beam, temperature, top_k, nucleus (default: nucleus)
    std::string strategy = "nucleus";

    // ============================================================
    // Training Metrics Service Configuration
    // ============================================================

    /// Enable training metrics service (default: true)
    bool enable_metrics_service = true;

    /// URL of metrics API daemon; empty string disables push reporting (default: http://localhost:8081)
    std::string metrics_server_url = "http://localhost:8081";

    // TODO: See TECHNICAL_DEBT.md TD-018 - Add METRICS_SESSION_KEY (string, default "0-default"),
    //   METRICS_MAX_LIVE_SESSIONS (int, default 16), METRICS_COMPLETED_TTL_SECONDS (int, default 3600),
    //   METRICS_SWEEP_INTERVAL_SECONDS (int, default 60); parse in Config.cpp alongside existing keys.

    /// HTTP timeout for pushing metrics in milliseconds (default: 1000)
    int metrics_push_timeout_ms = 1000;

    /// Enable persistence of metrics to disk (default: true)
    bool metrics_enable_persistence = true;

    /// Path to metrics JSONL file (default: training_sessions/metrics.jsonl)
    std::string metrics_file = "training_sessions/metrics.jsonl";

    /// Path to metrics summary JSON file (default: training_sessions/metrics_summary.json)
    std::string metrics_summary_file = "training_sessions/metrics_summary.json";

    /// Persist metrics every N samples (default: 100)
    int metrics_persist_every_samples = 100;

    /// Persist metrics every N seconds (default: 30)
    int metrics_persist_every_seconds = 30;

    /// Maximum records in memory (default: 10000)
    int metrics_max_records_in_memory = 10000;

    /// Maximum records on disk (default: 100000)
    int metrics_max_records_on_disk = 100000;

    /// Enable Prometheus format export (default: false)
    bool metrics_enable_prometheus = false;

    /// Path to Prometheus metrics file (default: training_sessions/metrics.prom)
    std::string metrics_prometheus_file = "training_sessions/metrics.prom";

    /// Session key for trainer-side session-scoped metrics routes.
    /// Empty value means the trainer should derive one at runtime.
    std::string metrics_session_key;

    /// Human-readable session label surfaced in /api/sessions responses (TD-021).
    /// Empty value means the trainer should auto-derive one at runtime.
    std::string metrics_session_label;

    /// Maximum number of live metrics sessions in the registry (default: 16).
    size_t metrics_max_live_sessions = 16;

    /// Retention for completed sessions in memory before eviction (default: 3600).
    int metrics_completed_ttl_seconds = 3600;

    /// Background sweep cadence for stale completed sessions (default: 60).
    int metrics_sweep_interval_seconds = 60;

    /// Seconds without a metrics ingest before a session is considered stale (default: 60).
    /// When a live session exceeds this threshold, is_stale=true and effective_is_training=false
    /// are set in snapshot reads and API responses so dashboards don't show false "running" state.
    int metrics_staleness_threshold_seconds = 60;

    /// Port for metrics API server daemon (default: 8081)
    int metrics_api_port = 8081;

    /// Allow control endpoints in metrics API (default: true)
    bool metrics_api_allow_control = true;

    // ============================================================
    // Metrics Database Persistence (TD-020)
    // ============================================================

    /// Storage backend: "file", "sqlite", "postgres", "sqlite+file", "postgres+file"
    std::string metrics_storage_backend = "sqlite+file";

    /// SQLite database file path (default: training_sessions/metrics.db)
    std::string metrics_db_path = "training_sessions/metrics.db";

    /// PostgreSQL connection URL (used when backend includes "postgres")
    std::string metrics_db_url;

    /// PostgreSQL connection pool size (default: 4)
    int metrics_db_pool_size = 4;

    // ============================================================
    // Distributed Registry Configuration (TD-028 Phase 9)
    // ============================================================

    /// URL of the registry_server daemon; empty string = local flat-file mode (default: "")
    std::string registry_server_url;

    /// Logical namespace for multi-project registry_server sharing.
    /// Defaults to the session_dir basename when not set.
    std::string run_group;

    /// Per-process training run identifier. Auto-derived from hostname+PID when empty.
    std::string run_id;

    /// HTTP timeout for registry_server calls in milliseconds (default: 5000)
    int registry_timeout_ms = 5000;

    // ============================================================
    // FTP Dataset Transport Configuration (Phase 10)
    // Server-side (registry_server): FTP listener and token settings.
    // Client-side (incremental_trainer): download directory and limits.
    // ============================================================

    /// FTP control port for the embedded FtpDataServer (default: 2121)
    int ftp_server_port = 2121;

    /// Lower bound of the PASV data port range (default: 50000)
    int ftp_pasv_port_min = 50000;

    /// Upper bound of the PASV data port range (default: 50099)
    int ftp_pasv_port_max = 50099;

    /// Per-file FTP token TTL in minutes (default: 30)
    int ftp_token_ttl_minutes = 30;

    /// HMAC key used for token signing (Phase 3); change in production
    std::string ftp_data_server_secret = "change-me-in-production";

    /// Trainer-owned directory for FTP downloads (default: "")
    /// Empty = disabled; FTP transport is skipped even when the server sends tokens.
    std::string download_dir;

    /// Maximum concurrent FTP connections in fetch_all() (default: 4)
    int max_parallel_downloads = 4;

    /// Log a warning when a file transfer exceeds this size in MB; 0 = disabled (default: 500)
    int large_file_warn_threshold_mb = 500;

    // ── Phase 3: Security hardening ───────────────────────────────────────

    /// Max concurrent FTP sessions per run_id; 0 = unlimited (default: 4)
    int ftp_max_sessions_per_run = 4;

    /// Enable FTPS (FTP over TLS) — encrypts credentials and data in transit (default: false)
    bool ftps_enabled = false;

    /// Path to PEM TLS certificate for the FTPS server; empty = generate self-signed
    std::string ftp_cert_file;

    /// Path to PEM TLS private key for the FTPS server; empty = generate self-signed
    std::string ftp_key_file;

    // ============================================================
    // Outlier Detection Configuration (TD-021)
    // ============================================================

    /// Flag sample if loss z-score exceeds this value (default: 3.0)
    float loss_outlier_z_threshold = 3.0f;

    /// Flag sample if gradient norm exceeds this value (default: 10.0)
    float grad_norm_outlier_threshold = 10.0f;

    /// Maximum number of abnormal samples reported per training run (default: 1000)
    int max_abnormal_samples = 1000;

    // ============================================================
    // Generation Quality Metrics Configuration
    // ============================================================

    /// Enable BLEU/ROUGE generation quality scoring during validation (default: false)
    bool enable_generation_quality_metrics = false;

    /// Number of validation samples to score per epoch (default: 10)
    int generation_quality_sample_size = 10;

    /// Max tokens per generate_response() call during scoring (default: 50)
    int generation_quality_max_tokens = 50;

    /// Minimum sample size to trigger async parallel scoring (default: 50, TD-023).
    /// Below this threshold the existing synchronous path is used.
    int generation_quality_async_threshold = 50;

    // ============================================================
    // RAG Configuration
    // ============================================================

    /// Enable Retrieval-Augmented Generation (default: false)
    bool rag_enabled = false;

    /// Path to directory containing .txt documents to index (default: "")
    std::string rag_docs_path;

    /// Number of documents to retrieve per query (default: 3)
    int rag_num_docs = 3;

    /// Minimum cosine similarity score for retrieval; 0 = no filter (default: 0.0)
    float rag_threshold = 0.0f;

    /// Maximum context length in tokens (default: 512)
    int rag_max_context_length = 512;

    // ============================================================
    // Model Name Service Configuration
    // ============================================================

    /// URL of the ModelNameService daemon; empty = local file-path mode (default: "")
    std::string name_service_url;

    /// Listen port for the mns_server daemon (default: 8083)
    int name_service_port = 8083;

    /// Storage directory for mns_server JSONL files (default: "name_service")
    std::string name_service_dir = "name_service";

    /// HTTP timeout for ModelNameClient calls in milliseconds (default: 5000)
    int name_service_timeout_ms = 5000;

    /// Human-readable model name registered in the MNS (default: "")
    std::string model_name;

    /// Role used for resolve_role() at startup (e.g. "chatbot"); empty = use model_name
    std::string model_role;

    // ============================================================
    // GPU / CUDA Configuration
    // Only used when the binary is built with -DENABLE_GPU=ON.
    // ============================================================

    /// Enable GPU acceleration at runtime (default: false — safe on CPU-only hosts)
    bool gpu_enabled = false;

    /// CUDA device index to use (default: 0)
    int gpu_device_id = 0;

    /**
     * @brief Fraction of total GPU memory ADAI may allocate (0.0–1.0, default: 0.5).
     *
     * Keeping this well below 1.0 leaves headroom for the display driver,
     * other ML frameworks, desktop compositing, and interactive GPU workloads
     * that share the same card.  A value of 0.5 is a reasonable starting point;
     * lower it (e.g. 0.25) if ADAI is running alongside a game or another
     * training job.
     */
    float gpu_memory_fraction = 0.5f;

    /**
     * @brief GPU scheduling strategy (default: BACKGROUND).
     *
     * BACKGROUND — low-priority CUDA stream; training yields to other GPU work
     *              (display driver, desktop compositing, other ML frameworks).
     *              Use on shared workstations.
     * FULL        — highest-priority CUDA stream; training is never preempted.
     *              Use on dedicated training machines. Pair with a higher
     *              gpu_memory_fraction (e.g. 0.9) to maximise throughput.
     */
    GPUStrategy gpu_strategy = GPUStrategy::BACKGROUND;

    // ============================================================
    // Tokenizer Configuration
    // ============================================================

    /// Use UTF-8 code-point tokenization (Unicode mode); false = byte-level ASCII mode (default).
    ///
    /// ASCII mode  — each raw byte is one BPE unit.  Fast, backward-compatible.
    /// Unicode mode — each UTF-8 code point is one BPE unit.  Learns better
    ///                subword merges for non-Latin scripts (CJK, Arabic, Cyrillic…).
    ///
    /// Config key: TOKENIZER_MODE = ascii | unicode
    /// Env var:    TOKENIZER_MODE=unicode
    bool unicode_tokenizer = false;

    /// Target vocabulary size when auto-building a new vocabulary from training data.
    /// 0 (default) = auto-size based on corpus, model architecture, and script analysis.
    /// Any positive value overrides automatic sizing.
    /// Config key: VOCAB_BUILD_SIZE = <int>   (0 = auto)
    /// Env var:    VOCAB_BUILD_SIZE=<int>
    int vocab_build_size = 0;
};

/**
 * @brief Configuration loader for the ADAI service.
 *
 * Loads configuration from environment variables and/or a configuration file.
 */
class ConfigLoader {
   public:
    /**
     * @brief Load configuration with defaults.
     */
    static ServiceConfig load();

    /**
     * @brief Load configuration from a file and environment variables.
     *
     * @param config_file_path Path to the configuration file
     * @return ServiceConfig The loaded configuration
     */
    static ServiceConfig load(const std::string& config_file_path);

    /**
     * @brief Print the current configuration to stdout.
     *
     * @param config The configuration to print
     */
    static void print(const ServiceConfig& config);

    /**
     * @brief Reload configuration from file with validation.
     *
     * This method is thread-safe and validates the new configuration
     * before applying it. If validation fails, the current config remains unchanged.
     *
     * @param config Current configuration to update
     * @param config_file_path Path to the configuration file
     * @param mutex Mutex to protect the config during update
     * @return true if reload succeeded, false if validation failed
     */
    static bool reload(ServiceConfig& config, const std::string& config_file_path,
                       std::mutex& mutex);

    /**
     * @brief Validate configuration parameters.
     *
     * @param config Configuration to validate
     * @param errors Vector to store validation error messages
     * @return true if configuration is valid, false otherwise
     */
    static bool validate(const ServiceConfig& config, std::vector<std::string>& errors);

    /**
     * @brief Detect and log configuration changes.
     *
     * @param old_config Previous configuration
     * @param new_config New configuration
     * @return Vector of change descriptions
     */
    static std::vector<std::string> detect_changes(const ServiceConfig& old_config,
                                                   const ServiceConfig& new_config);

   private:
    /**
     * @brief Load configuration from environment variables.
     *
     * @param config Configuration struct to update
     */
    static void load_from_env(ServiceConfig& config);

    /**
     * @brief Load configuration from a file.
     *
     * Supports simple key=value format.
     *
     * @param config Configuration struct to update
     * @param file_path Path to the configuration file
     */
    static void load_from_file(ServiceConfig& config, const std::string& file_path);

    /**
     * @brief Get environment variable as string.
     *
     * @param var_name Environment variable name
     * @return std::optional<std::string> The value if present
     */
    static std::optional<std::string> get_env(const std::string& var_name);

    /**
     * @brief Get environment variable as integer.
     *
     * @param var_name Environment variable name
     * @return std::optional<int> The value if present and valid
     */
    static std::optional<int> get_env_int(const std::string& var_name);

    /**
     * @brief Get environment variable as size_t.
     *
     * @param var_name Environment variable name
     * @return std::optional<size_t> The value if present and valid
     */
    static std::optional<size_t> get_env_size_t(const std::string& var_name);

    /**
     * @brief Get environment variable as float.
     *
     * @param var_name Environment variable name
     * @return std::optional<float> The value if present and valid
     */
    static std::optional<float> get_env_float(const std::string& var_name);

    /**
     * @brief Get environment variable as boolean.
     *
     * @param var_name Environment variable name
     * @return std::optional<bool> The value if present and valid
     */
    static std::optional<bool> get_env_bool(const std::string& var_name);
};

}  // namespace adai
