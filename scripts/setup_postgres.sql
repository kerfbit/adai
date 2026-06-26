-- ============================================================================
-- ADAI PostgreSQL Database Setup
--
-- Creates all tables for the metrics API server and the model name service.
-- Run once against a fresh database:
--
--   createdb adai
--   psql -d adai -f scripts/setup_postgres.sql
--
-- Idempotent — safe to re-run (uses IF NOT EXISTS / ON CONFLICT).
-- ============================================================================

BEGIN;

-- ============================================================================
-- Schema version tracking
-- Shared migration table. Each subsystem inserts its own version row.
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    subsystem  TEXT        NOT NULL,
    version    INTEGER     NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (subsystem, version)
);

-- ============================================================================
-- Training Metrics (TD-020)
--
-- Persistent storage for the metrics_api_server.  Replaces the flat
-- JSONL / JSON files with indexed, queryable tables.
-- ============================================================================

-- One row per training session.  Upserted on every persist_summary() call.
CREATE TABLE IF NOT EXISTS sessions (
    key                  TEXT    PRIMARY KEY,
    session_id           INTEGER NOT NULL,
    label                TEXT    NOT NULL DEFAULT '',
    config_json          TEXT,
    is_training          BOOLEAN NOT NULL DEFAULT TRUE,
    created_at           TIMESTAMPTZ NOT NULL,
    ended_at             TIMESTAMPTZ,
    last_update_at       TIMESTAMPTZ NOT NULL,
    total_epochs         INTEGER NOT NULL DEFAULT 0,
    total_samples        INTEGER NOT NULL DEFAULT 0,
    best_validation_loss REAL,
    best_epoch           INTEGER
);

-- Per-sample metrics history.  One row per PersistentMetricsRecord.
CREATE TABLE IF NOT EXISTS metrics_history (
    id              SERIAL      PRIMARY KEY,
    session_key     TEXT        NOT NULL REFERENCES sessions(key),
    recorded_at     TIMESTAMPTZ NOT NULL,
    epoch           INTEGER     NOT NULL,
    sample          INTEGER     NOT NULL,
    loss            REAL,
    validation_loss REAL,
    learning_rate   REAL,
    gradient_norm   REAL,
    perplexity      REAL
);

CREATE INDEX IF NOT EXISTS idx_metrics_history_session_time
    ON metrics_history (session_key, recorded_at);

-- BLEU / ROUGE generation-quality scores.  One row per epoch where
-- generation-quality evaluation ran; epochs without scoring produce no row.
CREATE TABLE IF NOT EXISTS generation_quality (
    id          SERIAL      PRIMARY KEY,
    session_key TEXT        NOT NULL REFERENCES sessions(key),
    recorded_at TIMESTAMPTZ NOT NULL,
    epoch       INTEGER     NOT NULL,
    bleu1       REAL,
    bleu2       REAL,
    bleu4       REAL,
    rouge1      REAL,
    rouge2      REAL,
    "rougeL"    REAL
);

CREATE INDEX IF NOT EXISTS idx_generation_quality_session_epoch
    ON generation_quality (session_key, epoch);

-- Flagged outlier training samples (loss or gradient-norm anomalies).
CREATE TABLE IF NOT EXISTS abnormal_samples (
    id          SERIAL      PRIMARY KEY,
    session_key TEXT        NOT NULL REFERENCES sessions(key),
    epoch       INTEGER     NOT NULL,
    sample_id   INTEGER     NOT NULL,
    loss        REAL,
    grad_norm   REAL,
    reason      TEXT,
    input_text  TEXT,
    target_text TEXT,
    recorded_at TIMESTAMPTZ NOT NULL
);

-- ============================================================================
-- Model Name Service (MNS)
--
-- Identity registry for trained models.  Tracks model metadata, architecture
-- dimensions, artifact locations, role assignments, and training history.
-- ============================================================================

-- One row per registered model.  model_name is the human-facing identifier;
-- model_id is a UUID assigned at registration time.
CREATE TABLE IF NOT EXISTS models (
    model_id           TEXT PRIMARY KEY,
    model_name         TEXT UNIQUE NOT NULL,
    role               TEXT    DEFAULT '',
    state              TEXT    NOT NULL DEFAULT 'initializing',
    run_id             TEXT    DEFAULT '',
    created_utc        TIMESTAMPTZ NOT NULL,
    updated_utc        TIMESTAMPTZ NOT NULL,
    artifact_host      TEXT    DEFAULT '',
    artifact_path      TEXT    DEFAULT '',
    artifact_checksum  TEXT    DEFAULT '',
    artifact_format    TEXT    DEFAULT 'adai-native',
    d_model            INTEGER DEFAULT 0,
    num_heads          INTEGER DEFAULT 0,
    d_ff               INTEGER DEFAULT 0,
    num_encoder_layers INTEGER DEFAULT 0,
    num_decoder_layers INTEGER DEFAULT 0,
    max_seq_length     INTEGER DEFAULT 0,
    tags_json          TEXT    DEFAULT '{}'
);

-- Training-run history.  One row per completed (or in-progress) training run
-- for a given model.  Links to metrics_session_key for cross-referencing
-- with the metrics tables above.
CREATE TABLE IF NOT EXISTS training_history (
    id                  SERIAL  PRIMARY KEY,
    model_name          TEXT    NOT NULL,
    run_id              TEXT    DEFAULT '',
    metrics_session_key TEXT    DEFAULT '',
    dataset_group       TEXT    DEFAULT '',
    epochs              INTEGER DEFAULT 0,
    final_loss          REAL    DEFAULT 0.0,
    started_utc         TIMESTAMPTZ DEFAULT NULL,
    finished_utc        TIMESTAMPTZ DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_training_history_model
    ON training_history (model_name);

-- Role → model mapping.  One active model per role; roles are reassigned
-- via PUT /api/models/:model_name/role.
CREATE TABLE IF NOT EXISTS roles (
    role       TEXT PRIMARY KEY,
    model_name TEXT NOT NULL
);

-- ============================================================================
-- Seed schema_version
-- ============================================================================

INSERT INTO schema_version (subsystem, version)
VALUES ('metrics', 1), ('mns', 1)
ON CONFLICT (subsystem, version) DO NOTHING;

COMMIT;
