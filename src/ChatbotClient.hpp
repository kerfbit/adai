#pragma once

// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-10

#include <memory>
#include <set>
#include <string>

// Only ChatbotClient.cpp needs all of httplib.h (TD-259).
namespace httplib {
class Client;
}

/**
 * @brief The generation settings a client sends with each message (TD-254).
 *
 * Only settings marked overridden are sent; the server's own defaults apply to the rest, so the
 * initial values here are placeholders. Keys are the request's JSON names: strategy, max_length,
 * temperature, top_p, top_k, beam_width.
 */
struct GenerationOverrides {
    std::string strategy;
    int max_length{100};
    float temperature{1.0f};
    float top_p{0.9f};
    int top_k{50};
    int beam_width{5};

    bool is_overridden(const std::string& key) const {
        return overridden.count(key) > 0;
    }
    /// Stop sending this setting, so the server's default applies again.
    void reset(const std::string& key) {
        overridden.erase(key);
    }
    void reset_all() {
        overridden.clear();
    }

    void set_strategy(const std::string& value) {
        strategy = value;
        overridden.insert("strategy");
    }
    void set_max_length(int value) {
        max_length = value;
        overridden.insert("max_length");
    }
    void set_temperature(float value) {
        temperature = value;
        overridden.insert("temperature");
    }
    void set_top_p(float value) {
        top_p = value;
        overridden.insert("top_p");
    }
    void set_top_k(int value) {
        top_k = value;
        overridden.insert("top_k");
    }
    void set_beam_width(int value) {
        beam_width = value;
        overridden.insert("beam_width");
    }

   private:
    std::set<std::string> overridden;
};

/**
 * @brief An HTTP session with chatbot_api_server, shared by the `chatbot` CLI and `chatbot_gui`
 * (TD-260: the GUI used to run its own copy of the model instead of talking to the server).
 *
 * It holds the server-issued session id (TD-232) and the generation overrides, and prints nothing:
 * each call returns what happened and the caller tells the user. One request at a time; cancel()
 * is the only call safe from another thread.
 */
class ChatbotClient {
   public:
    explicit ChatbotClient(std::string server_url);
    ~ChatbotClient();

    ChatbotClient(const ChatbotClient&) = delete;
    ChatbotClient& operator=(const ChatbotClient&) = delete;
    ChatbotClient(ChatbotClient&&) noexcept;
    ChatbotClient& operator=(ChatbotClient&&) noexcept;

    /// GET /health's answer. `reachable` with status 503 and `model_loaded` false is a server
    /// serving an untrained model (ALLOW_UNTRAINED_MODEL, TD-246).
    struct Health {
        bool reachable = false;
        int status = 0;
        bool model_loaded = false;
        std::string error;  // why the server couldn't be reached
    };

    /// Creates the connection (10 s connect, 300 s read, 30 s write timeouts) without contacting the
    /// server. connect() does this itself; a caller that cancels from another thread opens first, so
    /// the connection isn't created while cancel() reads it.
    void open();
    /// open()s if needed, then asks GET /health.
    Health connect();
    bool is_connected() const;

    struct Reply {
        bool ok = false;
        /// The model's reply, or "Error: <status> - <error> (request id N)" when the turn failed (TD-255).
        std::string text;
        /// The server no longer had the session (TD-232), so the message went to a new one.
        bool session_lost = false;
    };

    /// POST /chat/session with the overridden settings (TD-254).
    Reply send_message(const std::string& message);

    enum class ClearResult { Cleared, NoSession, SessionLost, Failed };
    /// POST /clear-session. SessionLost: the server no longer had it; the id is forgotten.
    ClearResult clear_session();

    struct Export {
        enum class Status { Ok, NoSession, NotConnected, Unreachable, SessionLost, Failed };
        Status status = Status::Failed;
        std::string data;   // the serialized conversation when Ok
        std::string error;  // the server's error when Failed
    };
    /// POST /chat/session/export. SessionLost: the server no longer had it; the id is forgotten.
    Export export_session();

    struct Import {
        bool ok = false;
        std::string error;
        std::string message_count;  // as the server wrote it; may be empty
    };
    /// POST /chat/session/import, into this session or, if there is none or the server lost it, a new one.
    Import import_session(const std::string& data);

    /// Stops a request in flight (from another thread): it fails as unreachable.
    void cancel();

    const std::string& server_url() const {
        return url;
    }
    const std::string& session_id() const {
        return session;
    }
    void forget_session() {
        session.clear();
    }

    GenerationOverrides& settings() {
        return overrides;
    }
    const GenerationOverrides& settings() const {
        return overrides;
    }

    /// The JSON body POST /chat/session sends for `message` (TD-254: only overridden settings).
    std::string chat_turn_body(const std::string& message) const;

   private:
    std::string url;
    std::string session;
    GenerationOverrides overrides;
    std::unique_ptr<httplib::Client> client;
};

namespace chatbot_client {

/// Writes `data` to a temporary file beside `path` and renames it over `path`, so a failed or
/// interrupted save never leaves a partial file (TD-258). Returns an error message, empty on success.
std::string write_file_atomically(const std::string& path, const std::string& data);

}  // namespace chatbot_client
