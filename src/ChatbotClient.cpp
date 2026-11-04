// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-04

#include "ChatbotClient.hpp"
#include <httplib.h>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>
#include "nlohmann/json.hpp"

namespace {

// TD-259: request bodies are built and replies read with nlohmann/json, not hand-written helpers.
// Invalid UTF-8 in the user's text is sent as U+FFFD instead of making the body invalid JSON.
std::string to_body(const nlohmann::ordered_json& body) {
    return body.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace);
}

// A setting float as its shortest decimal form, so 0.7 is sent as 0.7, not 0.699999988079071.
double wire_number(float value) {
    char text[32];
    const auto result = std::to_chars(text, text + sizeof text, value);
    return std::strtod(std::string(text, result.ptr).c_str(), nullptr);
}

// A reply's JSON object; empty when the body isn't one (an HTML error page, say).
nlohmann::json parse_reply(const std::string& body) {
    nlohmann::json reply = nlohmann::json::parse(body, nullptr, false);
    return reply.is_object() ? reply : nlohmann::json::object();
}

// A reply field as text: a string as is, a number or boolean as written, anything else empty.
std::string reply_field(const nlohmann::json& reply, const char* key) {
    const auto it = reply.find(key);
    if (it == reply.end() || it->is_null()) {
        return {};
    }
    if (it->is_string()) {
        return it->get<std::string>();
    }
    return it->is_primitive() ? it->dump() : std::string{};
}

}  // namespace

ChatbotClient::ChatbotClient(std::string server_url) : url(std::move(server_url)) {}

ChatbotClient::~ChatbotClient() = default;
ChatbotClient::ChatbotClient(ChatbotClient&&) noexcept = default;
ChatbotClient& ChatbotClient::operator=(ChatbotClient&&) noexcept = default;

void ChatbotClient::open() {
    if (client) {
        return;
    }
    client = std::make_unique<httplib::Client>(url);
    client->set_connection_timeout(10, 0);  // 10s connection timeout
    client->set_read_timeout(300, 0);       // 300s read timeout (generation can be VERY slow)
    client->set_write_timeout(30, 0);       // 30s write timeout
}

ChatbotClient::Health ChatbotClient::connect() {
    Health health;
    try {
        open();
    } catch (const std::exception& e) {
        client.reset();
        health.error = e.what();
        return health;
    }

    auto res = client->Get("/health");
    if (!res) {
        health.error = httplib::to_string(res.error());
        return health;
    }
    health.reachable = true;
    health.status = res->status;
    health.model_loaded = reply_field(parse_reply(res->body), "model_loaded") == "true";
    return health;
}

bool ChatbotClient::is_connected() const {
    return client != nullptr;
}

ChatbotClient::Reply ChatbotClient::send_message(const std::string& message) {
    Reply reply;
    if (!client) {
        reply.text = "Error: Client not initialized";
        return reply;
    }

    // TD-232: session ids are server-issued. Once the server no longer holds ours (expired,
    // evicted, or the server restarted) it answers 404; start a new session and send again.
    auto post_chat_turn = [&] {
        return client->Post("/chat/session", chat_turn_body(message), "application/json");
    };
    httplib::Result res = post_chat_turn();
    if (res && res->status == 404 && !session.empty()) {
        reply.session_lost = true;
        session.clear();
        res = post_chat_turn();
    }

    // TD-255: an error reply is {"success":false,"error":...} -- with HTTP 400/500 since TD-236, with 200
    // from older servers. Show its error field, never an empty reply or the raw JSON.
    if (!res) {
        reply.text = "Error: Connection failed";
        return reply;
    }
    const nlohmann::json body = parse_reply(res->body);
    if (res->status == 200 && reply_field(body, "success") != "false") {
        reply.ok = true;
        reply.text = reply_field(body, "response");
        const std::string new_sid = reply_field(body, "session_id");
        if (!new_sid.empty()) {
            session = new_sid;
        }
        return reply;
    }
    std::stringstream err;
    err << "Error: ";
    if (res->status != 200) {
        err << res->status;
    }
    const std::string error = reply_field(body, "error");
    if (!error.empty()) {
        err << (res->status != 200 ? " - " : "") << error;
    } else if (res->status == 200) {
        err << "the server reported a failure without a reason";
    }
    const std::string request_id = reply_field(body, "request_id");
    if (!request_id.empty()) {
        err << " (request id " << request_id << ")";
    }
    reply.text = err.str();
    return reply;
}

ChatbotClient::ClearResult ChatbotClient::clear_session() {
    if (session.empty()) {
        return ClearResult::NoSession;
    }
    if (!client) {
        return ClearResult::Failed;
    }
    // TD-257: escaped like every request body's session_id.
    auto res = client->Post("/clear-session", to_body({{"session_id", session}}), "application/json");
    if (res && res->status == 404) {  // TD-232: the server no longer holds it
        session.clear();
        return ClearResult::SessionLost;
    }
    return res && res->status == 200 ? ClearResult::Cleared : ClearResult::Failed;
}

ChatbotClient::Export ChatbotClient::export_session() {
    Export result;
    if (session.empty()) {
        result.status = Export::Status::NoSession;
        return result;
    }
    if (!client) {
        result.status = Export::Status::NotConnected;
        return result;
    }
    auto res = client->Post("/chat/session/export", to_body({{"session_id", session}}), "application/json");
    if (!res) {
        result.status = Export::Status::Unreachable;
        return result;
    }
    if (res->status == 404) {  // TD-232/TD-258: nothing left on the server to save
        session.clear();
        result.status = Export::Status::SessionLost;
        return result;
    }
    const nlohmann::json reply = parse_reply(res->body);
    if (reply_field(reply, "success") != "true") {
        result.error = reply_field(reply, "error");
        return result;
    }
    result.status = Export::Status::Ok;
    result.data = reply_field(reply, "data");
    return result;
}

ChatbotClient::Import ChatbotClient::import_session(const std::string& data) {
    Import result;
    if (!client) {
        result.error = "Client not initialized";
        return result;
    }
    auto post_import = [&] {
        nlohmann::ordered_json body;
        if (!session.empty()) {
            body["session_id"] = session;
        }
        body["data"] = data;
        return client->Post("/chat/session/import", to_body(body), "application/json");
    };
    auto res = post_import();
    if (res && res->status == 404 && !session.empty()) {  // TD-232: import into a new session
        session.clear();
        res = post_import();
    }
    if (!res) {
        result.error = "Failed to reach server to import conversation";
        return result;
    }
    const nlohmann::json reply = parse_reply(res->body);
    if (reply_field(reply, "success") != "true") {
        result.error = reply_field(reply, "error");
        if (result.error.empty()) {
            result.error = "Import failed";
        }
        return result;
    }
    const std::string new_sid = reply_field(reply, "session_id");
    if (!new_sid.empty()) {
        session = new_sid;
    }
    result.ok = true;
    result.message_count = reply_field(reply, "message_count");
    return result;
}

void ChatbotClient::cancel() {
    if (client) {
        client->stop();
    }
}

std::string ChatbotClient::chat_turn_body(const std::string& message) const {
    nlohmann::ordered_json body;
    if (!session.empty()) {
        body["session_id"] = session;  // TD-257
    }
    body["message"] = message;
    // TD-254: only the overridden settings; the server's defaults apply to the rest.
    const GenerationOverrides& s = overrides;
    if (s.is_overridden("max_length")) body["max_length"] = s.max_length;
    if (s.is_overridden("temperature")) body["temperature"] = wire_number(s.temperature);
    if (s.is_overridden("top_p")) body["top_p"] = wire_number(s.top_p);
    if (s.is_overridden("top_k")) body["top_k"] = s.top_k;
    if (s.is_overridden("beam_width")) body["beam_width"] = s.beam_width;
    if (s.is_overridden("strategy")) body["strategy"] = s.strategy;
    return to_body(body);
}

namespace chatbot_client {

std::string write_file_atomically(const std::string& path, const std::string& data) {
    namespace fs = std::filesystem;
    const std::string temp =
        path + ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::error_code ec;
    {
        std::ofstream out(temp, std::ios::trunc);
        if (!out.is_open()) {
            return "Failed to create '" + temp + "' to save into '" + path + "'";
        }
        out << data;
        out.flush();
        if (!out) {
            out.close();
            fs::remove(temp, ec);
            return "Failed to write '" + temp + "'";
        }
    }
    fs::rename(temp, path, ec);
    if (ec) {
        std::error_code ignored;
        fs::remove(temp, ignored);
        return "Failed to replace '" + path + "': " + ec.message();
    }
    return {};
}

}  // namespace chatbot_client
