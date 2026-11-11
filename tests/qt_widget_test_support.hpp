#pragma once

// TD-037: shared helpers for the Qt Test widget suites (chatbot_gui_widget_test.cpp,
// mns_manager_gui_widget_test.cpp). Each suite builds the real window against an in-process
// httplib server, drives its widgets with QTest, and checks what was sent and what is shown.

#include <httplib.h>
#include <QAbstractButton>
#include <QApplication>
#include <QFileDialog>
#include <QMessageBox>
#include <QPointer>
#include <QPushButton>
#include <QString>
#include <QTimer>
#include <QWidget>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace qt_widget_test {

/// The window's child with this object name (nullptr if none).
template <typename T>
T* child(QWidget* window, const char* name) {
    return window->findChild<T*>(QString::fromLatin1(name));
}

/// The window's push button with exactly this label (nullptr if none).
inline QPushButton* button(QWidget* window, const QString& text) {
    for (QPushButton* b : window->findChildren<QPushButton*>()) {
        if (b->text() == text) {
            return b;
        }
    }
    return nullptr;
}

/// What a modal dialog showed, as answered by DialogAnswerer.
struct SeenDialog {
    QString title;
    QString text;  // a message box's text; empty for a file dialog
};

/**
 * Answers the next modal dialog the window opens, from a timer: a click opens it with a static
 * QMessageBox/QFileDialog call that blocks until the dialog closes, so the answer must be armed
 * before the click. Gives up after `timeout_ms` (the dialog never opened).
 */
class DialogAnswerer {
   public:
    /// Clicks `button` on the next QMessageBox.
    void answer_message(QMessageBox::StandardButton b, int timeout_ms = 5000) {
        arm(timeout_ms, [this, b](QWidget* w) {
            auto* box = qobject_cast<QMessageBox*>(w);
            if (!box) {
                return false;
            }
            seen_.push_back({box->windowTitle(), box->text()});
            if (QAbstractButton* answer = box->button(b)) {
                answer->click();
            } else {
                box->reject();
            }
            return true;
        });
    }

    /// Chooses `path` in the next QFileDialog, or cancels it when `path` is empty.
    void answer_file(const QString& path, int timeout_ms = 5000) {
        arm(timeout_ms, [this, path](QWidget* w) {
            auto* dialog = qobject_cast<QFileDialog*>(w);
            if (!dialog) {
                return false;
            }
            seen_.push_back({dialog->windowTitle(), QString()});
            if (path.isEmpty()) {
                dialog->reject();
            } else {
                dialog->selectFile(path);
                static_cast<QDialog*>(dialog)->accept();  // QFileDialog::accept() is protected
            }
            return true;
        });
    }

    const std::vector<SeenDialog>& seen() const {
        return seen_;
    }

   private:
    void arm(int timeout_ms, std::function<bool(QWidget*)> handle) {
        // Owned here, so a timer whose dialog never opened can't answer another test's dialog
        // after this answerer is gone.
        timers_.push_back(std::make_unique<QTimer>());
        QTimer* timer = timers_.back().get();
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        QObject::connect(timer, &QTimer::timeout, [timer, handle, deadline] {
            QWidget* modal = QApplication::activeModalWidget();
            if ((modal && handle(modal)) || std::chrono::steady_clock::now() > deadline) {
                timer->stop();
            }
        });
        timer->start(10);
    }

    std::vector<std::unique_ptr<QTimer>> timers_;
    std::vector<SeenDialog> seen_;
};

/// The first entry, or an empty string: a failed check then reports the values instead of
/// throwing out of the test run. QString, so QCOMPARE prints both sides.
inline QString first(const std::vector<std::string>& entries) {
    return entries.empty() ? QString() : QString::fromStdString(entries.front());
}
inline QString last(const std::vector<std::string>& entries) {
    return entries.empty() ? QString() : QString::fromStdString(entries.back());
}
inline QString first_title(const DialogAnswerer& dialogs) {
    return dialogs.seen().empty() ? QString() : dialogs.seen().front().title;
}

/// A canned HTTP reply.
struct Reply {
    int status = 200;
    std::string body;
};

/**
 * An httplib server on a free loopback port that records every request as
 * "METHOD path[?query] body" and answers from a handler the test sets per route.
 */
class RecordingServer {
   public:
    using Handler = std::function<Reply(const httplib::Request&)>;

    RecordingServer() {
        // Catch-all routes, not a pre-routing handler: httplib reads the body only before routing.
        auto handle = [this](const httplib::Request& req, httplib::Response& res) {
            Handler handler;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                requests_.push_back(describe(req));
                handler = find_handler(req.method, req.path);
            }
            const Reply reply = handler ? handler(req) : Reply{404, R"({"error":"no route"})"};
            res.status = reply.status;
            res.set_content(reply.body, "application/json");
        };
        server_.Get(".*", handle);
        server_.Post(".*", handle);
        server_.Put(".*", handle);
        server_.Delete(".*", handle);
        port_ = server_.bind_to_any_port("127.0.0.1");
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        server_.wait_until_ready();
    }
    ~RecordingServer() {
        stop();
    }
    RecordingServer(const RecordingServer&) = delete;
    RecordingServer& operator=(const RecordingServer&) = delete;

    void stop() {
        server_.stop();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    std::string url() const {
        return "http://127.0.0.1:" + std::to_string(port_);
    }

    /// Answers `method` requests whose path starts with `path_prefix` (the longest prefix wins).
    void on(const std::string& method, const std::string& path_prefix, Handler handler) {
        std::lock_guard<std::mutex> lock(mutex_);
        handlers_[{method, path_prefix}] = std::move(handler);
    }
    void on(const std::string& method, const std::string& path_prefix, Reply reply) {
        on(method, path_prefix, [reply](const httplib::Request&) { return reply; });
    }

    std::vector<std::string> requests() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return requests_;
    }
    void clear_requests() {
        std::lock_guard<std::mutex> lock(mutex_);
        requests_.clear();
    }
    /// The requests starting with `prefix` (e.g. "PUT /models/m/state").
    std::vector<std::string> requests_starting(const std::string& prefix) const {
        std::vector<std::string> out;
        for (const auto& r : requests()) {
            if (r.rfind(prefix, 0) == 0) {
                out.push_back(r);
            }
        }
        return out;
    }

   private:
    static std::string describe(const httplib::Request& req) {
        std::string line = req.method + " " + req.path;
        char sep = '?';
        for (const auto& [k, v] : req.params) {
            line += sep + k + "=" + v;
            sep = '&';
        }
        if (!req.body.empty()) {
            line += " " + req.body;
        }
        return line;
    }

    Handler find_handler(const std::string& method, const std::string& path) const {
        Handler best;
        size_t best_len = 0;
        for (const auto& [key, handler] : handlers_) {
            if (key.first == method && path.rfind(key.second, 0) == 0 &&
                key.second.size() >= best_len) {
                best = handler;
                best_len = key.second.size();
            }
        }
        return best;
    }

    mutable std::mutex mutex_;
    std::vector<std::string> requests_;
    std::map<std::pair<std::string, std::string>, Handler> handlers_;
    httplib::Server server_;
    std::thread thread_;
    int port_ = 0;
};

}  // namespace qt_widget_test
