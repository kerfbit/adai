// TD-037: widget tests for ChatbotGUI (chatbot_gui). Each test builds the real window against an
// in-process stand-in for chatbot_api_server, drives the widgets with QTest (typing, clicks, Enter,
// spin boxes, the confirmation and file dialogs) and checks the requests the server received and
// what the chat pane shows. Requests run on the window's worker thread (TD-263), so results are
// awaited with QTRY_*. Runs on the offscreen Qt platform (QT_QPA_PLATFORM=offscreen, set by ctest).

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QTemporaryDir>
#include <QTest>
#include <QTextEdit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include "ChatbotGUI.hpp"
#include "nlohmann/json.hpp"
#include "qt_widget_test_support.hpp"

using qt_widget_test::child;
using qt_widget_test::DialogAnswerer;
using qt_widget_test::first;
using qt_widget_test::first_title;
using qt_widget_test::RecordingServer;
using qt_widget_test::Reply;

namespace {

/// chatbot_api_server's routes. /chat/session echoes the message; hold() makes it wait.
class FakeChatServer {
   public:
    FakeChatServer() {
        server.on("GET", "/health", [this](const httplib::Request&) {
            std::lock_guard<std::mutex> lock(mutex);
            return Reply{health_status,
                         nlohmann::json{{"status", health_status == 200 ? "ok" : "untrained"},
                                        {"model_loaded", health_status == 200}}
                             .dump()};
        });
        server.on("POST", "/chat/session", [this](const httplib::Request& req) {
            const auto body = nlohmann::json::parse(req.body);
            const std::string sid = body.value("session_id", "");
            std::unique_lock<std::mutex> lock(mutex);
            if (holding) {
                waiting = true;
                changed.notify_all();
                changed.wait_for(lock, std::chrono::seconds(30), [this] { return !holding; });
                waiting = false;
            }
            if (chat_error) {
                return Reply{500,
                             R"({"success":false,"error":"generation failed","request_id":"7"})"};
            }
            if (!sid.empty() && sid == lost_session) {
                return Reply{404, R"({"success":false,"error":"Unknown session_id"})"};
            }
            return Reply{200, nlohmann::json{{"success", true},
                                             {"response", "echo: " + body.value("message", "")},
                                             {"session_id",
                                              sid.empty() ? "s" + std::to_string(++sessions) : sid}}
                                  .dump()};
        });
        server.on("POST", "/clear-session", Reply{200, R"({"success":true})"});
        server.on("POST", "/chat/session/export", [this](const httplib::Request&) {
            std::lock_guard<std::mutex> lock(mutex);
            return Reply{200, nlohmann::json{{"success", true}, {"data", export_data}}.dump()};
        });
        server.on("POST", "/chat/session/import",
                  Reply{200, R"({"success":true,"session_id":"s9","message_count":2})"});
    }
    ~FakeChatServer() {
        release();
    }

    void set_health(int status) {
        std::lock_guard<std::mutex> lock(mutex);
        health_status = status;
    }
    void set_chat_error(bool on) {
        std::lock_guard<std::mutex> lock(mutex);
        chat_error = on;
    }
    void lose_session(const std::string& sid) {
        std::lock_guard<std::mutex> lock(mutex);
        lost_session = sid;
    }
    void set_export(const std::string& data) {
        std::lock_guard<std::mutex> lock(mutex);
        export_data = data;
    }
    void hold() {
        std::lock_guard<std::mutex> lock(mutex);
        holding = true;
    }
    bool is_waiting() {
        std::lock_guard<std::mutex> lock(mutex);
        return waiting;
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        holding = false;
        changed.notify_all();
    }
    /// The JSON bodies of every /chat/session request, oldest first.
    std::vector<nlohmann::json> chat_bodies() const {
        std::vector<nlohmann::json> out;
        for (const auto& r : server.requests_starting("POST /chat/session {")) {
            out.push_back(nlohmann::json::parse(r.substr(r.find('{'))));
        }
        return out;
    }

    RecordingServer server;

   private:
    std::mutex mutex;
    std::condition_variable changed;
    int health_status = 200;
    bool chat_error = false;
    bool holding = false;
    bool waiting = false;
    int sessions = 0;
    std::string lost_session;
    std::string export_data = "MAX_MESSAGES:20\n---\n";
};

}  // namespace

class ChatbotGuiWidgetTest : public QObject {
    Q_OBJECT

   private:
    std::unique_ptr<FakeChatServer> fake;
    std::unique_ptr<ChatbotGUI> window;

    void open_window(const std::string& url) {
        window = std::make_unique<ChatbotGUI>(url);
        window->show();
        QVERIFY(QTest::qWaitForWindowExposed(window.get()));
    }
    QString chat() {
        return child<QTextEdit>(window.get(), "chatDisplay")->toPlainText();
    }
    QPushButton* send_button() {
        return child<QPushButton>(window.get(), "sendButton");
    }
    QLineEdit* input() {
        return child<QLineEdit>(window.get(), "inputField");
    }
    /// Opens the window on the fake server and waits until connecting has finished.
    void open_connected() {
        open_window(fake->server.url());
        QTRY_VERIFY(send_button()->isEnabled());
    }
    void send(const QString& text) {
        QTest::keyClicks(input(), text);
        QTest::mouseClick(send_button(), Qt::LeftButton);
    }

   private slots:
    void initTestCase() {
        // Qt's own file dialog, which the tests can answer, wherever a native one would be used.
        QCoreApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    }
    void init() {
        fake = std::make_unique<FakeChatServer>();
    }
    void cleanup() {
        if (fake) {
            fake->release();
        }
        window.reset();
        fake.reset();
    }

    void connectsAndEnablesInput() {
        open_connected();
        QVERIFY(chat().contains("Welcome to ADAI Transformer Chatbot"));
        QVERIFY(chat().contains("Connected to chatbot_api_server at " +
                                QString::fromStdString(fake->server.url())));
        QCOMPARE(send_button()->text(), QString("Send"));
        QVERIFY(input()->isEnabled());
        QCOMPARE(fake->server.requests_starting("GET /health").size(), size_t{1});
    }

    void reportsAnUnreachableServer() {
        const std::string url = fake->server.url();
        fake.reset();
        open_window(url);
        QTRY_VERIFY(chat().contains("Could not reach chatbot_api_server at " +
                                    QString::fromStdString(url)));
    }

    void reportsAServerRunningAnUntrainedModel() {
        fake->set_health(503);
        open_connected();
        QVERIFY(chat().contains("the server is running an untrained model"));
    }

    void anInvalidUrlDisablesSending() {
        open_window("ftp://example.com");
        QVERIFY(chat().contains("Invalid server URL ftp://example.com"));
        QVERIFY(!send_button()->isEnabled());
        QCOMPARE(send_button()->text(), QString("No server"));
        QVERIFY(!input()->isEnabled());
    }

    // TD-263: the request runs on the worker; the controls are disabled until it finishes.
    void sendShowsTheReplyAndDisablesTheControlsMeanwhile() {
        open_connected();
        fake->hold();
        send("hello");
        QTRY_VERIFY(fake->is_waiting());
        QVERIFY(chat().contains("hello"));
        QVERIFY(input()->text().isEmpty());
        QCOMPARE(send_button()->text(), QString("Generating..."));
        QVERIFY(!send_button()->isEnabled());
        QVERIFY(!input()->isEnabled());
        QVERIFY(!child<QWidget>(window.get(), "settingsPanel")->isEnabled());
        QVERIFY(!child<QPushButton>(window.get(), "clearChatButton")->isEnabled());
        fake->release();
        QTRY_VERIFY(chat().contains("echo: hello"));
        QTRY_VERIFY(send_button()->isEnabled());
        QCOMPARE(send_button()->text(), QString("Send"));
        QVERIFY(child<QWidget>(window.get(), "settingsPanel")->isEnabled());
        const auto bodies = fake->chat_bodies();
        QCOMPARE(bodies.size(), size_t{1});
        QCOMPARE(QString::fromStdString(bodies[0].value("message", "")), QString("hello"));
    }

    void enterSendsAndTheSessionIsKept() {
        open_connected();
        QTest::keyClicks(input(), "first");
        QTest::keyClick(input(), Qt::Key_Return);
        QTRY_VERIFY(chat().contains("echo: first"));
        QTRY_VERIFY(input()->isEnabled());
        QTest::keyClicks(input(), "second");
        QTest::keyClick(input(), Qt::Key_Return);
        QTRY_VERIFY(chat().contains("echo: second"));
        const auto bodies = fake->chat_bodies();
        QCOMPARE(bodies.size(), size_t{2});
        QVERIFY(!bodies[0].contains("session_id"));  // TD-232: the server issues it
        QCOMPARE(QString::fromStdString(bodies[1].value("session_id", "")), QString("s1"));
    }

    void blankInputSendsNothing() {
        open_connected();
        send("   ");
        QTest::qWait(200);
        QVERIFY(fake->chat_bodies().empty());
    }

    // TD-254/TD-260: only settings moved off "server default" are sent.
    void onlyChangedSettingsAreSent() {
        open_connected();
        send("plain");
        QTRY_VERIFY(chat().contains("echo: plain"));
        QTRY_VERIFY(send_button()->isEnabled());

        auto* temperature = child<QDoubleSpinBox>(window.get(), "temperatureSpin");
        auto* top_k = child<QSpinBox>(window.get(), "topKSpin");
        QTest::keyClick(temperature, Qt::Key_Up);  // server default -> 0.00
        QTest::keyClick(temperature, Qt::Key_Up);  // -> 0.05
        QTest::keyClick(top_k, Qt::Key_Up);        // server default -> 1
        child<QComboBox>(window.get(), "strategyCombo")->setCurrentIndex(3);  // Greedy
        send("tuned");
        QTRY_VERIFY(chat().contains("echo: tuned"));
        QTRY_VERIFY(send_button()->isEnabled());

        QTest::keyClick(top_k, Qt::Key_Down);  // back to server default
        child<QComboBox>(window.get(), "strategyCombo")->setCurrentIndex(0);
        send("again");
        QTRY_VERIFY(chat().contains("echo: again"));

        const auto bodies = fake->chat_bodies();
        QCOMPARE(bodies.size(), size_t{3});
        for (const char* key :
             {"temperature", "top_p", "top_k", "max_length", "beam_width", "strategy"}) {
            QVERIFY2(!bodies[0].contains(key), key);
        }
        QCOMPARE(QString::fromStdString(bodies[1].value("strategy", "")), QString("greedy"));
        QCOMPARE(bodies[1].value("top_k", 0), 1);
        QVERIFY(std::abs(bodies[1].value("temperature", -1.0) - 0.05) < 1e-6);
        QVERIFY(!bodies[1].contains("top_p"));
        QVERIFY(!bodies[2].contains("top_k"));
        QVERIFY(!bodies[2].contains("strategy"));
        QVERIFY(bodies[2].contains("temperature"));
    }

    // TD-255: a failed turn shows the server's error as a system message.
    void aServerErrorIsShownAsASystemMessage() {
        open_connected();
        fake->set_chat_error(true);
        send("break");
        QTRY_VERIFY(chat().contains("Error: 500 - generation failed (request id 7)"));
        QVERIFY(!chat().contains("echo: break"));
    }

    void aLostSessionIsReported() {
        open_connected();
        send("one");
        QTRY_VERIFY(chat().contains("echo: one"));
        QTRY_VERIFY(send_button()->isEnabled());
        fake->lose_session("s1");
        send("two");
        QTRY_VERIFY(chat().contains("echo: two"));
        QVERIFY(chat().contains("The server no longer had this conversation"));
    }

    void clearChatAsksAndClearsOnYes() {
        open_connected();
        send("keep me");
        QTRY_VERIFY(chat().contains("echo: keep me"));
        QTRY_VERIFY(send_button()->isEnabled());

        DialogAnswerer no;
        no.answer_message(QMessageBox::No);
        QTest::mouseClick(child<QPushButton>(window.get(), "clearChatButton"), Qt::LeftButton);
        QCOMPARE(first_title(no), QString("Clear Conversation"));
        QVERIFY(fake->server.requests_starting("POST /clear-session").empty());
        QVERIFY(chat().contains("echo: keep me"));

        DialogAnswerer yes;
        yes.answer_message(QMessageBox::Yes);
        QTest::mouseClick(child<QPushButton>(window.get(), "clearInputButton"), Qt::LeftButton);
        QTRY_VERIFY(chat().contains("Conversation cleared."));
        QVERIFY(!chat().contains("echo: keep me"));
        const auto clears = fake->server.requests_starting("POST /clear-session");
        QCOMPARE(clears.size(), size_t{1});
        QVERIFY(clears[0].find(R"("session_id":"s1")") != std::string::npos);
    }

    void saveBeforeAnyMessageSaysThereIsNothingToSave() {
        open_connected();
        DialogAnswerer dialogs;
        dialogs.answer_file(QString(), 500);  // cancels a file dialog, which shouldn't open
        QTest::mouseClick(child<QPushButton>(window.get(), "saveButton"), Qt::LeftButton);
        QVERIFY(chat().contains("Nothing to save yet: send a message first."));
        QTest::qWait(600);
        QVERIFY(dialogs.seen().empty());
    }

    void saveWritesTheServersCopyToTheChosenFile() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString path = dir.filePath("saved.txt");
        const std::string data = "MAX_MESSAGES:20\n---\nuser|1|hi\nassistant|2|echo: hi\n";
        fake->set_export(data);
        open_connected();
        send("hi");
        QTRY_VERIFY(chat().contains("echo: hi"));
        QTRY_VERIFY(send_button()->isEnabled());

        DialogAnswerer dialogs;
        dialogs.answer_file(path);
        QTest::mouseClick(child<QPushButton>(window.get(), "saveButton"), Qt::LeftButton);
        QTRY_VERIFY(chat().contains("Conversation saved to " + path));
        QCOMPARE(first_title(dialogs), QString("Save Conversation"));
        QFile saved(path);
        QVERIFY(saved.open(QIODevice::ReadOnly));
        QCOMPARE(saved.readAll().toStdString(), data);
    }

    // TD-266: after loading, the pane shows the server's copy of the conversation.
    void loadImportsTheFileAndShowsTheServersCopy() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString path = dir.filePath("load.txt");
        const std::string file_data = "MAX_MESSAGES:20\n---\nuser|1|from the file\n";
        {
            QFile f(path);
            QVERIFY(f.open(QIODevice::WriteOnly));
            f.write(file_data.c_str());
        }
        fake->set_export("MAX_MESSAGES:20\n---\nuser|2|hello there\nassistant|2|general kenobi\n");
        open_connected();

        DialogAnswerer dialogs;
        dialogs.answer_file(path);
        QTest::mouseClick(child<QPushButton>(window.get(), "loadButton"), Qt::LeftButton);
        QTRY_VERIFY(chat().contains("Conversation loaded from " + path + " (2 messages)"));
        QVERIFY(chat().contains("hello there"));
        QVERIFY(chat().contains("general kenobi"));
        QVERIFY(!chat().contains("Welcome to ADAI"));  // the pane was replaced
        const auto imports = fake->server.requests_starting("POST /chat/session/import");
        QCOMPARE(imports.size(), size_t{1});
        QCOMPARE(
            QString::fromStdString(
                nlohmann::json::parse(imports[0].substr(imports[0].find('{'))).value("data", "")),
            QString::fromStdString(file_data));
    }

    void loadingAnEmptyFileShowsAnError() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString path = dir.filePath("empty.txt");
        {
            QFile f(path);
            QVERIFY(f.open(QIODevice::WriteOnly));
        }
        open_connected();
        DialogAnswerer file;
        DialogAnswerer error;
        file.answer_file(path);
        error.answer_message(QMessageBox::Ok);
        QTest::mouseClick(child<QPushButton>(window.get(), "loadButton"), Qt::LeftButton);
        QCOMPARE(error.seen().size(), size_t{1});
        QVERIFY(error.seen()[0].text.contains("is empty"));
        QVERIFY(fake->server.requests_starting("POST /chat/session/import").empty());
    }

    // The destructor cancels a request in flight instead of waiting for the server (300 s timeout).
    void closingDuringARequestDoesNotWaitForIt() {
        open_connected();
        fake->hold();  // the server would answer only after 30 s
        send("slow");
        QTRY_VERIFY(fake->is_waiting());
        const auto start = std::chrono::steady_clock::now();
        window.reset();
        const auto took = std::chrono::steady_clock::now() - start;
        QVERIFY2(took < std::chrono::seconds(5), "closing the window waited for the request");
    }
};

QTEST_MAIN(ChatbotGuiWidgetTest)
#include "chatbot_gui_widget_test.moc"
