// TD-037: widget tests for MnsManagerGUI (mns_manager_gui). Each test builds the real window
// against an in-process stand-in for mns_server, drives the widgets with QTest (clicks, typing,
// table rows, modal dialogs) and checks the requests the server received and what the window shows.
// Requests run on the window's worker thread (TD-455), so each action waits until the window's
// controls are enabled again. Runs on the offscreen Qt platform (QT_QPA_PLATFORM=offscreen, set by
// ctest).

#include <QCheckBox>
#include <QComboBox>
#include <QFile>
#include <QLabel>
#include <QLineEdit>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTest>
#include <QTextEdit>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include "MnsManagerGUI.hpp"
#include "nlohmann/json.hpp"
#include "qt_widget_test_support.hpp"

using qt_widget_test::button;
using qt_widget_test::child;
using qt_widget_test::DialogAnswerer;
using qt_widget_test::first;
using qt_widget_test::first_title;
using qt_widget_test::last;
using qt_widget_test::RecordingServer;
using qt_widget_test::Reply;

namespace {

const std::string kTrainingRecord =
    R"({"model_name":"m1","model_id":"0123456789abcdef","role":"chat","state":"training",)"
    R"("run_id":"run-03","updated_utc":"2026-11-10T10:00:00Z"})";
const std::string kModelsList =
    R"({"models":[{"model_name":"m1","model_id":"0123456789abcdef","role":"chat","state":"training",)"
    R"("updated_utc":"2026-11-10T10:00:00Z"},{"model_name":"m2","model_id":"fedcba9876543210",)"
    R"("role":"","state":"candidate","updated_utc":"2026-11-09T09:00:00Z"}]})";
const std::string kRolesList =
    R"({"roles":[{"role":"chat","production_model":"m2"},{"role":"spare","production_model":null}]})";

/// mns_server's routes, answering as a server holding m1 (training, run-03) and m2 (candidate).
void serve_mns(RecordingServer& server) {
    server.on("GET", "/health", Reply{200, R"({"status":"ok","model_count":2})"});
    server.on("GET", "/models", Reply{200, kModelsList});
    server.on("GET", "/models/m1", Reply{200, kTrainingRecord});
    server.on(
        "GET", "/models/m2",
        Reply{200, R"({"model_name":"m2","role":"chat","state":"candidate","run_id":"run-01"})"});
    server.on("GET", "/roles", Reply{200, kRolesList});
}

/// The JSON body of a recorded request ("METHOD target body").
nlohmann::json body_of(const std::string& request) {
    const auto brace = request.find('{');
    return brace == std::string::npos ? nlohmann::json()
                                      : nlohmann::json::parse(request.substr(brace));
}

}  // namespace

class MnsManagerGuiWidgetTest : public QObject {
    Q_OBJECT

   private:
    std::unique_ptr<RecordingServer> server;
    std::unique_ptr<MnsManagerGUI> window;

    void open_window(const std::string& url) {
        window = std::make_unique<MnsManagerGUI>(url);
        window->show();
        QVERIFY(QTest::qWaitForWindowExposed(window.get()));
        wait_idle();
    }
    /// Waits until no request runs (the controls are enabled again).
    void wait_idle() {
        QTRY_VERIFY(child<QWidget>(window.get(), "content")->isEnabled());
    }
    QString status() {
        return child<QLabel>(window.get(), "statusLabel")->text();
    }
    void click_model_row(int row) {
        auto* table = child<QTableWidget>(window.get(), "modelsTable");
        QTableWidgetItem* item = table->item(row, 0);
        QVERIFY(item);
        QTest::mouseClick(table->viewport(), Qt::LeftButton, {},
                          table->visualItemRect(item).center());
        wait_idle();
    }
    void click(const QString& label) {
        QPushButton* b = button(window.get(), label);
        QVERIFY2(b, qPrintable("no button " + label));
        QTest::mouseClick(b, Qt::LeftButton);
        wait_idle();
    }
    void click_named(const char* name) {
        QTest::mouseClick(child<QPushButton>(window.get(), name), Qt::LeftButton);
        wait_idle();
    }
    void type(const char* field, const QString& text) {
        auto* edit = child<QLineEdit>(window.get(), field);
        edit->clear();
        QTest::keyClicks(edit, text);
    }
    void open_register_tab(const QString& kind) {
        window->findChild<QTabWidget*>()->setCurrentIndex(2);
        auto* kinds = child<QComboBox>(window.get(), "regKind");
        kinds->setCurrentIndex(kinds->findText(kind));
    }

   private slots:
    void init() {
        server = std::make_unique<RecordingServer>();
        serve_mns(*server);
        open_window(server->url());
    }
    void cleanup() {
        window.reset();
        server.reset();
    }

    void startupConnectsAndFillsBothTables() {
        QCOMPARE(status(), QString("Connected"));
        auto* models = child<QTableWidget>(window.get(), "modelsTable");
        QCOMPARE(models->rowCount(), 2);
        QCOMPARE(models->item(0, 0)->text(), QString("m1"));
        QCOMPARE(models->item(0, 2)->text(), QString("training"));
        QCOMPARE(models->item(0, 3)->text(), QString("01234567..."));
        QCOMPARE(models->item(1, 2)->text(), QString("candidate"));
        auto* roles = child<QTableWidget>(window.get(), "rolesTable");
        QCOMPARE(roles->rowCount(), 2);
        QCOMPARE(roles->item(0, 1)->text(), QString("m2"));
        QCOMPARE(roles->item(1, 1)->text(), QString("(none)"));
    }

    void startupWithoutAServerShowsDisconnected() {
        const std::string url = server->url();
        window.reset();
        server.reset();  // nothing listens on that port now
        open_window(url);
        QVERIFY2(status().startsWith("Disconnected ("), qPrintable(status()));
        QCOMPARE(child<QTableWidget>(window.get(), "modelsTable")->rowCount(), 0);
    }

    void theServerFieldShowsTheStartupUrl() {
        QCOMPARE(child<QLineEdit>(window.get(), "serverUrl")->text(),
                 QString::fromStdString(server->url()));
    }

    // TD-452: httplib here has no TLS, so https:// used to go out as plain HTTP to port 8083.
    void anHttpsUrlIsRefusedWithAReason() {
        server->clear_requests();
        type("serverUrl", "https://mns.example.com");
        click_named("connectButton");
        QVERIFY2(status().contains("https:// isn't supported"), qPrintable(status()));
        QVERIFY(server->requests().empty());
    }

    void connectButtonSwitchesToTheTypedServer() {
        RecordingServer other;
        other.on("GET", "/health", Reply{200, R"({"status":"ok"})"});
        other.on("GET", "/models",
                 Reply{200, R"({"models":[{"model_name":"z9","state":"retired"}]})"});
        other.on("GET", "/roles", Reply{200, R"({"roles":[]})"});
        type("serverUrl", QString::fromStdString(other.url()));
        click_named("connectButton");
        QCOMPARE(status(), QString("Connected"));
        auto* models = child<QTableWidget>(window.get(), "modelsTable");
        QCOMPARE(models->rowCount(), 1);
        QCOMPARE(models->item(0, 0)->text(), QString("z9"));
        QCOMPARE(other.requests_starting("GET /health").size(), size_t{1});
    }

    void refreshReloadsModelsAndRoles() {
        server->clear_requests();
        click_named("refreshButton");
        QCOMPARE(server->requests_starting("GET /models").size(), size_t{1});
        QCOMPARE(server->requests_starting("GET /roles").size(), size_t{1});
    }

    // TD-455: the role filter lists the server's roles.
    void theRoleFilterOffersTheServersRoles() {
        auto* role = child<QComboBox>(window.get(), "roleFilter");
        QCOMPARE(role->count(), 3);
        QCOMPARE(role->itemText(1), QString("chat"));
        QCOMPARE(role->itemText(2), QString("spare"));
    }

    void filtersRequestMatchingModels() {
        server->clear_requests();
        child<QComboBox>(window.get(), "stateFilter")->setCurrentIndex(2);  // "training"
        wait_idle();
        QCOMPARE(last(server->requests()), QString("GET /models?state=training"));
        auto* role = child<QComboBox>(window.get(), "roleFilter");
        role->setCurrentIndex(role->findText("chat"));
        wait_idle();
        QCOMPARE(last(server->requests()), QString("GET /models?state=training&role=chat"));
    }

    // GET /roles lists only roles with a production model, so a typed role (one still in
    // training) must survive the role list being refilled, or the table stays filtered while the
    // filter shows "(all)".
    void aTypedRoleFilterSurvivesARefresh() {
        auto* role = child<QComboBox>(window.get(), "roleFilter");
        role->setFocus();
        role->lineEdit()->clear();
        QTest::keyClicks(role, "draft");
        QTest::keyClick(role, Qt::Key_Return);
        wait_idle();
        QCOMPARE(last(server->requests()), QString("GET /models?role=draft"));
        click_named("refreshButton");
        QCOMPARE(role->currentText(), QString("draft"));
        server->clear_requests();
        click_named("refreshButton");
        QCOMPARE(first(server->requests()), QString("GET /models?role=draft"));
    }

    // TD-455: names typed into the URL are encoded.
    void namesAreUrlEncoded() {
        server->on("GET", "/models",
                   Reply{200, R"({"models":[{"model_name":"a b/c","state":"training"}]})"});
        child<QComboBox>(window.get(), "stateFilter")->setCurrentIndex(1);
        wait_idle();
        auto* role = child<QComboBox>(window.get(), "roleFilter");
        role->addItem("x&y");
        role->setCurrentIndex(role->findText("x&y"));
        wait_idle();
        QCOMPARE(last(server->requests()), QString("GET /models?state=initializing&role=x%26y"));
        click_model_row(0);
        QCOMPARE(last(server->requests()), QString("GET /models/a%20b%2Fc"));
    }

    void clickingAModelRowLoadsItsRecordAndFillsTheActions() {
        click_model_row(0);
        QVERIFY(child<QTextEdit>(window.get(), "detailView")->toPlainText().contains("\"run-03\""));
        QCOMPARE(child<QLineEdit>(window.get(), "runIdField")->text(), QString("run-03"));
        QCOMPARE(child<QLineEdit>(window.get(), "promoteModelField")->text(), QString("m1"));
        QCOMPARE(child<QLineEdit>(window.get(), "promoteRoleField")->text(), QString("chat"));
        QCOMPARE(server->requests_starting("GET /models/m1").size(), size_t{1});
    }

    // TD-568: only a mouse click used to load the record.
    void movingTheSelectionWithTheKeyboardLoadsTheRecord() {
        click_model_row(0);
        auto* table = child<QTableWidget>(window.get(), "modelsTable");
        table->setFocus();
        QTest::keyClick(table, Qt::Key_Down);
        wait_idle();
        QCOMPARE(last(server->requests()), QString("GET /models/m2"));
        QVERIFY(child<QTextEdit>(window.get(), "detailView")->toPlainText().contains("candidate"));
        QCOMPARE(child<QLineEdit>(window.get(), "promoteModelField")->text(), QString("m2"));
    }

    // TD-568: a refilled table keeps the selected model highlighted, not whichever model now sits
    // in its old row, and refilling loads nothing; Refresh reloads the selected record.
    void aRefreshKeepsTheSelectedModelHighlightedAndReloadsIt() {
        click_model_row(1);  // m2
        server->on("GET", "/models",
                   Reply{200, R"({"models":[{"model_name":"m0","state":"training"},)"
                              R"({"model_name":"m1","state":"training"},)"
                              R"({"model_name":"m2","state":"candidate"}]})"});
        server->on("GET", "/models/m2",
                   Reply{200, R"({"model_name":"m2","state":"production","run_id":"run-01"})"});
        server->clear_requests();
        click_named("refreshButton");
        auto* table = child<QTableWidget>(window.get(), "modelsTable");
        QCOMPARE(table->currentRow(), 2);
        QCOMPARE(table->item(table->currentRow(), 0)->text(), QString("m2"));
        QCOMPARE(server->requests_starting("GET /models/m2").size(), size_t{1});
        QVERIFY(child<QTextEdit>(window.get(), "detailView")->toPlainText().contains("production"));
        QCOMPARE(server->requests_starting("GET /models/m1").size(), size_t{0});
        QCOMPARE(server->requests_starting("GET /models/m0").size(), size_t{0});
    }

    void aRefreshWithNothingSelectedSelectsNothing() {
        server->clear_requests();
        click_named("refreshButton");
        QCOMPARE(child<QTableWidget>(window.get(), "modelsTable")->currentRow(), -1);
        QCOMPARE(server->requests_starting("GET /models/").size(), size_t{0});
    }

    void clickingARoleRowFillsPromoteAndShowsItsModel() {
        auto* roles = child<QTableWidget>(window.get(), "rolesTable");
        window->findChild<QTabWidget*>()->setCurrentIndex(1);
        QTest::mouseClick(roles->viewport(), Qt::LeftButton, {},
                          roles->visualItemRect(roles->item(0, 0)).center());
        wait_idle();
        QCOMPARE(child<QLineEdit>(window.get(), "promoteRoleField")->text(), QString("chat"));
        QCOMPARE(child<QLineEdit>(window.get(), "promoteModelField")->text(), QString("m2"));
        QVERIFY(child<QTextEdit>(window.get(), "detailView")->toPlainText().contains("candidate"));
    }

    void actionsWithoutASelectedModelWarnAndSendNothing() {
        server->clear_requests();
        for (const QString label : {"Set Training", "Set Candidate", "Retire", "Delete"}) {
            DialogAnswerer dialogs;
            dialogs.answer_message(QMessageBox::Ok);
            click(label);
            QCOMPARE(first_title(dialogs), QString("No Model Selected"));
        }
        QVERIFY(server->requests().empty());
    }

    // TD-454: Set Training sends no run ID; the run MNS allocates fills Set Candidate's field.
    void setTrainingContinuesTheRunAndShowsTheAllocatedRunId() {
        server->on("PUT", "/models/m1/state",
                   Reply{200, R"({"model_name":"m1","state":"training","run_id":"run-03"})"});
        click_model_row(0);
        child<QLineEdit>(window.get(), "runIdField")->clear();
        click("Set Training");
        QCOMPARE(first(server->requests_starting("PUT /models/m1/state")),
                 QString(R"(PUT /models/m1/state {"state":"training"})"));
        QCOMPARE(child<QLineEdit>(window.get(), "runIdField")->text(), QString("run-03"));
        QCOMPARE(status(), QString("m1 -> training (run-03)"));
    }

    void setTrainingWithNewRunTickedAsksForTheNextRun() {
        server->on("PUT", "/models/m1/state",
                   Reply{200, R"({"model_name":"m1","state":"training","run_id":"run-04"})"});
        click_model_row(0);
        auto* new_run = child<QCheckBox>(window.get(), "newRunCheck");
        QTest::mouseClick(new_run, Qt::LeftButton);
        QVERIFY(new_run->isChecked());
        click("Set Training");
        QCOMPARE(first(server->requests_starting("PUT /models/m1/state")),
                 QString(R"(PUT /models/m1/state {"state":"training","new_run":true})"));
        QCOMPARE(child<QLineEdit>(window.get(), "runIdField")->text(), QString("run-04"));
        QVERIFY(!new_run->isChecked());  // a later Set Training continues run-04
    }

    void setCandidateSendsTheRunIdAndArtifact() {
        server->on("PUT", "/models/m1/state",
                   Reply{200, R"({"model_name":"m1","state":"candidate"})"});
        click_model_row(0);
        QTest::keyClicks(child<QLineEdit>(window.get(), "artifactField"), "/models/m1.bin");
        click("Set Candidate");
        const auto body =
            body_of(first(server->requests_starting("PUT /models/m1/state")).toStdString());
        QCOMPARE(QString::fromStdString(body.value("state", "")), QString("candidate"));
        QCOMPARE(QString::fromStdString(body.value("run_id", "")), QString("run-03"));
        QCOMPARE(QString::fromStdString(body["artifact"].value("path", "")),
                 QString("/models/m1.bin"));
        QCOMPARE(status(), QString("m1 -> candidate"));
    }

    void aRefusedTransitionShowsTheServersError() {
        server->on("PUT", "/models/m1/state",
                   Reply{409, R"({"error":"run_id does not match the active run"})"});
        click_model_row(0);
        DialogAnswerer dialogs;
        dialogs.answer_message(QMessageBox::Ok);
        click("Set Candidate");
        QCOMPARE(first_title(dialogs), QString("State Transition Failed"));
        QCOMPARE(dialogs.seen()[0].text,
                 QString("run_id does not match the active run (HTTP 409)"));
    }

    // TD-455: success is the HTTP status, not the word "error" in the body.
    void aFailureWithoutAnErrorFieldIsStillAFailure() {
        server->on("PUT", "/models/m1/state", Reply{500, "internal failure"});
        click_model_row(0);
        DialogAnswerer dialogs;
        dialogs.answer_message(QMessageBox::Ok);
        click("Set Training");
        QCOMPARE(first_title(dialogs), QString("State Transition Failed"));
        QCOMPARE(dialogs.seen()[0].text, QString("HTTP 500: internal failure"));
        QCOMPARE(status(), QString("State Transition Failed"));
    }

    void aSuccessMentioningErrorIsStillASuccess() {
        server->on("PUT", "/models/m1/state",
                   Reply{200, R"({"model_name":"m1","state":"training","run_id":"run-05",)"
                              R"("tags":{"last_check":"error"}})"});
        click_model_row(0);
        DialogAnswerer dialogs;
        dialogs.answer_message(QMessageBox::Ok, 500);
        click("Set Training");
        QCOMPARE(status(), QString("m1 -> training (run-05)"));
        QTest::qWait(600);
        QVERIFY(dialogs.seen().empty());
    }

    void aLostServerIsReportedAsAConnectionFailure() {
        click_model_row(0);
        server->stop();
        click("Set Training");
        QVERIFY2(status().startsWith("Connection failed ("), qPrintable(status()));
    }

    // TD-455: a request runs off the GUI thread: the window keeps painting with its controls
    // disabled, and closing it doesn't wait for the server.
    void aSlowServerDoesNotFreezeTheWindow() {
        std::mutex mutex;
        std::condition_variable released;
        bool hold = true;
        bool waiting = false;
        server->on("GET", "/models/m1", [&](const httplib::Request&) {
            std::unique_lock<std::mutex> lock(mutex);
            waiting = true;
            released.wait_for(lock, std::chrono::seconds(30), [&] { return !hold; });
            return Reply{200, kTrainingRecord};
        });
        auto* table = child<QTableWidget>(window.get(), "modelsTable");
        const auto clicked = std::chrono::steady_clock::now();
        QTest::mouseClick(table->viewport(), Qt::LeftButton, {},
                          table->visualItemRect(table->item(0, 0)).center());
        // The click returns at once: the request waits on the worker, not the GUI thread.
        QVERIFY(std::chrono::steady_clock::now() - clicked < std::chrono::seconds(2));
        QTRY_VERIFY([&] {
            std::lock_guard<std::mutex> lock(mutex);
            return waiting;
        }());
        QVERIFY(!child<QWidget>(window.get(), "content")->isEnabled());
        QCOMPARE(status(), QString("Loading m1..."));
        QTest::qWait(100);  // the event loop runs meanwhile
        const auto start = std::chrono::steady_clock::now();
        window.reset();
        QVERIFY2(std::chrono::steady_clock::now() - start < std::chrono::seconds(5),
                 "closing the window waited for the server");
        {
            std::lock_guard<std::mutex> lock(mutex);
            hold = false;
        }
        released.notify_all();
    }

    // TD-452: the Register tab sends kind and run group, through mns_cli's own request builder.
    void registerSendsAChatbotsFormAfterConfirmation() {
        server->on("POST", "/models",
                   Reply{201, R"({"model_name":"new1","state":"initializing"})"});
        open_register_tab("chatbot");
        type("regName", "new1");
        type("regRole", "chat");
        type("regRunGroup", "group-a");
        type("regDModel", "256");
        type("regHeads", "4");
        type("regDff", "1024");
        type("regMaxSeq", "128");
        type("regEncLayers", "2");
        type("regDecLayers", "3");
        type("regTags", "owner=ops");
        DialogAnswerer confirm;
        confirm.answer_message(QMessageBox::Yes);
        click_named("registerButton");
        QCOMPARE(first_title(confirm), QString("Confirm Registration"));
        QVERIFY(confirm.seen()[0].text.contains("d_model=256"));
        const auto body = body_of(first(server->requests_starting("POST /models")).toStdString());
        QCOMPARE(QString::fromStdString(body.value("model_name", "")), QString("new1"));
        QCOMPARE(QString::fromStdString(body.value("kind", "")), QString("chatbot"));
        QCOMPARE(QString::fromStdString(body.value("run_group", "")), QString("group-a"));
        QCOMPARE(body["arch"].value("d_model", 0), 256);
        QCOMPARE(body["arch"].value("num_encoder_layers", 0), 2);
        QCOMPARE(body["arch"].value("num_decoder_layers", 0), 3);
        QCOMPARE(QString::fromStdString(body["tags"].value("owner", "")), QString("ops"));
        QCOMPARE(status(), QString("Registered: new1"));
        QVERIFY(child<QLineEdit>(window.get(), "regName")->text().isEmpty());
    }

    void registerAStandaloneKindSendsItsOwnLayerCount() {
        server->on("POST", "/models", Reply{201, R"({"model_name":"enc1"})"});
        open_register_tab("world_model");
        QVERIFY(child<QLineEdit>(window.get(), "regLayers")->isVisible());
        QVERIFY(!child<QLineEdit>(window.get(), "regEncLayers")->isVisible());
        type("regName", "wm1");
        type("regRole", "world");
        type("regDModel", "64");
        type("regHeads", "2");
        type("regDff", "128");
        type("regMaxSeq", "32");
        type("regLayers", "3");
        type("regSigregSketches", "16");
        DialogAnswerer confirm;
        confirm.answer_message(QMessageBox::Yes);
        click_named("registerButton");
        const auto body = body_of(first(server->requests_starting("POST /models")).toStdString());
        QCOMPARE(QString::fromStdString(body.value("kind", "")), QString("world_model"));
        QCOMPARE(body["arch"].value("num_encoder_layers", -1), 3);
        QCOMPARE(body["arch"].value("num_decoder_layers", -1), 0);
        QCOMPARE(body["connection"].value("sigreg_num_sketches", 0), 16);
    }

    void registerALinkedChatbotSendsTheConnection() {
        server->on("POST", "/models", Reply{201, R"({"model_name":"pair"})"});
        open_register_tab("chatbot");
        type("regName", "pair");
        type("regRole", "chat");
        type("regEncoderName", "enc1");
        type("regDecoderName", "dec1");
        DialogAnswerer confirm;
        confirm.answer_message(QMessageBox::Yes);
        click_named("registerButton");
        QVERIFY(confirm.seen().at(0).text.contains("linked to encoder enc1 and decoder dec1"));
        const auto body = body_of(first(server->requests_starting("POST /models")).toStdString());
        QVERIFY(!body.contains("arch"));
        QCOMPARE(QString::fromStdString(body["connection"].value("encoder_name", "")),
                 QString("enc1"));
        QCOMPARE(QString::fromStdString(body["connection"].value("decoder_name", "")),
                 QString("dec1"));
    }

    // TD-455: the architecture starts blank and is required.
    void registerWithoutTheWholeArchitectureIsRefused() {
        server->clear_requests();
        open_register_tab("chatbot");
        QVERIFY(child<QLineEdit>(window.get(), "regDModel")->text().isEmpty());
        type("regName", "new1");
        type("regRole", "chat");
        type("regDModel", "256");
        DialogAnswerer dialogs;
        dialogs.answer_message(QMessageBox::Ok);
        click_named("registerButton");
        QCOMPARE(first_title(dialogs), QString("Invalid Registration"));
        QVERIFY2(dialogs.seen()[0].text.contains("Missing: --num-heads"),
                 qPrintable(dialogs.seen()[0].text));
        QVERIFY(server->requests_starting("POST").empty());
    }

    void declinedConfirmationRegistersNothing() {
        server->clear_requests();
        open_register_tab("encoder");
        type("regName", "e");
        type("regRole", "r");
        type("regDModel", "64");
        type("regHeads", "2");
        type("regDff", "128");
        type("regMaxSeq", "32");
        type("regLayers", "1");
        DialogAnswerer no;
        no.answer_message(QMessageBox::No);
        click_named("registerButton");
        QCOMPARE(first_title(no), QString("Confirm Registration"));
        QVERIFY(server->requests_starting("POST").empty());
    }

    void loadFromConfigFillsTheArchitecture() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString path = dir.filePath("config.trainer.conf");
        {
            QFile f(path);
            QVERIFY(f.open(QIODevice::WriteOnly));
            f.write(
                "D_MODEL=384\nNUM_HEADS=6\nD_FF=1536\nNUM_ENCODER_LAYERS=4\nNUM_DECODER_LAYERS=5\n"
                "MAX_SEQ_LENGTH=256\n");
        }
        QCoreApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
        open_register_tab("decoder");
        DialogAnswerer file;
        file.answer_file(path);
        click_named("loadConfigButton");
        QCOMPARE(child<QLineEdit>(window.get(), "regDModel")->text(), QString("384"));
        QCOMPARE(child<QLineEdit>(window.get(), "regHeads")->text(), QString("6"));
        QCOMPARE(child<QLineEdit>(window.get(), "regMaxSeq")->text(), QString("256"));
        QCOMPARE(child<QLineEdit>(window.get(), "regLayers")->text(),
                 QString("5"));  // a decoder's own
        QVERIFY(status().startsWith("Architecture loaded from"));
    }

    void registerWithoutANameOrRoleWarns() {
        server->clear_requests();
        open_register_tab("chatbot");
        type("regName", "new1");
        DialogAnswerer dialogs;
        dialogs.answer_message(QMessageBox::Ok);
        click_named("registerButton");
        QCOMPARE(first_title(dialogs), QString("Missing Fields"));
        QVERIFY(server->requests_starting("POST").empty());
    }

    void promoteSendsTheModelToTheRole() {
        server->on("PUT", "/roles/chat/production", Reply{200, R"({"promoted":"m1"})"});
        click_model_row(0);
        click("Promote");
        QCOMPARE(first(server->requests_starting("PUT /roles/chat/production")),
                 QString(R"(PUT /roles/chat/production {"model_name":"m1"})"));
        QCOMPARE(status(), QString("m1 promoted to chat"));
    }

    void retireAsksFirstAndSendsOnlyOnYes() {
        server->on("PUT", "/models/m1/state",
                   Reply{200, R"({"model_name":"m1","state":"retired"})"});
        click_model_row(0);
        DialogAnswerer no;
        no.answer_message(QMessageBox::No);
        click("Retire");
        QCOMPARE(first_title(no), QString("Confirm Retire"));
        QVERIFY(server->requests_starting("PUT /models/m1/state").empty());

        DialogAnswerer yes;
        yes.answer_message(QMessageBox::Yes);
        click("Retire");
        QCOMPARE(first(server->requests_starting("PUT /models/m1/state")),
                 QString(R"(PUT /models/m1/state {"state":"retired"})"));
        QCOMPARE(status(), QString("m1 retired"));
    }

    void deleteRemovesTheModelAndClearsTheDetail() {
        server->on("DELETE", "/models/m1", Reply{200, R"({"deleted":"m1"})"});
        click_model_row(0);
        DialogAnswerer yes;
        yes.answer_message(QMessageBox::Yes);
        click("Delete");
        QCOMPARE(first_title(yes), QString("Confirm Delete"));
        QCOMPARE(server->requests_starting("DELETE /models/m1").size(), size_t{1});
        QCOMPARE(status(), QString("m1 deleted"));
        QVERIFY(child<QTextEdit>(window.get(), "detailView")->toPlainText().isEmpty());

        // The selection went with it: another action asks for a model again.
        DialogAnswerer warn;
        warn.answer_message(QMessageBox::Ok);
        click("Set Training");
        QCOMPARE(first_title(warn), QString("No Model Selected"));
    }
};

QTEST_MAIN(MnsManagerGuiWidgetTest)
#include "mns_manager_gui_widget_test.moc"
