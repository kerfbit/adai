// TD-037: widget tests for MnsManagerGUI (mns_manager_gui). Each test builds the real window
// against an in-process stand-in for mns_server, drives the widgets with QTest (clicks, typing,
// table rows, modal dialogs) and checks the requests the server received and what the window shows.
// Runs on the offscreen Qt platform (QT_QPA_PLATFORM=offscreen, set by ctest).

#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QLineEdit>
#include <QSpinBox>
#include <QTableWidget>
#include <QTest>
#include <QTextEdit>
#include <memory>
#include "MnsManagerGUI.hpp"
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
    R"({"roles":[{"role":"chat","production_model":"m2"},{"role":"spare","production_model":""}]})";

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
    }
    void click(const QString& label) {
        QPushButton* b = button(window.get(), label);
        QVERIFY2(b, qPrintable("no button " + label));
        QTest::mouseClick(b, Qt::LeftButton);
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
        QCOMPARE(status(), QString("Disconnected"));
        QCOMPARE(child<QTableWidget>(window.get(), "modelsTable")->rowCount(), 0);
    }

    void connectButtonSwitchesToTheTypedServer() {
        RecordingServer other;
        other.on("GET", "/health", Reply{200, R"({"status":"ok"})"});
        other.on("GET", "/models",
                 Reply{200, R"({"models":[{"model_name":"z9","state":"retired"}]})"});
        other.on("GET", "/roles", Reply{200, R"({"roles":[]})"});
        auto* url = child<QLineEdit>(window.get(), "serverUrl");
        url->clear();
        QTest::keyClicks(url, QString::fromStdString(other.url()));
        QTest::mouseClick(child<QPushButton>(window.get(), "connectButton"), Qt::LeftButton);
        QCOMPARE(status(), QString("Connected"));
        auto* models = child<QTableWidget>(window.get(), "modelsTable");
        QCOMPARE(models->rowCount(), 1);
        QCOMPARE(models->item(0, 0)->text(), QString("z9"));
        QCOMPARE(other.requests_starting("GET /health").size(), size_t{1});
    }

    void refreshReloadsModelsAndRoles() {
        server->clear_requests();
        QTest::mouseClick(child<QPushButton>(window.get(), "refreshButton"), Qt::LeftButton);
        QCOMPARE(server->requests_starting("GET /models").size(), size_t{1});
        QCOMPARE(server->requests_starting("GET /roles").size(), size_t{1});
    }

    void filtersRequestMatchingModels() {
        server->clear_requests();
        child<QComboBox>(window.get(), "stateFilter")->setCurrentIndex(2);  // "training"
        QCOMPARE(last(server->requests()), QString("GET /models?state=training"));
        auto* role = child<QComboBox>(window.get(), "roleFilter");
        role->addItem("chat");
        role->setCurrentIndex(1);
        QCOMPARE(last(server->requests()), QString("GET /models?role=chat&state=training"));
    }

    void clickingAModelRowLoadsItsRecordAndFillsTheActions() {
        click_model_row(0);
        QVERIFY(child<QTextEdit>(window.get(), "detailView")->toPlainText().contains("\"run-03\""));
        QCOMPARE(child<QLineEdit>(window.get(), "runIdField")->text(), QString("run-03"));
        QCOMPARE(child<QLineEdit>(window.get(), "promoteModelField")->text(), QString("m1"));
        QCOMPARE(child<QLineEdit>(window.get(), "promoteRoleField")->text(), QString("chat"));
        QCOMPARE(server->requests_starting("GET /models/m1").size(), size_t{1});
    }

    void clickingARoleRowFillsPromoteAndShowsItsModel() {
        auto* roles = child<QTableWidget>(window.get(), "rolesTable");
        window->findChild<QTabWidget*>()->setCurrentIndex(1);
        QTest::mouseClick(roles->viewport(), Qt::LeftButton, {},
                          roles->visualItemRect(roles->item(0, 0)).center());
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
            QCOMPARE(dialogs.seen().size(), size_t{1});
            QCOMPARE(dialogs.seen()[0].title, QString("No Model Selected"));
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
        const auto puts = server->requests_starting("PUT /models/m1/state");
        QCOMPARE(puts.size(), size_t{1});
        QCOMPARE(QString::fromStdString(puts[0]),
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
        const auto puts = server->requests_starting("PUT /models/m1/state");
        QCOMPARE(puts.size(), size_t{1});
        QVERIFY(puts[0].find(R"("state":"candidate","run_id":"run-03")") != std::string::npos);
        QVERIFY(puts[0].find(R"("artifact":{"path":"/models/m1.bin")") != std::string::npos);
        QCOMPARE(status(), QString("m1 -> candidate"));
    }

    void aRefusedTransitionShowsTheServersError() {
        server->on("PUT", "/models/m1/state",
                   Reply{409, R"({"error":"run_id does not match the active run"})"});
        click_model_row(0);
        DialogAnswerer dialogs;
        dialogs.answer_message(QMessageBox::Ok);
        click("Set Candidate");
        QCOMPARE(dialogs.seen().size(), size_t{1});
        QCOMPARE(dialogs.seen()[0].title, QString("State Transition Failed"));
        QVERIFY(dialogs.seen()[0].text.contains("does not match the active run"));
    }

    void aLostServerIsReportedAsAConnectionFailure() {
        click_model_row(0);
        server->stop();
        click("Set Training");
        QCOMPARE(status(), QString("Connection failed"));
    }

    void registerSendsTheFormAndClearsTheName() {
        server->on("POST", "/models",
                   Reply{201, R"({"model_name":"new1","state":"initializing"})"});
        window->findChild<QTabWidget*>()->setCurrentIndex(2);
        QTest::keyClicks(child<QLineEdit>(window.get(), "regName"), "new1");
        QTest::keyClicks(child<QLineEdit>(window.get(), "regRole"), "chat");
        auto* d_model = child<QSpinBox>(window.get(), "regDModel");
        d_model->selectAll();
        QTest::keyClicks(d_model, "256");
        QTest::keyClicks(child<QLineEdit>(window.get(), "regTags"), "owner=ops");
        QTest::mouseClick(child<QPushButton>(window.get(), "registerButton"), Qt::LeftButton);
        const auto posts = server->requests_starting("POST /models");
        QCOMPARE(posts.size(), size_t{1});
        QVERIFY(posts[0].find(R"("model_name":"new1","role":"chat")") != std::string::npos);
        QVERIFY(posts[0].find(R"("d_model":256)") != std::string::npos);
        QVERIFY(posts[0].find(R"("owner":"ops")") != std::string::npos);
        QCOMPARE(status(), QString("Registered: new1"));
        QVERIFY(child<QLineEdit>(window.get(), "regName")->text().isEmpty());
    }

    void registerWithoutANameOrRoleWarns() {
        server->clear_requests();
        window->findChild<QTabWidget*>()->setCurrentIndex(2);
        QTest::keyClicks(child<QLineEdit>(window.get(), "regName"), "new1");
        DialogAnswerer dialogs;
        dialogs.answer_message(QMessageBox::Ok);
        QTest::mouseClick(child<QPushButton>(window.get(), "registerButton"), Qt::LeftButton);
        QCOMPARE(dialogs.seen().size(), size_t{1});
        QCOMPARE(dialogs.seen()[0].title, QString("Missing Fields"));
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
