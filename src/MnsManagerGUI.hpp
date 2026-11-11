#pragma once

// @adai-status: beta        (widget tests: mnsManagerGuiWidgetTests)
// @adai-version: 0.9.1
// @adai-reviewed: 2026-11-11

#include <QCheckBox>
#include <QComboBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QPushButton>
#include <QSplitter>
#include <QTabWidget>
#include <QTableWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace httplib {
class Client;
}

/**
 * @brief Qt GUI for mns_server (the Model Name Service): browse models and roles, register models,
 * and move a model through its lifecycle.
 *
 * TD-455: every request runs on a worker thread, one at a time, with the window's controls disabled
 * until it finishes, so a slow or absent server never freezes the window. Success is judged by the
 * HTTP status, and a failure shows the server's own error. TD-452: the Register tab registers every
 * kind (chatbot, encoder, decoder, world model) with run group, linked encoder/decoder and SIGReg
 * settings, building the request with mns_cli's own `adai::build_register_request()`.
 */
class MnsManagerGUI : public QMainWindow {
    Q_OBJECT

   public:
    explicit MnsManagerGUI(const std::string& server_url = "http://localhost:8083",
                           QWidget* parent = nullptr);
    /// Stops a request in flight and waits for the worker thread.
    ~MnsManagerGUI() override;

   private slots:
    void onConnect();
    void onRefreshModels();
    void onModelSelected(int row, int col);
    void onRoleSelected(int row, int col);
    void onRegisterKindChanged(int index);
    void onLoadRegisterConfig();
    void onRegisterModel();
    void onSetTraining();
    void onSetCandidate();
    void onPromote();
    void onRetire();
    void onDelete();

   private:
    /// One HTTP exchange: `reached` is false when no reply came back (`error` says why).
    struct HttpReply {
        bool reached = false;
        int status = 0;
        std::string body;
        std::string error;
        bool ok() const {
            return reached && status >= 200 && status < 300;
        }
    };

    void setupUI();
    void applyStylesheet();

    QWidget* createToolbar();
    QWidget* createModelsTab();
    QWidget* createRolesTab();
    QWidget* createRegisterTab();
    QWidget* createDetailPanel();

    /// Points the window at `url`; false (with the reason in the status line) when it can't be
    /// used.
    bool setServer(const std::string& url);
    /// GET /health, then the models and roles, on the worker.
    void connectToServer();

    /// Sends one request (worker thread only). `path` is relative to the server URL.
    HttpReply http(const std::string& method, const std::string& path,
                   const std::string& body = {});
    /// Runs `work` on the worker thread, then the function it returns on the GUI thread; the
    /// controls are disabled until then.
    void runRequest(const QString& busyText, std::function<std::function<void()>()> work);
    void setBusy(bool busy, const QString& busyText = QString());

    /// Shows a failed reply: "Connection failed" when nothing came back, else a warning dialog
    /// titled `title` with the server's error.
    void reportFailure(const QString& title, const HttpReply& reply);
    /// GET /models with the current filters (path built on the GUI thread).
    std::string modelsPath() const;
    static std::string modelPath(const QString& name);

    void populateModelsTable(const std::string& json);
    void populateRolesTable(const std::string& json);
    void showModelDetail(const std::string& json);
    void setStatusMessage(const QString& msg, bool error = false);

    // Connection: set on the GUI thread only while no request runs.
    std::string server_url_;
    std::string host_;
    int port_ = 8083;
    std::string base_path_;
    bool server_usable_ = false;
    std::string server_problem_;

    // The worker and the connection it is using, which the destructor stops.
    std::thread worker_;
    std::mutex client_mutex_;
    std::shared_ptr<httplib::Client> active_client_;
    bool closing_ = false;  // guarded by client_mutex_

    // Toolbar
    QLineEdit* urlField_;
    QPushButton* connectBtn_;
    QPushButton* refreshBtn_;
    QLabel* statusLabel_;
    QString statusText_;  // shown again after a request's busy text
    bool statusError_ = false;

    // Everything a request disables
    QWidget* content_;

    // Models tab
    QTableWidget* modelsTable_;
    QComboBox* stateFilter_;
    QComboBox* roleFilter_;

    // Roles tab
    QTableWidget* rolesTable_;

    // Register tab
    QLineEdit* regName_;
    QLineEdit* regRole_;
    QComboBox* regKind_;
    QLineEdit* regRunGroup_;
    QLineEdit* regDModel_;
    QLineEdit* regHeads_;
    QLineEdit* regDff_;
    QLineEdit* regMaxSeq_;
    QLineEdit* regEncLayers_;  // chatbot
    QLineEdit* regDecLayers_;  // chatbot
    QLineEdit* regLayers_;     // encoder, decoder, world model
    QLineEdit* regEncoderName_;
    QLineEdit* regDecoderName_;
    QLineEdit* regSigregLambda_;
    QLineEdit* regSigregSketches_;
    QLineEdit* regTags_;
    QWidget* regChatbotLayersRow_;
    QWidget* regLayersRow_;
    QWidget* regLinkGroup_;
    QWidget* regSigregGroup_;

    // Detail panel
    QTextEdit* detailView_;

    // Action panel
    QCheckBox* actionNewRun_;  // TD-454: Set Training's new_run
    QLineEdit* actionRunId_;   // Set Candidate's run ID: the record's, or the one Set Training got
    QLineEdit* actionArtifactPath_;
    QLineEdit* actionPromoteRole_;
    QLineEdit* actionPromoteModel_;

    // Tabs
    QTabWidget* tabWidget_;

    // Currently selected model name
    QString selectedModel_;
};
