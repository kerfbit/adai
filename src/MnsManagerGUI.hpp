#ifndef MNS_MANAGER_GUI_HPP
#define MNS_MANAGER_GUI_HPP

// @adai-status: beta        (Qt GUI, no dedicated test file)
// @adai-version: 0.7.0
// @adai-reviewed: 2026-09-07


#include <QComboBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QPushButton>
#include <QSpinBox>
#include <QSplitter>
#include <QTabWidget>
#include <QTableWidget>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>
#include <string>

class MnsManagerGUI : public QMainWindow {
    Q_OBJECT

   public:
    explicit MnsManagerGUI(const std::string& server_url = "http://localhost:8083",
                           QWidget* parent = nullptr);
    ~MnsManagerGUI() override;

   private slots:
    void onRefreshModels();
    void onRefreshRoles();
    void onModelSelected(int row, int col);
    void onRoleSelected(int row, int col);
    void onRegisterModel();
    void onSetTraining();
    void onSetCandidate();
    void onPromote();
    void onRetire();
    void onDelete();
    void onHealthCheck();

   private:
    void setupUI();
    void applyStylesheet();

    QWidget* createToolbar();
    QWidget* createModelsTab();
    QWidget* createRolesTab();
    QWidget* createRegisterTab();
    QWidget* createDetailPanel();

    void populateModelsTable(const std::string& json);
    void populateRolesTable(const std::string& json);
    void showModelDetail(const std::string& json);
    void setStatusMessage(const QString& msg, bool error = false);

    std::string httpGet(const std::string& path);
    std::string httpPost(const std::string& path, const std::string& body);
    std::string httpPut(const std::string& path, const std::string& body);
    std::string httpDelete(const std::string& path);

    // Connection
    std::string server_url_;
    std::string host_;
    int port_;
    void parseUrl();

    // Toolbar
    QLineEdit* urlField_;
    QPushButton* connectBtn_;
    QLabel* statusLabel_;

    // Models tab
    QTableWidget* modelsTable_;
    QComboBox* stateFilter_;
    QComboBox* roleFilter_;

    // Roles tab
    QTableWidget* rolesTable_;

    // Register tab
    QLineEdit* regName_;
    QLineEdit* regRole_;
    QSpinBox* regDModel_;
    QSpinBox* regHeads_;
    QSpinBox* regDff_;
    QSpinBox* regEncLayers_;
    QSpinBox* regDecLayers_;
    QSpinBox* regMaxSeq_;
    QLineEdit* regTags_;

    // Detail panel
    QTextEdit* detailView_;

    // Action panel
    QLineEdit* actionRunId_;
    QLineEdit* actionArtifactPath_;
    QLineEdit* actionPromoteRole_;
    QLineEdit* actionPromoteModel_;

    // Tabs
    QTabWidget* tabWidget_;

    // Currently selected model name
    QString selectedModel_;
};

#endif  // MNS_MANAGER_GUI_HPP
