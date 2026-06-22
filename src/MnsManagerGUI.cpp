#include "MnsManagerGUI.hpp"
#include "MnsJsonHelpers.hpp"
#include <QApplication>
#include <QMessageBox>
#include <QScrollBar>
#include <httplib.h>
#include <sstream>

using mns_gui::json_value;
using mns_gui::json_array_objects;
using mns_gui::json_escape;
using mns_gui::json_pretty;

// ============================================================================
// Construction
// ============================================================================

MnsManagerGUI::MnsManagerGUI(const std::string& server_url, QWidget* parent)
    : QMainWindow(parent), server_url_(server_url), port_(8083) {
    setWindowTitle("ADAI Model Name Service Manager");
    resize(1100, 700);
    parseUrl();
    setupUI();
    applyStylesheet();
    onHealthCheck();
}

MnsManagerGUI::~MnsManagerGUI() = default;

void MnsManagerGUI::parseUrl() {
    auto parsed = mns_gui::ParsedUrl::from(server_url_);
    host_ = parsed.host;
    port_ = parsed.port;
}

// ============================================================================
// HTTP helpers
// ============================================================================

std::string MnsManagerGUI::httpGet(const std::string& path) {
    httplib::Client c(host_, port_);
    c.set_connection_timeout(5, 0);
    c.set_read_timeout(10, 0);
    auto res = c.Get(path);
    if (!res) return {};
    return res->body;
}

std::string MnsManagerGUI::httpPost(const std::string& path, const std::string& body) {
    httplib::Client c(host_, port_);
    c.set_connection_timeout(5, 0);
    c.set_read_timeout(10, 0);
    auto res = c.Post(path, body, "application/json");
    if (!res) return {};
    return res->body;
}

std::string MnsManagerGUI::httpPut(const std::string& path, const std::string& body) {
    httplib::Client c(host_, port_);
    c.set_connection_timeout(5, 0);
    c.set_read_timeout(10, 0);
    auto res = c.Put(path, body, "application/json");
    if (!res) return {};
    return res->body;
}

std::string MnsManagerGUI::httpDelete(const std::string& path) {
    httplib::Client c(host_, port_);
    c.set_connection_timeout(5, 0);
    c.set_read_timeout(10, 0);
    auto res = c.Delete(path);
    if (!res) return {};
    return res->body;
}

// ============================================================================
// UI setup
// ============================================================================

void MnsManagerGUI::setupUI() {
    QWidget* central = new QWidget(this);
    setCentralWidget(central);
    QVBoxLayout* root = new QVBoxLayout(central);

    // Toolbar
    root->addWidget(createToolbar());

    // Main content: tabs on the left, detail on the right
    QSplitter* splitter = new QSplitter(Qt::Horizontal);

    // Left: tabs
    tabWidget_ = new QTabWidget();
    tabWidget_->addTab(createModelsTab(), "Models");
    tabWidget_->addTab(createRolesTab(), "Roles");
    tabWidget_->addTab(createRegisterTab(), "Register");
    splitter->addWidget(tabWidget_);

    // Right: detail + actions
    splitter->addWidget(createDetailPanel());

    splitter->setStretchFactor(0, 6);
    splitter->setStretchFactor(1, 4);
    root->addWidget(splitter);
}

QWidget* MnsManagerGUI::createToolbar() {
    QWidget* w = new QWidget();
    QHBoxLayout* lay = new QHBoxLayout(w);
    lay->setContentsMargins(4, 4, 4, 4);

    QLabel* lbl = new QLabel("Server:");
    urlField_ = new QLineEdit(QString::fromStdString(server_url_));
    urlField_->setMinimumWidth(250);

    connectBtn_ = new QPushButton("Connect");
    connect(connectBtn_, &QPushButton::clicked, this, [this]() {
        server_url_ = urlField_->text().toStdString();
        parseUrl();
        onHealthCheck();
    });

    QPushButton* refreshBtn = new QPushButton("Refresh");
    connect(refreshBtn, &QPushButton::clicked, this, [this]() {
        onRefreshModels();
        onRefreshRoles();
    });

    statusLabel_ = new QLabel("");
    statusLabel_->setMinimumWidth(200);

    lay->addWidget(lbl);
    lay->addWidget(urlField_);
    lay->addWidget(connectBtn_);
    lay->addWidget(refreshBtn);
    lay->addStretch();
    lay->addWidget(statusLabel_);
    return w;
}

QWidget* MnsManagerGUI::createModelsTab() {
    QWidget* w = new QWidget();
    QVBoxLayout* lay = new QVBoxLayout(w);

    // Filter bar
    QHBoxLayout* filterLay = new QHBoxLayout();
    filterLay->addWidget(new QLabel("State:"));
    stateFilter_ = new QComboBox();
    stateFilter_->addItems({"(all)", "initializing", "training", "candidate", "production", "retired"});
    connect(stateFilter_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { onRefreshModels(); });
    filterLay->addWidget(stateFilter_);

    filterLay->addWidget(new QLabel("Role:"));
    roleFilter_ = new QComboBox();
    roleFilter_->setEditable(true);
    roleFilter_->addItem("(all)");
    connect(roleFilter_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { onRefreshModels(); });
    filterLay->addWidget(roleFilter_);
    filterLay->addStretch();
    lay->addLayout(filterLay);

    // Table
    modelsTable_ = new QTableWidget(0, 5);
    modelsTable_->setHorizontalHeaderLabels({"Name", "Role", "State", "Model ID", "Updated"});
    modelsTable_->horizontalHeader()->setStretchLastSection(true);
    modelsTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    modelsTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    modelsTable_->setSelectionMode(QAbstractItemView::SingleSelection);
    modelsTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    modelsTable_->setAlternatingRowColors(true);
    connect(modelsTable_, &QTableWidget::cellClicked, this, &MnsManagerGUI::onModelSelected);
    lay->addWidget(modelsTable_);

    return w;
}

QWidget* MnsManagerGUI::createRolesTab() {
    QWidget* w = new QWidget();
    QVBoxLayout* lay = new QVBoxLayout(w);

    rolesTable_ = new QTableWidget(0, 2);
    rolesTable_->setHorizontalHeaderLabels({"Role", "Production Model"});
    rolesTable_->horizontalHeader()->setStretchLastSection(true);
    rolesTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    rolesTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    rolesTable_->setSelectionMode(QAbstractItemView::SingleSelection);
    rolesTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    rolesTable_->setAlternatingRowColors(true);
    connect(rolesTable_, &QTableWidget::cellClicked, this, &MnsManagerGUI::onRoleSelected);
    lay->addWidget(rolesTable_);

    return w;
}

QWidget* MnsManagerGUI::createRegisterTab() {
    QWidget* w = new QWidget();
    QVBoxLayout* lay = new QVBoxLayout(w);

    auto addField = [&](const QString& label, QWidget* field) {
        QHBoxLayout* row = new QHBoxLayout();
        QLabel* lbl = new QLabel(label);
        lbl->setMinimumWidth(120);
        row->addWidget(lbl);
        row->addWidget(field);
        lay->addLayout(row);
    };

    regName_ = new QLineEdit();
    regName_->setPlaceholderText("e.g. adai-chatbot-v3");
    addField("Model Name:", regName_);

    regRole_ = new QLineEdit();
    regRole_->setPlaceholderText("e.g. chatbot");
    addField("Role:", regRole_);

    // Architecture group
    QGroupBox* archGroup = new QGroupBox("Architecture");
    QVBoxLayout* archLay = new QVBoxLayout(archGroup);

    auto addSpin = [&](const QString& label, int def, int min, int max) -> QSpinBox* {
        QHBoxLayout* row = new QHBoxLayout();
        QLabel* lbl = new QLabel(label);
        lbl->setMinimumWidth(120);
        QSpinBox* spin = new QSpinBox();
        spin->setRange(min, max);
        spin->setValue(def);
        row->addWidget(lbl);
        row->addWidget(spin);
        archLay->addLayout(row);
        return spin;
    };

    regDModel_ = addSpin("d_model:", 128, 16, 4096);
    regHeads_ = addSpin("num_heads:", 4, 1, 64);
    regDff_ = addSpin("d_ff:", 512, 16, 16384);
    regEncLayers_ = addSpin("encoder_layers:", 6, 1, 48);
    regDecLayers_ = addSpin("decoder_layers:", 6, 1, 48);
    regMaxSeq_ = addSpin("max_seq_length:", 1024, 32, 8192);
    lay->addWidget(archGroup);

    regTags_ = new QLineEdit();
    regTags_->setPlaceholderText("key1=val1, key2=val2");
    addField("Tags:", regTags_);

    QPushButton* regBtn = new QPushButton("Register Model");
    regBtn->setMinimumHeight(36);
    connect(regBtn, &QPushButton::clicked, this, &MnsManagerGUI::onRegisterModel);
    lay->addWidget(regBtn);

    lay->addStretch();
    return w;
}

QWidget* MnsManagerGUI::createDetailPanel() {
    QWidget* w = new QWidget();
    QVBoxLayout* lay = new QVBoxLayout(w);

    // Detail view
    QLabel* detailLabel = new QLabel("Model Detail");
    QFont f = detailLabel->font();
    f.setPointSize(12);
    f.setBold(true);
    detailLabel->setFont(f);
    lay->addWidget(detailLabel);

    detailView_ = new QTextEdit();
    detailView_->setReadOnly(true);
    detailView_->setFont(QFont("Monospace", 10));
    lay->addWidget(detailView_);

    // Actions group
    QGroupBox* actGroup = new QGroupBox("Actions");
    QVBoxLayout* actLay = new QVBoxLayout(actGroup);

    // Set Training
    QHBoxLayout* trainRow = new QHBoxLayout();
    actionRunId_ = new QLineEdit();
    actionRunId_->setPlaceholderText("run-id");
    QPushButton* trainBtn = new QPushButton("Set Training");
    trainBtn->setObjectName("actionBtn");
    connect(trainBtn, &QPushButton::clicked, this, &MnsManagerGUI::onSetTraining);
    trainRow->addWidget(new QLabel("Run ID:"));
    trainRow->addWidget(actionRunId_);
    trainRow->addWidget(trainBtn);
    actLay->addLayout(trainRow);

    // Set Candidate
    QHBoxLayout* candRow = new QHBoxLayout();
    actionArtifactPath_ = new QLineEdit();
    actionArtifactPath_->setPlaceholderText("/path/to/weights.bin");
    QPushButton* candBtn = new QPushButton("Set Candidate");
    candBtn->setObjectName("actionBtn");
    connect(candBtn, &QPushButton::clicked, this, &MnsManagerGUI::onSetCandidate);
    candRow->addWidget(new QLabel("Artifact:"));
    candRow->addWidget(actionArtifactPath_);
    candRow->addWidget(candBtn);
    actLay->addLayout(candRow);

    // Promote
    QHBoxLayout* promRow = new QHBoxLayout();
    actionPromoteRole_ = new QLineEdit();
    actionPromoteRole_->setPlaceholderText("role");
    actionPromoteModel_ = new QLineEdit();
    actionPromoteModel_->setPlaceholderText("model-name");
    QPushButton* promBtn = new QPushButton("Promote");
    promBtn->setObjectName("promoteBtn");
    connect(promBtn, &QPushButton::clicked, this, &MnsManagerGUI::onPromote);
    promRow->addWidget(new QLabel("Role:"));
    promRow->addWidget(actionPromoteRole_);
    promRow->addWidget(new QLabel("Model:"));
    promRow->addWidget(actionPromoteModel_);
    promRow->addWidget(promBtn);
    actLay->addLayout(promRow);

    // Retire / Delete
    QHBoxLayout* dangerRow = new QHBoxLayout();
    QPushButton* retireBtn = new QPushButton("Retire");
    retireBtn->setObjectName("dangerBtn");
    connect(retireBtn, &QPushButton::clicked, this, &MnsManagerGUI::onRetire);
    QPushButton* deleteBtn = new QPushButton("Delete");
    deleteBtn->setObjectName("dangerBtn");
    connect(deleteBtn, &QPushButton::clicked, this, &MnsManagerGUI::onDelete);
    dangerRow->addStretch();
    dangerRow->addWidget(retireBtn);
    dangerRow->addWidget(deleteBtn);
    actLay->addLayout(dangerRow);

    lay->addWidget(actGroup);
    return w;
}

// ============================================================================
// Slots
// ============================================================================

void MnsManagerGUI::onHealthCheck() {
    auto body = httpGet("/health");
    if (body.empty()) {
        setStatusMessage("Disconnected", true);
    } else {
        setStatusMessage("Connected");
        onRefreshModels();
        onRefreshRoles();
    }
}

void MnsManagerGUI::onRefreshModels() {
    std::string path = "/models";
    std::string sep = "?";
    if (stateFilter_ && stateFilter_->currentIndex() > 0) {
        path += sep + "state=" + stateFilter_->currentText().toStdString();
        sep = "&";
    }
    if (roleFilter_ && roleFilter_->currentIndex() > 0) {
        path += sep + "role=" + roleFilter_->currentText().toStdString();
    }

    auto body = httpGet(path);
    if (!body.empty()) populateModelsTable(body);
}

void MnsManagerGUI::onRefreshRoles() {
    auto body = httpGet("/roles");
    if (!body.empty()) populateRolesTable(body);
}

void MnsManagerGUI::onModelSelected(int row, int /*col*/) {
    auto item = modelsTable_->item(row, 0);
    if (!item) return;
    selectedModel_ = item->text();

    // Pre-fill action fields
    actionPromoteModel_->setText(selectedModel_);
    auto roleItem = modelsTable_->item(row, 1);
    if (roleItem) actionPromoteRole_->setText(roleItem->text());

    // Fetch full record
    auto body = httpGet("/models/" + selectedModel_.toStdString());
    if (!body.empty()) showModelDetail(body);
}

void MnsManagerGUI::onRoleSelected(int row, int /*col*/) {
    auto roleItem = rolesTable_->item(row, 0);
    auto modelItem = rolesTable_->item(row, 1);
    if (!roleItem) return;

    actionPromoteRole_->setText(roleItem->text());
    if (modelItem && !modelItem->text().isEmpty() && modelItem->text() != "null") {
        selectedModel_ = modelItem->text();
        actionPromoteModel_->setText(selectedModel_);
        auto body = httpGet("/models/" + selectedModel_.toStdString());
        if (!body.empty()) showModelDetail(body);
    }
}

void MnsManagerGUI::onRegisterModel() {
    std::string name = regName_->text().toStdString();
    std::string role = regRole_->text().toStdString();
    if (name.empty() || role.empty()) {
        QMessageBox::warning(this, "Missing Fields", "Model name and role are required.");
        return;
    }

    std::ostringstream body;
    body << "{\"model_name\":\"" << json_escape(name) << "\""
         << ",\"role\":\"" << json_escape(role) << "\""
         << ",\"arch\":{"
            << "\"d_model\":" << regDModel_->value()
            << ",\"num_heads\":" << regHeads_->value()
            << ",\"d_ff\":" << regDff_->value()
            << ",\"num_encoder_layers\":" << regEncLayers_->value()
            << ",\"num_decoder_layers\":" << regDecLayers_->value()
            << ",\"max_seq_length\":" << regMaxSeq_->value()
         << "}";

    // Parse tags
    std::string tags_str = regTags_->text().toStdString();
    if (!tags_str.empty()) {
        body << ",\"tags\":{";
        std::istringstream ss(tags_str);
        std::string tok;
        bool first = true;
        while (std::getline(ss, tok, ',')) {
            auto eq = tok.find('=');
            if (eq == std::string::npos) continue;
            std::string k = tok.substr(0, eq);
            std::string v = tok.substr(eq + 1);
            // trim
            while (!k.empty() && k.front() == ' ') k.erase(k.begin());
            while (!k.empty() && k.back() == ' ') k.pop_back();
            while (!v.empty() && v.front() == ' ') v.erase(v.begin());
            while (!v.empty() && v.back() == ' ') v.pop_back();
            if (!first) body << ',';
            first = false;
            body << '"' << json_escape(k) << "\":\"" << json_escape(v) << '"';
        }
        body << "}";
    }
    body << "}";

    auto resp = httpPost("/models", body.str());
    if (resp.empty()) {
        setStatusMessage("Connection failed", true);
        return;
    }
    if (resp.find("\"error\"") != std::string::npos) {
        QMessageBox::warning(this, "Registration Failed", QString::fromStdString(resp));
    } else {
        setStatusMessage("Registered: " + QString::fromStdString(name));
        regName_->clear();
        onRefreshModels();
    }
}

void MnsManagerGUI::onSetTraining() {
    if (selectedModel_.isEmpty()) {
        QMessageBox::warning(this, "No Model Selected", "Select a model from the list first.");
        return;
    }
    std::string run_id = actionRunId_->text().toStdString();
    if (run_id.empty()) {
        QMessageBox::warning(this, "Missing Run ID", "Enter a run ID.");
        return;
    }

    std::string body = "{\"state\":\"training\",\"run_id\":\"" + json_escape(run_id) + "\"}";
    auto resp = httpPut("/models/" + selectedModel_.toStdString() + "/state", body);
    if (resp.empty()) { setStatusMessage("Connection failed", true); return; }
    if (resp.find("\"error\"") != std::string::npos) {
        QMessageBox::warning(this, "State Transition Failed", QString::fromStdString(resp));
    } else {
        setStatusMessage(selectedModel_ + " -> training");
        showModelDetail(resp);
        onRefreshModels();
    }
}

void MnsManagerGUI::onSetCandidate() {
    if (selectedModel_.isEmpty()) {
        QMessageBox::warning(this, "No Model Selected", "Select a model from the list first.");
        return;
    }
    std::string run_id = actionRunId_->text().toStdString();
    std::string art_path = actionArtifactPath_->text().toStdString();

    std::ostringstream body;
    body << "{\"state\":\"candidate\"";
    if (!run_id.empty())
        body << ",\"run_id\":\"" << json_escape(run_id) << "\"";
    if (!art_path.empty()) {
        body << ",\"artifact\":{\"path\":\"" << json_escape(art_path)
             << "\",\"host\":\"\",\"checksum\":\"\",\"format\":\"adai-native\"}";
    }
    body << "}";

    auto resp = httpPut("/models/" + selectedModel_.toStdString() + "/state", body.str());
    if (resp.empty()) { setStatusMessage("Connection failed", true); return; }
    if (resp.find("\"error\"") != std::string::npos) {
        QMessageBox::warning(this, "State Transition Failed", QString::fromStdString(resp));
    } else {
        setStatusMessage(selectedModel_ + " -> candidate");
        showModelDetail(resp);
        onRefreshModels();
    }
}

void MnsManagerGUI::onPromote() {
    std::string role = actionPromoteRole_->text().toStdString();
    std::string model = actionPromoteModel_->text().toStdString();
    if (role.empty() || model.empty()) {
        QMessageBox::warning(this, "Missing Fields", "Role and model name are required.");
        return;
    }

    std::string body = "{\"model_name\":\"" + json_escape(model) + "\"}";
    auto resp = httpPut("/roles/" + role + "/production", body);
    if (resp.empty()) { setStatusMessage("Connection failed", true); return; }
    if (resp.find("\"error\"") != std::string::npos) {
        QMessageBox::warning(this, "Promotion Failed", QString::fromStdString(resp));
    } else {
        setStatusMessage(QString::fromStdString(model) + " promoted to " + QString::fromStdString(role));
        onRefreshModels();
        onRefreshRoles();
    }
}

void MnsManagerGUI::onRetire() {
    if (selectedModel_.isEmpty()) {
        QMessageBox::warning(this, "No Model Selected", "Select a model from the list first.");
        return;
    }
    auto answer = QMessageBox::question(this, "Confirm Retire",
        "Retire model \"" + selectedModel_ + "\"?\nThis cannot be undone without re-promoting.");
    if (answer != QMessageBox::Yes) return;

    std::string body = "{\"state\":\"retired\"}";
    auto resp = httpPut("/models/" + selectedModel_.toStdString() + "/state", body);
    if (resp.empty()) { setStatusMessage("Connection failed", true); return; }
    if (resp.find("\"error\"") != std::string::npos) {
        QMessageBox::warning(this, "Retire Failed", QString::fromStdString(resp));
    } else {
        setStatusMessage(selectedModel_ + " retired");
        showModelDetail(resp);
        onRefreshModels();
        onRefreshRoles();
    }
}

void MnsManagerGUI::onDelete() {
    if (selectedModel_.isEmpty()) {
        QMessageBox::warning(this, "No Model Selected", "Select a model from the list first.");
        return;
    }
    auto answer = QMessageBox::question(this, "Confirm Delete",
        "Permanently delete model \"" + selectedModel_ + "\"?\n"
        "Only allowed for initializing or retired models.\n"
        "Weight files are NOT deleted from disk.");
    if (answer != QMessageBox::Yes) return;

    auto resp = httpDelete("/models/" + selectedModel_.toStdString());
    if (resp.empty()) { setStatusMessage("Connection failed", true); return; }
    if (resp.find("\"error\"") != std::string::npos) {
        QMessageBox::warning(this, "Delete Failed", QString::fromStdString(resp));
    } else {
        setStatusMessage(selectedModel_ + " deleted");
        selectedModel_.clear();
        detailView_->clear();
        onRefreshModels();
        onRefreshRoles();
    }
}

// ============================================================================
// Table population
// ============================================================================

void MnsManagerGUI::populateModelsTable(const std::string& json) {
    auto objects = json_array_objects(json, "models");
    modelsTable_->setRowCount(static_cast<int>(objects.size()));

    for (int i = 0; i < static_cast<int>(objects.size()); ++i) {
        const auto& obj = objects[i];
        modelsTable_->setItem(i, 0, new QTableWidgetItem(QString::fromStdString(json_value(obj, "model_name"))));
        modelsTable_->setItem(i, 1, new QTableWidgetItem(QString::fromStdString(json_value(obj, "role"))));

        auto stateItem = new QTableWidgetItem(QString::fromStdString(json_value(obj, "state")));
        std::string state = json_value(obj, "state");
        if (state == "production")   stateItem->setForeground(QColor("#2e7d32"));
        else if (state == "training") stateItem->setForeground(QColor("#1565c0"));
        else if (state == "candidate") stateItem->setForeground(QColor("#e65100"));
        else if (state == "retired")  stateItem->setForeground(QColor("#9e9e9e"));
        modelsTable_->setItem(i, 2, stateItem);

        std::string id = json_value(obj, "model_id");
        if (id.size() > 8) id = id.substr(0, 8) + "...";
        modelsTable_->setItem(i, 3, new QTableWidgetItem(QString::fromStdString(id)));
        modelsTable_->setItem(i, 4, new QTableWidgetItem(QString::fromStdString(json_value(obj, "updated_utc"))));
    }
}

void MnsManagerGUI::populateRolesTable(const std::string& json) {
    auto objects = json_array_objects(json, "roles");
    rolesTable_->setRowCount(static_cast<int>(objects.size()));

    for (int i = 0; i < static_cast<int>(objects.size()); ++i) {
        const auto& obj = objects[i];
        rolesTable_->setItem(i, 0, new QTableWidgetItem(QString::fromStdString(json_value(obj, "role"))));
        std::string prod = json_value(obj, "production_model");
        if (prod.empty()) prod = "(none)";
        rolesTable_->setItem(i, 1, new QTableWidgetItem(QString::fromStdString(prod)));
    }
}

void MnsManagerGUI::showModelDetail(const std::string& json) {
    detailView_->setPlainText(QString::fromStdString(json_pretty(json)));
}

void MnsManagerGUI::setStatusMessage(const QString& msg, bool error) {
    statusLabel_->setText(msg);
    if (error) {
        statusLabel_->setStyleSheet("color: #c62828; font-weight: bold;");
    } else {
        statusLabel_->setStyleSheet("color: #2e7d32; font-weight: bold;");
    }
}

// ============================================================================
// Stylesheet
// ============================================================================

void MnsManagerGUI::applyStylesheet() {
    setStyleSheet(R"(
        QMainWindow {
            background-color: #f5f5f5;
        }
        QTableWidget {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 12px;
            gridline-color: #eee;
        }
        QTableWidget::item:selected {
            background-color: #e3f2fd;
            color: #1565c0;
        }
        QHeaderView::section {
            background-color: #fafafa;
            border: 1px solid #ddd;
            padding: 6px;
            font-weight: bold;
            font-size: 12px;
        }
        QTextEdit {
            background-color: #fafafa;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 8px;
            font-family: 'Monospace', 'Courier New', monospace;
            font-size: 11px;
        }
        QLineEdit {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 6px 10px;
            font-size: 12px;
        }
        QLineEdit:focus {
            border: 1px solid #2196F3;
        }
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 16px;
            font-size: 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1976D2;
        }
        QPushButton:pressed {
            background-color: #0D47A1;
        }
        QPushButton#promoteBtn {
            background-color: #2e7d32;
        }
        QPushButton#promoteBtn:hover {
            background-color: #1b5e20;
        }
        QPushButton#dangerBtn {
            background-color: #c62828;
        }
        QPushButton#dangerBtn:hover {
            background-color: #b71c1c;
        }
        QPushButton#actionBtn {
            background-color: #e65100;
        }
        QPushButton#actionBtn:hover {
            background-color: #bf360c;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ddd;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 14px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QComboBox {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 4px 8px;
            min-height: 24px;
        }
        QSpinBox {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 4px;
            min-height: 24px;
        }
        QTabWidget::pane {
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
        }
        QTabBar::tab {
            background: #fafafa;
            border: 1px solid #ddd;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 8px 20px;
            font-size: 12px;
        }
        QTabBar::tab:selected {
            background: white;
            font-weight: bold;
        }
        QTabBar::tab:hover {
            background: #e3f2fd;
        }
        QLabel {
            font-size: 12px;
        }
    )");
}
