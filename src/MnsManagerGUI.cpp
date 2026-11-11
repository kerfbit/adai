// @adai-status: beta        (widget tests: mnsManagerGuiWidgetTests)
// @adai-version: 0.11.0
// @adai-reviewed: 2026-11-11

#include "MnsManagerGUI.hpp"
#include <httplib.h>
#include <QApplication>
#include <QDoubleValidator>
#include <QFileDialog>
#include <QIntValidator>
#include <QMessageBox>
#include <QMetaObject>
#include <QScrollArea>
#include <QSignalBlocker>
#include <fstream>
#include <utility>
#include <vector>
#include "Config.hpp"
#include "MnsCliCommands.hpp"
#include "MnsJsonHelpers.hpp"

using mns_gui::build_set_candidate_body;
using mns_gui::build_set_training_body;
using mns_gui::json_array_objects;
using mns_gui::json_pretty;
using mns_gui::json_value;

// ============================================================================
// Construction
// ============================================================================

MnsManagerGUI::MnsManagerGUI(const std::string& server_url, QWidget* parent)
    : QMainWindow(parent), server_url_(server_url) {  // before setupUI(), which shows it
    setWindowTitle("ADAI Model Name Service Manager");
    resize(1100, 760);
    setupUI();
    applyStylesheet();
    onRegisterKindChanged(regKind_->currentIndex());
    if (setServer(server_url)) {
        connectToServer();
    }
}

MnsManagerGUI::~MnsManagerGUI() {
    // TD-455: a request may be waiting on the server (up to the 10 s read timeout): stop it so
    // closing the window doesn't wait. Its result is dropped with the window.
    {
        std::lock_guard<std::mutex> lock(client_mutex_);
        closing_ = true;
        if (active_client_) {
            active_client_->stop();
        }
    }
    if (worker_.joinable()) {
        worker_.join();
    }
}

// ============================================================================
// Server connection and requests (TD-455)
// ============================================================================

bool MnsManagerGUI::setServer(const std::string& url) {
    server_url_ = url;
    server_usable_ = false;
    const adai::ParsedUrl parsed = adai::parse_url(url);
    if (!parsed.error.empty()) {
        server_problem_ = parsed.error;
    } else if (parsed.https) {
        // TD-452: httplib is built without TLS here, so https:// used to be sent as plain HTTP.
        server_problem_ =
            "https:// isn't supported by mns_manager_gui; use the server's http:// address "
            "(mns_cli supports https://)";
    } else {
        host_ = parsed.host;
        port_ = parsed.port;
        base_path_ = parsed.base_path;
        server_usable_ = true;
        server_problem_.clear();
    }
    if (!server_usable_) {
        setStatusMessage("Disconnected: " + QString::fromStdString(server_problem_), true);
    }
    return server_usable_;
}

void MnsManagerGUI::connectToServer() {
    const std::string models = modelsPath();
    runRequest("Connecting...", [this, models]() -> std::function<void()> {
        const HttpReply health = http("GET", "/health");
        if (!health.ok()) {
            return [this, health] {
                const std::string why =
                    health.reached ? "HTTP " + std::to_string(health.status) : health.error;
                setStatusMessage("Disconnected (" + QString::fromStdString(why) + ")", true);
            };
        }
        const HttpReply models_reply = http("GET", models);
        const HttpReply roles_reply = http("GET", "/roles");
        return [this, models_reply, roles_reply] {
            setStatusMessage("Connected");
            if (models_reply.ok())
                populateModelsTable(models_reply.body);
            if (roles_reply.ok())
                populateRolesTable(roles_reply.body);
        };
    });
}

MnsManagerGUI::HttpReply MnsManagerGUI::http(const std::string& method, const std::string& path,
                                             const std::string& body) {
    HttpReply reply;
    if (!server_usable_) {
        reply.error = server_problem_;
        return reply;
    }
    auto client = std::make_shared<httplib::Client>(host_, port_);
    client->set_connection_timeout(5, 0);
    client->set_read_timeout(10, 0);
    client->set_write_timeout(10, 0);
    {
        std::lock_guard<std::mutex> lock(client_mutex_);
        if (closing_) {
            reply.error = "the window is closing";
            return reply;
        }
        active_client_ = client;
    }
    const std::string target = base_path_ + path;
    httplib::Result res = method == "GET"    ? client->Get(target)
                          : method == "POST" ? client->Post(target, body, "application/json")
                          : method == "PUT"  ? client->Put(target, body, "application/json")
                                             : client->Delete(target);
    {
        std::lock_guard<std::mutex> lock(client_mutex_);
        active_client_.reset();
    }
    if (!res) {
        reply.error = httplib::to_string(res.error());
        return reply;
    }
    reply.reached = true;
    reply.status = res->status;
    reply.body = res->body;
    return reply;
}

void MnsManagerGUI::runRequest(const QString& busyText,
                               std::function<std::function<void()>()> work) {
    if (worker_.joinable()) {
        worker_.join();  // the previous request has finished: its result already ran setBusy(false)
    }
    setBusy(true, busyText);
    worker_ = std::thread([this, work = std::move(work)] {
        std::function<void()> done;
        try {
            done = work();
        } catch (const std::exception& e) {
            const QString error = e.what();
            done = [this, error] { setStatusMessage("Error: " + error, true); };
        }
        QMetaObject::invokeMethod(
            this,
            [this, done = std::move(done)] {
                setBusy(false);
                if (done) {
                    done();
                }
            },
            Qt::QueuedConnection);
    });
}

void MnsManagerGUI::setBusy(bool busy, const QString& busyText) {
    content_->setEnabled(!busy);
    urlField_->setEnabled(!busy);
    connectBtn_->setEnabled(!busy);
    refreshBtn_->setEnabled(!busy);
    if (busy) {
        statusLabel_->setText(busyText);
        statusLabel_->setStyleSheet("color: #555; font-weight: bold;");
    } else {
        setStatusMessage(statusText_, statusError_);
    }
}

void MnsManagerGUI::reportFailure(const QString& title, const HttpReply& reply) {
    if (!reply.reached) {
        setStatusMessage("Connection failed (" + QString::fromStdString(reply.error) + ")", true);
        return;
    }
    // TD-455: judged by the HTTP status; the server's own error explains it when it gives one.
    const std::string error = json_value(reply.body, "error");
    const std::string status = "HTTP " + std::to_string(reply.status);
    const std::string text = error.empty() ? status + (reply.body.empty() ? "" : ": " + reply.body)
                                           : error + " (" + status + ")";
    setStatusMessage(title, true);
    QMessageBox::warning(this, title, QString::fromStdString(text));
}

std::string MnsManagerGUI::modelsPath() const {
    std::string path = "/models";
    char sep = '?';
    if (stateFilter_->currentIndex() > 0) {
        path += sep + std::string("state=") +
                adai::url_encode(stateFilter_->currentText().toStdString());
        sep = '&';
    }
    if (roleFilter_->currentIndex() > 0) {
        path += sep + std::string("role=") +
                adai::url_encode(roleFilter_->itemText(roleFilter_->currentIndex()).toStdString());
    }
    return path;
}

std::string MnsManagerGUI::modelPath(const QString& name) {
    return "/models/" + adai::url_encode(name.toStdString());
}

// ============================================================================
// UI setup
// ============================================================================

void MnsManagerGUI::setupUI() {
    QWidget* central = new QWidget(this);
    setCentralWidget(central);
    QVBoxLayout* root = new QVBoxLayout(central);

    root->addWidget(createToolbar());

    // Main content: tabs on the left, detail on the right. Disabled while a request runs.
    QSplitter* splitter = new QSplitter(Qt::Horizontal);
    content_ = splitter;
    content_->setObjectName("content");

    tabWidget_ = new QTabWidget();
    tabWidget_->addTab(createModelsTab(), "Models");
    tabWidget_->addTab(createRolesTab(), "Roles");
    tabWidget_->addTab(createRegisterTab(), "Register");
    splitter->addWidget(tabWidget_);
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
    urlField_->setObjectName("serverUrl");
    urlField_->setMinimumWidth(250);

    connectBtn_ = new QPushButton("Connect");
    connectBtn_->setObjectName("connectButton");
    connect(connectBtn_, &QPushButton::clicked, this, &MnsManagerGUI::onConnect);

    refreshBtn_ = new QPushButton("Refresh");
    refreshBtn_->setObjectName("refreshButton");
    connect(refreshBtn_, &QPushButton::clicked, this, [this]() {
        // TD-568: also the selected model's record, as selecting its row again doesn't reload it.
        const std::string models = modelsPath();
        const std::string record = selectedModel_.isEmpty() ? "" : modelPath(selectedModel_);
        runRequest("Refreshing...", [this, models, record]() -> std::function<void()> {
            const HttpReply models_reply = http("GET", models);
            const HttpReply roles_reply = http("GET", "/roles");
            const HttpReply record_reply = record.empty() ? HttpReply{} : http("GET", record);
            return [this, models_reply, roles_reply, record_reply] {
                if (!models_reply.ok()) {
                    reportFailure("Refresh Failed", models_reply);
                    return;
                }
                populateModelsTable(models_reply.body);
                if (roles_reply.ok())
                    populateRolesTable(roles_reply.body);
                if (record_reply.ok())
                    showModelDetail(record_reply.body);
            };
        });
    });

    statusLabel_ = new QLabel("");
    statusLabel_->setObjectName("statusLabel");
    statusLabel_->setMinimumWidth(200);

    lay->addWidget(lbl);
    lay->addWidget(urlField_);
    lay->addWidget(connectBtn_);
    lay->addWidget(refreshBtn_);
    lay->addStretch();
    lay->addWidget(statusLabel_);
    return w;
}

QWidget* MnsManagerGUI::createModelsTab() {
    QWidget* w = new QWidget();
    QVBoxLayout* lay = new QVBoxLayout(w);

    QHBoxLayout* filterLay = new QHBoxLayout();
    filterLay->addWidget(new QLabel("State:"));
    stateFilter_ = new QComboBox();
    stateFilter_->setObjectName("stateFilter");
    stateFilter_->addItems(
        {"(all)", "initializing", "training", "candidate", "production", "retired"});
    connect(stateFilter_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { onRefreshModels(); });
    filterLay->addWidget(stateFilter_);

    // TD-455: filled with the server's roles (populateRolesTable()); a typed role is added with
    // Enter.
    filterLay->addWidget(new QLabel("Role:"));
    roleFilter_ = new QComboBox();
    roleFilter_->setObjectName("roleFilter");
    roleFilter_->setEditable(true);
    roleFilter_->addItem("(all)");
    connect(roleFilter_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { onRefreshModels(); });
    filterLay->addWidget(roleFilter_);
    filterLay->addStretch();
    lay->addLayout(filterLay);

    modelsTable_ = new QTableWidget(0, 5);
    modelsTable_->setObjectName("modelsTable");
    modelsTable_->setHorizontalHeaderLabels({"Name", "Role", "State", "Model ID", "Updated"});
    modelsTable_->horizontalHeader()->setStretchLastSection(true);
    modelsTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    modelsTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    modelsTable_->setSelectionMode(QAbstractItemView::SingleSelection);
    modelsTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    modelsTable_->setAlternatingRowColors(true);
    // TD-568: the record loads when the current row changes, by mouse or keyboard.
    connect(modelsTable_, &QTableWidget::currentCellChanged, this,
            [this](int row, int, int previous, int) {
                if (row >= 0 && row != previous)
                    onModelSelected(row, 0);
            });
    lay->addWidget(modelsTable_);

    return w;
}

QWidget* MnsManagerGUI::createRolesTab() {
    QWidget* w = new QWidget();
    QVBoxLayout* lay = new QVBoxLayout(w);

    rolesTable_ = new QTableWidget(0, 2);
    rolesTable_->setObjectName("rolesTable");
    rolesTable_->setHorizontalHeaderLabels({"Role", "Production Model"});
    rolesTable_->horizontalHeader()->setStretchLastSection(true);
    rolesTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    rolesTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    rolesTable_->setSelectionMode(QAbstractItemView::SingleSelection);
    rolesTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    rolesTable_->setAlternatingRowColors(true);
    connect(rolesTable_, &QTableWidget::currentCellChanged, this,
            [this](int row, int, int previous, int) {
                if (row >= 0 && row != previous)
                    onRoleSelected(row, 0);
            });
    lay->addWidget(rolesTable_);

    return w;
}

QWidget* MnsManagerGUI::createRegisterTab() {
    // TD-452: every kind, with run group, linked encoder/decoder and SIGReg settings. TD-455: the
    // architecture starts blank and is required (it can't be changed after registering); "Load from
    // config..." fills it from a trainer config, as mns_cli's --from-config does.
    QWidget* form = new QWidget();
    QVBoxLayout* lay = new QVBoxLayout(form);

    auto field = [](const char* name, const QString& placeholder) {
        QLineEdit* edit = new QLineEdit();
        edit->setObjectName(name);
        edit->setPlaceholderText(placeholder);
        return edit;
    };
    auto count_field = [&field](const char* name, int min) {
        QLineEdit* edit = field(name, "required");
        edit->setValidator(new QIntValidator(min, 1000000, edit));
        return edit;
    };
    auto row = [](QBoxLayout* into, const QString& label, QWidget* control) {
        QWidget* holder = new QWidget();
        QHBoxLayout* h = new QHBoxLayout(holder);
        h->setContentsMargins(0, 0, 0, 0);
        QLabel* lbl = new QLabel(label);
        lbl->setMinimumWidth(140);
        h->addWidget(lbl);
        h->addWidget(control);
        into->addWidget(holder);
        return holder;
    };

    regName_ = field("regName", "e.g. adai-chatbot-v3");
    row(lay, "Model Name:", regName_);
    regRole_ = field("regRole", "e.g. chatbot");
    row(lay, "Role:", regRole_);
    regKind_ = new QComboBox();
    regKind_->setObjectName("regKind");
    regKind_->addItems({"chatbot", "encoder", "decoder", "world_model"});
    connect(regKind_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &MnsManagerGUI::onRegisterKindChanged);
    row(lay, "Kind:", regKind_);
    regRunGroup_ = field("regRunGroup", "optional: the dataset registry group");
    row(lay, "Run group:", regRunGroup_);

    QGroupBox* archGroup = new QGroupBox("Architecture (can't be changed after registering)");
    QVBoxLayout* archLay = new QVBoxLayout(archGroup);
    regDModel_ = count_field("regDModel", 1);
    row(archLay, "d_model:", regDModel_);
    regHeads_ = count_field("regHeads", 1);
    row(archLay, "num_heads:", regHeads_);
    regDff_ = count_field("regDff", 1);
    row(archLay, "d_ff:", regDff_);
    regMaxSeq_ = count_field("regMaxSeq", 1);
    row(archLay, "max_seq_length:", regMaxSeq_);
    regLayers_ = count_field("regLayers", 1);
    regLayersRow_ = row(archLay, "Layers:", regLayers_);
    QWidget* chatbotLayers = new QWidget();
    QVBoxLayout* chatbotLayersLay = new QVBoxLayout(chatbotLayers);
    chatbotLayersLay->setContentsMargins(0, 0, 0, 0);
    regEncLayers_ = count_field("regEncLayers", 0);
    row(chatbotLayersLay, "Encoder layers:", regEncLayers_);
    regDecLayers_ = count_field("regDecLayers", 0);
    row(chatbotLayersLay, "Decoder layers:", regDecLayers_);
    archLay->addWidget(chatbotLayers);
    regChatbotLayersRow_ = chatbotLayers;
    QPushButton* loadConfigBtn = new QPushButton("Load from config...");
    loadConfigBtn->setObjectName("loadConfigButton");
    connect(loadConfigBtn, &QPushButton::clicked, this, &MnsManagerGUI::onLoadRegisterConfig);
    archLay->addWidget(loadConfigBtn);
    lay->addWidget(archGroup);

    QGroupBox* linkGroup = new QGroupBox("Or link registered encoder and decoder (chatbot only)");
    QVBoxLayout* linkLay = new QVBoxLayout(linkGroup);
    regEncoderName_ = field("regEncoderName", "encoder model name; leave the architecture blank");
    row(linkLay, "Encoder:", regEncoderName_);
    regDecoderName_ = field("regDecoderName", "decoder model name");
    row(linkLay, "Decoder:", regDecoderName_);
    lay->addWidget(linkGroup);
    regLinkGroup_ = linkGroup;

    QGroupBox* sigregGroup = new QGroupBox("SIGReg (world model only)");
    QVBoxLayout* sigregLay = new QVBoxLayout(sigregGroup);
    regSigregLambda_ = field("regSigregLambda", "default 1.0");
    regSigregLambda_->setValidator(new QDoubleValidator(0.0, 1e9, 6, regSigregLambda_));
    row(sigregLay, "Lambda:", regSigregLambda_);
    regSigregSketches_ = field("regSigregSketches", "default 64");
    regSigregSketches_->setValidator(new QIntValidator(1, 1000000, regSigregSketches_));
    row(sigregLay, "Sketches:", regSigregSketches_);
    lay->addWidget(sigregGroup);
    regSigregGroup_ = sigregGroup;

    regTags_ = field("regTags", "key1=val1, key2=val2");
    row(lay, "Tags:", regTags_);

    QPushButton* regBtn = new QPushButton("Register Model");
    regBtn->setObjectName("registerButton");
    regBtn->setMinimumHeight(36);
    connect(regBtn, &QPushButton::clicked, this, &MnsManagerGUI::onRegisterModel);
    lay->addWidget(regBtn);
    lay->addStretch();

    QScrollArea* scroll = new QScrollArea();
    scroll->setWidgetResizable(true);
    scroll->setWidget(form);
    return scroll;
}

QWidget* MnsManagerGUI::createDetailPanel() {
    QWidget* w = new QWidget();
    QVBoxLayout* lay = new QVBoxLayout(w);

    QLabel* detailLabel = new QLabel("Model Detail");
    QFont f = detailLabel->font();
    f.setPointSize(12);
    f.setBold(true);
    detailLabel->setFont(f);
    lay->addWidget(detailLabel);

    detailView_ = new QTextEdit();
    detailView_->setObjectName("detailView");
    detailView_->setReadOnly(true);
    detailView_->setFont(QFont("Monospace", 10));
    lay->addWidget(detailView_);

    QGroupBox* actGroup = new QGroupBox("Actions");
    QVBoxLayout* actLay = new QVBoxLayout(actGroup);

    // Set Training. TD-454: MNS allocates the run ID (run-NN); the checkbox asks for the next run
    // instead of continuing the current one.
    QHBoxLayout* trainRow = new QHBoxLayout();
    actionNewRun_ = new QCheckBox("Start a new run");
    actionNewRun_->setObjectName("newRunCheck");
    QPushButton* trainBtn = new QPushButton("Set Training");
    trainBtn->setObjectName("actionBtn");
    connect(trainBtn, &QPushButton::clicked, this, &MnsManagerGUI::onSetTraining);
    trainRow->addWidget(actionNewRun_);
    trainRow->addStretch();
    trainRow->addWidget(trainBtn);
    actLay->addLayout(trainRow);

    // Set Candidate. Leaving training needs the active run's ID, so it's filled in from the
    // selected record and from Set Training's reply.
    QHBoxLayout* candRow = new QHBoxLayout();
    actionRunId_ = new QLineEdit();
    actionRunId_->setObjectName("runIdField");
    actionRunId_->setPlaceholderText("run-NN");
    actionArtifactPath_ = new QLineEdit();
    actionArtifactPath_->setObjectName("artifactField");
    actionArtifactPath_->setPlaceholderText("/path/to/weights.bin");
    QPushButton* candBtn = new QPushButton("Set Candidate");
    candBtn->setObjectName("actionBtn");
    connect(candBtn, &QPushButton::clicked, this, &MnsManagerGUI::onSetCandidate);
    candRow->addWidget(new QLabel("Run ID:"));
    candRow->addWidget(actionRunId_);
    candRow->addWidget(new QLabel("Artifact:"));
    candRow->addWidget(actionArtifactPath_);
    candRow->addWidget(candBtn);
    actLay->addLayout(candRow);

    QHBoxLayout* promRow = new QHBoxLayout();
    actionPromoteRole_ = new QLineEdit();
    actionPromoteRole_->setObjectName("promoteRoleField");
    actionPromoteRole_->setPlaceholderText("role");
    actionPromoteModel_ = new QLineEdit();
    actionPromoteModel_->setObjectName("promoteModelField");
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

void MnsManagerGUI::onConnect() {
    if (setServer(urlField_->text().trimmed().toStdString())) {
        connectToServer();
    }
}

void MnsManagerGUI::onRefreshModels() {
    const std::string path = modelsPath();
    runRequest("Loading models...", [this, path]() -> std::function<void()> {
        const HttpReply reply = http("GET", path);
        return [this, reply] {
            if (!reply.ok()) {
                reportFailure("Refresh Failed", reply);
                return;
            }
            populateModelsTable(reply.body);
        };
    });
}

void MnsManagerGUI::onModelSelected(int row, int /*col*/) {
    auto item = modelsTable_->item(row, 0);
    if (!item)
        return;
    selectedModel_ = item->text();

    // Pre-fill action fields
    actionPromoteModel_->setText(selectedModel_);
    auto roleItem = modelsTable_->item(row, 1);
    if (roleItem)
        actionPromoteRole_->setText(roleItem->text());

    const std::string path = modelPath(selectedModel_);
    runRequest("Loading " + selectedModel_ + "...", [this, path]() -> std::function<void()> {
        const HttpReply reply = http("GET", path);
        return [this, reply] {
            if (!reply.ok()) {
                reportFailure("Load Failed", reply);
                return;
            }
            showModelDetail(reply.body);
            actionRunId_->setText(QString::fromStdString(json_value(reply.body, "run_id")));
        };
    });
}

void MnsManagerGUI::onRoleSelected(int row, int /*col*/) {
    auto roleItem = rolesTable_->item(row, 0);
    auto modelItem = rolesTable_->item(row, 1);
    if (!roleItem)
        return;

    actionPromoteRole_->setText(roleItem->text());
    if (!modelItem || modelItem->text().isEmpty() || modelItem->text() == "(none)")
        return;
    selectedModel_ = modelItem->text();
    actionPromoteModel_->setText(selectedModel_);
    const std::string path = modelPath(selectedModel_);
    runRequest("Loading " + selectedModel_ + "...", [this, path]() -> std::function<void()> {
        const HttpReply reply = http("GET", path);
        return [this, reply] {
            if (!reply.ok()) {
                reportFailure("Load Failed", reply);
                return;
            }
            showModelDetail(reply.body);
            actionRunId_->setText(QString::fromStdString(json_value(reply.body, "run_id")));
        };
    });
}

void MnsManagerGUI::onRegisterKindChanged(int /*index*/) {
    const QString kind = regKind_->currentText();
    const bool chatbot = kind == "chatbot";
    regChatbotLayersRow_->setVisible(chatbot);
    regLinkGroup_->setVisible(chatbot);
    regLayersRow_->setVisible(!chatbot);
    regSigregGroup_->setVisible(kind == "world_model");
}

void MnsManagerGUI::onLoadRegisterConfig() {
    const QString path =
        QFileDialog::getOpenFileName(this, "Load Architecture From a Trainer Config", QString(),
                                     "Config files (*.conf);;All Files (*)");
    if (path.isEmpty())
        return;
    if (!std::ifstream(path.toStdString())) {
        QMessageBox::warning(this, "Load Failed", "Can't read " + path);
        return;
    }
    const adai::ServiceConfig cfg = adai::ConfigLoader::load(path.toStdString());
    const bool decoder = regKind_->currentText() == "decoder";
    regDModel_->setText(QString::number(cfg.d_model));
    regHeads_->setText(QString::number(cfg.num_heads));
    regDff_->setText(QString::number(cfg.d_ff));
    regMaxSeq_->setText(QString::number(cfg.max_seq_length));
    regEncLayers_->setText(QString::number(cfg.num_encoder_layers));
    regDecLayers_->setText(QString::number(cfg.num_decoder_layers));
    regLayers_->setText(QString::number(decoder ? cfg.num_decoder_layers : cfg.num_encoder_layers));
    setStatusMessage("Architecture loaded from " + path + "; check it before registering");
}

void MnsManagerGUI::onRegisterModel() {
    const std::string name = regName_->text().trimmed().toStdString();
    const std::string role = regRole_->text().trimmed().toStdString();
    if (name.empty() || role.empty()) {
        QMessageBox::warning(this, "Missing Fields", "Model name and role are required.");
        return;
    }
    // TD-452: the same rules and body as `mns_cli register`, from the fields shown for this kind.
    const std::string kind = regKind_->currentText().toStdString();
    std::vector<std::string> args{name, role, "--kind", kind};
    auto add = [&args](const char* flag, const QLineEdit* edit) {
        const std::string value = edit->text().trimmed().toStdString();
        if (!value.empty()) {
            args.push_back(flag);
            args.push_back(value);
        }
    };
    add("--run-group", regRunGroup_);
    add("--d-model", regDModel_);
    add("--num-heads", regHeads_);
    add("--d-ff", regDff_);
    add("--max-seq-length", regMaxSeq_);
    if (kind == "chatbot") {
        add("--encoder-layers", regEncLayers_);
        add("--decoder-layers", regDecLayers_);
        add("--encoder", regEncoderName_);
        add("--decoder", regDecoderName_);
    } else {
        add("--num-layers", regLayers_);
    }
    if (kind == "world_model") {
        add("--sigreg-lambda", regSigregLambda_);
        add("--sigreg-num-sketches", regSigregSketches_);
    }
    for (const auto& [k, v] : mns_gui::parse_tags(regTags_->text().toStdString())) {
        args.push_back("--tag");
        args.push_back(k + "=" + v);
    }
    const adai::MnsCliRequest request = adai::build_register_request(args, adai::ServiceConfig{});
    if (request.error) {
        QMessageBox::warning(this, "Invalid Registration",
                             QString::fromStdString(request.error_message));
        return;
    }
    const QString summary =
        request.notice.empty()
            ? QString("Registering %1 (chatbot) linked to encoder %2 and decoder %3.")
                  .arg(QString::fromStdString(name), regEncoderName_->text().trimmed(),
                       regDecoderName_->text().trimmed())
            : QString::fromStdString(request.notice) + ".";
    if (QMessageBox::question(this, "Confirm Registration",
                              summary + "\n\nA model's architecture can't be changed after it is "
                                        "registered. Register it?") != QMessageBox::Yes)
        return;

    const std::string body = request.body;
    const std::string models = modelsPath();
    runRequest("Registering...", [this, body, models]() -> std::function<void()> {
        const HttpReply reply = http("POST", "/models", body);
        const HttpReply list = reply.ok() ? http("GET", models) : HttpReply{};
        return [this, reply, list] {
            if (!reply.ok()) {
                reportFailure("Registration Failed", reply);
                return;
            }
            setStatusMessage("Registered: " + regName_->text().trimmed());
            regName_->clear();
            if (list.ok())
                populateModelsTable(list.body);
        };
    });
}

void MnsManagerGUI::onSetTraining() {
    if (selectedModel_.isEmpty()) {
        QMessageBox::warning(this, "No Model Selected", "Select a model from the list first.");
        return;
    }
    const std::string path = modelPath(selectedModel_) + "/state";
    const std::string body = build_set_training_body(actionNewRun_->isChecked());
    const std::string models = modelsPath();
    runRequest("Updating...", [this, path, body, models]() -> std::function<void()> {
        const HttpReply reply = http("PUT", path, body);
        const HttpReply list = reply.ok() ? http("GET", models) : HttpReply{};
        return [this, reply, list] {
            if (!reply.ok()) {
                reportFailure("State Transition Failed", reply);
                return;
            }
            const std::string run_id = json_value(reply.body, "run_id");
            actionRunId_->setText(QString::fromStdString(run_id));
            actionNewRun_->setChecked(false);
            setStatusMessage(selectedModel_ + " -> training (" + QString::fromStdString(run_id) +
                             ")");
            showModelDetail(reply.body);
            if (list.ok())
                populateModelsTable(list.body);
        };
    });
}

void MnsManagerGUI::onSetCandidate() {
    if (selectedModel_.isEmpty()) {
        QMessageBox::warning(this, "No Model Selected", "Select a model from the list first.");
        return;
    }
    const std::string path = modelPath(selectedModel_) + "/state";
    const std::string body =
        build_set_candidate_body(actionRunId_->text().trimmed().toStdString(),
                                 actionArtifactPath_->text().trimmed().toStdString());
    const std::string models = modelsPath();
    runRequest("Updating...", [this, path, body, models]() -> std::function<void()> {
        const HttpReply reply = http("PUT", path, body);
        const HttpReply list = reply.ok() ? http("GET", models) : HttpReply{};
        return [this, reply, list] {
            if (!reply.ok()) {
                reportFailure("State Transition Failed", reply);
                return;
            }
            setStatusMessage(selectedModel_ + " -> candidate");
            showModelDetail(reply.body);
            if (list.ok())
                populateModelsTable(list.body);
        };
    });
}

void MnsManagerGUI::onPromote() {
    const QString role = actionPromoteRole_->text().trimmed();
    const QString model = actionPromoteModel_->text().trimmed();
    if (role.isEmpty() || model.isEmpty()) {
        QMessageBox::warning(this, "Missing Fields", "Role and model name are required.");
        return;
    }
    const std::string path = "/roles/" + adai::url_encode(role.toStdString()) + "/production";
    const std::string body =
        "{\"model_name\":\"" + mns_gui::json_escape(model.toStdString()) + "\"}";
    const std::string models = modelsPath();
    runRequest("Promoting...", [this, path, body, models, role, model]() -> std::function<void()> {
        const HttpReply reply = http("PUT", path, body);
        const HttpReply list = reply.ok() ? http("GET", models) : HttpReply{};
        const HttpReply roles = reply.ok() ? http("GET", "/roles") : HttpReply{};
        return [this, reply, list, roles, role, model] {
            if (!reply.ok()) {
                reportFailure("Promotion Failed", reply);
                return;
            }
            setStatusMessage(model + " promoted to " + role);
            if (list.ok())
                populateModelsTable(list.body);
            if (roles.ok())
                populateRolesTable(roles.body);
        };
    });
}

void MnsManagerGUI::onRetire() {
    if (selectedModel_.isEmpty()) {
        QMessageBox::warning(this, "No Model Selected", "Select a model from the list first.");
        return;
    }
    auto answer = QMessageBox::question(
        this, "Confirm Retire",
        "Retire model \"" + selectedModel_ + "\"?\nThis cannot be undone without re-promoting.");
    if (answer != QMessageBox::Yes)
        return;

    const std::string path = modelPath(selectedModel_) + "/state";
    const std::string models = modelsPath();
    runRequest("Retiring...", [this, path, models]() -> std::function<void()> {
        const HttpReply reply = http("PUT", path, R"({"state":"retired"})");
        const HttpReply list = reply.ok() ? http("GET", models) : HttpReply{};
        const HttpReply roles = reply.ok() ? http("GET", "/roles") : HttpReply{};
        return [this, reply, list, roles] {
            if (!reply.ok()) {
                reportFailure("Retire Failed", reply);
                return;
            }
            setStatusMessage(selectedModel_ + " retired");
            showModelDetail(reply.body);
            if (list.ok())
                populateModelsTable(list.body);
            if (roles.ok())
                populateRolesTable(roles.body);
        };
    });
}

void MnsManagerGUI::onDelete() {
    if (selectedModel_.isEmpty()) {
        QMessageBox::warning(this, "No Model Selected", "Select a model from the list first.");
        return;
    }
    auto answer = QMessageBox::question(this, "Confirm Delete",
                                        "Permanently delete model \"" + selectedModel_ +
                                            "\"?\n"
                                            "Only allowed for initializing or retired models.\n"
                                            "Weight files are NOT deleted from disk.");
    if (answer != QMessageBox::Yes)
        return;

    const std::string path = modelPath(selectedModel_);
    const std::string models = modelsPath();
    runRequest("Deleting...", [this, path, models]() -> std::function<void()> {
        const HttpReply reply = http("DELETE", path);
        const HttpReply list = reply.ok() ? http("GET", models) : HttpReply{};
        const HttpReply roles = reply.ok() ? http("GET", "/roles") : HttpReply{};
        return [this, reply, list, roles] {
            if (!reply.ok()) {
                reportFailure("Delete Failed", reply);
                return;
            }
            setStatusMessage(selectedModel_ + " deleted");
            selectedModel_.clear();
            detailView_->clear();
            actionRunId_->clear();
            if (list.ok())
                populateModelsTable(list.body);
            if (roles.ok())
                populateRolesTable(roles.body);
        };
    });
}

// ============================================================================
// Table population
// ============================================================================

void MnsManagerGUI::populateModelsTable(const std::string& json) {
    auto objects = json_array_objects(json, "models");
    // TD-568: refilling doesn't count as selecting a row; the selected model stays highlighted
    // (below) rather than whichever model now sits in its old row.
    QSignalBlocker block(modelsTable_);
    modelsTable_->setRowCount(static_cast<int>(objects.size()));
    int selected_row = -1;

    for (int i = 0; i < static_cast<int>(objects.size()); ++i) {
        const auto& obj = objects[i];
        const QString name = QString::fromStdString(json_value(obj, "model_name"));
        if (!selectedModel_.isEmpty() && name == selectedModel_)
            selected_row = i;
        modelsTable_->setItem(i, 0, new QTableWidgetItem(name));
        modelsTable_->setItem(
            i, 1, new QTableWidgetItem(QString::fromStdString(json_value(obj, "role"))));

        auto stateItem = new QTableWidgetItem(QString::fromStdString(json_value(obj, "state")));
        std::string state = json_value(obj, "state");
        if (state == "production")
            stateItem->setForeground(QColor("#2e7d32"));
        else if (state == "training")
            stateItem->setForeground(QColor("#1565c0"));
        else if (state == "candidate")
            stateItem->setForeground(QColor("#e65100"));
        else if (state == "retired")
            stateItem->setForeground(QColor("#9e9e9e"));
        modelsTable_->setItem(i, 2, stateItem);

        std::string id = json_value(obj, "model_id");
        if (id.size() > 8)
            id = id.substr(0, 8) + "...";
        modelsTable_->setItem(i, 3, new QTableWidgetItem(QString::fromStdString(id)));
        modelsTable_->setItem(
            i, 4, new QTableWidgetItem(QString::fromStdString(json_value(obj, "updated_utc"))));
    }
    if (selected_row >= 0) {
        modelsTable_->setCurrentCell(selected_row, 0);
    } else {
        modelsTable_->setCurrentCell(-1, -1);
        modelsTable_->clearSelection();
    }
}

void MnsManagerGUI::populateRolesTable(const std::string& json) {
    auto objects = json_array_objects(json, "roles");
    QSignalBlocker block_table(rolesTable_);  // TD-568: refilling doesn't select a row
    rolesTable_->setRowCount(static_cast<int>(objects.size()));
    rolesTable_->setCurrentCell(-1, -1);
    rolesTable_->clearSelection();
    QStringList roles;

    for (int i = 0; i < static_cast<int>(objects.size()); ++i) {
        const auto& obj = objects[i];
        const QString role = QString::fromStdString(json_value(obj, "role"));
        roles << role;
        rolesTable_->setItem(i, 0, new QTableWidgetItem(role));
        std::string prod = json_value(obj, "production_model");
        if (prod.empty() || prod == "null")
            prod = "(none)";
        rolesTable_->setItem(i, 1, new QTableWidgetItem(QString::fromStdString(prod)));
    }

    // TD-455: the role filter offers the server's roles, which are only those with a production
    // model, so the current choice is kept even when it isn't one of them (a typed role); else the
    // filter would show "(all)" over a filtered table.
    const QString current = roleFilter_->currentIndex() > 0
                                ? roleFilter_->itemText(roleFilter_->currentIndex())
                                : QString();
    if (!current.isEmpty() && !roles.contains(current))
        roles << current;
    QSignalBlocker block(roleFilter_);
    roleFilter_->clear();
    roleFilter_->addItem("(all)");
    roleFilter_->addItems(roles);
    roleFilter_->setCurrentIndex(current.isEmpty() ? 0 : roleFilter_->findText(current));
}

void MnsManagerGUI::showModelDetail(const std::string& json) {
    detailView_->setPlainText(QString::fromStdString(json_pretty(json)));
}

void MnsManagerGUI::setStatusMessage(const QString& msg, bool error) {
    statusText_ = msg;
    statusError_ = error;
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
