// @adai-status: beta        (capped by TD-037 — no Qt Test infrastructure in this repo; onStrategyChanged's mapping extracted to ChatbotGuiLogic.hpp, see TECHNICAL_DEBT.md)
// @adai-version: 0.9.0
// @adai-reviewed: 2026-11-04

#include "ChatbotGUI.hpp"
#include <QDateTime>
#include <QFile>
#include <QFileDialog>
#include <QMessageBox>
#include <QMetaObject>
#include <QScrollBar>
#include <memory>
#include <utility>
#include "ChatbotGuiLogic.hpp"
#include "GenerationSettings.hpp"

ChatbotGUI::ChatbotGUI(const std::string& server_url, QWidget* parent)
    : QMainWindow(parent), client(server_url), conversation_save_path("conversation_history.txt") {
    setWindowTitle("ADAI Transformer Chatbot");
    resize(1000, 700);

    // Set up the user interface
    setupUI();
    applyStylesheet();

    // Add welcome message
    addMessage("System",
               "Welcome to ADAI Transformer Chatbot!\n"
               "Type your message and press Send or Enter to chat.",
               false);

    // TD-260: the GUI is a client of chatbot_api_server, which serves the trained model. Connect on
    // the worker thread so a slow or absent server doesn't hold up the window.
    try {
        client.open();
    } catch (const std::exception& e) {
        addMessage("System", QString("Invalid server URL %1: %2").arg(QString::fromStdString(server_url), e.what()),
                   false);
        setBusy(true, "No server");  // nothing can be sent without a connection
        return;
    }
    runRequest("Connecting...", [this] {
        const ChatbotClient::Health health = client.connect();
        return [this, health] {
            const auto notice = chatbot_gui::connection_notice(client.server_url(), health);
            addMessage("System", QString::fromStdString(notice.text), false);
        };
    });
}

ChatbotGUI::~ChatbotGUI() {
    // A request may be waiting on the server (up to the 300 s read timeout): stop it so closing the
    // window doesn't wait for it. Its result is never shown: Qt drops events posted to a deleted object.
    client.cancel();
    if (worker.joinable()) {
        worker.join();
    }
}

void ChatbotGUI::runRequest(const QString& busyText, std::function<std::function<void()>()> work) {
    if (worker.joinable()) {
        worker.join();  // the previous request has finished: its result already ran setBusy(false)
    }
    setBusy(true, busyText);
    worker = std::thread([this, work = std::move(work)] {
        std::function<void()> done;
        try {
            done = work();
        } catch (const std::exception& e) {
            const QString error = e.what();
            done = [this, error] { addMessage("System", "Error: " + error, false); };
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

void ChatbotGUI::setBusy(bool busy, const QString& busyText) {
    inputField->setEnabled(!busy);
    sendButton->setEnabled(!busy);
    sendButton->setText(busy ? busyText : "Send");
    clearButton->setEnabled(!busy);
    clearInputButton->setEnabled(!busy);
    saveButton->setEnabled(!busy);
    loadButton->setEnabled(!busy);
    settingsPanel->setEnabled(!busy);
    if (!busy) {
        inputField->setFocus();
    }
}

void ChatbotGUI::setupUI() {
    // Create main widget and layout
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);

    // Create splitter for resizable areas
    QSplitter* splitter = new QSplitter(Qt::Horizontal, this);

    // Left side: Chat area
    QWidget* chatWidget = new QWidget();
    QVBoxLayout* chatLayout = new QVBoxLayout(chatWidget);
    chatLayout->addWidget(createChatArea());
    chatLayout->addWidget(createInputArea());

    splitter->addWidget(chatWidget);

    // Right side: Settings panel
    splitter->addWidget(createSettingsPanel());

    // Set initial sizes (70% chat, 30% settings)
    splitter->setStretchFactor(0, 7);
    splitter->setStretchFactor(1, 3);

    mainLayout->addWidget(splitter);
}

QWidget* ChatbotGUI::createChatArea() {
    QWidget* widget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(widget);

    // Chat display
    chatDisplay = new QTextEdit();
    chatDisplay->setReadOnly(true);
    chatDisplay->setMinimumHeight(400);

    layout->addWidget(chatDisplay);

    // Action buttons
    QHBoxLayout* buttonLayout = new QHBoxLayout();

    clearButton = new QPushButton("Clear Chat");
    saveButton = new QPushButton("Save Conversation");
    loadButton = new QPushButton("Load Conversation");

    connect(clearButton, &QPushButton::clicked, this, &ChatbotGUI::onClearConversation);
    connect(saveButton, &QPushButton::clicked, this, &ChatbotGUI::onSaveConversation);
    connect(loadButton, &QPushButton::clicked, this, &ChatbotGUI::onLoadConversation);

    buttonLayout->addWidget(clearButton);
    buttonLayout->addWidget(saveButton);
    buttonLayout->addWidget(loadButton);
    buttonLayout->addStretch();

    layout->addLayout(buttonLayout);

    return widget;
}

QWidget* ChatbotGUI::createInputArea() {
    QWidget* widget = new QWidget();
    QHBoxLayout* layout = new QHBoxLayout(widget);

    inputField = new QLineEdit();
    inputField->setPlaceholderText("Type your message here...");
    inputField->setMinimumHeight(40);

    // Hygiene fix: this used to assign into the `clearButton` member — the
    // same member createChatArea() already set to the chat area's
    // "Clear Chat" button. Both buttons' own connect()s still worked (Qt
    // binds to the object, not the member holding it), so this was never a
    // functional bug, but `clearButton` silently ended up referring only to
    // this second, redundant button afterward, discarding any live handle
    // to the first one. `clearButton` means the chat area's clear button,
    // matching its siblings saveButton/loadButton; this one has its own
    // member so setBusy() can disable it (TD-263).
    clearInputButton = new QPushButton("Clear");
    clearInputButton->setMinimumWidth(80);
    clearInputButton->setMinimumHeight(40);
    clearInputButton->setToolTip("Clear conversation history");

    sendButton = new QPushButton("Send");
    sendButton->setMinimumWidth(100);
    sendButton->setMinimumHeight(40);

    // Connect signals
    connect(sendButton, &QPushButton::clicked, this, &ChatbotGUI::onSendMessage);
    connect(clearInputButton, &QPushButton::clicked, this, &ChatbotGUI::onClearConversation);
    connect(inputField, &QLineEdit::returnPressed, this, &ChatbotGUI::onSendMessage);

    layout->addWidget(inputField);
    layout->addWidget(clearInputButton);
    layout->addWidget(sendButton);

    return widget;
}

namespace {

// TD-260: the settings use the server's ranges (adai::GenerationLimits, which it checks each request
// against, TD-254). Each spin box starts one step below its range, shown as "server default": the
// setting isn't sent until changed, so the server's configured default applies.
constexpr double kRealStep = 0.05;
const QString kServerDefault = "server default";

QSpinBox* make_int_setting(int min, int max, int step) {
    auto* spin = new QSpinBox();
    spin->setRange(min - 1, max);
    spin->setSingleStep(step);
    spin->setSpecialValueText(kServerDefault);
    spin->setValue(min - 1);
    return spin;
}

QDoubleSpinBox* make_real_setting(double min, double max) {
    auto* spin = new QDoubleSpinBox();
    spin->setDecimals(2);
    spin->setRange(min - kRealStep, max);
    spin->setSingleStep(kRealStep);
    spin->setSpecialValueText(kServerDefault);
    spin->setValue(min - kRealStep);
    return spin;
}

QGroupBox* setting_group(const QString& title, QWidget* control, const QString& description) {
    auto* group = new QGroupBox(title);
    auto* layout = new QVBoxLayout();
    layout->addWidget(control);
    if (!description.isEmpty()) {
        auto* label = new QLabel(description);
        label->setStyleSheet("font-size: 10px; color: #666;");
        label->setWordWrap(true);
        layout->addWidget(label);
    }
    group->setLayout(layout);
    return group;
}

}  // namespace

QWidget* ChatbotGUI::createSettingsPanel() {
    using Limits = adai::GenerationLimits;
    settingsPanel = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(settingsPanel);

    // Title
    QLabel* titleLabel = new QLabel("Generation Settings");
    QFont titleFont = titleLabel->font();
    titleFont.setPointSize(14);
    titleFont.setBold(true);
    titleLabel->setFont(titleFont);
    layout->addWidget(titleLabel);

    QLabel* note = new QLabel("Settings left at \"server default\" use chatbot_api_server's own settings.");
    note->setStyleSheet("font-size: 10px; color: #666;");
    note->setWordWrap(true);
    layout->addWidget(note);

    // Strategy selection; the order matches chatbot_gui::generation_strategy_for_index().
    strategyCombo = new QComboBox();
    strategyCombo->addItems({"Server default", "Nucleus (Top-p)", "Top-k Sampling", "Greedy",
                             "Beam Search", "Sampling (Temperature)"});
    connect(strategyCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &ChatbotGUI::onStrategyChanged);
    layout->addWidget(setting_group("Generation Strategy", strategyCombo, QString()));

    temperatureSpin = make_real_setting(Limits::kMinTemperature, Limits::kMaxTemperature);
    connect(temperatureSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &ChatbotGUI::onTemperatureChanged);
    layout->addWidget(setting_group("Temperature", temperatureSpin, "Higher = more random"));

    topPSpin = make_real_setting(Limits::kMinTopP, Limits::kMaxTopP);
    connect(topPSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &ChatbotGUI::onTopPChanged);
    layout->addWidget(setting_group("Top-p (Nucleus)", topPSpin, "Probability mass cutoff"));

    topKSpin = make_int_setting(Limits::kMinTopK, Limits::kMaxTopK, 1);
    connect(topKSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &ChatbotGUI::onTopKChanged);
    layout->addWidget(setting_group("Top-k", topKSpin, "Number of top tokens"));

    maxLengthSpin = make_int_setting(Limits::kMinMaxLength, Limits::kMaxMaxLength, 10);
    connect(maxLengthSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &ChatbotGUI::onMaxLengthChanged);
    layout->addWidget(setting_group("Max Response Length", maxLengthSpin, "New tokens"));

    beamWidthSpin = make_int_setting(Limits::kMinBeamWidth, Limits::kMaxBeamWidth, 1);
    connect(beamWidthSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &ChatbotGUI::onBeamWidthChanged);
    layout->addWidget(setting_group("Beam Width", beamWidthSpin, "Used in beam search"));

    // Add stretch to push everything to the top
    layout->addStretch();

    return settingsPanel;
}

void ChatbotGUI::addMessage(const QString& sender, const QString& message, bool isUser) {
    // Get current time
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");

    // Create HTML formatted message
    QString color = isUser ? "#2196F3" : "#4CAF50";
    QString bgColor = isUser ? "#E3F2FD" : "#E8F5E9";

    if (sender == "System") {
        color = "#FF9800";
        bgColor = "#FFF3E0";
    }

    QString html =
        QString(
            "<div style='margin: 10px; padding: 10px; background-color: %1; border-radius: 8px;'>"
            "<span style='font-weight: bold; color: %2;'>%3</span> "
            "<span style='color: #666; font-size: 10px;'>%4</span><br>"
            "<span style='color: #333; margin-top: 5px; display: block;'>%5</span>"
            "</div>")
            .arg(bgColor, color, sender, timestamp, message.toHtmlEscaped());

    chatDisplay->append(html);

    // Auto-scroll to bottom
    QScrollBar* scrollBar = chatDisplay->verticalScrollBar();
    scrollBar->setValue(scrollBar->maximum());
}

void ChatbotGUI::onSendMessage() {
    const QString userMessage = inputField->text().trimmed();
    if (userMessage.isEmpty()) {
        return;
    }
    addMessage("You", userMessage, true);
    inputField->clear();

    // TD-263: the request runs on the worker thread; the window stays responsive.
    runRequest("Generating...", [this, message = userMessage.toStdString()] {
        const ChatbotClient::Reply reply = client.send_message(message);
        return [this, reply] {
            if (reply.session_lost) {
                addMessage("System", QString::fromStdString(chatbot_gui::session_lost_notice()), false);
            }
            // TD-255: a failed turn shows the server's error, as a system message.
            addMessage(reply.ok ? "Bot" : "System", QString::fromStdString(reply.text), false);
        };
    });
}

void ChatbotGUI::onClearConversation() {
    auto answer = QMessageBox::question(this, "Clear Conversation",
                                        "Are you sure you want to clear the conversation history?",
                                        QMessageBox::Yes | QMessageBox::No);
    if (answer != QMessageBox::Yes) {
        return;
    }
    runRequest("Clearing...", [this] {
        const ChatbotClient::ClearResult result = client.clear_session();
        return [this, result] {
            if (result == ChatbotClient::ClearResult::Failed) {
                addMessage("System", "The server didn't clear the conversation.", false);
                return;
            }
            chatDisplay->clear();
            addMessage("System", "Conversation cleared.", false);
        };
    });
}

void ChatbotGUI::onSaveConversation() {
    if (client.session_id().empty()) {
        addMessage("System", "Nothing to save yet: send a message first.", false);
        return;
    }
    const QString fileName =
        QFileDialog::getSaveFileName(this, "Save Conversation", conversation_save_path.c_str(),
                                     "Text Files (*.txt);;All Files (*)");
    if (fileName.isEmpty()) {
        return;
    }
    // TD-053: the conversation is kept by the server; saving exports it and writes the file.
    runRequest("Saving...", [this, path = fileName.toStdString()] {
        using Status = ChatbotClient::Export::Status;
        const ChatbotClient::Export exported = client.export_session();
        std::string error;
        switch (exported.status) {
            case Status::Ok:
                error = chatbot_client::write_file_atomically(path, exported.data);  // TD-258
                break;
            case Status::NoSession:
            case Status::NotConnected:
                error = "Nothing to save yet: send a message first.";
                break;
            case Status::Unreachable:
                error = "Couldn't reach the server to export the conversation.";
                break;
            case Status::SessionLost:
                error = "The server no longer has this conversation (it expired or the server "
                        "restarted), so there is nothing to save.";
                break;
            case Status::Failed:
                error = exported.error.empty() ? "The server didn't export the conversation." : exported.error;
                break;
        }
        return [this, path, error] {
            if (!error.empty()) {
                QMessageBox::critical(this, "Error",
                                      QString("Failed to save conversation:\n%1").arg(QString::fromStdString(error)));
                return;
            }
            conversation_save_path = path;
            addMessage("System", QString("Conversation saved to %1").arg(QString::fromStdString(path)), false);
        };
    });
}

void ChatbotGUI::onLoadConversation() {
    const QString fileName =
        QFileDialog::getOpenFileName(this, "Load Conversation", conversation_save_path.c_str(),
                                     "Text Files (*.txt);;All Files (*)");
    if (fileName.isEmpty()) {
        return;
    }
    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, "Error", QString("Failed to open %1:\n%2").arg(fileName, file.errorString()));
        return;
    }
    const std::string data = file.readAll().toStdString();
    if (data.empty()) {
        QMessageBox::critical(this, "Error", QString("%1 is empty.").arg(fileName));
        return;
    }
    // TD-053: the server imports the conversation into this session (or a new one).
    runRequest("Loading...", [this, data, path = fileName.toStdString()] {
        const ChatbotClient::Import imported = client.import_session(data);
        return [this, imported, path] {
            if (!imported.ok) {
                QMessageBox::critical(this, "Error",
                                      QString("Failed to load conversation:\n%1")
                                          .arg(QString::fromStdString(imported.error)));
                return;
            }
            conversation_save_path = path;
            chatDisplay->clear();
            // TODO: See TD-266 in TECHNICAL_DEBT.md - loaded messages are not redisplayed.
            QString text = QString("Conversation loaded from %1").arg(QString::fromStdString(path));
            if (!imported.message_count.empty()) {
                text += QString(" (%1 messages)").arg(QString::fromStdString(imported.message_count));
            }
            addMessage("System", text + ". The server continues it from here.", false);
        };
    });
}

// TD-260: each change is recorded in the client's overrides, sent with the next message (TD-254).
// The settings panel is disabled while a request runs, so the worker never reads them mid-change.
void ChatbotGUI::onStrategyChanged(int index) {
    chatbot_gui::apply_strategy(client.settings(), index);
}

void ChatbotGUI::onTemperatureChanged(double value) {
    chatbot_gui::apply_real_setting(client.settings(), "temperature", value, temperatureSpin->minimum());
}

void ChatbotGUI::onTopPChanged(double value) {
    chatbot_gui::apply_real_setting(client.settings(), "top_p", value, topPSpin->minimum());
}

void ChatbotGUI::onTopKChanged(int value) {
    chatbot_gui::apply_int_setting(client.settings(), "top_k", value, topKSpin->minimum());
}

void ChatbotGUI::onMaxLengthChanged(int value) {
    chatbot_gui::apply_int_setting(client.settings(), "max_length", value, maxLengthSpin->minimum());
}

void ChatbotGUI::onBeamWidthChanged(int value) {
    chatbot_gui::apply_int_setting(client.settings(), "beam_width", value, beamWidthSpin->minimum());
}

void ChatbotGUI::applyStylesheet() {
    QString styleSheet = R"(
        QMainWindow {
            background-color: #f5f5f5;
        }
        
        QTextEdit {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 13px;
        }
        
        QLineEdit {
            background-color: white;
            border: 2px solid #ddd;
            border-radius: 20px;
            padding: 8px 15px;
            font-size: 13px;
        }
        
        QLineEdit:focus {
            border: 2px solid #2196F3;
        }
        
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 20px;
            padding: 8px 20px;
            font-size: 13px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #1976D2;
        }
        
        QPushButton:pressed {
            background-color: #0D47A1;
        }
        
        QPushButton:disabled {
            background-color: #ccc;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 2px solid #ddd;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
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
            padding: 5px;
            min-height: 25px;
        }
        
        QSpinBox, QDoubleSpinBox {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            min-height: 25px;
        }
        
        QScrollBar:vertical {
            border: none;
            background: #f0f0f0;
            width: 10px;
            margin: 0;
        }
        
        QScrollBar::handle:vertical {
            background: #c0c0c0;
            border-radius: 5px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: #a0a0a0;
        }
    )";

    setStyleSheet(styleSheet);
}
