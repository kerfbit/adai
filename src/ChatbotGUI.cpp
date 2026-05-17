#include "ChatbotGUI.hpp"
#include <QApplication>
#include <QDateTime>
#include <QFileDialog>
#include <QMessageBox>
#include <QScrollBar>
#include <fstream>
#include "BPETokenizer.hpp"
#include "Config.hpp"
#include "ConversationContext.hpp"
#include "EncoderDecoderModel.hpp"

ChatbotGUI::ChatbotGUI(const std::string& vocab_file, const std::string& model_file,
                       QWidget* parent)
    : QMainWindow(parent),
      vocab_path(vocab_file),
      model_path(model_file),
      conversation_save_path("conversation_history.txt"),
      generation_strategy("nucleus"),
      max_response_length(100),
      temperature(1.0f),
      top_p(0.9f),
      top_k(50),
      beam_width(5) {
    setWindowTitle("ADAI Transformer Chatbot");
    resize(1000, 700);

    // Initialize chatbot components
    if (!initializeChatbot()) {
        QMessageBox::critical(this, "Initialization Error",
                              "Failed to initialize chatbot components.\n"
                              "Please check that vocab and model files exist.");
    }

    // Set up the user interface
    setupUI();
    applyStylesheet();

    // Add welcome message
    addMessage("System",
               "Welcome to ADAI Transformer Chatbot!\n"
               "Type your message and press Send or Enter to chat.",
               false);
}

ChatbotGUI::~ChatbotGUI() = default;

bool ChatbotGUI::initializeChatbot() {
    try {
        // Load tokenizer
        tokenizer = std::make_unique<BPETokenizer>();
        tokenizer->load_vocab(vocab_path);

        // Load architecture from config (vocab/model paths still come from constructor args)
        adai::ServiceConfig svc = adai::ConfigLoader::load();

        // Initialize model
        model = std::make_unique<EncoderDecoderModel>(
            tokenizer->get_vocab_size(),               // vocab_size
            static_cast<int>(svc.d_model),             // d_model
            static_cast<int>(svc.num_encoder_layers),  // encoder_layers
            static_cast<int>(svc.num_decoder_layers),  // decoder_layers
            static_cast<int>(svc.num_heads),           // num_heads
            static_cast<int>(svc.d_ff),                // d_ff
            static_cast<int>(svc.max_seq_length)       // max_seq_length
        );

        // Transfer tokenizer ownership to model
        model->set_tokenizer(tokenizer.release());

        // Load pre-trained weights if available
        std::ifstream model_file(model_path);
        if (model_file.good()) {
            model->load_model(model_path);
        }

        // Create conversation context with max_tokens=480 to stay under model's max_len=512
        context = std::make_unique<ConversationContext>(20, 480);

        return true;
    } catch (const std::exception& e) {
        QMessageBox::warning(nullptr, "Initialization Warning",
                             QString("Error during initialization: %1\n"
                                     "Using default/random initialization.")
                                 .arg(e.what()));
        return false;
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

    clearButton = new QPushButton("Clear");
    clearButton->setMinimumWidth(80);
    clearButton->setMinimumHeight(40);
    clearButton->setToolTip("Clear conversation history");

    sendButton = new QPushButton("Send");
    sendButton->setMinimumWidth(100);
    sendButton->setMinimumHeight(40);

    // Connect signals
    connect(sendButton, &QPushButton::clicked, this, &ChatbotGUI::onSendMessage);
    connect(clearButton, &QPushButton::clicked, this, &ChatbotGUI::onClearConversation);
    connect(inputField, &QLineEdit::returnPressed, this, &ChatbotGUI::onSendMessage);

    layout->addWidget(inputField);
    layout->addWidget(clearButton);
    layout->addWidget(sendButton);

    return widget;
}

QWidget* ChatbotGUI::createSettingsPanel() {
    QWidget* widget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(widget);

    // Title
    QLabel* titleLabel = new QLabel("Generation Settings");
    QFont titleFont = titleLabel->font();
    titleFont.setPointSize(14);
    titleFont.setBold(true);
    titleLabel->setFont(titleFont);
    layout->addWidget(titleLabel);

    // Strategy selection
    QGroupBox* strategyGroup = new QGroupBox("Generation Strategy");
    QVBoxLayout* strategyLayout = new QVBoxLayout();

    strategyCombo = new QComboBox();
    strategyCombo->addItems(
        {"Nucleus (Top-p)", "Top-k Sampling", "Greedy", "Beam Search", "Sampling"});
    strategyCombo->setCurrentText("Nucleus (Top-p)");
    connect(strategyCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &ChatbotGUI::onStrategyChanged);

    strategyLayout->addWidget(strategyCombo);
    strategyGroup->setLayout(strategyLayout);
    layout->addWidget(strategyGroup);

    // Temperature
    QGroupBox* tempGroup = new QGroupBox("Temperature");
    QVBoxLayout* tempLayout = new QVBoxLayout();

    temperatureSpin = new QDoubleSpinBox();
    temperatureSpin->setRange(0.1, 2.0);
    temperatureSpin->setSingleStep(0.1);
    temperatureSpin->setValue(1.0);
    temperatureSpin->setDecimals(1);
    connect(temperatureSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &ChatbotGUI::onTemperatureChanged);

    QLabel* tempDesc = new QLabel("Higher = more random");
    tempDesc->setStyleSheet("font-size: 10px; color: #666;");

    tempLayout->addWidget(temperatureSpin);
    tempLayout->addWidget(tempDesc);
    tempGroup->setLayout(tempLayout);
    layout->addWidget(tempGroup);

    // Top-p (nucleus)
    QGroupBox* topPGroup = new QGroupBox("Top-p (Nucleus)");
    QVBoxLayout* topPLayout = new QVBoxLayout();

    topPSpin = new QDoubleSpinBox();
    topPSpin->setRange(0.1, 1.0);
    topPSpin->setSingleStep(0.05);
    topPSpin->setValue(0.9);
    topPSpin->setDecimals(2);
    connect(topPSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &ChatbotGUI::onTopPChanged);

    QLabel* topPDesc = new QLabel("Probability mass cutoff");
    topPDesc->setStyleSheet("font-size: 10px; color: #666;");

    topPLayout->addWidget(topPSpin);
    topPLayout->addWidget(topPDesc);
    topPGroup->setLayout(topPLayout);
    layout->addWidget(topPGroup);

    // Top-k
    QGroupBox* topKGroup = new QGroupBox("Top-k");
    QVBoxLayout* topKLayout = new QVBoxLayout();

    topKSpin = new QSpinBox();
    topKSpin->setRange(1, 200);
    topKSpin->setValue(50);
    connect(topKSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &ChatbotGUI::onTopKChanged);

    QLabel* topKDesc = new QLabel("Number of top tokens");
    topKDesc->setStyleSheet("font-size: 10px; color: #666;");

    topKLayout->addWidget(topKSpin);
    topKLayout->addWidget(topKDesc);
    topKGroup->setLayout(topKLayout);
    layout->addWidget(topKGroup);

    // Max length
    QGroupBox* maxLenGroup = new QGroupBox("Max Response Length");
    QVBoxLayout* maxLenLayout = new QVBoxLayout();

    maxLengthSpin = new QSpinBox();
    maxLengthSpin->setRange(10, 500);
    maxLengthSpin->setValue(100);
    connect(maxLengthSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &ChatbotGUI::onMaxLengthChanged);

    maxLenLayout->addWidget(maxLengthSpin);
    maxLenGroup->setLayout(maxLenLayout);
    layout->addWidget(maxLenGroup);

    // Beam width (for beam search)
    QGroupBox* beamGroup = new QGroupBox("Beam Width");
    QVBoxLayout* beamLayout = new QVBoxLayout();

    beamWidthSpin = new QSpinBox();
    beamWidthSpin->setRange(1, 10);
    beamWidthSpin->setValue(5);
    connect(beamWidthSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &ChatbotGUI::onBeamWidthChanged);

    QLabel* beamDesc = new QLabel("Used in beam search");
    beamDesc->setStyleSheet("font-size: 10px; color: #666;");

    beamLayout->addWidget(beamWidthSpin);
    beamLayout->addWidget(beamDesc);
    beamGroup->setLayout(beamLayout);
    layout->addWidget(beamGroup);

    // Add stretch to push everything to the top
    layout->addStretch();

    return widget;
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
    QString userMessage = inputField->text().trimmed();

    if (userMessage.isEmpty()) {
        return;
    }

    // Add user message to display
    addMessage("You", userMessage, true);

    // Clear input field
    inputField->clear();

    // Disable input while generating
    inputField->setEnabled(false);
    sendButton->setEnabled(false);
    sendButton->setText("Generating...");

    // Process events to update UI
    QApplication::processEvents();

    // Generate response
    std::string response = generateResponse(userMessage.toStdString());

    // Add bot response to display
    addMessage("Bot", QString::fromStdString(response), false);

    // Re-enable input
    inputField->setEnabled(true);
    sendButton->setEnabled(true);
    sendButton->setText("Send");
    inputField->setFocus();
}

void ChatbotGUI::onClearConversation() {
    auto reply = QMessageBox::question(this, "Clear Conversation",
                                       "Are you sure you want to clear the conversation history?",
                                       QMessageBox::Yes | QMessageBox::No);

    if (reply == QMessageBox::Yes) {
        chatDisplay->clear();
        if (context) {
            context->clear();
        }
        addMessage("System", "Conversation cleared.", false);
    }
}

void ChatbotGUI::onSaveConversation() {
    QString fileName =
        QFileDialog::getSaveFileName(this, "Save Conversation", conversation_save_path.c_str(),
                                     "Text Files (*.txt);;All Files (*)");

    if (!fileName.isEmpty()) {
        try {
            if (context) {
                context->save_to_file(fileName.toStdString());
                QMessageBox::information(this, "Success", "Conversation saved successfully!");
            }
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "Error",
                                  QString("Failed to save conversation:\n%1").arg(e.what()));
        }
    }
}

void ChatbotGUI::onLoadConversation() {
    QString fileName =
        QFileDialog::getOpenFileName(this, "Load Conversation", conversation_save_path.c_str(),
                                     "Text Files (*.txt);;All Files (*)");

    if (!fileName.isEmpty()) {
        try {
            if (context) {
                context->load_from_file(fileName.toStdString());
                chatDisplay->clear();
                addMessage("System", "Conversation loaded successfully!", false);

                // Display loaded messages
                // Note: This would require ConversationContext to expose message history
                addMessage("System", "Previous conversation has been loaded into context.", false);
            }
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "Error",
                                  QString("Failed to load conversation:\n%1").arg(e.what()));
        }
    }
}

void ChatbotGUI::onStrategyChanged(int index) {
    switch (index) {
        case 0:
            generation_strategy = "nucleus";
            break;
        case 1:
            generation_strategy = "top-k";
            break;
        case 2:
            generation_strategy = "greedy";
            break;
        case 3:
            generation_strategy = "beam";
            break;
        case 4:
            generation_strategy = "sampling";
            break;
        default:
            generation_strategy = "nucleus";
            break;
    }
}

void ChatbotGUI::onTemperatureChanged(double value) {
    temperature = static_cast<float>(value);
}

void ChatbotGUI::onTopPChanged(double value) {
    top_p = static_cast<float>(value);
}

void ChatbotGUI::onTopKChanged(int value) {
    top_k = value;
}

void ChatbotGUI::onMaxLengthChanged(int value) {
    max_response_length = value;
}

void ChatbotGUI::onBeamWidthChanged(int value) {
    beam_width = value;
}

std::string ChatbotGUI::generateResponse(const std::string& user_input) {
    if (!model || !context) {
        return "[Error: Chatbot not initialized]";
    }

    try {
        // Add user message to context
        context->add_user_message(user_input);

        // Format context for model
        std::string formatted_context = context->format_with_special_tokens();

        // Generate response using the model
        std::string response = model->generate_response_with_strategy(
            formatted_context, max_response_length, generation_strategy, temperature, top_k, top_p,
            beam_width);

        // Add assistant response to context
        context->add_assistant_message(response);

        return response;

    } catch (const std::exception& e) {
        return "[Error: " + std::string(e.what()) + "]";
    }
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
