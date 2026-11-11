#ifndef CHATBOT_GUI_HPP
#define CHATBOT_GUI_HPP

// @adai-status: stable        (widget tests: chatbotGuiWidgetTests)
// @adai-version: 1.0.0
// @adai-reviewed: 2026-11-10

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QSplitter>
#include <QTextEdit>
#include <QVBoxLayout>
#include <functional>
#include <string>
#include <thread>
#include "ChatbotClient.hpp"

/**
 * @brief Qt-based GUI client for chatbot_api_server
 *
 * TD-260: the GUI talks to the served model over HTTP, through the same ChatbotClient as the `chatbot`
 * CLI; it used to load its own copy of the model from local files. It provides:
 * - Chat message display with scrolling
 * - User input field
 * - Generation settings, sent with each message when changed from the server's defaults (TD-254)
 * - Conversation management (save/load/clear), kept by the server
 *
 * TD-263: each request runs on a worker thread, one at a time, so the window stays responsive; input,
 * buttons and settings are disabled until it finishes.
 */
class ChatbotGUI : public QMainWindow {
    Q_OBJECT

   public:
    /**
     * @brief Construct the chatbot GUI and start connecting to the server
     *
     * @param server_url URL of chatbot_api_server (e.g. "http://localhost:8080")
     * @param parent Parent widget (nullptr for main window)
     */
    explicit ChatbotGUI(const std::string& server_url = "http://localhost:8080",
                        QWidget* parent = nullptr);

    /**
     * @brief Destructor: stops a request in flight and waits for its worker thread
     */
    ~ChatbotGUI() override;

   private slots:
    /**
     * @brief Handle send button click or Enter key press
     */
    void onSendMessage();

    /**
     * @brief Clear conversation history
     */
    void onClearConversation();

    /**
     * @brief Save conversation to file
     */
    void onSaveConversation();

    /**
     * @brief Load conversation from file
     */
    void onLoadConversation();

    /**
     * @brief Update generation strategy
     */
    void onStrategyChanged(int index);

    /**
     * @brief Update temperature parameter
     */
    void onTemperatureChanged(double value);

    /**
     * @brief Update top_p parameter
     */
    void onTopPChanged(double value);

    /**
     * @brief Update top_k parameter
     */
    void onTopKChanged(int value);

    /**
     * @brief Update max response length
     */
    void onMaxLengthChanged(int value);

    /**
     * @brief Update beam width for beam search
     */
    void onBeamWidthChanged(int value);

   private:
    /**
     * @brief Set up the user interface
     */
    void setupUI();

    /**
     * @brief Create the chat display area
     */
    QWidget* createChatArea();

    /**
     * @brief Create the input area with text field and send button
     */
    QWidget* createInputArea();

    /**
     * @brief Create the settings panel
     */
    QWidget* createSettingsPanel();

    /**
     * @brief Add a message to the chat display
     *
     * @param sender "User" or "Bot"
     * @param message The message text
     * @param isUser true for user messages, false for bot messages
     */
    void addMessage(const QString& sender, const QString& message, bool isUser);

    /**
     * @brief Run `work` on the worker thread, then the function it returns on the GUI thread
     *
     * Only `work` touches `client`, and only one request runs at a time: the controls are disabled
     * (setBusy) until the returned function has run.
     */
    void runRequest(const QString& busyText, std::function<std::function<void()>()> work);

    /**
     * @brief Enable or disable input, buttons and settings while a request runs
     */
    void setBusy(bool busy, const QString& busyText = QString());

    /**
     * @brief Apply stylesheet for modern appearance
     */
    void applyStylesheet();

    // UI Components
    QTextEdit* chatDisplay;
    QLineEdit* inputField;
    QPushButton* sendButton;
    QPushButton* clearButton;       // the chat area's "Clear Chat"
    QPushButton* clearInputButton;  // "Clear" beside Send
    QPushButton* saveButton;
    QPushButton* loadButton;

    // Settings controls
    QWidget* settingsPanel;
    QComboBox* strategyCombo;
    QDoubleSpinBox* temperatureSpin;
    QDoubleSpinBox* topPSpin;
    QSpinBox* topKSpin;
    QSpinBox* maxLengthSpin;
    QSpinBox* beamWidthSpin;

    // TD-260: the server session and the settings sent with each message. Used only by the worker
    // thread while a request runs, and by the GUI thread otherwise.
    ChatbotClient client;
    std::thread worker;
    std::string conversation_save_path;
};

#endif  // CHATBOT_GUI_HPP
