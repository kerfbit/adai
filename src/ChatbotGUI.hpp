#ifndef CHATBOT_GUI_HPP
#define CHATBOT_GUI_HPP

// @adai-status: beta        (Qt GUI, no dedicated test file)
// @adai-version: 0.7.0
// @adai-reviewed: 2026-09-07


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
#include <memory>
#include <string>

// Forward declarations
class BPETokenizer;
class EncoderDecoderModel;
class ConversationContext;

/**
 * @brief Qt-based GUI for the ADAI transformer chatbot
 *
 * Provides a modern graphical interface with:
 * - Chat message display with scrolling
 * - User input field
 * - Real-time response generation
 * - Settings panel for generation parameters
 * - Conversation management (save/load/clear)
 */
class ChatbotGUI : public QMainWindow {
    Q_OBJECT

   public:
    /**
     * @brief Construct the chatbot GUI
     *
     * @param vocab_file Path to BPE vocabulary file
     * @param model_file Path to pre-trained model weights
     * @param parent Parent widget (nullptr for main window)
     */
    explicit ChatbotGUI(const std::string& vocab_file = "vocab.txt",
                        const std::string& model_file = "chatbot_model.bin",
                        QWidget* parent = nullptr);

    /**
     * @brief Destructor
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
     * @brief Initialize the chatbot components (tokenizer, model, context)
     * @return true if successful, false otherwise
     */
    bool initializeChatbot();

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
     * @brief Generate a response to user input
     *
     * @param user_input The user's message
     * @return The generated response
     */
    std::string generateResponse(const std::string& user_input);

    /**
     * @brief Apply stylesheet for modern appearance
     */
    void applyStylesheet();

    // UI Components
    QTextEdit* chatDisplay;
    QLineEdit* inputField;
    QPushButton* sendButton;
    QPushButton* clearButton;
    QPushButton* saveButton;
    QPushButton* loadButton;

    // Settings controls
    QComboBox* strategyCombo;
    QDoubleSpinBox* temperatureSpin;
    QDoubleSpinBox* topPSpin;
    QSpinBox* topKSpin;
    QSpinBox* maxLengthSpin;
    QSpinBox* beamWidthSpin;

    // Chatbot components
    std::unique_ptr<BPETokenizer> tokenizer;
    std::unique_ptr<EncoderDecoderModel> model;
    std::unique_ptr<ConversationContext> context;

    // Configuration
    std::string vocab_path;
    std::string model_path;
    std::string conversation_save_path;

    // Generation parameters
    std::string generation_strategy;
    int max_response_length;
    float temperature;
    float top_p;
    int top_k;
    int beam_width;
};

#endif  // CHATBOT_GUI_HPP
