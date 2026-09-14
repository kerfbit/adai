#pragma once

// @adai-status: beta        (TD-037 — covered by chatbotguilogicTests)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13

#include <string>

namespace chatbot_gui {

// TD-037: ChatbotGUI's generation-strategy QComboBox (see ChatbotGUI.cpp's createSettingsPanel())
// lists strategies in this exact order — "Nucleus (Top-p)", "Top-k Sampling", "Greedy",
// "Beam Search", "Sampling" — and onStrategyChanged(int index) used to map the combo box's
// selected index to EncoderDecoderModel::generate_response_with_strategy()'s strategy-name
// argument via a switch statement inline in the widget class. Extracted here (TD-037's own
// Description calls this out as the tractable near-term step ahead of any QTest/widget-testing
// framework decision) so it's testable without a QApplication/QComboBox at all. Any out-of-range
// index (there shouldn't be one — Qt only ever calls back with a real item index — but the
// original switch had one) falls back to "nucleus", preserved here for behavioral parity.
inline std::string generation_strategy_for_index(int index) {
    switch (index) {
        case 0:
            return "nucleus";
        case 1:
            return "top-k";
        case 2:
            return "greedy";
        case 3:
            return "beam";
        case 4:
            return "sampling";
        default:
            return "nucleus";
    }
}

}  // namespace chatbot_gui
