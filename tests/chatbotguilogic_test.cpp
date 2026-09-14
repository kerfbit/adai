// TD-037: ChatbotGuiLogic.hpp's generation_strategy_for_index() was extracted from
// ChatbotGUI::onStrategyChanged() specifically so it's testable without a QApplication/QComboBox
// — see that header's own doc comment for the full rationale.
#include "ChatbotGuiLogic.hpp"
#include <gtest/gtest.h>

using chatbot_gui::generation_strategy_for_index;

TEST(GenerationStrategyForIndexTest, MapsEachComboBoxIndexToItsStrategyName) {
    // Order matches ChatbotGUI.cpp's createSettingsPanel() combo box items exactly:
    // "Nucleus (Top-p)", "Top-k Sampling", "Greedy", "Beam Search", "Sampling".
    EXPECT_EQ(generation_strategy_for_index(0), "nucleus");
    EXPECT_EQ(generation_strategy_for_index(1), "top-k");
    EXPECT_EQ(generation_strategy_for_index(2), "greedy");
    EXPECT_EQ(generation_strategy_for_index(3), "beam");
    EXPECT_EQ(generation_strategy_for_index(4), "sampling");
}

TEST(GenerationStrategyForIndexTest, FallsBackToNucleusForAnOutOfRangeIndex) {
    EXPECT_EQ(generation_strategy_for_index(-1), "nucleus");
    EXPECT_EQ(generation_strategy_for_index(5), "nucleus");
    EXPECT_EQ(generation_strategy_for_index(999), "nucleus");
}
