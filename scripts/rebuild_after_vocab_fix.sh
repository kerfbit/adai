#!/bin/bash
# Complete rebuild and retrain script after vocabulary fix
# Usage: ./rebuild_after_vocab_fix.sh [epochs] [learning_rate]

set -e  # Exit on error

EPOCHS=${1:-20}
LEARNING_RATE=${2:-0.001}
VOCAB_FILE="vocab.txt"
TRAINING_DATA="sample_training_data.txt"
NEW_MODEL="chatbot_model_fixed.bin"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}  Rebuilding After Vocabulary Fix${NC}"
echo -e "${BLUE}=======================================================================${NC}"

# Step 1: Verify vocabulary
echo -e "\n${YELLOW}Step 1: Verifying vocabulary...${NC}"
if [ ! -f "$VOCAB_FILE" ]; then
    echo -e "${RED}Error: $VOCAB_FILE not found!${NC}"
    exit 1
fi

python3 diagnose_generation.py "$VOCAB_FILE" > /tmp/vocab_check.txt 2>&1
if grep -q "ISSUES FOUND" /tmp/vocab_check.txt; then
    echo -e "${RED}Vocabulary has issues! Please fix first:${NC}"
    cat /tmp/vocab_check.txt
    exit 1
fi
echo -e "${GREEN}✓ Vocabulary is valid${NC}"

# Step 2: Clean old build
echo -e "\n${YELLOW}Step 2: Cleaning old build...${NC}"
if [ -d "build" ]; then
    echo "Removing old build directory..."
    rm -rf build
fi
echo -e "${GREEN}✓ Clean complete${NC}"

# Step 3: Rebuild project
echo -e "\n${YELLOW}Step 3: Rebuilding project...${NC}"
mkdir -p build
cd build
echo "Running CMake..."
cmake .. > /tmp/cmake.log 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}CMake failed! Check /tmp/cmake.log${NC}"
    tail -20 /tmp/cmake.log
    exit 1
fi

echo "Compiling with $(nproc) cores..."
make -j$(nproc) > /tmp/make.log 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}Compilation failed! Check /tmp/make.log${NC}"
    tail -20 /tmp/make.log
    exit 1
fi
cd ..
echo -e "${GREEN}✓ Compilation successful${NC}"

# Step 4: Remove old models
echo -e "\n${YELLOW}Step 4: Removing old model files...${NC}"
OLD_MODELS=$(find . -maxdepth 1 -name "chatbot_model*.bin*" -type f 2>/dev/null | wc -l)
if [ "$OLD_MODELS" -gt 0 ]; then
    echo "Found $OLD_MODELS old model file(s)"
    find . -maxdepth 1 -name "chatbot_model*.bin*" -type f -exec rm -f {} \;
    echo -e "${GREEN}✓ Removed $OLD_MODELS old model file(s)${NC}"
else
    echo "No old model files found"
fi

# Step 5: Verify training data
echo -e "\n${YELLOW}Step 5: Verifying training data...${NC}"
if [ ! -f "$TRAINING_DATA" ]; then
    echo -e "${RED}Error: $TRAINING_DATA not found!${NC}"
    exit 1
fi

LINE_COUNT=$(wc -l < "$TRAINING_DATA")
echo "Training data has $LINE_COUNT lines"
if [ "$LINE_COUNT" -lt 10 ]; then
    echo -e "${YELLOW}Warning: Training data seems very small (< 10 lines)${NC}"
    echo -e "${YELLOW}Consider adding more training examples for better results${NC}"
fi
echo -e "${GREEN}✓ Training data found${NC}"

# Step 6: Train new model
echo -e "\n${YELLOW}Step 6: Training new model from scratch...${NC}"
echo "Configuration:"
echo "  - Vocabulary: $VOCAB_FILE"
echo "  - Training data: $TRAINING_DATA"
echo "  - Output model: $NEW_MODEL"
echo "  - Epochs: $EPOCHS"
echo "  - Learning rate: $LEARNING_RATE"
echo ""

if [ ! -f "build/bin/chatbot_trainer" ]; then
    echo -e "${RED}Error: chatbot_trainer executable not found!${NC}"
    exit 1
fi

echo "Starting training (this may take a while)..."
./build/bin/chatbot_trainer \
    --data "$TRAINING_DATA" \
    --vocab "$VOCAB_FILE" \
    --output "$NEW_MODEL" \
    --epochs "$EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --batch-size 32

if [ $? -ne 0 ]; then
    echo -e "${RED}Training failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Training complete${NC}"

# Step 7: Verify model was created
echo -e "\n${YELLOW}Step 7: Verifying model files...${NC}"
if [ ! -f "$NEW_MODEL.config" ]; then
    echo -e "${RED}Error: Model config file not created!${NC}"
    exit 1
fi

if [ ! -f "$NEW_MODEL.vocab" ]; then
    echo -e "${RED}Error: Model vocab file not created!${NC}"
    exit 1
fi

echo "Model files created:"
ls -lh "$NEW_MODEL"* | awk '{print "  " $9 " (" $5 ")"}'
echo -e "${GREEN}✓ All model files created${NC}"

# Step 8: Test the model
echo -e "\n${YELLOW}Step 8: Testing model with simple input...${NC}"
echo -e "${BLUE}You can now test the model with:${NC}"
echo ""
echo "  ./build/bin/chatbot $VOCAB_FILE $NEW_MODEL"
echo ""
echo -e "${BLUE}Try inputs like:${NC}"
echo "  - Hello"
echo "  - How are you?"
echo "  - What is your name?"
echo ""

# Success summary
echo -e "\n${GREEN}=======================================================================${NC}"
echo -e "${GREEN}  ✓ Rebuild Complete!${NC}"
echo -e "${GREEN}=======================================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Test the model as shown above"
echo "  2. If still getting <unk> tokens, try:"
echo "     - Increase epochs: $0 50 0.001"
echo "     - Lower learning rate: $0 30 0.0005"
echo "     - Add more training data to $TRAINING_DATA"
echo ""
echo -e "${GREEN}Model ready: $NEW_MODEL${NC}"
