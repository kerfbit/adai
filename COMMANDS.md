# ADAI - Command Cheat Sheet

## One-Line Commands (Copy & Paste)

### Build
```bash
# Quick build (release mode)
cd /home/rodney/Repos/adai && ./build_and_vocab.sh build

# Interactive menu
cd /home/rodney/Repos/adai && ./build_and_vocab.sh
```

### Create Vocabulary
```bash
# From sample training data (2000 tokens)
cd /home/rodney/Repos/adai && ./build/bin/vocab_builder --input sample_training_data.txt --output vocab.txt --vocab-size 2000 --format pairs --stats

# From sample training data (5000 tokens)
cd /home/rodney/Repos/adai && ./build/bin/vocab_builder --input sample_training_data.txt --output vocab.txt --vocab-size 5000 --format pairs --stats

# From sample training data (10000 tokens)
cd /home/rodney/Repos/adai && ./build/bin/vocab_builder --input sample_training_data.txt --output vocab.txt --vocab-size 10000 --format pairs --stats
```

### Train Model
```bash
# Quick test (5 epochs)
cd /home/rodney/Repos/adai && ./build/bin/chatbot_trainer --data sample_training_data.txt --vocab vocab.txt --output test_model.bin --epochs 5

# Standard training (10 epochs)
cd /home/rodney/Repos/adai && ./build/bin/chatbot_trainer --data sample_training_data.txt --vocab vocab.txt --output chatbot_model.bin --epochs 10

# Production training (50 epochs)
cd /home/rodney/Repos/adai && ./build/bin/chatbot_trainer --data sample_training_data.txt --vocab vocab.txt --output prod_model.bin --epochs 50 --learning-rate 0.0005
```

### Run Chatbot
```bash
# With default model
cd /home/rodney/Repos/adai && ./build/bin/chatbot --vocab vocab.txt --model chatbot_model.bin

# With specific epoch
cd /home/rodney/Repos/adai && ./build/bin/chatbot --vocab vocab.txt --model chatbot_model.bin.epoch10

# With GUI
cd /home/rodney/Repos/adai && ./build/bin/chatbot_gui --vocab vocab.txt --model chatbot_model.bin
```

### Full Workflow (Everything)
```bash
cd /home/rodney/Repos/adai && ./build_and_vocab.sh full
```

### Clean Build
```bash
cd /home/rodney/Repos/adai && rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
```

## Aliases (Add to ~/.bashrc)
```bash
# Add these to your ~/.bashrc for convenience
alias adai-build='cd /home/rodney/Repos/adai && ./build_and_vocab.sh build'
alias adai-vocab='cd /home/rodney/Repos/adai && ./build/bin/vocab_builder'
alias adai-train='cd /home/rodney/Repos/adai && ./build/bin/chatbot_trainer'
alias adai-chat='cd /home/rodney/Repos/adai && ./build/bin/chatbot'
alias adai-menu='cd /home/rodney/Repos/adai && ./build_and_vocab.sh'
```

After adding aliases:
```bash
source ~/.bashrc
adai-build                    # Build the project
adai-vocab --help             # Vocabulary builder help
adai-train --help             # Trainer help
adai-chat --help              # Chatbot help
adai-menu                     # Interactive menu
```

## Quick Fixes

### Model generates gibberish?
```bash
# Test with greedy decoding and low temperature
cd /home/rodney/Repos/adai && ./build/bin/chatbot --vocab vocab.txt --model chatbot_model.bin
# Then in chatbot:
# /set strategy greedy
# /set temperature 0.3
```

### Need to retrain?
```bash
cd /home/rodney/Repos/adai && ./build/bin/chatbot_trainer --data sample_training_data.txt --vocab vocab.txt --output new_model.bin --epochs 20 --learning-rate 0.001
```

### Rebuild everything from scratch?
```bash
cd /home/rodney/Repos/adai && rm -rf build && ./build_and_vocab.sh full
```
