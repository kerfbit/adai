# ADAI - Command Cheat Sheet

## Build

```bash
# Quick build (release mode)
cd /home/rodney/Repos/adai
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Rebuild a specific target
cd /home/rodney/Repos/adai/build && cmake --build . --target incremental_trainer -j$(nproc)

# Clean rebuild
cd /home/rodney/Repos/adai && rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
```

## Create Vocabulary

```bash
# From training pairs file (2000 tokens)
cd /home/rodney/Repos/adai && ./build/bin/vocab_builder --input sample_training_data.txt --output vocab.txt --vocab-size 2000 --format pairs --stats

# From training pairs file (5000 tokens)
cd /home/rodney/Repos/adai && ./build/bin/vocab_builder --input sample_training_data.txt --output vocab.txt --vocab-size 5000 --format pairs --stats

# From training pairs file (10000 tokens)
cd /home/rodney/Repos/adai && ./build/bin/vocab_builder --input sample_training_data.txt --output vocab.txt --vocab-size 10000 --format pairs --stats

# From multiple input files
cd /home/rodney/Repos/adai && ./build/bin/vocab_builder --input file1.txt --input file2.txt --output vocab.txt --vocab-size 10000 --stats
```

## Dataset Management

```bash
# Add a local training file to the pending queue
cd /home/rodney/Repos/adai && ./build/bin/dataset_manager add sample_training_data.txt

# Download a Gutenberg book and queue it (500 pairs)
cd /home/rodney/Repos/adai && ./build/bin/dataset_manager gutenberg 1342 500

# Batch-download multiple Gutenberg books
cd /home/rodney/Repos/adai && ./build/bin/dataset_manager gutenberg-batch 1342,11,84,1661 300

# Download a HuggingFace dataset
cd /home/rodney/Repos/adai && ./build/bin/dataset_manager huggingface daily_dialog 500
cd /home/rodney/Repos/adai && ./build/bin/dataset_manager huggingface tatsu-lab/alpaca 300 train instruction output

# Check queue status
cd /home/rodney/Repos/adai && ./build/bin/dataset_manager status

# List pending / trained files
cd /home/rodney/Repos/adai && ./build/bin/dataset_manager list-pending
cd /home/rodney/Repos/adai && ./build/bin/dataset_manager list-trained

# Clear the pending queue
cd /home/rodney/Repos/adai && ./build/bin/dataset_manager clear-pending
```

## Train Model

All training commands fork into the background automatically. Follow progress in the log.

```bash
# Initialize the trainer (first time only)
cd /home/rodney/Repos/adai && ./build/bin/incremental_trainer init

# Train on pending files (incremental — only new data)
cd /home/rodney/Repos/adai && ./build/bin/incremental_trainer train 25

# Full retrain from scratch on all queued data
cd /home/rodney/Repos/adai && ./build/bin/incremental_trainer retrain 25

# Resume an interrupted session
cd /home/rodney/Repos/adai && ./build/bin/incremental_trainer resume

# Check training status and pending queue
cd /home/rodney/Repos/adai && ./build/bin/incremental_trainer status

# Show full session history and data registry
cd /home/rodney/Repos/adai && ./build/bin/incremental_trainer history

# Reset model weights to config architecture (keeps data registry)
cd /home/rodney/Repos/adai && ./build/bin/incremental_trainer reset --keep-data --yes

# Follow training log
cd /home/rodney/Repos/adai && tail -f chatbot_server.log
```

Hyperparameters (`LEARNING_RATE`, `NUM_EPOCHS`, `BATCH_SIZE`, etc.) are set in `config.conf`.

## Training Metrics

```bash
# Start the metrics daemon (run before training to enable live monitoring)
cd /home/rodney/Repos/adai && ./build/bin/metrics_api_server

# Custom port
cd /home/rodney/Repos/adai && ./build/bin/metrics_api_server --port 9090

# Poll current metrics
curl http://localhost:8081/api/metrics/current

# List tracked sessions
curl http://localhost:8081/api/sessions

# Aggregated summary
curl http://localhost:8081/api/metrics/summary
```

## Run Chatbot

The CLI chatbot connects to the API server over HTTP. Start the API server first.

```bash
# Start the API server
cd /home/rodney/Repos/adai && ./build/src/chatbot_api_server -c config.conf

# Connect CLI client (default: http://localhost:8080)
cd /home/rodney/Repos/adai && ./build/bin/chatbot

# Connect to a non-default server
cd /home/rodney/Repos/adai && ./build/bin/chatbot http://localhost:9000

# Save conversation history to a file
cd /home/rodney/Repos/adai && ./build/bin/chatbot http://localhost:8080 my_conversation.txt
```

## Run GUI Chatbot

```bash
# Launch GUI (loads vocab.txt and chatbot_model.bin from current directory)
cd /home/rodney/Repos/adai && ./build/src/chatbot_gui

# Specify vocab and model explicitly
cd /home/rodney/Repos/adai && ./build/src/chatbot_gui vocab.txt chatbot_model.bin
```

## Model Service (script wrapper for API server)

```bash
# Start service (builds if needed, daemonises)
cd /home/rodney/Repos/adai && ./scripts/model_service.sh start

# Start with specific model and port
cd /home/rodney/Repos/adai && ./scripts/model_service.sh start --model models/chatbot_model.bin --port 9000

# Run in foreground (no daemonise)
cd /home/rodney/Repos/adai && ./scripts/model_service.sh start --foreground

# Stop / restart / status
cd /home/rodney/Repos/adai && ./scripts/model_service.sh stop
cd /home/rodney/Repos/adai && ./scripts/model_service.sh restart
cd /home/rodney/Repos/adai && ./scripts/model_service.sh status

# Health check and live logs
cd /home/rodney/Repos/adai && ./scripts/model_service.sh health
cd /home/rodney/Repos/adai && ./scripts/model_service.sh logs
```

## Full Workflow (New Model from Scratch)

```bash
cd /home/rodney/Repos/adai

# 1. Build
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && cd ..

# 2. Create vocabulary
./build/bin/vocab_builder --input sample_training_data.txt --output vocab.txt --vocab-size 5000 --format pairs --stats

# 3. Queue training data
./build/bin/dataset_manager add sample_training_data.txt

# 4. Initialize and train (runs in background)
./build/bin/incremental_trainer init
./build/bin/incremental_trainer train 25

# 5. Follow training log
tail -f chatbot_server.log

# 6. Start the API server and chat
./build/src/chatbot_api_server -c config.conf &
./build/bin/chatbot
```

## Aliases (Add to ~/.bashrc)

```bash
alias adai-build='cd /home/rodney/Repos/adai && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)'
alias adai-vocab='cd /home/rodney/Repos/adai && ./build/bin/vocab_builder'
alias adai-data='cd /home/rodney/Repos/adai && ./build/bin/dataset_manager'
alias adai-train='cd /home/rodney/Repos/adai && ./build/bin/incremental_trainer'
alias adai-metrics='cd /home/rodney/Repos/adai && ./build/bin/metrics_api_server'
alias adai-serve='cd /home/rodney/Repos/adai && ./scripts/model_service.sh'
alias adai-chat='cd /home/rodney/Repos/adai && ./build/bin/chatbot'
```

After adding aliases:

```bash
source ~/.bashrc
adai-build                          # Build the project
adai-data add my_data.txt           # Queue a training file
adai-train train 25                 # Train on pending data
adai-metrics                        # Start metrics daemon
adai-serve start                    # Start API server
adai-chat                           # Connect CLI client
```

## Quick Fixes

### Model generates gibberish?

```bash
# Verify the API server is running and healthy
curl http://localhost:8080/health

# Check model files are present
ls -lh chatbot_model.bin*
```

### Need to add more training data and retrain?

```bash
cd /home/rodney/Repos/adai
./build/bin/dataset_manager add new_data.txt
./build/bin/incremental_trainer train 15
tail -f chatbot_server.log
```

### Need a full retrain from scratch?

```bash
cd /home/rodney/Repos/adai
./build/bin/incremental_trainer retrain 25
tail -f chatbot_server.log
```

### Loss not improving / high perplexity?

Lower `LEARNING_RATE` in `config.conf`, then retrain:

```ini
LEARNING_RATE=0.0001
GRADIENT_CLIP=0.5
```

```bash
./build/bin/incremental_trainer retrain 25
```

### Rebuild everything from scratch?

```bash
cd /home/rodney/Repos/adai && rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
```
