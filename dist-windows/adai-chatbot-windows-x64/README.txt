ADAI Chatbot - Windows Edition
================================

This package contains Windows native executables for the ADAI transformer-based chatbot.

Contents:
---------
  chatbot.exe          - Interactive chatbot CLI
  chatbot_trainer.exe  - Model training tool
  vocab.txt            - Vocabulary file
  sample_training_data.txt - Sample training data
  chatbot_model.bin*   - Pre-trained model files (if included)

Quick Start:
------------
1. Open Command Prompt or PowerShell
2. Navigate to this directory
3. Run the chatbot:

   chatbot.exe

   Or with custom files:
   
   chatbot.exe vocab.txt chatbot_model.bin

Training a Model:
-----------------
To train a new model:

   chatbot_trainer.exe --data sample_training_data.txt --vocab vocab.txt --output my_model.bin

For more options:

   chatbot_trainer.exe --help

Interactive Commands:
---------------------
Once the chatbot is running, you can use these commands:

  /help         - Show available commands
  /save         - Save conversation
  /load         - Load conversation
  /stats        - Show conversation statistics
  /settings     - Show current settings
  /set <param>  - Change generation parameter
  /exit         - Exit the chatbot

Generation Strategies:
----------------------
  greedy   - Always pick most likely token
  beam     - Beam search (multiple hypotheses)
  sampling - Random sampling
  top-k    - Sample from top K tokens
  nucleus  - Sample from top P probability mass

Examples:
---------
Change strategy:
  /set strategy greedy

Adjust temperature:
  /set temperature 0.7

Set max response length:
  /set length 150

Requirements:
-------------
- Windows 7 or later (64-bit)
- No additional dependencies required (statically linked)

Troubleshooting:
----------------
If the executable won't run:
1. Make sure you're using 64-bit Windows
2. Try running from Command Prompt to see error messages
3. Check that all .bin files are in the same directory

For more information, visit:
  https://github.com/rjv717/adai

Built with MinGW-w64 cross-compiler
