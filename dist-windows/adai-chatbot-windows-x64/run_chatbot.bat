@echo off
echo ========================================
echo ADAI Chatbot Launcher
echo ========================================
echo.

REM Check if chatbot.exe exists
if not exist chatbot.exe (
    echo ERROR: chatbot.exe not found!
    echo Please make sure you're running this from the correct directory.
    pause
    exit /b 1
)

REM Check if vocab file exists
if not exist vocab.txt (
    echo WARNING: vocab.txt not found!
    echo The chatbot may not work without a vocabulary file.
    echo.
)

REM Check if model exists
if not exist chatbot_model.bin (
    echo WARNING: chatbot_model.bin not found!
    echo You may need to train a model first using chatbot_trainer.exe
    echo.
)

echo Starting chatbot...
echo.
chatbot.exe

pause
