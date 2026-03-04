#!/bin/bash

# Configuration
SERVER_HOST="localhost"
SERVER_PORT="8080"
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
# Determine directories relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${ROOT_DIR}/build"
CLIENT_BIN="${BUILD_DIR}/src/chatbot"
SERVER_BIN="${BUILD_DIR}/src/chatbot_api_server"
CONFIG_FILE="${ROOT_DIR}/config.conf"
LOG_FILE="${ROOT_DIR}/chatbot_server.log"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🤖 ADAI Chatbot Launcher${NC}"
echo "=================================================="

# Check if client binary exists
if [ ! -f "$CLIENT_BIN" ]; then
    echo -e "${RED}Error: Chatbot client binary not found at:${NC}"
    echo "  $CLIENT_BIN"
    echo "Please build the project first:"
    echo "  cd build && cmake .. && make chatbot"
    exit 1
fi

# Function to check if server is running
check_server() {
    if command -v curl >/dev/null 2>&1; then
        curl -s -f "${SERVER_URL}/health" >/dev/null 2>&1
        return $?
    else
        echo -e "${YELLOW}Warning: curl not found, assuming server might be down.${NC}"
        return 1
    fi
}

# Check server status
echo -n "Checking server status at ${SERVER_URL}... "
if check_server; then
    echo -e "${GREEN}Running${NC}"
else
    echo -e "${YELLOW}Not running${NC}"

    # Check if server binary exists
    if [ ! -f "$SERVER_BIN" ]; then
        echo -e "${RED}Error: Chatbot server binary not found at:${NC}"
        echo "  $SERVER_BIN"
        echo "Please build the server:"
        echo "  cd build && cmake .. && make chatbot_api_server"
        exit 1
    fi

    # Start server
    echo -e "${YELLOW}Starting server...${NC}"
    
    SERVER_CMD="$SERVER_BIN"
    if [ -f "$CONFIG_FILE" ]; then
        echo "Using configuration: $CONFIG_FILE"
        # Force small CPU-friendly configuration to prevent timeouts during testing
        # echo "Forcing CPU-optimized parameters (override config file)..."
        # SERVER_CMD="$SERVER_CMD --config $CONFIG_FILE --d-model 64 --num-heads 4 --d-ff 256 --enc-layers 2 --dec-layers 2 --max-seq-len 128"
    else
        echo "Warning: config.conf not found, using defaults"
        SERVER_CMD="$SERVER_CMD --d-model 64 --num-heads 4 --d-ff 256 --enc-layers 2 --dec-layers 2 --max-seq-len 128"
    fi
    
    # Run in background, redirect logs
    nohup $SERVER_CMD > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    
    echo "Server process started (PID: $SERVER_PID)"
    echo "Logs redirecting to: $LOG_FILE"
    
    # Wait for server to become ready
    echo "Waiting for server to initialize (this may take a few minutes)..."
    MAX_RETRIES=300  # 5 minutes timeout
    COUNT=0
    SERVER_READY=false
    
    # Store previous log line count to only show new lines
    LAST_LOG_LINE=0
    
    while [ $COUNT -lt $MAX_RETRIES ]; do
        sleep 1
        
        # Show new logs while waiting
        CURRENT_LOG_LINE=$(wc -l < "$LOG_FILE")
        if [ "$CURRENT_LOG_LINE" -gt "$LAST_LOG_LINE" ]; then
            tail -n $((CURRENT_LOG_LINE - LAST_LOG_LINE)) "$LOG_FILE" | grep -v "^$" | head -n 5
            LAST_LOG_LINE=$CURRENT_LOG_LINE
        fi
        
        if check_server; then
            SERVER_READY=true
            break
        fi
        # Only print dot if no logs were shown to avoid clutter
        if [ "$CURRENT_LOG_LINE" -eq "$LAST_LOG_LINE" ]; then
             echo -n "."
        fi
        COUNT=$((COUNT+1))
    done
    echo ""
    
    if [ "$SERVER_READY" = true ]; then
        echo -e "${GREEN}Server is ready!${NC}"
    else
        echo -e "${RED}Server failed to start within $MAX_RETRIES seconds.${NC}"
        echo "Check $LOG_FILE for details."
        
        # Check if process is still running
        if ps -p $SERVER_PID > /dev/null; then
             echo "Process is still running but unresponsive/initializing. Killing it."
             kill $SERVER_PID
        fi
        exit 1
    fi
    
    echo -e "${YELLOW}Note: Server will keep running in background (PID $SERVER_PID).${NC}"
    echo "To stop it later: kill $SERVER_PID"
fi

echo "=================================================="
echo -e "${GREEN}Starting Chatbot Client...${NC}"
echo "=================================================="

# Run client
"$CLIENT_BIN" "$SERVER_URL"
