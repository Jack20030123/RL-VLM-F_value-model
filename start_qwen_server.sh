#!/bin/bash
#
# Start Qwen3-VL HTTP Inference Server
#
# This script starts the Qwen HTTP server in the qwen conda environment.
# The server must be running before starting rlvlmf training with Qwen VLM.
#
# Usage:
#   ./start_qwen_server.sh [--preload] [--port PORT]
#
# Options:
#   --preload    Preload the model at startup (recommended, takes ~200s)
#   --port PORT  Port number (default: 8000)
#
# Example:
#   ./start_qwen_server.sh --preload --port 8000
#

# Default values
PORT=8000
PRELOAD=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --preload)
            PRELOAD="--preload"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--preload] [--port PORT]"
            exit 1
            ;;
    esac
done

echo "=================================================="
echo "Starting Qwen3-VL HTTP Inference Server"
echo "=================================================="
echo "Port: $PORT"
echo "Preload model: ${PRELOAD:-no}"
echo "=================================================="
echo ""

# Activate qwen environment and start server
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen

echo "Qwen environment activated"
echo "Python version: $(python --version)"
echo ""

# Check if Flask is installed
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Flask is not installed in the qwen environment"
    echo "Please install Flask: pip install flask"
    exit 1
fi

# Check if requests is installed (for testing)
python -c "import requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: requests library not found (needed for client calls)"
    echo "Install with: pip install requests"
fi

echo "Starting server..."
echo ""

# Create log directory if it doesn't exist
mkdir -p qwen_server_logs

# Generate log filename with timestamp
LOG_FILE="qwen_server_logs/qwen_server_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo ""

# Start the server and redirect output to log file
python vlms/servers/qwen_server.py --host 127.0.0.1 --port $PORT $PRELOAD 2>&1 | tee "$LOG_FILE"
