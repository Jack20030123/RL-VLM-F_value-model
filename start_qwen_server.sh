#!/bin/bash
#
# Start Qwen3-VL HTTP Inference Server
#
# This script starts the Qwen HTTP server in the qwen conda environment.
# The server must be running before starting rlvlmf training with Qwen VLM.
#
# Usage:
#   ./start_qwen_server.sh [--preload] [--host HOST] [--port PORT] [--model MODEL]
#
# Options:
#   --preload      Preload the model at startup (recommended, takes ~200s for 8B, ~400s for 32B)
#   --host HOST    Host to bind (default: 127.0.0.1)
#                  For cross-node, use 0.0.0.0
#   --port PORT    Port number (default: 8000)
#   --model MODEL  Model name: 8B or 32B (default: 8B)
#                  8B = Qwen/Qwen3-VL-8B-Instruct
#                  32B = Qwen/Qwen3-VL-32B-Instruct
#
# Examples:
#   ./start_qwen_server.sh --preload --host 127.0.0.1 --port 8000 --model 8B
#   ./start_qwen_server.sh --preload --host 0.0.0.0   --port 8000 --model 32B
#

# Default values
HOST="127.0.0.1"
PORT=8000
PRELOAD=""
MODEL="8B"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --preload)
            PRELOAD="--preload"
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--preload] [--host HOST] [--port PORT] [--model MODEL]"
            exit 1
            ;;
    esac
done

# Convert short model name to full HuggingFace model name
case $MODEL in
    8B|8b)
        MODEL_FULL="Qwen/Qwen3-VL-8B-Instruct"
        ;;
    32B|32b)
        MODEL_FULL="Qwen/Qwen3-VL-32B-Instruct"
        ;;
    *)
        # Assume it's already a full model name (e.g., Qwen/Qwen3-VL-8B-Instruct)
        MODEL_FULL="$MODEL"
        ;;
esac

echo "=================================================="
echo "Starting Qwen3-VL HTTP Inference Server"
echo "=================================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Model: $MODEL_FULL"
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
python vlms/servers/qwen_server.py --host "$HOST" --port $PORT --model "$MODEL_FULL" $PRELOAD 2>&1 | tee "$LOG_FILE"
