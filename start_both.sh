#!/bin/bash
# Start BOTH backends simultaneously
# Moshi: port 8998 (English)
# Alternative: port 8999 (Vietnamese + RAG)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure UV is in PATH
export PATH="$HOME/.local/bin:$PATH"

echo "Starting both backends..."

# Start Moshi in background
echo "Starting Moshi on port 8998..."
cd "$SCRIPT_DIR"
export HF_TOKEN=${HF_TOKEN:-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}
python3 -m moshi.server --host 0.0.0.0 --port 8998 --static ./client/dist &
MOSHI_PID=$!

# Start Alternative in background
echo "Starting Alternative on port 8999..."
cd "$SCRIPT_DIR/alternative_server"
source "$SCRIPT_DIR/.venv-alternative/bin/activate"

# Ensure Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &
    sleep 5
fi

uv run alternative-server \
    --host 0.0.0.0 \
    --port 8999 \
    --static "$SCRIPT_DIR/client/dist" \
    --ollama-url http://localhost:11434 \
    --model ${OLLAMA_MODEL:-llama3.2} \
    --whisper-model ${WHISPER_MODEL:-small} \
    --device cuda &
ALT_PID=$!

echo ""
echo "=========================================="
echo "Both backends running!"
echo "=========================================="
echo "Moshi (English):       http://<ip>:8998"
echo "Alternative (Viet):    http://<ip>:8999"
echo ""
echo "Client with selector:  http://<ip>:8998"
echo "=========================================="
echo ""
echo "PIDs: Moshi=$MOSHI_PID, Alternative=$ALT_PID"
echo "Stop with: kill $MOSHI_PID $ALT_PID"

# Wait for both
wait $MOSHI_PID $ALT_PID
