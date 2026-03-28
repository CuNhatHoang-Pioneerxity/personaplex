#!/bin/bash
# Start Alternative Server (Whisper + Kokoro + Ollama - Vietnamese support)
# Runs on port 8999 (Moshi uses 8998)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure UV is in PATH
export PATH="$HOME/.local/bin:$PATH"

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
    --device cuda
