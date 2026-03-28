#!/bin/bash
# Start Alternative Server (Whisper + Kokoro + Ollama - Vietnamese support)
# Runs on port 8999 (Moshi uses 8998)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure UV is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Check for UV
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed."
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Then reload your shell or run: source ~/.bashrc"
    exit 1
fi

# Build client if needed
if [ ! -d "$SCRIPT_DIR/alternative_server/client/dist" ]; then
    echo "Building client..."
    cd "$SCRIPT_DIR/alternative_server/client"
    npm install
    npm run build
fi

cd "$SCRIPT_DIR/alternative_server"

# Create venv if it doesn't exist
if [ ! -d "$SCRIPT_DIR/.venv-alternative" ]; then
    echo "Creating virtual environment..."
    uv venv "$SCRIPT_DIR/.venv-alternative"
fi

source "$SCRIPT_DIR/.venv-alternative/bin/activate"

# Install dependencies
echo "Installing Python dependencies..."
uv sync

# Check for Ollama (optional - can use remote)
if ! command -v ollama &> /dev/null; then
    echo "WARNING: ollama is not installed locally."
    echo "You can install it from: https://ollama.ai"
    echo "Or set OLLAMA_URL to a remote server."
fi

# Ensure Ollama is running (if installed)
if command -v ollama &> /dev/null; then
    if ! pgrep -x "ollama" > /dev/null; then
        echo "Starting Ollama..."
        ollama serve &
        sleep 5
    fi
fi

uv run alternative-server \
    --host 0.0.0.0 \
    --port 8999 \
    --static "$SCRIPT_DIR/alternative_server/client/dist" \
    --ollama-url ${OLLAMA_URL:-http://localhost:11434} \
    --model ${OLLAMA_MODEL:-llama3.2} \
    --whisper-model ${WHISPER_MODEL:-small} \
    --device cuda
