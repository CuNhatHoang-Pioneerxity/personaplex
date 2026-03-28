#!/bin/bash
# Start Alternative Server (Whisper + Kokoro + Ollama - Vietnamese support)
# Runs on port 8999 (Moshi uses 8998)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure UV and NVM are in PATH
export PATH="$HOME/.local/bin:$PATH"
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# Check for UV
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed."
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Then reload your shell or run: source ~/.bashrc"
    exit 1
fi

# Build client if needed (in the correct location for Alternative server)
CLIENT_DIST="$SCRIPT_DIR/alternative_server/client/dist"
if [ ! -d "$CLIENT_DIST" ]; then
    echo "Building client in alternative_server/client..."
    cd "$SCRIPT_DIR/alternative_server/client"
    if ! command -v npm &> /dev/null; then
        echo "ERROR: npm not found. Ensure Node.js is installed via NVM."
        echo "Run: source ~/.nvm/nvm.sh && nvm use --lts"
        exit 1
    fi
    npm install
    npm run build
    if [ ! -d "$CLIENT_DIST" ]; then
        echo "ERROR: Client build failed - dist directory not created"
        exit 1
    fi
    echo "Client built successfully at $CLIENT_DIST"
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
