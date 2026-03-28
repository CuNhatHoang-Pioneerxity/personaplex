#!/bin/bash
# RunPod All-in-One Setup Script
# Run this script from the PersonaPlex root directory
# Sets up PersonaPlex client + Moshi server + Alternative server (Whisper+Kokoro+Ollama)

set -e

echo "=========================================="
echo "PersonaPlex Complete Setup - RunPod"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
export DEBIAN_FRONTEND=noninteractive
OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.2}
WHISPER_MODEL=${WHISPER_MODEL:-small}
HF_TOKEN=${HF_TOKEN:-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}

echo "Script Dir: $SCRIPT_DIR"
echo "Ollama Model: $OLLAMA_MODEL"
echo "Whisper Model: $WHISPER_MODEL"

# ==========================================
# STEP 1: System Dependencies
# ==========================================
echo ""
echo "[1/6] Installing system dependencies..."
apt-get update
apt-get install -y \
    git curl wget ffmpeg \
    libopus-dev libsndfile1 portaudio19-dev \
    build-essential zstd \
    python3.11 python3.11-venv python3-pip

# ==========================================
# STEP 2: Install NVM + Node
# ==========================================
echo ""
echo "[2/6] Installing NVM and Node.js..."
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm install --lts
nvm use --lts

# ==========================================
# STEP 3: UV Package Manager
# ==========================================
echo ""
echo "[3/6] Installing UV..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# ==========================================
# STEP 4: Install Moshi + Build Client
# ==========================================
echo ""
echo "[4/6] Installing Moshi and building client..."
cd "$SCRIPT_DIR/moshi"
pip install -e .

cd "$SCRIPT_DIR/client"
npm install
npm run build

# ==========================================
# STEP 5: Ollama (for Alternative)
# ==========================================
echo ""
echo "[5/6] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
sleep 10
ollama pull "$OLLAMA_MODEL"

# ==========================================
# STEP 6: Setup Alternative Server
# ==========================================
echo ""
echo "[6/6] Setting up Alternative server..."
cd "$SCRIPT_DIR/alternative_server"
uv venv "$SCRIPT_DIR/.venv-alternative"
source "$SCRIPT_DIR/.venv-alternative/bin/activate"
uv sync
deactivate

# ==========================================
# Done
# ==========================================

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start:"
echo "  ./start_both.sh        # Start both backends"
echo "  ./start_moshi.sh       # Moshi only (port 8998)"
echo "  ./start_alternative.sh # Alternative only (port 8999)"
echo ""
echo "Client: http://<pod-ip>:8998"
