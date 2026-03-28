#!/bin/bash
# RunPod Setup Script with Service Selection
# Usage: ./setup_runpod.sh [both|moshi|alt]
# Sets up PersonaPlex client + selected backend(s)

set -e

# Get service parameter
SERVICE=${1:-both}
case $SERVICE in
    both|moshi|alt)
        echo "Setting up for service: $SERVICE"
        ;;
    *)
        echo "Error: Invalid service '$SERVICE'"
        echo "Usage: $0 [both|moshi|alt]"
        echo "  both - Setup both Moshi and Alternative servers"
        echo "  moshi - Setup Moshi server only"
        echo "  alt   - Setup Alternative server only"
        exit 1
        ;;
esac

echo "=========================================="
echo "PersonaPlex Setup - RunPod ($SERVICE)"
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
echo "[1/7] Installing system dependencies..."
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
echo "[2/7] Installing NVM and Node.js..."
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm install --lts
nvm use --lts

# ==========================================
# STEP 3: UV Package Manager
# ==========================================
echo ""
echo "[3/7] Installing UV..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# ==========================================
# STEP 4: Build Client
# ==========================================
echo ""
echo "[4/7] Building client..."
cd "$SCRIPT_DIR/client"
npm install
npm run build

# ==========================================
# STEP 5: Setup Selected Services
# ==========================================
echo ""
echo "[5/7] Setting up selected services..."

# Setup Moshi if needed
if [[ "$SERVICE" == "both" || "$SERVICE" == "moshi" ]]; then
    echo "Setting up Moshi..."
    cd "$SCRIPT_DIR/moshi"
    pip install -e .
fi

# Setup Alternative if needed
if [[ "$SERVICE" == "both" || "$SERVICE" == "alt" ]]; then
    echo "Setting up Alternative server..."
    cd "$SCRIPT_DIR/alternative_server"
    uv venv "$SCRIPT_DIR/.venv-alternative"
    source "$SCRIPT_DIR/.venv-alternative/bin/activate"
    uv sync
    deactivate
fi

# ==========================================
# STEP 6: Ollama (for Alternative)
# ==========================================
if [[ "$SERVICE" == "both" || "$SERVICE" == "alt" ]]; then
    echo ""
    echo "[6/7] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve &
    sleep 10
    ollama pull "$OLLAMA_MODEL"
else
    echo ""
    echo "[6/7] Skipping Ollama (not needed for Moshi-only)..."
fi

# ==========================================
# STEP 7: Start Selected Service
# ==========================================
echo ""
echo "[7/7] Starting selected service..."

if [[ "$SERVICE" == "both" ]]; then
    echo "Starting both backends..."
    exec "$SCRIPT_DIR/start_both.sh"
elif [[ "$SERVICE" == "moshi" ]]; then
    echo "Starting Moshi backend..."
    exec "$SCRIPT_DIR/start_moshi.sh"
elif [[ "$SERVICE" == "alt" ]]; then
    echo "Starting Alternative backend..."
    exec "$SCRIPT_DIR/start_alternative.sh"
fi
